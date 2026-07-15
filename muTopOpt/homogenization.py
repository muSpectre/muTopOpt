#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
FFT-accelerated FE micromechanical homogenization on a regular grid.

:class:`Homogenization` owns the muGrid machinery -- the FFT engine (which
doubles as the ghosted domain decomposition), the fused
``IsotropicStiffnessOperator`` (the system matrix ``K``), the per-pixel Lamé
fields, and the J-FFT (Green-Jacobi) preconditioner -- and exposes the three
operations the optimizer needs:

* :meth:`solve_macro`   -- solve ``K(rho) u = -Bᵀ C:Ē`` for a macro strain ``Ē``;
* :meth:`solve_rhs`     -- solve ``K(rho) x = b`` for an arbitrary rhs (adjoint);
* :meth:`homogenized_stress` -- the cell-averaged stress ``⟨σ⟩``.

Everything is dimension-agnostic (``dim`` = 2 or 3): the only per-dimension
detail is which ``IsotropicStiffnessOperator{2,3}D`` to instantiate. Fields live
on the engine's real-space collection so the solver, the preconditioner FFTs and
the operators all share one ghosted, MPI-decomposed, (optionally) device-resident
layout.
"""

import os

import numpy as np

import muGrid
from muGrid.Preconditioners import (
    make_green_jacobi_preconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import ConvergenceError, conjugate_gradients

from .material import SimpMaterial

# muGrid's ``conjugate_gradients`` gained an optional ``residual`` out-field
# (the final r = b - Kx, needed for the adjoint-corrected objective) after
# 0.110. Detect it once so we work against released muGrid too, falling back
# to one extra operator apply when the argument is unavailable.
import inspect as _inspect

_CG_HAS_RESIDUAL = "residual" in _inspect.signature(
    conjugate_gradients).parameters


class _CGStagnation(Exception):
    """Internal control flow of :meth:`Homogenization.solve_rhs`: raised from
    the CG iteration callback when the solve has stopped making progress (the
    finite-precision residual floor), so the solve can be cut short and the
    best iterate returned instead of burning the full iteration budget."""


def _local_rank(comm):
    """Node-local rank of this process, used to pick a per-rank GPU.

    Prefers the launcher-provided local-rank env var (OpenMPI, MPICH/Hydra,
    MVAPICH, Slurm); falls back to the global communicator rank (correct on a
    single node, and the best available guess otherwise).
    """
    for var in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return comm.rank if comm is not None else 0


def _gpu_device_count():
    """Number of visible GPUs, or 1 if it cannot be determined."""
    try:
        import cupy

        return max(1, cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return 1


def _resolve_device(device, comm=None):
    """Map a user-facing device spec to a ``muGrid.Device`` (or ``None`` = CPU).

    Accepts ``None``/``"cpu"`` (host), ``"gpu"``/``"rocm"``/``"cuda"`` (an
    accelerator), an explicit ``"gpu:N"``/``"rocm:N"``/``"cuda:N"`` (a specific
    device by id), or a ready-made ``muGrid.Device``. The string forms are
    parsed by ``muGrid.Device.from_string`` (the inverse of a device's
    ``device_string``); muTopOpt adds only the MPI placement policy on top: a
    *bare* accelerator alias (no ``:N``) binds each rank to GPU
    ``local_rank % n_gpus`` so an MPI run spreads across the node's
    accelerators instead of oversubscribing device 0. An explicit id pins
    *every* rank to that one device; use it for single-rank runs or to override
    the automatic placement (e.g. ``"rocm:2"``).
    """
    if device is None:
        return None
    if not isinstance(device, str):
        return device  # assume a muGrid.Device instance
    if device == "cpu":
        return None
    # A bare accelerator alias gets muTopOpt's per-rank placement; rewrite it to
    # an explicit "<kind>:<id>". Everything else (explicit ids, bad spellings)
    # is handed verbatim to muGrid's parser.
    if device.lower() in ("gpu", "rocm", "cuda"):
        device = f"{device.lower()}:{_local_rank(comm) % _gpu_device_count()}"
    return muGrid.Device.from_string(device)


# Process-global guard: the managed allocator, and muGrid's routing to it, must
# be installed exactly once per process. Re-registering muGrid's allocator while
# device fields are alive would drop the keepalive of their buffers and free
# them out from under muGrid.
_MANAGED_ALLOCATOR_INSTALLED = False


def _enable_managed_device_allocator():
    """Route muGrid's device allocations (and cupy's) through a *managed*
    memory pool, returning ``True`` on success.

    On a unified-memory accelerator the default device allocator
    (raw ``hipMalloc``/``cudaMalloc``) is capped at the coarse-grained
    device-local window -- on an AMD MI300A APU that is ~half the 128 GB
    package (~62.8 GiB), even though the HBM is physically shared with the
    host. Managed allocations (``hipMallocManaged``/``cudaMallocManaged``) draw
    from the whole unified pool instead, so device fields can span the full
    HBM. Routing cupy's default pool through managed memory and then pointing
    muGrid at that pool (``use_cupy_allocator``) puts *all* device memory --
    fields, CG scratch, the preconditioner -- on the managed path.

    Installed once per process (idempotent); a no-op returning ``False`` when
    cupy is unavailable."""
    global _MANAGED_ALLOCATOR_INSTALLED
    if _MANAGED_ALLOCATOR_INSTALLED:
        return True
    try:
        import cupy
    except Exception:
        return False
    cupy.cuda.set_allocator(
        cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
    muGrid.use_cupy_allocator()
    _MANAGED_ALLOCATOR_INSTALLED = True
    return True


class Homogenization:
    def __init__(
        self,
        nb_grid_pts,
        material: SimpMaterial,
        comm=None,
        domain_lengths=None,
        element="q1",
        preconditioner="green-jacobi",
        cg_tol=1e-8,
        cg_maxiter=2000,
        cg_verbose=False,
        device=None,
        dtype=np.float64,
        managed_memory=None,
    ):
        """``dtype`` (``np.float64`` default, or ``np.float32``) is the
        precision of all grid fields and hence of the forward/adjoint solves,
        the preconditioner FFTs and the sensitivity kernels. The outer
        optimizer and the host-side reductions stay in double precision.

        ``managed_memory`` controls whether GPU device memory is drawn from a
        *managed* (unified) pool rather than the default coarse-grained
        ``hipMalloc``/``cudaMalloc`` window. On a unified-memory APU (e.g. AMD
        MI300A) the default window is only part of the physical HBM (~62.8 GiB
        of 128 GB on MI300A), so managed memory is what lets large grids (e.g.
        512^3 in double precision) use the full package. ``None`` (default)
        enables it automatically whenever a GPU device is selected; pass
        ``False`` (or set ``MUTOPOPT_MANAGED=0``) to force the default
        allocator. Ignored on CPU."""
        self.dim = len(nb_grid_pts)
        if self.dim not in (2, 3):
            raise ValueError("nb_grid_pts must be 2- or 3-dimensional")
        # Scalar precision of the on-grid fields. muGrid's FFTEngine and fused
        # operators dispatch on the field dtype, so allocating every field at
        # this dtype runs the whole forward/adjoint/sensitivity path in single
        # or double precision. The host-side optimizer stays in float64.
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError(
                f"dtype must be float64 or float32, got {self.dtype}")
        self.nb_grid_pts = tuple(int(n) for n in nb_grid_pts)
        self.material = material
        self.comm = comm if comm is not None else muGrid.Communicator()

        if domain_lengths is None:
            domain_lengths = [1.0] * self.dim
        self.domain_lengths = [float(x) for x in domain_lengths]
        self.grid_spacing = [
            L / n for L, n in zip(self.domain_lengths, self.nb_grid_pts)
        ]
        self.domain_volume = float(np.prod(self.domain_lengths))

        self.device = _resolve_device(device, self.comm)
        # On a GPU, default to the managed (unified) allocator so device fields
        # can use the full HBM of a unified-memory APU rather than the smaller
        # coarse-grained window (see _enable_managed_device_allocator). Must be
        # installed before the engine allocates any device field. Opt out with
        # managed_memory=False or MUTOPOPT_MANAGED=0.
        if managed_memory is None:
            managed_memory = os.environ.get("MUTOPOPT_MANAGED", "1") != "0"
        self.managed_memory = False
        if self.device is not None and managed_memory:
            self.managed_memory = _enable_managed_device_allocator()
        ghosts = (1,) * self.dim
        self.engine = muGrid.FFTEngine(
            self.nb_grid_pts, self.comm,
            nb_ghosts_left=ghosts, nb_ghosts_right=ghosts,
            device=self.device,
        )
        self.fc = self.engine.real_space_collection

        elem = (
            muGrid.FEMElement.q1 if element == "q1" else muGrid.FEMElement.p1
        )
        self.element = elem  # muGrid.FEMElement, shared with the regularization
        self.element_name = element
        OpCls = (
            muGrid.IsotropicStiffnessOperator2D
            if self.dim == 2
            else muGrid.IsotropicStiffnessOperator3D
        )
        self.op = OpCls(self.grid_spacing, elem)

        # Per-pixel material (scalar fields) and solver scratch.
        self.lam = self.fc.real_field("to_lambda", dtype=self.dtype)
        self.mu = self.fc.real_field("to_mu", dtype=self.dtype)
        self._rhs = self.fc.real_field("to_rhs", (self.dim,), dtype=self.dtype)
        self._Ku = self.fc.real_field("to_Ku", (self.dim,), dtype=self.dtype)

        # Array module of the (possibly device-resident) fields: cupy on a GPU
        # device, numpy on host. `on_device` gates the host<->device copies at
        # the optimizer boundary.
        self.on_device = "cupy" in type(self.lam.p).__module__
        if self.on_device:
            import cupy as _cp

            self._xp = _cp
        else:
            self._xp = np

        self.preconditioner_kind = preconditioner
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        # Stagnation safeguard of the CG solves (see solve_rhs): a solve whose
        # residual has not improved by at least `cg_stagnation_rel` (relative)
        # within the last `cg_stagnation_patience` iterations has hit its
        # finite-precision floor (or is diverging from it -- float32 CG on a
        # round-off-level rhs can blow the residual up by orders of magnitude
        # before the iteration cap); it is aborted and the best iterate seen so
        # far is returned, with the true residual recomputed. Preconditioned
        # solves here converge in O(100) iterations, so 100 dead iterations
        # reliably means the floor, not a plateau.
        #
        # A *cold-started* solve that has never improved on its initial
        # residual |b| is aborted much sooner -- after
        # `cg_no_progress_patience` iterations, or immediately once the
        # residual diverges `cg_divergence_factor`x (in the norm) above it.
        # That is the signature of a round-off-level rhs (float32 above all):
        # the very first CG step is rounding garbage (|r| jumps by ~1e6 on a
        # semi-definite operator) and the solve never gets back below |b|, so
        # waiting out the full patience wastes ~100 iterations per solve --
        # and the correct answer at that precision is x = 0 anyway. A healthy
        # cold-started solve improves on |b| within the first few iterations;
        # warm-started solves are exempt (a good initial guess makes early
        # residual excursions above the small r0 legitimate).
        self.cg_stagnation_patience = 100
        self.cg_stagnation_rel = 1e-2
        self.cg_no_progress_patience = 25
        self.cg_divergence_factor = 1e3
        # Number of solves cut short by the safeguard (diagnostic).
        self.cg_stagnation_count = 0
        self._x_best = self.fc.real_field(
            "to_cg_best", (self.dim,), dtype=self.dtype)
        # When True, print a live per-CG-iteration residual on rank 0 during
        # every solve, so convergence can be watched as it happens (opt-in;
        # verbose over a full optimization). See ``solve_rhs``.
        self.cg_verbose = cg_verbose
        self._prec = None
        self._nb_pixels = tuple(self.engine.nb_subdomain_grid_pts)

    # -- geometry helpers ---------------------------------------------------
    @property
    def nb_pixels(self):
        """Local (owned) pixel grid shape -- the shape of an element-wise
        density array on this rank."""
        return self._nb_pixels

    def to_host(self, a):
        """Return ``a`` (a field ``.p`` view or array) as a host NumPy array,
        copying off the device if necessary."""
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def to_device(self, a):
        """Return ``a`` as an array in the fields' module (cupy on device, numpy
        on host) at the grid precision, so it can be assigned into a field's
        ``.p``/``.s`` view."""
        return self._xp.asarray(a, dtype=self.dtype)

    def scalar_field(self, name):
        return self.fc.real_field(name, dtype=self.dtype)

    def vector_field(self, name):
        return self.fc.real_field(name, (self.dim,), dtype=self.dtype)

    # -- material update ----------------------------------------------------
    def set_density(self, rho):
        """Interpolate rho -> (lambda, mu), fill ghosts, and (re)build/refresh
        the preconditioner. ``rho`` has shape :attr:`nb_pixels`."""
        lam, mu = self.material.lame(np.asarray(rho))
        self.lam.p[...] = self.to_device(lam)
        self.mu.p[...] = self.to_device(mu)
        self.engine.communicate_ghosts(self.lam)
        self.engine.communicate_ghosts(self.mu)
        # Stiffness scale (max |λ|, |2μ|), reduced across ranks, used to detect a
        # negligible (round-off-only) right-hand side.
        local = max(float(np.abs(lam).max()), float(2.0 * np.abs(mu).max()))
        self._mat_scale = self.comm.max(local)
        self._update_preconditioner()

    def _reference_lame(self):
        # Global (MPI-reduced) means -> a rank-consistent reference material.
        n = self.comm.sum(int(self.lam.p.size))
        lam_ref = self.comm.sum(float(self._xp.sum(self.lam.p))) / n
        mu_ref = self.comm.sum(float(self._xp.sum(self.mu.p))) / n
        return lam_ref, mu_ref

    def _update_preconditioner(self):
        if self.preconditioner_kind is None:
            self._prec = None
            return
        if self._prec is None:
            lam_ref, mu_ref = self._reference_lame()
            # ``dtype=self.dtype`` builds a preconditioner at the field
            # precision, so a single-precision (float32) material yields a
            # single-precision preconditioner (complex64 FFT buffers, float32
            # kernel/diagonal) and the whole forward/adjoint solve stays in
            # single precision. Requires muGrid that threads dtype through the
            # preconditioner factories (see the muGrid pin in pyproject.toml).
            if self.preconditioner_kind == "green-jacobi":
                self._prec = make_green_jacobi_preconditioner(
                    self.engine, self.op, self.lam, self.mu, self.dim,
                    reference_lambda=lam_ref, reference_mu=mu_ref,
                    dtype=self.dtype,
                )
            elif self.preconditioner_kind == "green":

                def apply_ref(u, f):
                    self.engine.communicate_ghosts(u)
                    self.op.apply_uniform(u, lam_ref, mu_ref, f)

                self._prec = make_reference_stiffness_preconditioner(
                    self.engine, apply_ref, self.dim, dtype=self.dtype
                )
            else:
                raise ValueError(
                    f"unknown preconditioner '{self.preconditioner_kind}'"
                )
        elif self.preconditioner_kind == "green-jacobi":
            # Reuse the (reference) Green part; refresh only the Jacobi diagonal
            # from the updated material.
            self._prec.refresh()

    # -- solves -------------------------------------------------------------
    def _hessp(self, u, Au):
        self.engine.communicate_ghosts(u)
        self.op.apply(u, self.lam, self.mu, Au)

    def solve_rhs(self, b, x, rtol=None, maxiter=None, rhs_scale=None,
                  residual=None, label=None, warm_start=False):
        """Solve ``K x = b`` in place; returns ``x``.

        A negligible right-hand side (e.g. a spatially uniform material, whose
        macro response needs no fluctuation, giving a round-off-level rhs) has
        the exact solution ``x = 0``; return it directly, since CG would divide
        by zero on the first step. ``rhs_scale`` sets the (absolute) force scale
        below which the rhs counts as zero; when omitted the material scale is
        used.

        If ``residual`` is a field, the final CG residual ``r = b - K x`` is
        copied into it (used for the adjoint-corrected objective).

        ``label`` tags this solve in the live convergence trace emitted when
        ``self.cg_verbose`` is set (e.g. ``"case 1 fwd"``).

        ``warm_start`` uses the current contents of ``x`` as the CG initial
        guess instead of zeroing it. For a sequence of solves with the same
        operator and slowly-varying right-hand side (e.g. the Hessian-vector
        products across trust-region CG iterations) this cuts the iteration
        count; the previous solution must already be in ``x``.

        This method never raises on non-convergence. A solve that stagnates
        (no relative residual improvement of :attr:`cg_stagnation_rel` within
        :attr:`cg_stagnation_patience` iterations -- the finite-precision
        residual floor), hits the iteration cap, or breaks down returns the
        *best iterate* seen, with its true residual ``b - K x`` recomputed
        (and never worse than the zero solution). The callers tolerate a
        less-accurate solve: the consistent objective is second-order in the
        state-solve residual and reports the remaining error, and the
        trust-region model only needs a bounded Hessian-vector product."""
        if not warm_start:
            x.set_zero()
        bp = b.p.ravel()
        b_norm = np.sqrt(self.comm.sum(float(self._xp.dot(bp, bp))))
        scale = rhs_scale if rhs_scale is not None else getattr(
            self, "_mat_scale", 1.0)
        verbose = self.cg_verbose and self.comm.rank == 0
        tag = f"{label}  " if label else ""
        # Negligible-rhs threshold, relative to the natural force scale of this
        # solve. The historic 1e-9 is calibrated for float64; in float32 the
        # assembly round-off alone is ~eps*scale ~ 1e-7*scale, so a rhs below a
        # few eps is indistinguishable from noise (and CG on pure noise
        # diverges -- the operator is only semi-definite for void_ratio=0).
        negligible = max(1e-9, 10.0 * float(np.finfo(self.dtype).eps)) * scale
        if b_norm <= negligible:
            # Exact solution x = 0; no CG iterations performed. The residual
            # of x = 0 is b itself (round-off-level by construction). Zero even
            # under warm_start: a negligible rhs really does mean x = 0.
            x.set_zero()
            if residual is not None:
                residual.s[...] = b.s
            self.last_cg_iters = 0
            if verbose:
                print(f"    cg-iter    0  {tag}skipped (negligible rhs, x=0)",
                      flush=True)
            return x
        cold_start = True
        if warm_start:
            # Guard the warm start: an initial guess left over from a previous
            # solve (a different Steihaug direction, or -- worse -- a previous
            # outer iterate whose material was less ill-conditioned) can be far
            # *worse* than zero. In the high-contrast SIMP regime ``K⁻¹`` has
            # huge norm, so a stale guess can give ``|b - K x₀| ≫ |b|``; CG then
            # cannot recover within any iteration budget. Keep the guess only
            # if it beats a cold start (``|b - K x₀| < |b|``), else zero it.
            # One extra operator apply -- negligible against the CG it saves.
            self._hessp(x, self._Ku)
            diff = bp - self._Ku.p.ravel()
            warm_norm = np.sqrt(self.comm.sum(float(self._xp.dot(diff, diff))))
            if not (warm_norm < b_norm):
                x.set_zero()
                if verbose:
                    print(f"    cg-iter    0  {tag}warm start rejected "
                          f"(|r0|/|b|={warm_norm / b_norm:.2e}); cold start",
                          flush=True)
            else:
                cold_start = False
        # Count CG iterations via the solver callback (fires once per
        # iteration). When verbose, the same callback prints one line per CG
        # iteration so the convergence of the inner solve can be watched step
        # by step. ``rr`` is the globally reduced squared residual ||r||^2; CG
        # converges at ||r|| <= rtol*||b||.
        #
        # The callback doubles as the stagnation safeguard: it snapshots the
        # best iterate seen so far and aborts the solve (via _CGStagnation)
        # once the residual has not beaten the reference by at least
        # ``cg_stagnation_rel`` for ``cg_stagnation_patience`` consecutive
        # iterations. In finite precision -- float32 above all -- the *true*
        # residual has a floor; past it the recursive CG residual stagnates or
        # diverges (on a round-off-level rhs it can grow by orders of
        # magnitude), and without the safeguard the solve burns its full
        # iteration budget and raises. ``guard['best']``/``guard['ref']`` are
        # squared norms; ``rel`` compares in the norm, hence the square.
        counter = {"n": 0}
        rtol_eff = self.cg_tol if rtol is None else rtol
        stall = (1.0 - self.cg_stagnation_rel) ** 2
        diverge = self.cg_divergence_factor ** 2
        guard = {"best": np.inf, "best_iter": 0, "ref": np.inf, "ref_iter": 0,
                 "saved": False}

        def _count(iteration, state):
            counter["n"] += 1
            rr = float(state["rr"])
            if verbose:
                res = np.sqrt(rr)
                rel = res / b_norm if b_norm > 0 else 0.0
                print(f"    cg-iter {iteration:4d}  {tag}|r|={res:.3e}  "
                      f"|r|/|b|={rel:.2e}  (rtol={rtol_eff:.1e})", flush=True)
            if rr < guard["best"]:
                guard["best"] = rr
                guard["best_iter"] = iteration
                self._x_best.s[...] = x.s
                guard["saved"] = True
            elif cold_start and guard["best_iter"] == 0 and (
                    iteration >= self.cg_no_progress_patience
                    or rr > diverge * guard["best"]):
                # A cold start that has never improved on |r0| = |b|: the rhs
                # is round-off noise and the solve cannot progress. Only ever
                # fires on cold starts -- a warm start's small r0 makes early
                # excursions above it legitimate.
                raise _CGStagnation()
            if rr < stall * guard["ref"]:
                guard["ref"] = rr
                guard["ref_iter"] = iteration
            elif iteration - guard["ref_iter"] >= self.cg_stagnation_patience:
                raise _CGStagnation()

        cg_kwargs = {}
        if residual is not None and _CG_HAS_RESIDUAL:
            cg_kwargs["residual"] = residual
        try:
            conjugate_gradients(
                self.comm, self.fc, b, x,
                hessp=self._hessp, prec=self._prec,
                rtol=rtol_eff,
                maxiter=self.cg_maxiter if maxiter is None else maxiter,
                callback=_count,
                **cg_kwargs,
            )
        except (_CGStagnation, ConvergenceError) as err:
            # Truncated solve (stagnation, iteration cap, or NaN breakdown):
            # fall back to the best iterate instead of failing the whole
            # optimization. The callers tolerate a solve that is merely less
            # accurate than requested -- the consistent (Lagrangian) objective
            # is second-order in the state-solve residual and reports the
            # remaining error through `corrections`, and the trust-region
            # model only needs a bounded Hessian-vector product -- whereas a
            # raised ConvergenceError kills the run.
            if not guard["saved"]:
                raise  # NaN before the first callback; nothing to salvage
            x.s[...] = self._x_best.s
            # True (non-recursive) residual of the returned iterate; the best
            # recursive rr can be far below it past the precision floor.
            self._hessp(x, self._Ku)
            self._Ku.s[...] = b.s - self._Ku.s
            rp = self._Ku.p.ravel()
            true_norm = np.sqrt(self.comm.sum(float(self._xp.dot(rp, rp))))
            if not (true_norm < b_norm):
                # Worse than doing nothing: the zero solution (residual b) is
                # the safest answer for a rhs this close to round-off.
                x.set_zero()
                self._Ku.s[...] = b.s
                true_norm = b_norm
            if residual is not None:
                residual.s[...] = self._Ku.s
            self.last_cg_iters = counter["n"]
            self.cg_stagnation_count += 1
            if verbose:
                why = ("stagnated" if isinstance(err, _CGStagnation)
                       else str(err))
                print(f"    cg {tag}{why} after {counter['n']} iterations; "
                      f"accepted best iterate at |r|/|b|="
                      f"{true_norm / b_norm:.2e} (target {rtol_eff:.1e})",
                      flush=True)
            return x
        if residual is not None and not _CG_HAS_RESIDUAL:
            # Released muGrid without the residual out-field: recover
            # r = b - K x with one extra operator apply.
            self._hessp(x, self._Ku)
            residual.s[...] = b.s - self._Ku.s
        self.last_cg_iters = counter["n"]
        return x

    def solve_macro(self, E_macro, x, rtol=None, maxiter=None, residual=None,
                    label=None):
        """Solve the periodic homogenization problem ``K u = -Bᵀ C:Ē`` for the
        fluctuation displacement ``u`` under macro strain ``Ē``."""
        E_arr = np.asarray(E_macro, dtype=float)
        E = list(E_arr.ravel())
        self.op.apply_macro_rhs(self.lam, self.mu, E, self._rhs)
        self._rhs.s[...] *= -1.0
        scale = getattr(self, "_mat_scale", 1.0) * max(
            float(np.abs(E_arr).max()), 1e-300)
        return self.solve_rhs(
            self._rhs, x, rtol=rtol, maxiter=maxiter, rhs_scale=scale,
            residual=residual, label=label)

    @property
    def mat_scale(self):
        """Stiffness force scale (max |λ|, |2μ|) of the current material."""
        return getattr(self, "_mat_scale", 1.0)

    def macro_rhs_tensor(self, tensor, out, scale=1.0):
        """Assemble ``out = scale * Bᵀ C : T`` for a constant symmetric tensor
        ``T`` (used to build the adjoint right-hand side)."""
        T = list(np.asarray(tensor, dtype=float).ravel())
        self.op.apply_macro_rhs(self.lam, self.mu, T, out)
        if scale != 1.0:
            out.s[...] *= scale
        return out

    def homogenized_stress(self, u, E_macro, lam=None, mu=None):
        """Cell-averaged stress ``⟨σ⟩ = (1/V) ∫ C:(Ē + ∇u) dV`` as a
        ``dim x dim`` array (MPI-reduced).

        ``lam``/``mu`` override the material fields; passing *perturbed* Lamé
        fields ``dλ/dρ·v``/``dμ/dρ·v`` evaluates the material-derivative
        stress ``(1/V) ∫ δC:(Ē + ∇u) dV`` (the operator is linear in the
        material), as needed for Hessian-vector products."""
        E = list(np.asarray(E_macro, dtype=float).ravel())
        local = self.op.average_stress(
            u, self.lam if lam is None else lam,
            self.mu if mu is None else mu, E)
        glob = np.array([self.comm.sum(float(v)) for v in local])
        return glob.reshape(self.dim, self.dim) / self.domain_volume
