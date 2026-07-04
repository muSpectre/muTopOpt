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
from muGrid.Solvers import conjugate_gradients

from .material import SimpMaterial

# muGrid's ``conjugate_gradients`` gained an optional ``residual`` out-field
# (the final r = b - Kx, needed for the adjoint-corrected objective) after
# 0.110. Detect it once so we work against released muGrid too, falling back
# to one extra operator apply when the argument is unavailable.
import inspect as _inspect

_CG_HAS_RESIDUAL = "residual" in _inspect.signature(
    conjugate_gradients).parameters


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
    accelerator), or a ready-made ``muGrid.Device``. For the generic strings,
    each rank binds to GPU ``local_rank % n_gpus`` so an MPI run spreads across
    the node's accelerators instead of oversubscribing device 0. Pass an
    explicit ``muGrid.Device`` (e.g. ``muGrid.Device.rocm(2)``) to override.
    """
    if device is None or device == "cpu":
        return None
    if isinstance(device, str):
        key = device.lower()
        if key in ("gpu", "rocm", "cuda"):
            dev_id = _local_rank(comm) % _gpu_device_count()
            if key == "rocm":
                return muGrid.Device.rocm(dev_id)
            if key == "cuda":
                return muGrid.Device.cuda(dev_id)
            # "gpu" is the platform accelerator (ROCm or CUDA), chosen by the
            # build; pass the id so each rank binds its own device, not just 0.
            return muGrid.Device.gpu(dev_id)
        raise ValueError(f"unknown device '{device}'")
    return device  # assume a muGrid.Device instance


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
        device=None,
        dtype=np.float64,
    ):
        """``dtype`` (``np.float64`` default, or ``np.float32``) is the
        precision of all grid fields and hence of the forward/adjoint solves,
        the preconditioner FFTs and the sensitivity kernels. The outer
        optimizer and the host-side reductions stay in double precision."""
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
                  residual=None):
        """Solve ``K x = b`` in place; returns ``x``.

        A negligible right-hand side (e.g. a spatially uniform material, whose
        macro response needs no fluctuation, giving a round-off-level rhs) has
        the exact solution ``x = 0``; return it directly, since CG would divide
        by zero on the first step. ``rhs_scale`` sets the (absolute) force scale
        below which the rhs counts as zero; when omitted the material scale is
        used.

        If ``residual`` is a field, the final CG residual ``r = b - K x`` is
        copied into it (used for the adjoint-corrected objective)."""
        x.set_zero()
        bp = b.p.ravel()
        b_norm = np.sqrt(self.comm.sum(float(self._xp.dot(bp, bp))))
        scale = rhs_scale if rhs_scale is not None else getattr(
            self, "_mat_scale", 1.0)
        if b_norm <= 1e-9 * scale:
            # Exact solution x = 0; no CG iterations performed. The residual
            # of x = 0 is b itself (round-off-level by construction).
            if residual is not None:
                residual.s[...] = b.s
            self.last_cg_iters = 0
            return x
        # Count CG iterations via the solver callback (fires once per iteration).
        counter = {"n": 0}

        def _count(iteration, state):
            counter["n"] += 1

        cg_kwargs = {}
        if residual is not None and _CG_HAS_RESIDUAL:
            cg_kwargs["residual"] = residual
        conjugate_gradients(
            self.comm, self.fc, b, x,
            hessp=self._hessp, prec=self._prec,
            rtol=self.cg_tol if rtol is None else rtol,
            maxiter=self.cg_maxiter if maxiter is None else maxiter,
            callback=_count,
            **cg_kwargs,
        )
        if residual is not None and not _CG_HAS_RESIDUAL:
            # Released muGrid without the residual out-field: recover
            # r = b - K x with one extra operator apply.
            self._hessp(x, self._Ku)
            residual.s[...] = b.s - self._Ku.s
        self.last_cg_iters = counter["n"]
        return x

    def solve_macro(self, E_macro, x, rtol=None, maxiter=None, residual=None):
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
            residual=residual)

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

    def homogenized_stress(self, u, E_macro):
        """Cell-averaged stress ``⟨σ⟩ = (1/V) ∫ C:(Ē + ∇u) dV`` as a
        ``dim x dim`` array (MPI-reduced)."""
        E = list(np.asarray(E_macro, dtype=float).ravel())
        local = self.op.average_stress(u, self.lam, self.mu, E)
        glob = np.array([self.comm.sum(float(v)) for v in local])
        return glob.reshape(self.dim, self.dim) / self.domain_volume
