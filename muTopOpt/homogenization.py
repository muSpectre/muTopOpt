#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
FFT-accelerated FE micromechanical homogenization on a regular grid.

Cell problem. A periodic unit cell of volume ``V`` is loaded by a macro
strain ``Ē``; the total strain is ``ε = Ē + ∇ˢu`` with a periodic nodal
fluctuation ``u``. With the discrete symmetric gradient ``B`` (nodal values
-> quadrature-point strains), the quadrature weights ``W`` and the per-element
isotropic stiffness ``C(ρ)`` of :mod:`muTopOpt.material`, equilibrium
``Bᵀ W C (Ē + B u) = 0`` reads

    K u = −Bᵀ C:Ē,        K = Bᵀ W C B,

where on each element ``K_e = 2μ_e G + λ_e V`` with the constant geometric
matrices ``G`` (deviatoric/shear) and ``V`` (volumetric). ``K`` is symmetric,
and positive definite on fluctuations modulo rigid translations as long as
``λ, μ > 0`` everywhere. The homogenized stress is the cell average

    σ̄ = (1/V) ∫ C:(Ē + ∇ˢu) dV,

and ``Ē ↦ σ̄`` is the effective stiffness ``C̄``.

:class:`Homogenization` owns the muGrid machinery -- the FFT engine (which
doubles as the ghosted domain decomposition), the fused
``IsotropicStiffnessOperator`` (which applies ``K`` without assembling it),
the per-pixel Lamé fields, and the preconditioner ``M ≈ K⁻¹`` -- and exposes
the operations the optimizer needs:

* :meth:`solve_macro`   -- solve ``K(ρ) u = −Bᵀ C:Ē`` for a macro strain ``Ē``;
* :meth:`solve_rhs`     -- solve ``K(ρ) x = b`` for an arbitrary rhs (adjoint);
* :meth:`homogenized_stress` -- the cell-averaged stress ``σ̄``.

The default preconditioner is J-FFT (Green-Jacobi): the inverse of the
stiffness of a homogeneous reference medium -- the global means of ``λ`` and
``μ`` at the first :meth:`set_density` -- applied by FFT, combined with the
Jacobi diagonal of ``K``. The reference part is built once; the diagonal is
refreshed from the current design on every :meth:`set_density`.

Everything is dimension-agnostic (``d`` = 2 or 3): the only per-dimension
detail is which ``IsotropicStiffnessOperator{2,3}D`` to instantiate. Fields live
on the engine's real-space collection so the solver, the preconditioner FFTs and
the operators all share one ghosted, MPI-decomposed, (optionally) device-resident
layout.
"""

import os

import warnings

import numpy as np

import muGrid
from muGrid import linalg
from muGrid.Preconditioners import (
    GreenJacobiPreconditioner,
    HybridFourierTridiagonalPreconditioner,
    make_green_jacobi_preconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import ConvergenceError, conjugate_gradients

from .material import SimpMaterial
from .precision import cg_rtol_floor
from .timing import timer

# muGrid's ``conjugate_gradients`` accepts an optional ``residual`` out-field
# (the final r = b - Kx, needed for the adjoint-corrected objective) only in
# versions after 0.110. Detect it once; without it, the residual costs one
# extra operator apply.
import inspect as _inspect

_CG_HAS_RESIDUAL = "residual" in _inspect.signature(
    conjugate_gradients).parameters


#: Factor by which the squared CG residual must drop before the best iterate
#: is snapshot again (0.25: the residual halved), so that a solve that has to
#: fall back returns an iterate within a factor two of the best residual.
_BACKUP_REDUCTION = 0.25


class _CGStagnation(Exception):
    """Internal control flow of :meth:`Homogenization.solve_rhs`: raised from
    the CG iteration callback when the solve has stopped making progress (the
    finite-precision residual floor), so the solve can be cut short and the
    best iterate returned instead of burning the full iteration budget."""


def _local_rank(comm):
    """Node-local rank of this process, used to pick a per-rank GPU.

    muGrid's communicator knows it (``node_rank``, the rank among the
    processes sharing this node's memory, found when the communicator is
    made), so it holds for any MPI library and launcher. A serial run has
    node rank 0.
    """
    return 0 if comm is None else comm.node_rank


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


# --------------------------------------------------------------------------- #
# Device memory: plain, unless managed is asked for
#
# Every GPU field, the CG scratch and the preconditioner's arrays come from one
# allocator, chosen once per process before the first field exists. There are
# two candidates:
#
# * Plain device memory (hipMalloc / cudaMalloc) -- muGrid's own default, and
#   muTopOpt's on every GPU.
# * Managed memory (hipMallocManaged / cudaMallocManaged), installed by
#   _enable_managed_device_allocator, only on request.
#
# Discrete GPUs (NVIDIA cards, AMD MI100/MI200/MI300X, ...): host and device
# have separate physical memories, and a managed page lives in one of them at
# a time; whichever side touches it moves it there, over PCIe or the fabric.
# A solver that streams every field through the GPU on every iteration then
# migrates all of them, all the time: measured on an RTX PRO 500, 9x slower
# per CG iteration at 64^3, and 96^3 made no progress at all.
#
# Unified-memory APUs (AMD MI300A): CPU and GPU share one physical HBM, so
# nothing migrates, and kernels read managed memory at full speed -- but MPI
# does not. Measured on Hunter (HLRS; 4x MI300A per node, ROCm 7.0.2, Cray
# MPICH 9.1 with its GPU support enabled), an all-to-all between the four
# GPUs of a node, 64 MiB per peer, ms per exchange:
#
#     plain     0.9
#     managed  39
#     host     35   (malloc, for comparison)
#
#   MPI moves managed buffers as it moves host memory, with the CPU, ~40x
#   slower than plain device buffers. In muGrid's homogenization example at
#   256^3 on 4 GPUs that made a CG iteration 1.8x (hybrid) to 5.4x
#   (reference FFT) slower. On the same nodes plain hipMalloc reached 120 GiB
#   of the 128 GB package, with XNACK on or off, so managed memory does not
#   buy capacity there either.
#
# Override. managed_memory=True, or MUTOPOPT_MANAGED=1, installs managed
# memory: for a system whose plain allocations stop short of what one rank's
# fields need, at the price of slower MPI exchanges.
#
# MPI and NetCDF need nothing from muTopOpt for either kind of memory: muGrid
# (1.6.0 and newer) asks the MPI library whether it takes device buffers, and
# the GPU runtime whether the host can read each buffer, and stages through
# the host where it cannot.
# --------------------------------------------------------------------------- #


def gpu_aware_mpi():
    """Whether muGrid hands device buffers to MPI directly (True) or stages
    them through the host (False): the MPI library's own answer, or
    ``MUGRID_GPU_AWARE_MPI`` when set. Ask after MPI is initialised."""
    return bool(getattr(muGrid._muGrid.Communicator, "gpu_aware_mpi", False))


def _requested_managed_memory(managed_memory, environ=os.environ):
    """The caller's explicit choice -- ``True``, ``False`` -- or ``None`` for
    the default, plain device memory.

    The argument wins over the environment; ``MUTOPOPT_MANAGED=0`` asks for
    plain device memory and any other value for managed memory.
    """
    if managed_memory is not None:
        return bool(managed_memory)
    env = environ.get("MUTOPOPT_MANAGED")
    if env is None:
        return None
    return env != "0"


def _device_memory_policy(on_device, requested=None):
    """Choose between managed and plain device memory; see the notes above.

    Returns ``(managed, reason)``: whether to install the managed allocator,
    and one line saying why, for the run's log.
    """
    if not on_device:
        return False, "CPU run, no device memory"
    if requested:
        return True, "managed memory, as requested"
    if requested is None:
        return False, ("plain device memory: the default, since MPI is much "
                       "slower on managed buffers")
    return False, "plain device memory, as requested"


# Process-global guard: the managed allocator, and muGrid's routing to it, must
# be installed exactly once per process. Re-registering muGrid's allocator while
# device fields are alive would drop the keepalive of their buffers and free
# them out from under muGrid.
_MANAGED_ALLOCATOR_INSTALLED = False


def _enable_managed_device_allocator():
    """Route muGrid's device allocations (and cupy's) through a *managed*
    memory pool, returning ``True`` on success.

    Only on request (see the notes above). Routing cupy's default pool
    through managed memory and then pointing muGrid at that pool
    (``use_cupy_allocator``) puts *all* device memory -- fields, CG scratch,
    the preconditioner -- on the managed path.

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
        cg_tol=None,
        cg_maxiter=2000,
        cg_verbose=False,
        device=None,
        dtype=np.float64,
        managed_memory=None,
        timer=None,
    ):
        """Set up the grid, the stiffness operator and the solver state.

        The material fields are filled, and the preconditioner built, by the
        first :meth:`set_density`.

        Parameters
        ----------
        nb_grid_pts : sequence of int
            Global number of grid points ``N_α`` per axis (2 or 3 entries).
        material : SimpMaterial
            Maps the density to the Lamé parameters and their derivatives.
        comm : muGrid.Communicator, optional
            Communicator of the domain decomposition, typically
            ``muGrid.Communicator.world()`` (default: serial). An mpi4py
            communicator is accepted too and wrapped.
        domain_lengths : sequence of float, optional
            Cell size per axis (default 1 per axis); the grid spacing is
            ``h_α = L_α / N_α``.
        element : {"q1", "p1"}, optional
            Finite element of the stiffness operator (default ``"q1"``; any
            other value selects P1).
        preconditioner : str or None, optional
            ``"green-jacobi"`` (default): J-FFT, reference-medium FFT inverse
            combined with the Jacobi diagonal. ``"green"``: the
            reference-medium FFT inverse alone. ``"hybrid"``: the reference
            operator inverted by FFT along the rank-local axes and by a
            tridiagonal solve along the distributed one, which needs no
            all-to-all communication. ``"hybrid-jacobi"``: ``"hybrid"``
            combined with the Jacobi diagonal. ``None``: unpreconditioned CG.
            Any other value raises ``ValueError`` in the first
            :meth:`set_density`.
        cg_tol : float, optional
            Default relative CG tolerance ``‖r‖ ≤ cg_tol ‖b‖``. ``None``
            (default) gives ``max(1e-8, cg_rtol_floor(dtype))``, i.e. 1e-8 in
            double and the float32 floor (1e-6) in single precision. An
            explicit value below the floor is raised to it with a
            ``RuntimeWarning``.

            .. TODO: rationale for the default ``cg_tol = 1e-8``?
        cg_maxiter : int, optional
            Maximum number of CG iterations per solve (default 2000).
        cg_verbose : bool, optional
            Print the residual of every CG iteration on rank 0 (default
            False).
        device : str or muGrid.Device, optional
            ``None``/``"cpu"`` (default), ``"gpu"``/``"cuda"``/``"rocm"``
            (one GPU per rank, ``local_rank % n_gpus``), or an explicit
            ``"<kind>:N"`` (see :func:`_resolve_device`).
        dtype : {numpy.float64, numpy.float32}, optional
            Precision of all grid fields and hence of the forward/adjoint
            solves, the preconditioner FFTs and the sensitivity kernels
            (default float64). The outer optimizer and the host-side
            reductions stay in double precision.
        managed_memory : bool, optional
            Managed instead of plain device memory for everything on the GPU.
            ``None`` (default) and ``False`` mean plain device memory, on
            every GPU: managed pages migrate on discrete GPUs, and MPI is
            much slower on managed buffers on unified-memory APUs (AMD
            MI300A). ``True`` (or ``MUTOPOPT_MANAGED=1``) asks for managed
            memory. The choice and its reason are stored in
            :attr:`device_memory` and :attr:`device_memory_reason`; the
            device-memory notes in ``muTopOpt/homogenization.py`` give the
            measurements. Ignored on CPU.
        timer : muTimer.Timer, optional
            Handed to the CG solver and the preconditioner so a benchmark can
            read back where the *wall* time of a solve goes. On a GPU the
            phases are host-side: the operator and preconditioner regions
            measure only the cost of queueing their kernels, and the wait for
            that work lands in the next region that reads a reduction back to
            the host. The gap between these numbers and the GPU-busy time
            from a kernel trace is therefore the pipeline bubble. A
            :class:`muTopOpt.timing.Timer` with ``synchronize`` set waits for
            the device around every region instead, which charges each phase
            with its own kernels. The operator apply is split into
            ``"ghosts"`` and ``"stencil"``.

        Raises
        ------
        ValueError
            If ``nb_grid_pts`` is not 2- or 3-dimensional, or ``dtype`` is
            neither float64 nor float32.
        """
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
        # A muGrid communicator whatever was passed (None, an mpi4py
        # communicator): node_rank and the NuMPI protocol need one
        self.comm = muGrid.Communicator(comm)

        if domain_lengths is None:
            domain_lengths = [1.0] * self.dim
        self.domain_lengths = [float(x) for x in domain_lengths]
        self.grid_spacing = [
            L / n for L, n in zip(self.domain_lengths, self.nb_grid_pts)
        ]
        self.domain_volume = float(np.prod(self.domain_lengths))

        self.device = _resolve_device(device, self.comm)
        # cupy places arrays (and launches kernels) on its *current* device,
        # device 0 unless told otherwise. With one GPU per rank that would put
        # every cupy temporary of rank 1 on GPU 0, next to fields on GPU 1.
        if self.device is not None and self.device.is_device:
            import cupy

            cupy.cuda.Device(self.device.device_id).use()
        # Plain or managed device memory: see the notes above
        # _device_memory_policy. Decided before the engine allocates its
        # first device field.
        on_device = self.device is not None and self.device.is_device
        managed, reason = _device_memory_policy(
            on_device, _requested_managed_memory(managed_memory))
        self.managed_memory = managed and _enable_managed_device_allocator()
        if managed and not self.managed_memory:
            reason += " -- but cupy is unavailable, so plain device memory"
        elif not managed and on_device and _MANAGED_ALLOCATOR_INSTALLED:
            # The allocator is process-global and cannot be taken back while
            # fields allocated through it are alive (see its guard).
            self.managed_memory = True
            reason += (" -- but an earlier Homogenization in this process "
                       "installed managed memory, which stays")
        #: "managed", "plain" or "host": what this run actually allocates.
        self.device_memory = ("managed" if self.managed_memory
                              else "plain" if on_device else "host")
        #: One line saying why, for the run's log.
        self.device_memory_reason = reason
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
        self.timer = timer
        # Smallest relative residual this precision can actually reach; 0.0
        # in double, so everything below is a no-op there. See
        # muTopOpt.precision.
        self._rtol_floor = cg_rtol_floor(self.dtype)
        self._rtol_clamp_warned = False
        # Default is dtype-aware: 1e-8 is meaningless in float32, and a
        # default the caller never chose should not produce a warning.
        # An explicit value that cannot be reached does warn (_clamp_rtol).
        self.cg_tol = (max(1e-8, self._rtol_floor) if cg_tol is None
                       else self._clamp_rtol(cg_tol))
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

    @property
    def xp(self):
        """The array module of the fields: cupy on a device, numpy on the
        host."""
        return self._xp

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
        """Interpolate ``ρ -> (λ, μ)``, fill ghosts, and build (first call) or
        refresh the preconditioner. ``rho`` is the element-wise density of
        shape :attr:`nb_pixels`, on the host or in the fields' array module
        (:attr:`xp`); the interpolation runs in the latter.

        Raises ``ValueError`` on the first call if the ``preconditioner``
        given to the constructor is unknown."""
        with timer("set_density"):
            lam, mu = self.material.lame(self._xp.asarray(rho))
            self.lam.p[...] = self.to_device(lam)
            self.mu.p[...] = self.to_device(mu)
            self.engine.communicate_ghosts(self.lam)
            self.engine.communicate_ghosts(self.mu)
            # Stiffness scale (max |λ|, |2μ|), reduced across ranks, used to
            # detect a negligible (round-off-only) right-hand side.
            local = max(float(abs(lam).max()), float(2.0 * abs(mu).max()))
            self._mat_scale = self.comm.max(local)
            with timer("preconditioner"):
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
                    dtype=self.dtype, timer=self.timer,
                )
            elif self.preconditioner_kind == "green":

                def apply_ref(u, f):
                    self.engine.communicate_ghosts(u)
                    self.op.apply_uniform(u, lam_ref, mu_ref, f)

                self._prec = make_reference_stiffness_preconditioner(
                    self.engine, apply_ref, self.dim, dtype=self.dtype,
                    timer=self.timer,
                )
            elif self.preconditioner_kind in ("hybrid", "hybrid-jacobi"):
                # The same reference operator as 'green', inverted exactly by
                # FFT in the rank-local axes and a tridiagonal solve in the
                # distributed one -- no all-to-all. The engine's real-space
                # split is the slab ([1, ..., P]) it needs.
                green = HybridFourierTridiagonalPreconditioner(
                    self.engine, self.grid_spacing, lam_ref, mu_ref,
                    communicator=self.comm, element=self.element,
                    timer=self.timer, dtype=self.dtype,
                )
                if self.preconditioner_kind == "hybrid":
                    self._prec = green
                else:
                    diagonal = self.engine.real_space_field(
                        "to_hybrid_jacobi_diagonal", components=(self.dim,),
                        dtype=self.dtype)
                    self.op.assemble_diagonal(self.lam, self.mu, diagonal)
                    prec = GreenJacobiPreconditioner(
                        green, diagonal, timer=self.timer,
                        name="hybrid-jacobi", communicator=self.comm,
                    )

                    def refresh():
                        self.op.assemble_diagonal(self.lam, self.mu, diagonal)
                        prec.update_diagonal(diagonal)

                    prec.refresh = refresh
                    self._prec = prec
            else:
                raise ValueError(
                    f"unknown preconditioner '{self.preconditioner_kind}'"
                )
        elif self.preconditioner_kind in ("green-jacobi", "hybrid-jacobi"):
            # Reuse the (reference) Green part; refresh only the Jacobi diagonal
            # from the updated material.
            self._prec.refresh()

    # -- solves -------------------------------------------------------------
    def _hessp(self, u, Au):
        if self.timer is None:
            self.engine.communicate_ghosts(u)
            self.op.apply(u, self.lam, self.mu, Au)
            return
        # Split the halo exchange (inter-rank communication) from the stencil
        # (pure local bandwidth) for a solve-time breakdown.
        with self.timer("ghosts"):
            self.engine.communicate_ghosts(u)
        with self.timer("stencil"):
            self.op.apply(u, self.lam, self.mu, Au)

    def _clamp_rtol(self, rtol):
        """Raise ``rtol`` to what this precision can actually reach, warning
        once if it had to.

        A float32 solve asked for 1e-8 cannot get there: it burns
        ``cg_maxiter`` iterations and is then rescued by the stagnation guard,
        with a worse iterate than the reachable tolerance would have produced.
        The warning fires once per instance so a per-iteration solve loop does
        not drown the log.
        """
        if rtol is None or rtol >= self._rtol_floor:
            return rtol
        if not self._rtol_clamp_warned:
            self._rtol_clamp_warned = True
            warnings.warn(
                f"{type(self).__name__}: inner CG rtol={rtol:.1e} is below "
                f"the {np.dtype(self.dtype).name} solve accuracy floor "
                f"(~{self._rtol_floor:.0e}) and cannot be reached; using the "
                "floor instead. Pass dtype=np.float64 for a tighter solve.",
                RuntimeWarning,
                stacklevel=3,
            )
        return self._rtol_floor

    def solve_rhs(self, b, x, rtol=None, maxiter=None, rhs_scale=None,
                  residual=None, label=None, warm_start=False):
        """Solve ``K x = b`` by preconditioned CG in place; returns ``x``.

        Parameters
        ----------
        b : muGrid field
            Right-hand side (vector field on :attr:`fc`).
        x : muGrid field
            Solution, overwritten (and read as initial guess with
            ``warm_start``).
        rtol : float, optional
            Relative tolerance (default :attr:`cg_tol`), clamped to the
            precision floor (:mod:`muTopOpt.precision`).
        maxiter : int, optional
            Iteration cap (default :attr:`cg_maxiter`).
        rhs_scale : float, optional
            Absolute force scale for the negligible-rhs test (default: the
            material scale :attr:`mat_scale`).
        residual : muGrid field, optional
            Receives the final residual ``r = b − K x``.
        label : str, optional
            Tag of this solve in the ``cg_verbose`` trace (e.g.
            ``"case 1 fwd"``).
        warm_start : bool, optional
            Use the current ``x`` as initial guess (default False: zero).

        Notes
        -----
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

        This method does not raise on non-convergence. A solve that stagnates
        (no relative residual improvement of :attr:`cg_stagnation_rel` within
        :attr:`cg_stagnation_patience` iterations -- the finite-precision
        residual floor), hits the iteration cap, or breaks down returns an
        iterate within a factor two of the *best residual* seen (see
        :data:`_BACKUP_REDUCTION`), with its true residual ``b − K x`` recomputed
        (and never worse than the zero solution). The callers tolerate a
        less-accurate solve: the consistent objective is second-order in the
        state-solve residual and reports the remaining error, and the
        trust-region model only needs a bounded Hessian-vector product. The
        number of such solves is counted in :attr:`cg_stagnation_count`, the
        iterations of the last solve in :attr:`last_cg_iters`.

        Raises
        ------
        muGrid.Solvers.ConvergenceError
            Only if CG breaks down (NaN) before its first iteration, so that
            no iterate exists to fall back to."""
        if not warm_start:
            x.set_zero()
        # muGrid's reductions run over the interior in double precision and
        # return it. Reducing b.p.ravel() with xp.dot instead would (a) use
        # BLAS sdot for a float32 field, whose error grows *linearly* in N --
        # 4.6e-4 relative at 512^3 against 6.9e-8 at 64^3 -- and (b) silently
        # materialise a full contiguous copy of the field, because .p is a
        # strided view whenever the collection carries ghosts. It also makes
        # b_norm bit-identical to the ||b|| the CG itself converges against.
        b_norm = np.sqrt(self.comm.sum(float(linalg.norm_sq(b))))
        scale = rhs_scale if rhs_scale is not None else getattr(
            self, "_mat_scale", 1.0)
        verbose = self.cg_verbose and self.comm.rank == 0
        tag = f"{label}  " if label else ""
        # Negligible-rhs threshold, relative to the natural force scale of this
        # solve: 1e-9 in float64; in float32 the assembly round-off alone is
        # ~eps*scale ~ 1e-7*scale, so a rhs below a few eps is
        # indistinguishable from noise (and CG on pure noise diverges -- the
        # operator is only semi-definite for void_ratio=0), hence 10 eps.
        # TODO: rationale for the float64 threshold 1e-9?
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
            # _Ku <- K x - b, whose norm is |b - K x|; the fused axpy_norm_sq
            # does the update and the reduction in one pass and needs no
            # scratch field. _Ku is dead here -- every later use rewrites it
            # via _hessp first.
            warm_norm = np.sqrt(self.comm.sum(
                float(linalg.axpy_norm_sq(-1.0, b, self._Ku))))
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
        # best iterate seen so far (whenever the residual has halved since the
        # last snapshot) and aborts the solve (via _CGStagnation)
        # once the residual has not beaten the reference by at least
        # ``cg_stagnation_rel`` for ``cg_stagnation_patience`` consecutive
        # iterations. In finite precision -- float32 above all -- the *true*
        # residual has a floor; past it the recursive CG residual stagnates or
        # diverges (on a round-off-level rhs it can grow by orders of
        # magnitude), and without the safeguard the solve burns its full
        # iteration budget and raises. ``guard['best']``/``guard['ref']`` are
        # squared norms; ``rel`` compares in the norm, hence the square.
        counter = {"n": 0}
        # Clamp at the point of consumption, not at one branch of one
        # driver: every caller -- the fixed cg_tol, an adaptive controller's
        # current value, a Hessian-vector solve's own rtol -- passes through
        # here, so the float32 floor cannot be routed around. In double
        # precision the floor is 0.0 and this is a no-op.
        rtol_eff = self._clamp_rtol(self.cg_tol if rtol is None else rtol)
        stall = (1.0 - self.cg_stagnation_rel) ** 2
        diverge = self.cg_divergence_factor ** 2
        guard = {"best": np.inf, "best_iter": 0, "ref": np.inf, "ref_iter": 0,
                 "saved": False, "saved_rr": np.inf}

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
                if rr < _BACKUP_REDUCTION * guard["saved_rr"]:
                    # Snapshot only once the residual has halved since the
                    # last one: a full-field copy on every improving
                    # iteration cost ~2% of a solve, and the fallback only
                    # needs an iterate close to the best. Full-buffer copy
                    # (ghosts included): a contiguous copy, whereas
                    # `.s[...] = .s` goes through a strided, ghost-excluded
                    # view.
                    muGrid.linalg.copy(x, self._x_best)
                    guard["saved"] = True
                    guard["saved_rr"] = rr
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
                rtol=rtol_eff, timer=self.timer,
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
            muGrid.linalg.copy(self._x_best, x)
            # True (non-recursive) residual of the returned iterate; the best
            # recursive rr can be far below it past the precision floor.
            self._hessp(x, self._Ku)
            # _Ku <- b - K x, the true residual of the salvaged iterate.
            # axpby avoids the full-size temporary that `b.s - _Ku.s` builds,
            # and norm_sq reduces it in double (see solve_rhs's b_norm).
            linalg.axpby(1.0, b, -1.0, self._Ku)
            true_norm = np.sqrt(self.comm.sum(float(linalg.norm_sq(self._Ku))))
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
            # muGrid without the residual out-field: recover r = b - K x
            # with one extra operator apply.
            self._hessp(x, self._Ku)
            residual.s[...] = b.s - self._Ku.s
        self.last_cg_iters = counter["n"]
        return x

    def solve_macro(self, E_macro, x, rtol=None, maxiter=None, residual=None,
                    label=None):
        """Solve the periodic cell problem ``K u = −Bᵀ C:Ē`` for the
        fluctuation displacement ``u`` (written to ``x`` and returned) under
        the ``dim × dim`` macro strain ``E_macro``. The remaining arguments
        are those of :meth:`solve_rhs`; the negligible-rhs scale is
        ``mat_scale · max|Ē|``."""
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
        """Stiffness force scale ``max(max|λ|, max|2μ|)`` of the current
        material, MPI-reduced (1 before the first :meth:`set_density`)."""
        return getattr(self, "_mat_scale", 1.0)

    def macro_rhs_tensor(self, tensor, out, scale=1.0):
        """Assemble ``out = scale · Bᵀ C:T`` for a constant symmetric
        ``dim × dim`` tensor ``T`` (used to build the adjoint right-hand
        side); returns ``out``."""
        T = list(np.asarray(tensor, dtype=float).ravel())
        self.op.apply_macro_rhs(self.lam, self.mu, T, out)
        if scale != 1.0:
            out.s[...] *= scale
        return out

    def homogenized_stress(self, u, E_macro, lam=None, mu=None):
        """Homogenized stress ``σ̄ = (1/V) ∫ C:(Ē + ∇ˢu) dV`` as a
        ``dim × dim`` host array (MPI-reduced).

        ``lam``/``mu`` override the material fields; passing *perturbed* Lamé
        fields ``λ'δρ``/``μ'δρ`` evaluates the material-derivative stress
        ``(1/V) ∫ δC:(Ē + ∇ˢu) dV`` (the operator is linear in the
        material), as needed for Hessian-vector products."""
        E = list(np.asarray(E_macro, dtype=float).ravel())
        local = self.op.average_stress(
            u, self.lam if lam is None else lam,
            self.mu if mu is None else mu, E)
        glob = np.array([self.comm.sum(float(v)) for v in local])
        return glob.reshape(self.dim, self.dim) / self.domain_volume
