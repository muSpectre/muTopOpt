#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
FFT-accelerated FE homogenization of steady-state heat conduction (scalar
diffusion) on a regular periodic grid, for topology optimization.

Mathematical setup
------------------
On the periodic cell of volume ``V`` the temperature is split into an affine
part and a periodic nodal fluctuation ``u``, so the temperature gradient at
quadrature point ``q`` is ``Ē + (B u)_q`` for a prescribed macro gradient
``Ē`` (a ``d``-vector). ``B`` is muGrid's ``FEMGradientOperator.apply``
(nodal scalar -> gradient at the quadrature points of each P1/Q1 pixel) and
``Bᵀ W`` its ``transpose``, which applies the physical quadrature weights
``W_q`` (``Σ_q W_q`` over one pixel equals the pixel volume). With the
per-pixel SIMP conductivity ``κ(ρ)`` broadcast to all quadrature points of a
pixel, the discrete equilibrium is

    K(ρ) u = -Bᵀ W κ(ρ) Ē,      K(ρ) = Bᵀ W κ(ρ) B,

and the homogenized flux is

    q̄ = (1/V) Σ_q W_q κ_q (Ē + B u)_q.

``K`` is symmetric positive semi-definite for ``κ > 0`` (its null space is
the constant field), so the forward and adjoint problems are solved by
preconditioned CG with the same operator.

:class:`HomogenizationConductivity` is the scalar-diffusion sibling of
:class:`muTopOpt.homogenization.Homogenization`: same muGrid engine, same
periodic-domain / ghost / CG-solve machinery, but the system matrix ``K`` is
assembled from muGrid's dimension-templated, physics-agnostic
``FEMGradientOperator`` (nodal scalar field -> gradient at quadrature points,
and its quadrature-weighted transpose) instead of the fused
``IsotropicStiffnessOperator`` elasticity kernel. muGrid has no fused
conductivity operator; the composition

    K(rho) u = transpose(kappa(rho) * apply(u))

is exactly the FE operator ``Bᵀ W κ B`` above, because ``transpose``'s
default weighting already applies the physical (volume-scaled) quadrature
weights, so that ``vᵀ K u = Σ_q W_q κ_q (B v)_q · (B u)_q``. It costs one
``apply``, one ``transpose`` and two ghost exchanges per product, and is
symmetric by construction.

Only an isotropic base material is supported (see :class:`SimpConductivity`):
the *target* conductivity tensor in a topology-optimization objective may
still be anisotropic (see :mod:`muTopOpt.loadcases_conduction`), only the
physical solid/void material itself is isotropic.  Both a plain Green's-
function (``'green'``) and the Green-Jacobi (``'green-jacobi'``) preconditioner
are available.  The Jacobi diagonal of the conductivity operator is assembled
in closed form from the element shape-function gradients and the conductivity
(see :meth:`HomogenizationConductivity._assemble_diagonal`), exactly for every
grid and subdomain size.
"""

import inspect as _inspect

import warnings

import itertools

import numpy as np

import muGrid
from muGrid import linalg
from muGrid.Preconditioners import (
    GreenJacobiPreconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import ConvergenceError, conjugate_gradients

from .homogenization import _resolve_device
from .precision import cg_rtol_floor

_CG_HAS_RESIDUAL = "residual" in _inspect.signature(
    conjugate_gradients).parameters


class _CGStagnation(Exception):
    """Internal control flow of :meth:`HomogenizationConductivity.solve_rhs`;
    see :class:`muTopOpt.homogenization.Homogenization` for the rationale."""


def kappa_from_conductivity(k):
    """Return ``float(k)``: the counterpart of
    :func:`muTopOpt.material.lame_from_E_nu` for the conductivity material,
    which has a single parameter and therefore nothing to convert."""
    return float(k)


class SimpConductivity:
    """Power-law (SIMP) interpolation of an isotropic scalar conductivity,

        κ(ρ) = κ_void + ρ^p (κ_solid - κ_void),   κ_void = void_ratio · κ_solid.

    Parameters
    ----------
    kappa_solid : float
        Conductivity of the solid phase.
    penalty : float, optional
        SIMP exponent ``p`` (default 2). ``p > 1`` penalizes intermediate
        densities.
    void_ratio : float, optional
        Conductivity ratio of the void phase, ``kappa_void / kappa_solid``
        (default 1e-3). A small positive value keeps the operator SPD and the
        preconditioner well-behaved; set smaller to approach a true void.
    """

    def __init__(self, kappa_solid, penalty=2.0, void_ratio=1e-3):
        self.penalty = float(penalty)
        self.kappa_solid = float(kappa_solid)
        self.kappa_void = self.kappa_solid * float(void_ratio)

    def kappa(self, rho):
        """Return ``kappa(rho)`` as an array with the shape of ``rho``."""
        f = np.power(rho, self.penalty)
        return self.kappa_void + f * (self.kappa_solid - self.kappa_void)

    def dkappa(self, rho):
        """Return the density derivative ``dkappa/drho``."""
        df = self.penalty * np.power(rho, self.penalty - 1.0)
        return df * (self.kappa_solid - self.kappa_void)

    def d2kappa(self, rho):
        """Return the second density derivative ``d^2 kappa/drho^2``."""
        p = self.penalty
        d2f = p * (p - 1.0) * np.power(rho, p - 2.0)
        return d2f * (self.kappa_solid - self.kappa_void)


class HomogenizationConductivity:
    def __init__(
            self,
            nb_grid_pts,
            material: SimpConductivity,
            comm=None,
            domain_lengths=None,
            element="q1",
            preconditioner="green-jacobi",
            cg_tol=None,
            cg_maxiter=2000,
            cg_verbose=False,
            dtype=np.float64,
            device=None,
    ):
        """Homogenization engine for scalar conduction on a periodic grid.

        See :class:`muTopOpt.homogenization.Homogenization` for the shared
        conventions (grid, MPI, dtype, CG hardening).

        Parameters
        ----------
        nb_grid_pts : sequence of int
            Global grid ``(N_1, ..., N_d)``, ``d`` in {2, 3}.
        material : SimpConductivity
            Density-to-conductivity interpolation.
        comm : muGrid.Communicator, optional
            Communicator of the domain decomposition, typically
            ``muGrid.Communicator.world()`` (default: serial). An mpi4py
            communicator is accepted too and wrapped.
        domain_lengths : sequence of float, optional
            Physical cell edge lengths (default: 1 per axis); grid spacing
            ``h_α = L_α / N_α``.
        element : {'q1', 'p1'}, optional
            Finite element (default ``'q1'``); any value other than
            ``'q1'`` selects P1.
        preconditioner : {'green-jacobi', 'green'}, optional
            ``'green-jacobi'`` (default, J-FFT: Green operator of the mean
            conductivity combined with the Jacobi diagonal of ``K``) or
            ``'green'`` (Green operator of the mean conductivity alone).
        cg_tol : float, optional
            Default inner relative tolerance; ``None`` gives
            ``max(1e-8, cg_rtol_floor(dtype))``. An explicit value below the
            precision floor is raised to it with a ``RuntimeWarning``.
        cg_maxiter : int, optional
            Maximum CG iterations per solve (default 2000).
        cg_verbose : bool, optional
            Print every CG iteration on rank 0 (default False).
        dtype : {np.float64, np.float32}, optional
            Scalar type of all fields and solves (default float64).
        device : str or muGrid.Device, optional
            Where fields live and the solves run, with the conventions of
            :class:`~muTopOpt.homogenization.Homogenization`: ``None`` or
            ``"cpu"`` for the host, ``"gpu"`` for this rank's accelerator,
            ``"gpu:N"`` for a specific one, or a ``muGrid.Device``.

        Raises
        ------
        ValueError
            For a grid that is not 2D/3D, an unsupported ``dtype`` or an
            unknown ``preconditioner``.
        """
        self.dim = len(nb_grid_pts)
        if self.dim not in (2, 3):
            raise ValueError("nb_grid_pts must be 2- or 3-dimensional")
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
        if self.device is not None and self.device.is_device:
            import cupy

            cupy.cuda.Device(self.device.device_id).use()
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
        self.element = elem
        self.element_name = element
        self.grad = muGrid.FEMGradientOperator(
            self.dim, self.grid_spacing, elem)
        self.nb_quad = self.grad.nb_quad_pts
        # Physical (volume-scaled) quadrature weights W_q: muGrid's
        # FEMGradientOperator.quadrature_weights already include the pixel
        # volume from the grid_spacing given at construction
        # (sum(quadrature_weights) == prod(grid_spacing)), so no extra volume
        # factor is applied anywhere below.
        self.quad_weights = np.asarray(self.grad.quadrature_weights)
        self.fc.set_nb_sub_pts("quad", self.nb_quad)

        # Per-quadrature-point (piecewise-constant per pixel) conductivity and
        # solver scratch. `kappa` is a scalar field (no component axis: `.s`
        # shape (1, nb_quad, *nb_pixels)); `_g`/`_flux` carry the gradient's
        # `dim` components per quadrature point.
        self.kappa = self.fc.real_field(
            "to_kappa", (), "quad", dtype=self.dtype)
        self._g = self.fc.real_field(
            "to_grad", (self.dim,), "quad", dtype=self.dtype)
        self._flux = self.fc.real_field(
            "to_flux", (self.dim,), "quad", dtype=self.dtype)
        self._rhs = self.fc.real_field("to_rhs", dtype=self.dtype)
        self._Ku = self.fc.real_field("to_Ku", dtype=self.dtype)

        self.on_device = "cupy" in type(self.kappa.p).__module__
        if self.on_device:
            import cupy as _cp

            self._xp = _cp
        else:
            self._xp = np

        # Smallest relative residual this precision can actually reach; 0.0
        # in double, so everything below is a no-op there. See
        # muTopOpt.precision.
        self._rtol_floor = cg_rtol_floor(self.dtype)
        self._rtol_clamp_warned = False
        # Default is dtype-aware: 1e-8 is below the float32 floor, and a
        # default the caller never chose should not produce a warning.
        # An explicit value that cannot be reached does warn (_clamp_rtol).
        self.cg_tol = (max(1e-8, self._rtol_floor) if cg_tol is None
                       else self._clamp_rtol(cg_tol))
        self.cg_maxiter = cg_maxiter
        # Same CG hardening thresholds as Homogenization -- see its docstring
        # for the rationale; duplicated rather than shared to keep the two
        # homogenization engines independently modifiable.
        self.cg_stagnation_patience = 100
        self.cg_stagnation_rel = 1e-2
        self.cg_no_progress_patience = 25
        self.cg_divergence_factor = 1e3
        self.cg_stagnation_count = 0
        self._x_best = self.fc.real_field("to_cg_best", dtype=self.dtype)
        self.cg_verbose = cg_verbose
        if preconditioner not in ("green-jacobi", "green"):
            raise ValueError(
                f"preconditioner must be 'green-jacobi' or 'green', "
                f"got '{preconditioner}'"
            )
        self.preconditioner_kind = preconditioner
        self._prec = None
        self._nb_pixels = tuple(self.engine.nb_subdomain_grid_pts)

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
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def to_device(self, a):
        return self._xp.asarray(a, dtype=self.dtype)

    def scalar_field(self, name):
        return self.fc.real_field(name, dtype=self.dtype)

    def temperature_field(self, name):
        """A nodal scalar (temperature/fluctuation) field -- the "unknown"
        field of this problem, analogous to
        :meth:`Homogenization.vector_field`."""
        return self.fc.real_field(name, dtype=self.dtype)

    # -- material update ----------------------------------------------------
    def set_density(self, rho):
        """Interpolate rho -> kappa (broadcast to every quadrature point of
        each pixel) and (re)build the Green preconditioner. ``rho`` has shape
        :attr:`nb_pixels`."""
        kappa_pix = self.material.kappa(np.asarray(rho))
        kappa_dev = self.to_device(kappa_pix)
        # kappa.s has shape (nb_quad, *nb_pixels) -- a scalar (components=())
        # sub_pt field carries no leading dummy component axis, unlike a plain
        # scalar pixel field -- so the same per-pixel value must be broadcast
        # across the leading (quadrature) axis explicitly via a full slice.
        self.kappa.s[...] = kappa_dev[np.newaxis, ...]
        local = float(self._xp.abs(self.kappa.p).max()) if self.kappa.p.size \
            else 0.0
        self._mat_scale = self.comm.max(local)
        self._update_preconditioner()

    def _reference_kappa(self):
        n = self.comm.sum(int(self.kappa.p.size))
        return self.comm.sum(float(self._xp.sum(self.kappa.p))) / n

    def _assemble_diagonal(self, diag_field):
        """Assemble the diagonal of the conductivity stiffness matrix
        ``K = Bᵀ W κ B`` into ``diag_field``, exactly and in closed form.

        Node ``i`` is corner ``o ∈ {0,1}^d`` of the element (pixel) ``i − o``,
        so

        .. math::

            K_{ii} = \\sum_{o} \\sum_q W_q\\, \\kappa_{q,\\,i-o}
                     \\sum_\\alpha B_{q\\alpha o}^2,

        with ``B`` the shape-function gradients of the element (already
        divided by the grid spacing). This needs only one left ghost layer of
        ``κ``, so it is exact for every grid and subdomain size, including odd
        ones, where a two-colour probing of ``K`` would let the periodic or
        rank-boundary neighbours of a node share its colour. Cost: one ghost
        exchange of ``κ`` and ``2^d · nb_quad`` fused multiply-adds per node.
        """
        xp = self._xp
        self.engine.communicate_ghosts(self.kappa)
        coeff = np.asarray(self.grad.coefficients)  # (d, nb_quad, 1, 2,..,2)
        # per (quad point, corner): W_q Σ_α B_{qαo}²
        corner_weight = np.einsum(
            "q,dq...->q...", self.quad_weights, coeff[:, :, 0] ** 2)
        kappa = self.kappa.sg  # (nb_quad, *pixels with ghosts)
        left = tuple(self.fc.nb_ghosts_left)
        n = self._nb_pixels
        diag = xp.zeros(n, dtype=self.dtype)
        for offsets in itertools.product(range(2), repeat=self.dim):
            elements = tuple(
                slice(gl - o, gl - o + m) for gl, o, m in zip(left, offsets, n))
            for q in range(self.nb_quad):
                diag += self.dtype.type(corner_weight[(q,) + offsets]) * \
                    kappa[(q,) + elements]
        diag_field.p[...] = diag

    def _update_preconditioner(self):
        kappa_ref = self._reference_kappa()

        def apply_ref(u, f):
            self.engine.communicate_ghosts(u)
            self.grad.apply(u, self._g)
            self._flux.s[...] = kappa_ref * self._g.s
            self.engine.communicate_ghosts(self._flux)
            self.grad.transpose(self._flux, f)

        if self._prec is None:
            green = make_reference_stiffness_preconditioner(
                self.engine, apply_ref, 1, dtype=self.dtype
            )
            if self.preconditioner_kind == "green-jacobi":
                diag_field = self.fc.real_field(
                    "to_kappa_diag", dtype=self.dtype)
                self._assemble_diagonal(diag_field)
                self._prec = GreenJacobiPreconditioner(
                    green,
                    diag_field,
                    communicator=self.comm,
                )
            else:
                self._prec = green
        elif self.preconditioner_kind == "green-jacobi":
            # Refresh the Jacobi diagonal from the updated kappa field;
            # the Green (reference-stiffness) part is reused unchanged.
            diag_field = self.fc.real_field("to_kappa_diag", dtype=self.dtype)
            self._assemble_diagonal(diag_field)
            self._prec.update_diagonal(diag_field)

    # -- solves -------------------------------------------------------------
    def _hessp(self, u, Ku):
        self.engine.communicate_ghosts(u)
        self.grad.apply(u, self._g)
        self._flux.s[...] = self.kappa.s * self._g.s
        self.engine.communicate_ghosts(self._flux)
        self.grad.transpose(self._flux, Ku)

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
        """Solve ``K x = b`` in place; returns ``x``. Identical contract to
        :meth:`muTopOpt.homogenization.Homogenization.solve_rhs` -- see its
        docstring for the CG-hardening rationale (negligible-rhs shortcut,
        warm-start guard, stagnation safeguard); duplicated here rather than
        shared so the two engines stay independently modifiable."""
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
        negligible = max(1e-9, 10.0 * float(np.finfo(self.dtype).eps)) * scale
        if b_norm <= negligible:
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
            if not guard["saved"]:
                raise
            x.s[...] = self._x_best.s
            self._hessp(x, self._Ku)
            # _Ku <- b - K x, the true residual of the salvaged iterate.
            # axpby avoids the full-size temporary that `b.s - _Ku.s` builds,
            # and norm_sq reduces it in double (see solve_rhs's b_norm).
            linalg.axpby(1.0, b, -1.0, self._Ku)
            true_norm = np.sqrt(self.comm.sum(float(linalg.norm_sq(self._Ku))))
            if not (true_norm < b_norm):
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
            self._hessp(x, self._Ku)
            residual.s[...] = b.s - self._Ku.s
        self.last_cg_iters = counter["n"]
        return x

    def solve_macro(self, E_macro, x, rtol=None, maxiter=None, residual=None,
                    label=None):
        """Solve the periodic homogenization problem
        ``K u = -Bᵀ W κ Ē`` (``-transpose(kappa * E_macro)``) for the
        periodic temperature fluctuation ``u`` under the macro temperature
        gradient ``Ē = E_macro`` (a length-``dim`` vector), cold-started.
        Returns ``x`` (holding ``u``); the other arguments are as in
        :meth:`solve_rhs`."""
        E_arr = np.asarray(E_macro, dtype=float)
        self.macro_rhs_vector(E_arr, self._rhs, scale=-1.0)
        scale = getattr(self, "_mat_scale", 1.0) * max(
            float(np.abs(E_arr).max()), 1e-300)
        return self.solve_rhs(
            self._rhs, x, rtol=rtol, maxiter=maxiter, rhs_scale=scale,
            residual=residual, label=label)

    @property
    def mat_scale(self):
        """Conductivity force scale (max |kappa|) of the current material."""
        return getattr(self, "_mat_scale", 1.0)

    def macro_rhs_vector(self, E_macro, out, scale=1.0):
        """Assemble ``out = scale · Bᵀ W κ E_macro`` (i.e.
        ``scale * transpose(kappa * E_macro)``) for a constant vector
        ``E_macro``, used to build the homogenization and adjoint right-hand
        sides. Returns ``out``."""
        E = self.to_device(np.asarray(E_macro, dtype=float))
        shape = (self.dim,) + (1,) * (self._flux.s.ndim - 1)
        self._flux.s[...] = self.kappa.s * E.reshape(shape)
        self.engine.communicate_ghosts(self._flux)
        self.grad.transpose(self._flux, out)
        if scale != 1.0:
            out.s[...] *= scale
        return out

    def homogenized_flux(self, u, E_macro):
        """Cell-averaged flux
        ``q̄ = (1/V) Σ_q W_q κ_q (E_macro + B u)_q``, the quadrature of
        ``(1/V) ∫ κ (E_macro + ∇u) dV``, as a length-``dim`` host array
        (MPI-reduced).

        Sign convention: ``q`` is the *constitutive* flux ``kappa * grad T``
        (no Fourier's-law minus sign), matching the stress analogy
        ``sigma = C:(E_macro + grad u)`` this module mirrors, so that
        ``target_flux = kappa_target @ macro_gradient`` (see
        :mod:`muTopOpt.loadcases_conduction`) is the response you would
        actually measure."""
        self.engine.communicate_ghosts(u)
        self.grad.apply(u, self._g)
        E = self.to_device(np.asarray(E_macro, dtype=float))
        shape = (self.dim,) + (1,) * (self._g.s.ndim - 1)
        total_grad = self._g.s + E.reshape(shape)
        flux_q = self.to_host(self.kappa.s * total_grad)  # (dim, nb_quad, *nb_pixels)
        # Local (unnormalized) volume integral: contract the quadrature axis
        # with the physical (volume-scaled) weights, sum over owned pixels.
        weighted = np.tensordot(flux_q, self.quad_weights, axes=([1], [0]))
        local = weighted.reshape(self.dim, -1).sum(axis=1)
        glob = np.array([self.comm.sum(float(v)) for v in local])
        return glob / self.domain_volume

    def compute_sensitivity(self, forward_disp, forward_macro, costate_disp,
                            costate_macro):
        """Return the per-pixel geometry contraction of the topology-
        optimization material-derivative sensitivity, as a host array of
        shape :attr:`nb_pixels`:

            g = sum_q w_q  (forward_macro + grad(forward_disp))_q
                          . (costate_macro + grad(costate_disp))_q

        The sensitivity of ``vᵀ (Bᵀ W κ (Ē + B u))`` with respect to the
        density of a pixel is then ``dkappa/drho * g`` (the caller applies
        the chain rule); there is only one material coefficient here, unlike
        elasticity's shear/volumetric split."""
        self.engine.communicate_ghosts(forward_disp)
        self.engine.communicate_ghosts(costate_disp)
        self.grad.apply(forward_disp, self._g)
        g_fwd = self.to_host(self._g.s).copy()
        self.grad.apply(costate_disp, self._g)
        g_co = self.to_host(self._g.s)

        shape = (self.dim,) + (1,) * (g_fwd.ndim - 1)
        g_fwd = g_fwd + np.asarray(forward_macro, dtype=float).reshape(shape)
        g_co = g_co + np.asarray(costate_macro, dtype=float).reshape(shape)

        dot = np.einsum("i...,i...->...", g_fwd, g_co)  # (nb_quad, *nb_pixels)
        return np.einsum("q,q...->...", self.quad_weights, dot)

    def homogenized_tangent(self):
        """Effective conductivity tensor of the current density, as a
        ``(dim, dim)`` host array ``A`` whose row ``i`` is the homogenized
        flux :meth:`homogenized_flux` for the unit macro gradient
        ``Ē = e_i``, i.e. ``A[i, j] = q̄_j(e_i) = κ_eff[j, i]``.

        Each of the ``dim`` forward problems is solved with ``rtol=1e-8``
        (raised to the precision floor in float32), independent of
        :attr:`cg_tol`. Same sign convention as :meth:`homogenized_flux`.
        Uses the scratch field ``to_prob_u_homogenization``."""

        macro_gradients = np.eye(self.dim)
        _u = self.temperature_field("to_prob_u_homogenization")

        tangent = []
        cg_iters = []
        for i in np.arange(self.dim):
            macro_gradient = macro_gradients[i]

            u = self.solve_macro(
                macro_gradient, _u, rtol=1e-8,
                label=f"case {i + 1} fwd")
            cg_iters.append(self.last_cg_iters)
            q = self.homogenized_flux(u, macro_gradient)
            tangent.append(q)

        return np.asarray(tangent)
