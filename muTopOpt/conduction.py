#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
FFT-accelerated FE homogenization of steady-state heat conduction (scalar
diffusion) on a regular grid, for topology optimization.

:class:`HomogenizationConductivity` is the scalar-diffusion sibling of
:class:`muTopOpt.homogenization.Homogenization`: same muGrid engine, same
periodic-domain / ghost / CG-solve machinery, but the system matrix ``K`` is
assembled from muGrid's dimension-templated, physics-agnostic
``FEMGradientOperator`` (nodal scalar field -> gradient at quadrature points,
and its quadrature-weighted transpose) instead of the fused
``IsotropicStiffnessOperator`` elasticity kernel -- there is no fused
conductivity operator in muGrid (yet), but none is needed: the heterogeneous
operator is exactly

    K(rho) u = transpose(kappa(rho) * apply(u))

with a per-quadrature-point (piecewise-constant per pixel) scalar
conductivity ``kappa``, and ``transpose``'s default weighting already applies
the physical (volume-scaled) quadrature weights, verified against the
weighted weak form ``<v, K u> == sum_q w_q <grad v, grad u>_q`` for a random
heterogeneous ``kappa`` field, including self-adjointness.

Only an isotropic base material is supported (see :class:`SimpConductivity`):
the *target* conductivity tensor in a topology-optimization objective may
still be anisotropic (see :mod:`muTopOpt.loadcases_conduction`), only the
physical solid/void material itself is isotropic.  Both a plain Green's-
function (``'green'``) and the Green-Jacobi (``'green-jacobi'``) preconditioner
are available.  The Jacobi diagonal of the conductivity operator is assembled
via a ``2**dim``-colour graph-colouring scheme (applying the operator to one
bitmask colour at a time), which avoids the need for a fused
``assemble_diagonal`` kernel.
"""

import inspect as _inspect

import numpy as np

import muGrid
from muGrid.Preconditioners import (
    GreenJacobiPreconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import ConvergenceError, conjugate_gradients

_CG_HAS_RESIDUAL = "residual" in _inspect.signature(
    conjugate_gradients).parameters


class _CGStagnation(Exception):
    """Internal control flow of :meth:`HomogenizationConductivity.solve_rhs`;
    see :class:`muTopOpt.homogenization.Homogenization` for the rationale."""


def kappa_from_conductivity(k):
    """Identity helper kept for symmetry with
    :func:`muTopOpt.material.lame_from_E_nu` -- the conductivity material has
    only one parameter, so there is nothing to convert."""
    return float(k)


class SimpConductivity:
    """Power-law (SIMP) interpolation of an isotropic scalar conductivity.

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
        cg_tol=1e-8,
        cg_maxiter=2000,
        cg_verbose=False,
        dtype=np.float64,
    ):
        """See :class:`muTopOpt.homogenization.Homogenization` for the shared
        conventions (grid, MPI, dtype, CG hardening).  Two preconditioners are
        supported: ``'green-jacobi'`` (default, J-FFT Green-Jacobi) and
        ``'green'`` (plain reference-stiffness Green operator)."""
        self.dim = len(nb_grid_pts)
        if self.dim not in (2, 3):
            raise ValueError("nb_grid_pts must be 2- or 3-dimensional")
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

        ghosts = (1,) * self.dim
        self.engine = muGrid.FFTEngine(
            self.nb_grid_pts, self.comm,
            nb_ghosts_left=ghosts, nb_ghosts_right=ghosts,
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
        # Physical (volume-scaled) quadrature weights -- muGrid's
        # FEMGradientOperator.quadrature_weights already bakes in the cell
        # volume from the grid_spacing given at construction (Wfrac * vol_pixel,
        # verified empirically: sum(quadrature_weights) == prod(grid_spacing)),
        # so no extra volume factor is applied anywhere below.
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

        self.cg_tol = cg_tol
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
        # Scratch fields for _assemble_diagonal; allocated lazily on first use.
        self._diag_u = None
        self._diag_Ku = None
        self._nb_pixels = tuple(self.engine.nb_subdomain_grid_pts)

    @property
    def nb_pixels(self):
        """Local (owned) pixel grid shape -- the shape of an element-wise
        density array on this rank."""
        return self._nb_pixels

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
        ``diag(K)`` into ``diag_field`` using a ``2**dim``-colour graph-
        colouring scheme.

        Nodes coloured the same colour are not coupled by any element, so
        applying ``K`` to the characteristic vector of one colour selects
        exactly the diagonal entries of that colour: ``(K chi_c)[i] = K[i,i]``
        for every node ``i`` of colour ``c``."""
        import itertools
        # Allocate scratch fields lazily (once) and reuse across calls.
        if self._diag_u is None:
            self._diag_u = self.fc.real_field(
                "to_diag_u", dtype=self.dtype)
            self._diag_Ku = self.fc.real_field(
                "to_diag_Ku", dtype=self.dtype)
        diag_field.set_zero()
        for offsets in itertools.product(range(2), repeat=self.dim):
            self._diag_u.set_zero()
            slices = tuple(slice(o, None, 2) for o in offsets)
            self._diag_u.p[slices] = self._xp.ones(
                self._diag_u.p[slices].shape, dtype=self.dtype)
            self._hessp(self._diag_u, self._diag_Ku)
            diag_field.p[slices] = self._diag_Ku.p[slices]

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

    def solve_rhs(self, b, x, rtol=None, maxiter=None, rhs_scale=None,
                  residual=None, label=None, warm_start=False):
        """Solve ``K x = b`` in place; returns ``x``. Identical contract to
        :meth:`muTopOpt.homogenization.Homogenization.solve_rhs` -- see its
        docstring for the CG-hardening rationale (negligible-rhs shortcut,
        warm-start guard, stagnation safeguard); duplicated here rather than
        shared so the two engines stay independently modifiable."""
        if not warm_start:
            x.set_zero()
        bp = b.p.ravel()
        b_norm = np.sqrt(self.comm.sum(float(self._xp.dot(bp, bp))))
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
            self._Ku.s[...] = b.s - self._Ku.s
            rp = self._Ku.p.ravel()
            true_norm = np.sqrt(self.comm.sum(float(self._xp.dot(rp, rp))))
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
        ``K u = -transpose(kappa * E_macro)`` for the fluctuation temperature
        ``u`` under macro temperature gradient ``E_macro`` (a length-``dim``
        vector)."""
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
        """Assemble ``out = scale * transpose(kappa * E_macro)`` for a
        constant macro gradient vector ``E_macro`` (used to build the
        homogenization/adjoint right-hand sides)."""
        E = self.to_device(np.asarray(E_macro, dtype=float))
        shape = (self.dim,) + (1,) * (self._flux.s.ndim - 1)
        self._flux.s[...] = self.kappa.s * E.reshape(shape)
        self.engine.communicate_ghosts(self._flux)
        self.grad.transpose(self._flux, out)
        if scale != 1.0:
            out.s[...] *= scale
        return out

    def homogenized_flux(self, u, E_macro):
        """Cell-averaged flux ``<q> = (1/V) int kappa (E_macro + grad u) dV``
        as a length-``dim`` array (MPI-reduced).

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

        The material-derivative sensitivity of a SIMP-interpolated
        conductivity is then ``dkappa/drho * g`` (the caller applies the
        chain rule) -- there being only one material coefficient here, unlike
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
