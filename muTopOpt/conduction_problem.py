#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Flux-matching topology optimization problem (element-wise density) for
steady-state heat conduction.

Minimize, over the per-pixel density ``rho in [0, 1]``,

    f(rho) = Σ_Γ a_Γ ‖⟨q^Γ⟩ - q_target^Γ‖² / ‖q_target^Γ‖²   +   f_reg(rho)

subject to thermal equilibrium for each load case Γ (a prescribed macro
temperature gradient ``Ē^Γ``), solved by the FFT-accelerated FE solver of
:class:`muTopOpt.conduction.HomogenizationConductivity`. This is the scalar-
diffusion sibling of :class:`muTopOpt.problem.StressTargetProblem` -- same
adjoint construction (the operator is self-adjoint, so the adjoint solve
reuses the forward operator and preconditioner; its right-hand side is
``-(1/V) transpose(kappa * S^Γ)`` with ``S^Γ = ∂f/∂⟨q^Γ⟩``), same
Lagrangian-correction trick for the *consistent* objective under a truncated
inner solve, same :class:`muTopOpt.optimize.AdaptiveInnerTolerance` hook --
minus the nodal-density and Hessian-vector-product support
``StressTargetProblem`` also offers (not yet implemented for this physics;
see :mod:`muTopOpt.conduction`'s module docstring for the scope notes).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FluxLoadCase:
    """A prescribed macro temperature gradient and the flux it should
    produce."""
    macro_gradient: np.ndarray  # (dim,)
    target_flux: np.ndarray  # (dim,)
    weight: float = 1.0


class FluxTargetProblem:
    def __init__(self, homogenization, load_cases, regularization=None,
                 consistent_objective=True):
        """``consistent_objective=True`` (default) reports the *Lagrangian*
        ``L = f + Σ_Γ λ_Γᵀ (K u_Γ - b_Γ)`` instead of the raw objective, for
        the same reason as :class:`muTopOpt.problem.StressTargetProblem`: the
        adjoint-weighted residual of each (possibly truncated) forward solve
        cancels the first-order effect of the solve error, so the reported
        value is second-order accurate in the CG tolerance and consistent
        with the adjoint gradient. See that class's docstring for the full
        caveat about comparing values across different inner tolerances."""
        self.h = homogenization
        self.dim = homogenization.dim
        self.load_cases = [self._as_case(lc) for lc in load_cases]
        self.regularization = regularization
        self.consistent_objective = bool(consistent_objective)

        # Optional adaptive inner-solve tolerance controller (see
        # muTopOpt.optimize.AdaptiveInnerTolerance); attached by the optimizer
        # drivers, None = fixed HomogenizationConductivity.cg_tol.
        self.inner_tolerance = None
        # Mesh-invariant gradient scale V/V_e; see StressTargetProblem for the
        # rationale (matches the units of the optimizer drivers' gtol).
        self._gnorm_scale = (homogenization.domain_volume
                             / float(np.prod(homogenization.grid_spacing)))

        self._u = self.h.temperature_field("to_prob_u")
        self._adj = self.h.temperature_field("to_prob_adjoint")
        self._adj_rhs = self.h.temperature_field("to_prob_adjoint_rhs")
        if self.consistent_objective:
            self._res_u = self.h.temperature_field("to_prob_forward_residual")

        self.last = {}  # diagnostics from the most recent evaluation

    def _as_case(self, lc):
        d = self.dim
        E = np.asarray(lc.macro_gradient, dtype=float).reshape(d)
        q = np.asarray(lc.target_flux, dtype=float).reshape(d)
        return FluxLoadCase(E, q, float(lc.weight))

    def objective_and_gradient(self, rho):
        """Return (f, df/drho) for a density array of shape
        :attr:`HomogenizationConductivity.nb_pixels`."""
        rho = np.asarray(rho, dtype=float)
        h = self.h
        V = h.domain_volume
        h.set_density(rho)

        dkappa = h.material.dkappa(rho)  # SIMP derivative, per element

        rtol = (self.inner_tolerance.current
                if self.inner_tolerance is not None else None)

        f = 0.0
        grad = np.zeros_like(rho)
        fluxes = []
        corrections = []
        cg_iters = []
        for i, lc in enumerate(self.load_cases):
            norm = float(np.sum(lc.target_flux**2))
            u = h.solve_macro(
                lc.macro_gradient, self._u, rtol=rtol,
                residual=self._res_u if self.consistent_objective else None,
                label=f"case {i + 1} fwd")
            cg_iters.append(h.last_cg_iters)
            q = h.homogenized_flux(u, lc.macro_gradient)
            fluxes.append(q)
            diff = q - lc.target_flux
            f += lc.weight * float(np.sum(diff**2)) / norm

            # Adjoint: S = df/d<q>; rhs = -(1/V) transpose(kappa * S);
            # solve K adj = rhs.
            S = 2.0 * lc.weight * diff / norm
            h.macro_rhs_vector(S, self._adj_rhs, scale=-1.0 / V)
            adj_scale = h.mat_scale * max(float(np.abs(S / V).max()), 1e-300)
            adj = h.solve_rhs(self._adj_rhs, self._adj, rtol=rtol,
                              rhs_scale=adj_scale, label=f"case {i + 1} adj")
            cg_iters.append(h.last_cg_iters)

            if self.consistent_objective:
                corr = -h.comm.sum(float(
                    h._xp.sum(adj.p * self._res_u.p)))
                f += corr
                corrections.append(corr)

            g_sens = h.compute_sensitivity(
                u, list(lc.macro_gradient), adj, list(S / V))
            grad += dkappa * g_sens

        if self.regularization is not None:
            f_reg, g_reg = self.regularization.value_and_gradient(rho)
            f += f_reg
            grad += g_reg

        if self.inner_tolerance is not None:
            lo, hi = self.inner_tolerance.bounds
            tol_box = 1e-12
            r = np.abs(grad)
            if lo is not None:
                r[(rho <= lo + tol_box) & (grad >= 0.0)] = 0.0
            if hi is not None:
                r[(rho >= hi - tol_box) & (grad <= 0.0)] = 0.0
            local = float(r.max()) if r.size else 0.0
            gnorm = float(h.comm.max(local)) * self._gnorm_scale
            self.inner_tolerance.observe(gnorm)

        self.last = {"objective": f, "fluxes": fluxes,
                     "cg_iters": cg_iters, "corrections": corrections,
                     "cg_rtol": rtol}
        return f, grad
