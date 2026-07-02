#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Stress-matching topology optimization problem (element-wise density).

Minimize, over the per-pixel density ``rho in [0, 1]``,

    f(rho) = Σ_Γ a_Γ ‖⟨σ^Γ⟩ - σ_target^Γ‖² / ‖σ_target^Γ‖²   +   f_reg(rho)

subject to mechanical equilibrium for each load case Γ (a prescribed macro
strain ``Ē^Γ``), solved by the FFT-accelerated FE solver. The gradient is
obtained by the discrete adjoint method: because the system matrix is symmetric,
the adjoint problem uses the same operator and preconditioner as the forward
solve, and its right-hand side is ``-(1/V) Bᵀ C : S^Γ`` with
``S^Γ = ∂f/∂⟨σ^Γ⟩``. The per-pixel material-derivative sensitivity is assembled
by the fused ``compute_sensitivity`` kernel (which returns the geometry
contractions ``g_shear = ε(u):ε(costate)``, ``g_vol = tr·tr``) followed by the
SIMP chain rule ``d(2μ)/drho · g_shear + dλ/drho · g_vol``.

This holds in 2D and 3D unchanged; a 2D problem takes 3 independent load cases
to constrain the effective stiffness, a 3D problem 6.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadCase:
    """A prescribed macro strain and the stress it should produce."""
    macro_strain: np.ndarray  # (dim, dim)
    target_stress: np.ndarray  # (dim, dim)
    weight: float = 1.0


class StressTargetProblem:
    def __init__(self, homogenization, load_cases, regularization=None,
                 design="element"):
        """``design='element'`` optimizes a per-pixel density (the material of
        each element is ``SIMP(rho_e)`` directly). ``design='nodal'`` optimizes
        a *nodal* FE density: each element's material is ``SIMP(rho_e)`` with
        ``rho_e`` the exact element average of the nodal interpolant (see
        :class:`muTopOpt.nodal.NodalElementMap`), so every nodal degree of
        freedom couples its ``2^dim`` adjacent elements -- an implicit
        sensitivity filter. Pair the nodal design with
        :class:`~muTopOpt.regularization.NodalPhaseFieldRegularization`."""
        self.h = homogenization
        self.dim = homogenization.dim
        self.load_cases = [self._as_case(lc) for lc in load_cases]
        self.regularization = regularization
        if design not in ("element", "nodal"):
            raise ValueError(f"unknown design space '{design}'")
        self.design = design
        if design == "nodal":
            from .nodal import NodalElementMap

            self._nodal_map = NodalElementMap(homogenization)

        # Per-load-case solver fields, reused across iterations.
        self._u = self.h.vector_field("to_prob_u")
        self._adj = self.h.vector_field("to_prob_adjoint")
        self._adj_rhs = self.h.vector_field("to_prob_adjoint_rhs")
        self._g_shear = self.h.scalar_field("to_prob_g_shear")
        self._g_vol = self.h.scalar_field("to_prob_g_vol")

        self.last = {}  # diagnostics from the most recent evaluation

    def _as_case(self, lc):
        d = self.dim
        E = np.asarray(lc.macro_strain, dtype=float).reshape(d, d)
        S = np.asarray(lc.target_stress, dtype=float).reshape(d, d)
        return LoadCase(E, S, float(lc.weight))

    def objective_and_gradient(self, rho):
        """Return (f, df/drho) for a density array of shape
        :attr:`Homogenization.nb_pixels` (element-wise or nodal, per the
        ``design`` chosen at construction)."""
        rho = np.asarray(rho, dtype=float)
        h = self.h
        V = h.domain_volume
        if self.design == "nodal":
            # Element material from the exact element average of the nodal
            # interpolant; sensitivities are scattered back through the
            # transpose of this map after the load-case loop.
            rho_e = self._nodal_map.gather_mean(rho)
        else:
            rho_e = rho
        h.set_density(rho_e)

        dlam, dmu = h.material.dlame(rho_e)  # SIMP derivatives, per element

        f = 0.0
        grad = np.zeros_like(rho_e)
        stresses = []
        cg_iters = []  # CG iteration count of each (forward, adjoint) solve
        for lc in self.load_cases:
            norm = float(np.sum(lc.target_stress**2))
            # Forward equilibrium.
            u = h.solve_macro(lc.macro_strain, self._u)
            cg_iters.append(h.last_cg_iters)
            sigma = h.homogenized_stress(u, lc.macro_strain)
            stresses.append(sigma)
            diff = sigma - lc.target_stress
            f += lc.weight * float(np.sum(diff**2)) / norm

            # Adjoint: S = df/d<sigma>; rhs = -(1/V) Bᵀ C : S; solve K adj = rhs.
            S = 2.0 * lc.weight * diff / norm
            h.macro_rhs_tensor(S, self._adj_rhs, scale=-1.0 / V)
            adj_scale = h.mat_scale * max(float(np.abs(S / V).max()), 1e-300)
            adj = h.solve_rhs(self._adj_rhs, self._adj, rhs_scale=adj_scale)
            cg_iters.append(h.last_cg_iters)

            # Sensitivity: geometry contractions with total forward strain
            # (macro Ē) and total costate strain (fluctuation adj + macro S/V);
            # the S/V macro folds in the explicit d<sigma>/drho term.
            h.op.compute_sensitivity(
                u, list(lc.macro_strain.ravel()),
                adj, list((S / V).ravel()),
                self._g_shear, self._g_vol,
            )
            g_shear = h.to_host(self._g_shear.p)
            g_vol = h.to_host(self._g_vol.p)
            grad += 2.0 * dmu * g_shear + dlam * g_vol

        if self.design == "nodal":
            # Chain rule through the element average: the per-element
            # sensitivity is distributed onto the element's corner nodes with
            # the averaging weights (exact transpose of gather_mean).
            grad = self._nodal_map.scatter_mean(grad)

        if self.regularization is not None:
            # The regularization acts on the design variables themselves
            # (nodal for the nodal design).
            f_reg, g_reg = self.regularization.value_and_gradient(rho)
            f += f_reg
            grad += g_reg

        self.last = {"objective": f, "stresses": stresses,
                     "cg_iters": cg_iters}
        return f, grad
