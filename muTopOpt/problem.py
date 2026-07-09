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
                 design="element", consistent_objective=True, hessian=False):
        """``design='element'`` optimizes a per-pixel density (the material of
        each element is ``SIMP(rho_e)`` directly). ``design='nodal'`` optimizes
        a *nodal* FE density: each element's material is ``SIMP(rho_e)`` with
        ``rho_e`` the exact element average of the nodal interpolant (see
        :class:`muTopOpt.nodal.NodalElementMap`), so every nodal degree of
        freedom couples its ``2^dim`` adjacent elements -- an implicit
        sensitivity filter. Pair the nodal design with
        :class:`~muTopOpt.regularization.NodalPhaseFieldRegularization`.

        ``consistent_objective=True`` (default) reports the *Lagrangian*
        ``L = f + Σ_Γ λ_Γᵀ (K u_Γ - b_Γ)`` instead of the raw objective: the
        adjoint-weighted residual of each (possibly truncated) forward solve
        cancels the first-order effect of the solve error, making the reported
        value second-order accurate in the CG tolerance and *consistent* with
        the adjoint gradient (which is exactly ``∂L/∂ρ``). At converged solves
        the correction vanishes; with it, loose inner tolerances (e.g.
        ``cg_tol ~ 1e-3``) suffice for clean L-BFGS line searches.

        ``hessian=True`` enables exact (second-order-adjoint) Hessian-vector
        products (:meth:`hessian_vector_product`, for the trust-region
        Newton-CG optimizer). It allocates *four* vector fields per load case
        -- the forward/adjoint solutions that must survive the load-case loop,
        plus the sensitivity/co-state fields kept per case so the Hv solves can
        warm-start from the previous product -- and a handful of shared scratch
        fields, so it is opt-in."""
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
        self.consistent_objective = bool(consistent_objective)
        self.hessian = bool(hessian)

        # Optional adaptive inner-solve tolerance controller
        # (:class:`muTopOpt.optimize.AdaptiveInnerTolerance`). When set, every
        # forward/adjoint solve uses ``inner_tolerance.current`` as its CG
        # relative tolerance instead of the fixed ``Homogenization.cg_tol``,
        # and each evaluation reports its box-KKT gradient norm back to the
        # controller. Attached by the optimizer drivers; ``None`` = fixed tol.
        self.inner_tolerance = None

        # Per-load-case solver fields, reused across iterations.
        self._u = self.h.vector_field("to_prob_u")
        self._adj = self.h.vector_field("to_prob_adjoint")
        self._adj_rhs = self.h.vector_field("to_prob_adjoint_rhs")
        self._g_shear = self.h.scalar_field("to_prob_g_shear")
        self._g_vol = self.h.scalar_field("to_prob_g_vol")
        if self.consistent_objective:
            # Final residual of the forward solve, kept until the adjoint of
            # the same load case is available.
            self._res_u = self.h.vector_field("to_prob_forward_residual")

        if self.hessian:
            # Per-case solution/adjoint fields (the Hessian-vector product
            # needs u_G and lambda_G of *every* case, so they cannot share the
            # single scratch fields above) plus Hv scratch.
            n = len(self.load_cases)
            self._u_cases = [
                self.h.vector_field(f"to_prob_u_case{i}") for i in range(n)]
            self._adj_cases = [
                self.h.vector_field(f"to_prob_adjoint_case{i}")
                for i in range(n)]
            self._dlam_v = self.h.scalar_field("to_prob_hv_dlam")
            self._dmu_v = self.h.scalar_field("to_prob_hv_dmu")
            self._hv_rhs = self.h.vector_field("to_prob_hv_rhs")
            self._hv_tmp = self.h.vector_field("to_prob_hv_tmp")
            # Per-case sensitivity/co-state fields. Kept per case (not shared)
            # so each Hessian-vector product warm-starts its two solves from
            # the same load case's previous solution -- successive Hv products
            # (trust-region CG iterations) have a slowly-varying rhs, so the
            # warm start cuts the CG count. Zero-initialized, so the first Hv
            # cold-starts. (2*n_cases extra vector fields; see the docstring.)
            self._du_cases = [
                self.h.vector_field(f"to_prob_hv_du_case{i}")
                for i in range(n)]
            self._dadj_cases = [
                self.h.vector_field(f"to_prob_hv_dadj_case{i}")
                for i in range(n)]
        # Host-side snapshot of the last evaluation (design iterate, per-case
        # sensitivity kernels and stress data) that hessian_vector_product
        # differentiates around; None until the first evaluation.
        self._cache = None

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

        # Adaptive inner tolerance: the current (frozen-per-outer-iterate)
        # forcing-term value, or None to use the fixed Homogenization.cg_tol.
        rtol = (self.inner_tolerance.current
                if self.inner_tolerance is not None else None)

        f = 0.0
        grad = np.zeros_like(rho_e)
        stresses = []
        corrections = []  # adjoint-weighted forward residuals (one per case)
        cg_iters = []  # CG iteration count of each (forward, adjoint) solve
        hv_cache = ({"g_shear": [], "g_vol": [], "S": []}
                    if self.hessian else None)
        for i, lc in enumerate(self.load_cases):
            norm = float(np.sum(lc.target_stress**2))
            # Forward equilibrium. With Hessian support the solution must
            # survive the loop (one field per case).
            u = h.solve_macro(
                lc.macro_strain,
                self._u_cases[i] if self.hessian else self._u, rtol=rtol,
                residual=self._res_u if self.consistent_objective else None,
                label=f"case {i + 1} fwd")
            cg_iters.append(h.last_cg_iters)
            sigma = h.homogenized_stress(u, lc.macro_strain)
            stresses.append(sigma)
            diff = sigma - lc.target_stress
            f += lc.weight * float(np.sum(diff**2)) / norm

            # Adjoint: S = df/d<sigma>; rhs = -(1/V) Bᵀ C : S; solve K adj = rhs.
            S = 2.0 * lc.weight * diff / norm
            h.macro_rhs_tensor(S, self._adj_rhs, scale=-1.0 / V)
            adj_scale = h.mat_scale * max(float(np.abs(S / V).max()), 1e-300)
            adj = h.solve_rhs(self._adj_rhs,
                              self._adj_cases[i] if self.hessian
                              else self._adj, rtol=rtol,
                              rhs_scale=adj_scale, label=f"case {i + 1} adj")
            cg_iters.append(h.last_cg_iters)

            if self.consistent_objective:
                # Lagrangian correction λᵀ(K u - b) = -λᵀ r with the CG
                # residual r = b - K u: cancels the first-order effect of the
                # truncated forward solve on the objective (the adjoint
                # equation K λ = -∂f/∂u makes ∂f/∂u δu + λᵀ K δu vanish), so
                # the reported value is second-order accurate in the solve
                # error and exactly the function whose ρ-derivative the
                # gradient below is.
                corr = -h.comm.sum(float(
                    h._xp.sum(adj.p * self._res_u.p)))
                f += corr
                corrections.append(corr)

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

            if self.hessian:
                # Snapshot the per-case kernels and adjoint source for the
                # Hessian-vector product (the kernel fields are reused by the
                # next case, so copy to host).
                hv_cache["g_shear"].append(np.array(g_shear, copy=True))
                hv_cache["g_vol"].append(np.array(g_vol, copy=True))
                hv_cache["S"].append(S)

        if self.hessian:
            hv_cache["rho"] = np.array(rho, copy=True)
            hv_cache["rho_e"] = (np.array(rho_e, copy=True)
                                 if self.design == "nodal" else
                                 hv_cache["rho"])
            self._cache = hv_cache

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

        if self.inner_tolerance is not None:
            # Report the box-KKT gradient norm to the tolerance controller so
            # it can retune the inner accuracy at the next accepted outer
            # iterate. This is the *masked* infinity norm -- box-active
            # components whose gradient points further into the bound are
            # zeroed, matching the stationarity measure l_bfgs_bounded
            # converges on (NuMPI ``_kkt_residual``). ``rho`` is the design
            # variable (element or nodal) the optimizer bounds in [lo, hi].
            lo, hi = self.inner_tolerance.bounds
            tol_box = 1e-12
            r = np.abs(grad)
            if lo is not None:
                r[(rho <= lo + tol_box) & (grad >= 0.0)] = 0.0
            if hi is not None:
                r[(rho >= hi - tol_box) & (grad <= 0.0)] = 0.0
            local = float(r.max()) if r.size else 0.0
            gnorm = float(h.comm.max(local))
            self.inner_tolerance.observe(gnorm)

        self.last = {"objective": f, "stresses": stresses,
                     "cg_iters": cg_iters, "corrections": corrections,
                     "cg_rtol": rtol}
        return f, grad

    def ensure_state(self, rho):
        """Make the cached linearization point match ``rho``, re-evaluating
        the objective/gradient if it does not.

        The Hessian-vector product differentiates around the *last evaluated*
        iterate (cached per-case solutions, material state, preconditioner).
        A trust-region optimizer evaluates trial points that may be rejected,
        after which the cache belongs to the rejected point -- this guard
        re-primes it at the current iterate. The mismatch decision is
        MPI-reduced so all ranks take the same path."""
        if not self.hessian:
            raise RuntimeError(
                "Hessian support is off; construct the problem with "
                "hessian=True")
        rho = np.asarray(rho, dtype=float)
        stale = 1.0
        if self._cache is not None:
            c = self._cache["rho"]
            if c.shape == rho.shape and bool(np.array_equal(c, rho)):
                stale = 0.0
        if self.h.comm.max(stale) > 0.0:
            self.objective_and_gradient(rho)

    def hessian_vector_product(self, v, rtol=None):
        """Return the exact reduced-Hessian action ``H v`` on a design-space
        direction ``v`` by the second-order adjoint method, around the last
        evaluated iterate (see :meth:`ensure_state`).

        Per load case this costs two CG solves with the *same* operator and
        preconditioner as the forward problem:

        * forward sensitivity  ``K du = -(dK v) u - Bᵀ (dC v):Ē``;
        * second adjoint       ``K dl = -(dK v) l - (1/V) Bᵀ [(dC v):S + C:dS]``
          with ``dS = (2w/n) d<sigma>``, where
          ``d<sigma> = (1/V)∫[(dC v):(Ē+∇u) + C:∇du]``.

        The operator is linear in the Lamé fields, so ``(dK v) u`` is simply
        the fused stiffness ``apply`` with the perturbed fields
        ``dλ/dρ·v, dμ/dρ·v`` -- no new kernels are needed. The chain rule of
        the gradient assembly then gives

            H v = [2 d²μ·v g_shear + d²λ·v g_vol]                (pointwise)
                + kernel(du, 0;  l, S/V) + kernel(u, Ē;  dl, dS/V)
                + H_reg v,

        contracted with (2 dμ, dλ) as in the gradient. The model Hessian is
        exact up to the CG tolerance of the two solves; a trust-region method
        only requires it to be bounded, so ``rtol`` may be loose.

        Note ``W''`` of the double-well makes ``H`` genuinely indefinite --
        intended for optimizers that exploit negative curvature (Steihaug).
        """
        if self._cache is None:
            raise RuntimeError(
                "no cached linearization point; call ensure_state(rho) or "
                "objective_and_gradient(rho) first")
        h = self.h
        V = h.domain_volume
        d = self.dim
        cache = self._cache
        rho_e = cache["rho_e"]

        v = np.asarray(v, dtype=float)
        # The reduced Hessian is a linear operator, so H v = ||v|| * H (v/||v||)
        # exactly. Solve for the *unit* direction and rescale: this keeps every
        # Hv-solve right-hand side at a comparable magnitude regardless of the
        # trust-region CG direction's scale, which (a) makes warm-starting from
        # the previous product well-scaled and (b) makes the fixed relative CG
        # tolerance mean the same accuracy every time.
        vnorm = float(np.sqrt(h.comm.sum(float(np.sum(v * v)))))
        if vnorm == 0.0:
            self.last_hv_cg_iters = []
            return np.zeros_like(v)
        vhat = v / vnorm
        v_e = (self._nodal_map.gather_mean(vhat)
               if self.design == "nodal" else vhat)

        dlam, dmu = h.material.dlame(rho_e)
        d2lam, d2mu = h.material.d2lame(rho_e)

        # Perturbed Lamé fields: (dC v) as material fields for the fused ops.
        dlam_v = dlam * v_e
        dmu_v = dmu * v_e
        self._dlam_v.p[...] = h.to_device(dlam_v)
        self._dmu_v.p[...] = h.to_device(dmu_v)
        h.engine.communicate_ghosts(self._dlam_v)
        h.engine.communicate_ghosts(self._dmu_v)
        # Force scale of the perturbed material (for negligible-rhs detection
        # in the solves below), MPI-reduced.
        local = (max(float(np.abs(dlam_v).max()),
                     float(2.0 * np.abs(dmu_v).max()))
                 if dlam_v.size else 0.0)
        pert_scale = h.comm.max(local)

        hv_e = np.zeros_like(v_e)
        cg_iters = []
        zero_E = [0.0] * (d * d)
        for i, lc in enumerate(self.load_cases):
            u = self._u_cases[i]
            adj = self._adj_cases[i]
            du = self._du_cases[i]
            dadj = self._dadj_cases[i]
            E = lc.macro_strain
            norm = float(np.sum(lc.target_stress**2))
            S = cache["S"][i]

            # Forward sensitivity: K du = -(dK v) u - Bᵀ (dC v):Ē. Warm-started
            # from this case's previous product (rhs is O(1) in the unit
            # direction, so the guess is well-scaled).
            h.engine.communicate_ghosts(u)
            h.op.apply(u, self._dlam_v, self._dmu_v, self._hv_rhs)
            h.op.apply_macro_rhs(self._dlam_v, self._dmu_v,
                                 list(E.ravel()), self._hv_tmp)
            self._hv_rhs.s[...] = -self._hv_rhs.s - self._hv_tmp.s
            scale_fwd = pert_scale * max(float(np.abs(E).max()), 1e-300)
            h.solve_rhs(self._hv_rhs, du, rtol=rtol, rhs_scale=scale_fwd,
                        warm_start=True, label=f"case {i + 1} hv-fwd")
            cg_iters.append(h.last_cg_iters)

            # d<sigma> = (1/V)∫ (dC v):(Ē + ∇u) + (1/V)∫ C:∇du.
            dsigma = (
                h.homogenized_stress(u, E, lam=self._dlam_v, mu=self._dmu_v)
                + h.homogenized_stress(du, np.zeros((d, d)))
            )
            dS = 2.0 * lc.weight * dsigma / norm

            # Second adjoint:
            # K dl = -(dK v) l - (1/V) Bᵀ [(dC v):S + C:dS].
            h.engine.communicate_ghosts(adj)
            h.op.apply(adj, self._dlam_v, self._dmu_v, self._hv_rhs)
            h.op.apply_macro_rhs(self._dlam_v, self._dmu_v,
                                 list(S.ravel()), self._hv_tmp)
            self._hv_rhs.s[...] = -self._hv_rhs.s - self._hv_tmp.s / V
            h.op.apply_macro_rhs(h.lam, h.mu, list(dS.ravel()), self._hv_tmp)
            self._hv_rhs.s[...] -= self._hv_tmp.s / V
            scale_adj = max(
                pert_scale * float(np.abs(S / V).max()),
                h.mat_scale * float(np.abs(dS / V).max()), 1e-300)
            h.solve_rhs(self._hv_rhs, dadj, rtol=rtol, rhs_scale=scale_adj,
                        warm_start=True, label=f"case {i + 1} hv-adj")
            cg_iters.append(h.last_cg_iters)

            # Chain rule of grad = dC (Ē+∇u):(S/V+∇l), term by term:
            # (a) second material derivative, with the cached kernels;
            hv_e += ((2.0 * d2mu * cache["g_shear"][i]
                      + d2lam * cache["g_vol"][i]) * v_e)
            # (b) state variation: dC (∇du):(S/V + ∇l);
            h.engine.communicate_ghosts(du)
            h.op.compute_sensitivity(
                du, zero_E, adj, list((S / V).ravel()),
                self._g_shear, self._g_vol)
            hv_e += (2.0 * dmu * h.to_host(self._g_shear.p)
                     + dlam * h.to_host(self._g_vol.p))
            # (c) costate variation: dC (Ē+∇u):(dS/V + ∇dl).
            h.engine.communicate_ghosts(dadj)
            h.op.compute_sensitivity(
                u, list(E.ravel()), dadj, list((dS / V).ravel()),
                self._g_shear, self._g_vol)
            hv_e += (2.0 * dmu * h.to_host(self._g_shear.p)
                     + dlam * h.to_host(self._g_vol.p))

        if self.design == "nodal":
            hv = self._nodal_map.scatter_mean(hv_e)
        else:
            hv = hv_e

        if self.regularization is not None:
            # Also linear in the direction; evaluate on the unit direction to
            # match hv_e, then the common vnorm rescaling below recovers H v.
            hv = hv + self.regularization.hessian_vector_product(
                cache["rho"], vhat)

        self.last_hv_cg_iters = cg_iters
        # Undo the unit-direction normalization: H v = ||v|| * H (v/||v||).
        return vnorm * hv
