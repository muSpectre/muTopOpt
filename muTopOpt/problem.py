#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Stress-matching topology optimization problem.

Objective. For load cases Γ with prescribed macro strain ``Ē^Γ``, target
stress ``σ*^Γ`` and weight ``a_Γ``, minimize over the density ``ρ ∈ [0, 1]``

    f(ρ) = Σ_Γ a_Γ ‖σ̄^Γ(ρ) − σ*^Γ‖² / ‖σ*^Γ‖²  +  f_reg(ρ),

with Frobenius norms, the homogenized stress
``σ̄^Γ = (1/V) ∫ C(ρ):(Ē^Γ + ∇ˢu^Γ) dV`` and the fluctuation ``u^Γ`` solving
the cell problem ``K(ρ) u^Γ = b^Γ = −Bᵀ C(ρ):Ē^Γ`` (see
:mod:`muTopOpt.homogenization`). ``f_reg`` is the phase-field term of
:mod:`muTopOpt.regularization`. In 2D, 3 independent load cases constrain
the effective stiffness, in 3D 6 (:mod:`muTopOpt.loadcases`).

Adjoint gradient. With ``S^Γ = ∂f/∂σ̄^Γ = 2 a_Γ (σ̄^Γ − σ*^Γ)/‖σ*^Γ‖²`` the
Lagrangian

    L(ρ, u, v) = f + Σ_Γ v^Γᵀ (K u^Γ − b^Γ)

is stationary in ``u^Γ`` when the adjoint (costate) ``v^Γ`` solves

    K v^Γ = −∂f/∂u^Γ = −(1/V) Bᵀ C:S^Γ.

``K`` is symmetric, so the adjoint problem uses the same operator and the
same preconditioner as the forward problem. The total derivative is then

    df/dρ_e = ∫_e (Ē^Γ + ∇ˢu^Γ) : ∂C/∂ρ_e : (S^Γ/V + ∇ˢv^Γ) dV + ∂f_reg/∂ρ_e,

summed over Γ: the macro part ``S^Γ/V`` of the costate strain carries the
explicit dependence of ``σ̄^Γ`` on ``ρ``. Since ``∂C/∂ρ_e = 2 μ'(ρ_e) G +
λ'(ρ_e) V`` (the SIMP derivatives of :mod:`muTopOpt.material`), the fused
``compute_sensitivity`` kernel only returns the two geometric contractions
``g_shear = ∫_e ε(u):ε(v)`` and ``g_vol = ∫_e tr ε(u) tr ε(v)`` of the total
strains, and ``df/dρ_e = Σ_Γ [2 μ' g_shear + λ' g_vol]``.

Adjoint-corrected objective. The CG solves are truncated at a relative
tolerance, leaving residuals ``r^Γ = b^Γ − K u^Γ``. The reported objective is
``L = f − Σ_Γ v^Γᵀ r^Γ`` rather than ``f``: because the adjoint equation
makes ``∂f/∂u δu + vᵀ K δu`` vanish, the first-order effect of the forward
solve error cancels, so ``L`` differs from the exact objective by a term of
the order of the product of the forward and adjoint solve errors,
``O(rtol²)``, whereas ``f`` differs by ``O(rtol)``. ``L`` is also exactly
the function whose ``ρ``-derivative the adjoint gradient is, which keeps line
searches consistent with loose inner tolerances.

Hessian-vector products. :meth:`StressTargetProblem.hessian_vector_product`
differentiates the gradient once more along a direction ``δρ`` (second-order
adjoint): per load case one forward-sensitivity solve for ``δu`` and one
second-adjoint solve for ``δv``, again with ``K`` and its preconditioner.

Nodal design. With ``design="nodal"`` the design variable is a nodal density;
the element densities are its exact element averages
(:class:`muTopOpt.nodal.NodalElementMap`) and the element gradient is pulled
back with the transpose of that map.
"""

from dataclasses import dataclass

import numpy as np
from muGrid import linalg

from .timing import timer


@dataclass
class LoadCase:
    """A prescribed macro strain and the stress it should produce."""
    macro_strain: np.ndarray  # (dim, dim)
    target_stress: np.ndarray  # (dim, dim)
    weight: float = 1.0


class StressTargetProblem:
    def __init__(self, homogenization, load_cases, regularization=None,
                 design="element", consistent_objective=True, hessian=False,
                 hv_warm_start=False):
        """Stress-matching objective over a set of load cases (see the module
        docstring for the mathematics).

        Parameters
        ----------
        homogenization : Homogenization
            Solver of the cell problem; its material and preconditioner are
            updated on every evaluation.
        load_cases : sequence of LoadCase
            Macro strains, target stresses and weights (any object with
            ``macro_strain``, ``target_stress`` and ``weight`` attributes).
        regularization : object, optional
            A :class:`~muTopOpt.regularization.PhaseFieldRegularization` or
            :class:`~muTopOpt.regularization.NodalPhaseFieldRegularization`
            (default None: no regularization). Added to the objective,
            gradient and Hessian-vector product; it acts on the design
            variable itself (nodal for the nodal design).
        design : {"element", "nodal"}, optional
            ``"element"`` (default) optimizes a per-pixel density; the
            material of element ``e`` is ``C(ρ_e)``. ``"nodal"`` optimizes a
            *nodal* FE density whose exact element averages ``ρ_e`` (see
            :class:`muTopOpt.nodal.NodalElementMap`) set the material, so
            every nodal degree of freedom couples its ``2^d`` adjacent
            elements -- an implicit sensitivity filter. Pair it with
            :class:`~muTopOpt.regularization.NodalPhaseFieldRegularization`.
        consistent_objective : bool, optional
            If True (default), report the adjoint-corrected Lagrangian
            ``L = f − Σ_Γ v^Γᵀ r^Γ`` instead of ``f``. ``L`` is second-order
            accurate in the CG tolerance and is exactly the function whose
            derivative the adjoint gradient is, so loose inner tolerances
            (e.g. ``cg_tol ~ 1e-3``) still give consistent L-BFGS line
            searches. The correction vanishes for converged solves. The
            remaining ``O(rtol²)`` error means values computed at
            *different* inner tolerances are not directly comparable: when an
            adaptive controller tightens the tolerance between outer
            iterates, the reported objective can increase by up to the old
            tolerance's evaluation error (bounded by
            ``sum(|last['corrections']|)``) although the true objective
            decreased. Compare values at equal ``cg_rtol`` only (both are
            recorded in :attr:`last`).
        hessian : bool, optional
            If True, enable exact (second-order adjoint) Hessian-vector
            products (:meth:`hessian_vector_product`, used by the
            trust-region Newton-CG optimizer). This keeps the forward and
            adjoint solutions of every load case resident (two vector fields
            per case) plus shared scratch, so it is opt-in (default False).
        hv_warm_start : bool, optional
            Only with ``hessian=True``. If True, keep the two Hessian-vector
            solution fields per load case (another two vector fields per
            case) so that each product warm-starts its solves from the same
            case's previous product. Default False: the two fields are
            shared by all cases and every solve starts from zero (see the
            comment in the constructor for the measured trade-off).

        Raises
        ------
        ValueError
            If ``design`` is neither ``"element"`` nor ``"nodal"``.
        """
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
        # Mesh-invariant gradient scale V/V_e (= the global pixel count):
        # stationarity norms are reported on the volume-fraction derivative
        # ĝ = (V/V_e) ∂f/∂ρ, whose magnitude does not depend on the grid
        # resolution (the raw per-pixel gradient is O(V_e/V)). Matches the
        # units of the optimizer drivers' gtol (muTopOpt.optimize.
        # _gradient_scale is the reciprocal).
        self._gnorm_scale = (homogenization.domain_volume
                             / float(np.prod(homogenization.grid_spacing)))

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
            # needs u^Γ and v^Γ of *every* case, so they cannot share the
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
            # Sensitivity/co-state fields (du, dv) of the Hessian-vector
            # product. Kept per case with hv_warm_start=True, so each product
            # warm-starts its two solves from the same load case's previous
            # solution: successive Hv products (trust-region CG iterations)
            # have a slowly varying rhs. Zero-initialized, so the first Hv
            # cold-starts.
            #
            # Per case they are 2*n_cases vector fields, at scale the largest
            # block of device memory the optimizer holds (2x6 fields of
            # 3x98^3 floats incl. ghosts = 135 MB at 96^3 in single
            # precision, ~28% of the resident set). Shared fields with cold
            # starts cost 2-4% more inner CG iterations (3D runs, 32^3: 5779
            # -> 5989 iterations; wall time 11.14 s -> 11.15 s at 32^3 and
            # 33.6 s -> 34.0 s at 48^3). Hence shared fields by default.
            # TODO: record the hardware and command of these timings.
            self.hv_warm_start = bool(hv_warm_start)
            if self.hv_warm_start:
                self._du_cases = [
                    self.h.vector_field(f"to_prob_hv_du_case{i}")
                    for i in range(n)]
                self._dadj_cases = [
                    self.h.vector_field(f"to_prob_hv_dadj_case{i}")
                    for i in range(n)]
            else:
                du = self.h.vector_field("to_prob_hv_du")
                dadj = self.h.vector_field("to_prob_hv_dadj")
                self._du_cases = [du] * n
                self._dadj_cases = [dadj] * n
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
        """Return ``(f, df/dρ)`` for a density array of shape
        :attr:`Homogenization.nb_pixels` (element-wise or nodal, per the
        ``design`` chosen at construction).

        ``f`` is the MPI-reduced objective (the adjoint-corrected Lagrangian
        if ``consistent_objective``) including the regularization; the
        gradient is the rank-local slice. Diagnostics of the evaluation are
        stored in :attr:`last`: ``objective``, ``stresses`` (``σ̄^Γ``),
        ``cg_iters`` (forward and adjoint per case), ``corrections`` (the
        Lagrangian corrections ``−v^Γᵀ r^Γ``) and ``cg_rtol`` (``None`` for
        the fixed ``Homogenization.cg_tol``).
        """
        with timer("objective_and_gradient"):
            return self._objective_and_gradient(rho)

    def _objective_and_gradient(self, rho):
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
            with timer("forward solve"):
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
            with timer("adjoint solve"):
                adj = h.solve_rhs(self._adj_rhs,
                                  self._adj_cases[i] if self.hessian
                                  else self._adj, rtol=rtol,
                                  rhs_scale=adj_scale,
                                  label=f"case {i + 1} adj")
            cg_iters.append(h.last_cg_iters)

            if self.consistent_objective:
                # Lagrangian correction λᵀ(K u - b) = -λᵀ r with the CG
                # residual r = b - K u: cancels the first-order effect of the
                # truncated forward solve on the objective (the adjoint
                # equation K λ = -∂f/∂u makes ∂f/∂u δu + λᵀ K δu vanish), so
                # the reported value is second-order accurate in the solve
                # error and exactly the function whose ρ-derivative the
                # gradient below is.
                # vecdot reduces the interior in double and needs no
                # temporary, whereas `xp.sum(adj.p * res.p)` would build a
                # full-size product array (~dim*N*4 bytes in float32) and sum
                # it in the field precision. This value is the reported
                # objective *and* feeds the trust region's accuracy control,
                # so its error budget is the tightest in the package.
                corr = -h.comm.sum(float(linalg.vecdot(adj, self._res_u)))
                f += corr
                corrections.append(corr)

            # Sensitivity: geometry contractions with total forward strain
            # (macro Ē) and total costate strain (fluctuation adj + macro S/V);
            # the S/V macro folds in the explicit d<sigma>/drho term.
            with timer("sensitivity"):
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
            with timer("regularization"):
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
            # Reported in mesh-invariant ĝ units (see _gnorm_scale) so the
            # controller history is human-readable and consistent with the
            # drivers' gtol/max_grad; the relative forcing term itself is
            # unit-agnostic.
            gnorm = float(h.comm.max(local)) * self._gnorm_scale
            self.inner_tolerance.observe(gnorm)

        self.last = {"objective": f, "stresses": stresses,
                     "cg_iters": cg_iters, "corrections": corrections,
                     "cg_rtol": rtol}
        return f, grad

    def ensure_state(self, rho):
        """Make the cached linearization point match ``rho``, re-evaluating
        the objective/gradient if it does not. Raises ``RuntimeError`` if
        the problem was constructed with ``hessian=False``.

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
        """Return the reduced-Hessian action ``H δρ`` on a design-space
        direction ``δρ`` (the argument ``v``) by the second-order adjoint
        method, around the last evaluated iterate (see :meth:`ensure_state`).

        Let ``δC = ∂C/∂ρ · δρ`` (the perturbed Lamé fields ``λ'δρ, μ'δρ``) and
        ``δK`` the stiffness assembled from them. Per load case this costs
        two CG solves with the *same* operator and preconditioner as the
        forward problem:

        * forward sensitivity  ``K δu = −δK u − Bᵀ δC:Ē``;
        * second adjoint       ``K δv = −δK v − (1/V) Bᵀ [δC:S + C:δS]``,

        with ``δS = 2a (δσ̄)/‖σ*‖²`` and
        ``δσ̄ = (1/V) ∫ [δC:(Ē + ∇ˢu) + C:∇ˢδu] dV``. ``K`` is linear in the
        Lamé fields, so ``δK u`` is the fused stiffness ``apply`` with the
        perturbed fields -- no further kernels are needed. Differentiating
        the gradient ``∫_e (Ē + ∇ˢu):∂C/∂ρ_e:(S/V + ∇ˢv)`` gives, per
        element,

            (H δρ)_e = [2 μ'' g_shear + λ'' g_vol]_e δρ_e
                     + ∫_e ∇ˢδu : ∂C/∂ρ_e : (S/V + ∇ˢv)
                     + ∫_e (Ē + ∇ˢu) : ∂C/∂ρ_e : (δS/V + ∇ˢδv),

        summed over load cases, plus the regularization term ``H_reg δρ``;
        for the nodal design ``δρ`` is mapped to elements and the result back
        to nodes with :class:`~muTopOpt.nodal.NodalElementMap`. The product
        is exact up to the CG tolerance ``rtol`` of the two solves (``None``
        uses ``Homogenization.cg_tol``); a trust-region method only
        requires it to be bounded, so ``rtol`` may be loose. The solves use
        the unit direction ``δρ/‖δρ‖`` and the result is rescaled, so the
        relative tolerance means the same accuracy for every direction.

        ``W''(1/2) < 0`` of the double well makes ``H`` indefinite; the
        product is intended for optimizers that exploit negative curvature
        (Steihaug-CG).

        Raises
        ------
        RuntimeError
            If no linearization point has been evaluated yet.
        """
        with timer("hessian_vector_product"):
            return self._hessian_vector_product(v, rtol=rtol)

    def _hessian_vector_product(self, v, rtol=None):
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
            if not self.hv_warm_start:
                # Shared scratch: it holds the previous *case's* solution, not
                # this one's, so start from zero rather than from an unrelated
                # iterate.
                du.set_zero()
                dadj.set_zero()
            E = lc.macro_strain
            norm = float(np.sum(lc.target_stress**2))
            S = cache["S"][i]

            # Forward sensitivity: K du = -(dK v) u - Bᵀ (dC v):Ē. With
            # hv_warm_start, warm-started from this case's previous product
            # (the rhs is O(1) in the unit direction, so the guess is
            # well-scaled); otherwise du was zeroed above and solve_rhs's
            # warm start reduces to a cold start.
            with timer("hv forward solve"):
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
            with timer("hv adjoint solve"):
                h.engine.communicate_ghosts(adj)
                h.op.apply(adj, self._dlam_v, self._dmu_v, self._hv_rhs)
                h.op.apply_macro_rhs(self._dlam_v, self._dmu_v,
                                     list(S.ravel()), self._hv_tmp)
                self._hv_rhs.s[...] = -self._hv_rhs.s - self._hv_tmp.s / V
                h.op.apply_macro_rhs(h.lam, h.mu, list(dS.ravel()),
                                     self._hv_tmp)
                self._hv_rhs.s[...] -= self._hv_tmp.s / V
                scale_adj = max(
                    pert_scale * float(np.abs(S / V).max()),
                    h.mat_scale * float(np.abs(dS / V).max()), 1e-300)
                h.solve_rhs(self._hv_rhs, dadj, rtol=rtol,
                            rhs_scale=scale_adj, warm_start=True,
                            label=f"case {i + 1} hv-adj")
            cg_iters.append(h.last_cg_iters)

            # Chain rule of grad = dC (Ē+∇u):(S/V+∇v), term by term:
            with timer("hv sensitivity"):
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
                # (c) costate variation: dC (Ē+∇u):(dS/V + ∇dv).
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
            with timer("regularization"):
                hv = hv + self.regularization.hessian_vector_product(
                    cache["rho"], vhat)

        self.last_hv_cg_iters = cg_iters
        # Undo the unit-direction normalization: H v = ||v|| * H (v/||v||).
        return vnorm * hv
