#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Phase-field regularization of the density.

Following Bourdin/phase-field topology optimization, the density is regularized
by penalizing interfacial area,

    f_reg(rho) = eta * ∫ |∇rho|^2 dx  +  (well/eta) * ∫ rho^2 (1 - rho)^2 dx.

The gradient penalty smooths the design and removes mesh dependence; the
double-well drives rho toward {0, 1}. Together they yield sharp interfaces
without an explicit volume constraint.

Two variants of the gradient penalty are provided:

* :class:`PhaseFieldRegularization` (element-wise density) discretizes it with a
  finite-difference Laplacian on the pixel grid (:class:`muGrid.LaplaceOperator`;
  assumes cubic voxels).
* :class:`NodalPhaseFieldRegularization` (nodal FE density) uses the
  element-consistent H¹ seminorm ``rhoᵀ L rho`` with the scalar FE-Laplacian
  ``L = Bᵀ W B`` of the mechanics element (P1 or Q1). ``L`` is a constant
  ``3^dim`` stencil (recovered once from the element geometry) applied
  matrix-free by :class:`muGrid.GenericLinearOperator` -- no resident
  ``dim × nb_quad`` gradient field, so it is memory-lean at scale and handles
  anisotropic spacing.

The double-well term is pointwise (lumped nodal) in both.
"""

import numpy as np

import muGrid


class PhaseFieldRegularization:
    def __init__(self, homogenization, eta=1.0, well_weight=1.0):
        self.h = homogenization
        self.eta = float(eta)
        self.well = float(well_weight)

        h0 = self.h.grid_spacing[0]
        if not np.allclose(self.h.grid_spacing, h0):
            # The isotropic FD Laplacian assumes equal spacing; warn rather than
            # silently mis-scale the gradient penalty.
            import warnings

            warnings.warn(
                "PhaseFieldRegularization assumes cubic voxels; using "
                "grid_spacing[0] for the Laplacian.",
                RuntimeWarning,
            )
        # -Δ (positive definite) as a finite-difference stencil.
        self.laplace = muGrid.LaplaceOperator(self.h.dim, -1.0 / h0**2)
        self.vol_pixel = float(np.prod(self.h.grid_spacing))

        self._rho = self.h.scalar_field("to_reg_rho")
        self._lap = self.h.scalar_field("to_reg_lap")

    def value_and_gradient(self, rho):
        """Return (f_reg, df_reg/drho) for an element-wise density array."""
        rho = np.asarray(rho)
        self._rho.p[...] = rho
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._lap)  # (-Δ rho)
        lap = np.asarray(self._lap.p)

        grad_pen = self.h.comm.sum(float(np.sum(rho * lap))) * self.vol_pixel
        dwell = self.h.comm.sum(
            float(np.sum(rho**2 * (1.0 - rho) ** 2))
        ) * self.vol_pixel

        f = self.eta * grad_pen + (self.well / self.eta) * dwell

        # d/drho [rho^2 (1-rho)^2] = 2 rho (1 - rho)(1 - 2 rho)
        dwell_drho = 2.0 * rho * (1.0 - rho) * (1.0 - 2.0 * rho)
        g = (
            self.eta * 2.0 * lap * self.vol_pixel
            + (self.well / self.eta) * dwell_drho * self.vol_pixel
        )
        return f, g


def fe_laplacian_stencil(dim, grid_spacing, element):
    r"""Constant-coefficient stencil of the scalar FE-Laplacian
    ``L = Bᵀ W B`` (``∫ ∇φ_i · ∇φ_j``) for the given element (P1 or Q1) on a
    uniform grid with the given spacing.

    On a uniform periodic grid the assembled FE-Laplacian is translation-
    invariant, so it is a single ``3^dim`` stencil (a node couples only to
    neighbours sharing an element, i.e. within ±1 per axis). We recover it once,
    at set-up, by the impulse-response method: apply ``L`` to a unit nodal
    impulse on a small grid (via muGrid's ``FEMGradientOperator`` -- the only
    place the ``dim × nb_quad`` gradient is ever materialised, and only on a
    5^dim grid) and read the neighbourhood of the impulse. The returned
    ``(offset, stencil)`` are then applied on the full grid by a
    :class:`muGrid.GenericLinearOperator` -- a matrix-free, scalar-in/scalar-out
    convolution (host/GPU/MPI) that never allocates a full-grid quad field.
    """
    n = 5  # > stencil width (3); impulses stay in the interior, clear of ghosts
    ghosts = (1,) * dim
    grad = muGrid.FEMGradientOperator(dim, list(grid_spacing), element)
    nq = grad.nb_quad_pts
    fc = muGrid.GlobalFieldCollection(
        (n,) * dim, nb_ghosts_left=ghosts, nb_ghosts_right=ghosts,
        sub_pts={"quad": nq},
    )
    imp = fc.real_field("stencil_impulse", (1,))
    g = fc.real_field("stencil_grad", (grad.nb_output_components,), "quad")
    w = np.asarray(grad.quadrature_weights).reshape((1, nq) + (1,) * dim)

    def gradient_of_impulse(pos):
        # B applied to a unit nodal impulse at `pos` (interior -> no ghost
        # coupling); returns the (components, quad, *grid) gradient field.
        imp.p[...] = 0.0
        imp.p[(0, *pos)] = 1.0
        grad.apply(imp, g)
        return np.asarray(g.s).copy()

    # L = Bᵀ W B directly from `apply` (no `transpose`, which carries a 1/nq
    # normalisation): the stencil entry at offset k is the exact element-
    # Laplacian coupling  L_{0,k} = Σ_q w_q ∇φ_0·∇φ_k  =  <W B δ_0, B δ_k>.
    center = tuple(n // 2 for _ in range(dim))
    g0 = gradient_of_impulse(center)
    wg0 = w * g0
    stencil = np.zeros((3,) * dim)
    for off in np.ndindex(*((3,) * dim)):
        pos = tuple(center[d] + off[d] - 1 for d in range(dim))
        stencil[off] = float(np.sum(wg0 * gradient_of_impulse(pos)))
    return [-1] * dim, np.ascontiguousarray(stencil)


class NodalPhaseFieldRegularization:
    r"""Nodal-FE phase-field regularization.

    Same functional as :class:`PhaseFieldRegularization`,

        f_reg = eta * ∫|∇rho|^2 + (well/eta) * ∫ rho^2 (1 - rho)^2,

    but the density is a *nodal* finite-element field and the gradient penalty
    is the element-consistent H¹ seminorm ``eta * rhoᵀ L rho`` with the fused
    scalar FE-Laplacian ``L = Bᵀ W B`` of the mechanics element (P1 or Q1),
    rather than an ad-hoc finite-difference Laplacian. ``L`` is applied
    matrix-free by a :class:`muGrid.GenericLinearOperator` (no resident
    ``dim × nb_quad`` gradient field -- the memory-lean "fused" form), so the
    per-iteration memory is two scalar fields regardless of dimension or
    element. The double-well is a lumped nodal quadrature (pointwise), which
    needs no shape-function-value interpolation.

    ``L`` already carries the physical ``∫∇φ·∇φ`` scaling, so
    ``rhoᵀ L rho = ∫|∇rho|^2`` directly. The regularization is consistent with
    the mechanics discretization: it uses the *same* element as the stiffness
    operator (``homogenization.element``).
    """

    def __init__(self, homogenization, eta=1.0, well_weight=1.0):
        self.h = homogenization
        self.eta = float(eta)
        self.well = float(well_weight)

        offset, stencil = fe_laplacian_stencil(
            self.h.dim, self.h.grid_spacing, self.h.element
        )
        # Fused (matrix-free) scalar FE-Laplacian: L = Bᵀ W B as a convolution.
        self.laplace = muGrid.GenericLinearOperator(offset, stencil)
        self.vol_pixel = float(np.prod(self.h.grid_spacing))

        self._rho = self.h.scalar_field("to_nodal_reg_rho")
        self._Lrho = self.h.scalar_field("to_nodal_reg_Lrho")

    def value_and_gradient(self, rho):
        """Return (f_reg, df_reg/drho) for a nodal density array."""
        rho = np.asarray(rho)
        self._rho.p[...] = rho
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._Lrho)  # L rho
        Lrho = np.asarray(self._Lrho.p)

        # eta * rhoᵀ L rho  (L already includes the physical volume weighting).
        grad_pen = self.h.comm.sum(float(np.sum(rho * Lrho)))
        dwell = self.h.comm.sum(
            float(np.sum(rho**2 * (1.0 - rho) ** 2))
        ) * self.vol_pixel

        f = self.eta * grad_pen + (self.well / self.eta) * dwell

        dwell_drho = 2.0 * rho * (1.0 - rho) * (1.0 - 2.0 * rho)
        g = (
            self.eta * 2.0 * Lrho
            + (self.well / self.eta) * dwell_drho * self.vol_pixel
        )
        return f, g
