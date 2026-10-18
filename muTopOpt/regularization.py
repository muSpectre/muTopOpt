#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Phase-field regularization of the density.

Following Bourdin's phase-field approach to topology optimization, the
density is regularized by penalizing interfacial area with a normalized
Modica-Mortola functional,

    f_reg(ρ) = w / (c_W V^((d−1)/d)) · [ η ∫ |∇ρ|² dx + (1/η) ∫ W(ρ) dx ],

    c_W = 2 ∫_0^1 √W(s) ds,

with the double well ``W(ρ) = ρ² (1 − ρ)²``, for which ``c_W = 1/3``
(:data:`C_W`), ``V`` the volume of the unit cell and ``w`` the weight. The
gradient term smooths the design and removes mesh dependence; the double
well drives ``ρ`` toward {0, 1}.

Normalization. As ``η → 0`` the bracket Γ-converges to ``c_W`` times the
interfacial area (Modica-Mortola), so ``f_reg / w`` converges to the
interfacial area measured in units of ``L^(d−1)``, with ``L = V^(1/d)`` the
linear size of the cell (:func:`perimeter_prefactor`). ``f_reg`` is thus
dimensionless, and the optimized design does not depend on the absolute size
of the cell as long as ``η / L`` is held fixed. ``w`` is the penalty per unit
relative interfacial area, in units of the (normalized) stress objective of
:mod:`muTopOpt.problem`.

Interface width. ``η`` is the interface width, in the *physical length
units of the unit cell*: in 1D the minimizer of the bracket across an
interface satisfies ``η ρ' = √W(ρ)``, whose solution is the logistic profile
``ρ(x) = 1 / (1 + exp(−x/η))``. It defaults to one grid spacing. Choosing it
much larger than a few grid spacings makes the gradient penalty dominate and
the design collapses to a constant density.

Discretizations of the gradient term:

* :class:`PhaseFieldRegularization` (element-wise density) uses the
  finite-difference Laplacian on the pixel grid,
  ``∫ |∇ρ|² ≈ h^d ρᵀ(−Δ_h)ρ`` with ``h^d = ∏_α h_α`` the pixel volume
  (:class:`muGrid.LaplaceOperator` with the spacing of every axis; a muGrid
  older than 1.5.0 has a single-scale Laplacian, in which case cubic voxels
  are assumed and a warning is issued if they are not).
* :class:`NodalPhaseFieldRegularization` (nodal FE density) uses the
  element-consistent H¹ seminorm ``∫ |∇ρ|² = ρᵀ L ρ`` with the scalar
  FE-Laplacian ``L = Bᵀ W B`` of the mechanics element (P1 or Q1). ``L`` is a
  constant ``3^d`` stencil (computed once from the element geometry by
  :func:`fe_laplacian_stencil`) applied matrix-free by
  :class:`muGrid.GenericLinearOperator`, so no ``d × nb_quad`` gradient field
  is resident and anisotropic spacing is handled.

The double-well term is a lumped (pointwise) quadrature for the element-wise
density; for the nodal density it is either the exact integral of the
interpolant (default, :class:`muTopOpt.nodal.ConsistentDoubleWell`) or the
lumped quadrature.
"""

import numpy as np

import muGrid

#: Interfacial energy per unit area of the quartic double well
#: ``W(ρ) = ρ² (1 − ρ)²`` in the Modica-Mortola functional
#: ``∫ [η |∇ρ|² + W(ρ)/η] dx``:
#: ``c_W = 2 ∫_0^1 √W(s) ds = 2 ∫_0^1 s (1 − s) ds = 1/3``.
C_W = 1.0 / 3.0


def perimeter_prefactor(homogenization):
    """``1 / (c_W V^((d−1)/d))``: the factor that turns the raw Modica-Mortola
    functional into the interfacial area in units of ``L^(d−1)``, with ``V``
    the cell volume (``homogenization.domain_volume``) and ``L = V^(1/d)`` its
    linear size."""
    dim = homogenization.dim
    V = homogenization.domain_volume
    return 1.0 / (C_W * V ** ((dim - 1) / dim))


def _laplace_takes_grid_spacing():
    """Whether this muGrid's ``LaplaceOperator`` accepts a per-axis
    ``grid_spacing`` (added in muGrid 1.5.0)."""
    import inspect

    try:
        return "grid_spacing" in inspect.signature(
            muGrid.LaplaceOperator.__init__).parameters
    except (TypeError, ValueError):
        return False


class PhaseFieldRegularization:
    def __init__(self, homogenization, eta=None, weight=1.0):
        """Phase-field regularization of an element-wise (per-pixel) density.

        Evaluates the normalized Modica-Mortola functional of the module
        docstring with a finite-difference Laplacian for the gradient term
        and a lumped double well.

        Parameters
        ----------
        homogenization : Homogenization
            Provides the grid, its spacing, the cell volume, the
            communicator and the field collection.
        eta : float, optional
            Interface width ``η`` in the length units of the cell (default:
            ``homogenization.grid_spacing[0]``).
        weight : float, optional
            Penalty ``w`` per unit relative interfacial area (default 1).

        Warns
        -----
        RuntimeWarning
            If muGrid's ``LaplaceOperator`` has no per-axis spacing (muGrid
            older than 1.5.0) and the voxels are not cubic.
        """
        self.h = homogenization
        h0 = self.h.grid_spacing[0]
        # eta IS the interface width (Modica-Mortola normalization); it
        # defaults to one grid spacing.
        self.eta = h0 if eta is None else float(eta)
        # Penalty per unit relative interfacial area, in units of the
        # (normalized) stress objective.
        self.weight = float(weight)
        # 1 / (c_W V^((D-1)/D)): raw Modica-Mortola bracket -> relative
        # interfacial area.
        self.prefactor = perimeter_prefactor(self.h)

        # -Δ (positive semi-definite on the periodic grid) as a
        # finite-difference stencil with the grid spacing of every axis.
        # muGrid's LaplaceOperator accepts per-axis spacing from 1.5.0 on;
        # older versions take a single scale for all axes.
        grid_spacing = [float(h) for h in self.h.grid_spacing]
        if _laplace_takes_grid_spacing():
            self.laplace = muGrid.LaplaceOperator(
                self.h.dim, -1.0, grid_spacing=grid_spacing)
        else:
            if not np.allclose(grid_spacing, h0):
                import warnings

                warnings.warn(
                    "PhaseFieldRegularization needs muGrid >= 1.5.0 for "
                    "anisotropic voxels; this muGrid's LaplaceOperator has a "
                    "single scale, so grid_spacing[0] is used on every axis "
                    "and the gradient penalty is mis-scaled.",
                    RuntimeWarning,
                )
            self.laplace = muGrid.LaplaceOperator(self.h.dim, -1.0 / h0**2)
        self.vol_pixel = float(np.prod(self.h.grid_spacing))

        self._rho = self.h.scalar_field("to_reg_rho")
        self._lap = self.h.scalar_field("to_reg_lap")

    @property
    def scale(self):
        """``w / (c_W V^((d−1)/d))``, applied to the raw functional."""
        return self.weight * self.prefactor

    def value_and_gradient(self, rho):
        """Return ``(f_reg, ∂f_reg/∂ρ)`` for an element-wise density array.

        The value is MPI-reduced; the gradient is the local slice,
        ``w/(c_W V^((d−1)/d)) · h^d [2η (−Δ_h ρ) + W'(ρ)/η]`` with
        ``W'(ρ) = 2ρ(1 − ρ)(1 − 2ρ)``.
        """
        rho = np.asarray(rho)
        self._rho.p[...] = self.h.to_device(rho)
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._lap)  # (-Δ rho)
        lap = self.h.to_host(self._lap.p)

        grad_pen = self.h.comm.sum(float(np.sum(rho * lap))) * self.vol_pixel
        dwell = self.h.comm.sum(
            float(np.sum(rho**2 * (1.0 - rho) ** 2))
        ) * self.vol_pixel

        # Raw Modica-Mortola bracket eta * ∫|∇rho|^2 + (1/eta) * ∫ W; the
        # normalization weight / (c_W V^((D-1)/D)) is applied by `scale`.
        f = self.eta * grad_pen + dwell / self.eta

        # d/drho [rho^2 (1-rho)^2] = 2 rho (1 - rho)(1 - 2 rho)
        dwell_drho = 2.0 * rho * (1.0 - rho) * (1.0 - 2.0 * rho)
        g = (
            self.eta * 2.0 * lap * self.vol_pixel
            + dwell_drho * self.vol_pixel / self.eta
        )
        scale = self.scale
        return scale * f, scale * g

    def hessian_vector_product(self, rho, v):
        """Return ``H_reg v`` for a density direction ``v`` (exact).

        The gradient-penalty Hessian is the constant operator
        ``2η h^d (−Δ_h)``; the double-well Hessian is the pointwise diagonal
        ``h^d W''(ρ)/η`` with ``W''(ρ) = 2 (1 − 6ρ + 6ρ²)`` (both times the
        normalization :attr:`scale`). Since ``W''(1/2) < 0``, the
        regularization Hessian is indefinite in the interfacial region."""
        rho = np.asarray(rho)
        v = np.asarray(v)
        self._rho.p[...] = self.h.to_device(v)
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._lap)  # (-Delta v)
        lap_v = self.h.to_host(self._lap.p)

        d2well = 2.0 * (1.0 - 6.0 * rho + 6.0 * rho**2)
        hv = (
            self.eta * 2.0 * lap_v * self.vol_pixel
            + d2well * v * self.vol_pixel / self.eta
        )
        return self.scale * hv


def fe_laplacian_stencil(dim, grid_spacing, element):
    r"""Constant-coefficient stencil of the scalar FE-Laplacian
    ``L = Bᵀ W B`` (``L_ij = ∫ ∇φ_i · ∇φ_j``) for the given element (P1 or Q1)
    on a uniform grid with the given spacing.

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

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    grid_spacing : sequence of float
        Grid spacing ``h_α`` per axis.
    element : muGrid.FEMElement
        ``muGrid.FEMElement.p1`` or ``muGrid.FEMElement.q1``.

    Returns
    -------
    offset : list of int
        ``[-1] * dim``, the stencil offset for ``GenericLinearOperator``.
    stencil : numpy.ndarray
        Contiguous array of shape ``(3,) * dim``.
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

        f_reg = w / (c_W V^((d−1)/d)) · [ η ∫ |∇ρ|² + (1/η) ∫ ρ² (1 − ρ)² ],

    but the density is a *nodal* finite-element field ``ρ_h`` and the
    gradient term is the element-consistent H¹ seminorm ``η ρᵀ L ρ`` with the
    scalar FE-Laplacian ``L = Bᵀ W B`` of the mechanics element (P1 or Q1).
    ``L`` carries the physical ``∫ ∇φ·∇φ`` scaling, so ``ρᵀ L ρ`` is
    ``∫ |∇ρ_h|²`` evaluated with the element quadrature, and it uses the
    *same* element as the stiffness operator (``homogenization.element``), so
    regularization and mechanics share one discretization. ``L`` is applied
    matrix-free by a :class:`muGrid.GenericLinearOperator`; no
    ``d × nb_quad`` gradient field is resident, so the gradient term needs two
    scalar fields regardless of dimension or element.

    The double-well term is, by default, the exact Galerkin integral
    ``∫ W(ρ_h) dx`` of the interpolant (see
    :class:`muTopOpt.nodal.ConsistentDoubleWell`); ``dwell='lumped'`` selects
    the pointwise (lumped nodal) quadrature ``h^d Σ_n W(ρ_n)`` instead.

    Parameters
    ----------
    homogenization : Homogenization
        Provides the grid, element, cell volume, communicator and fields.
    eta : float, optional
        Interface width ``η`` in the length units of the cell (default:
        ``homogenization.grid_spacing[0]``).
    weight : float, optional
        Penalty ``w`` per unit relative interfacial area (default 1).
    dwell : {"consistent", "lumped"}, optional
        Quadrature of the double-well term (default ``"consistent"``).

    Raises
    ------
    ValueError
        If ``dwell`` is neither ``"consistent"`` nor ``"lumped"``.
    """

    def __init__(self, homogenization, eta=None, weight=1.0,
                 dwell="consistent"):
        self.h = homogenization
        # eta IS the interface width (Modica-Mortola normalization); it
        # defaults to one grid spacing.
        self.eta = (self.h.grid_spacing[0] if eta is None else float(eta))
        # Penalty per unit relative interfacial area.
        self.weight = float(weight)
        # 1 / (c_W V^((D-1)/D)): raw Modica-Mortola bracket -> relative
        # interfacial area.
        self.prefactor = perimeter_prefactor(self.h)
        if dwell not in ("consistent", "lumped"):
            raise ValueError(f"unknown double-well quadrature '{dwell}'")
        self.dwell_kind = dwell

        offset, stencil = fe_laplacian_stencil(
            self.h.dim, self.h.grid_spacing, self.h.element
        )
        # Fused (matrix-free) scalar FE-Laplacian: L = Bᵀ W B as a convolution.
        self.laplace = muGrid.GenericLinearOperator(offset, stencil)
        self.vol_pixel = float(np.prod(self.h.grid_spacing))

        self._rho = self.h.scalar_field("to_nodal_reg_rho")
        self._Lrho = self.h.scalar_field("to_nodal_reg_Lrho")

        if dwell == "consistent":
            from .nodal import ConsistentDoubleWell, NodalElementMap

            self._dwell = ConsistentDoubleWell(NodalElementMap(self.h))
        else:
            self._dwell = None

    @property
    def scale(self):
        """``w / (c_W V^((d−1)/d))``, applied to the raw functional."""
        return self.weight * self.prefactor

    def value_and_gradient(self, rho):
        """Return ``(f_reg, ∂f_reg/∂ρ)`` for a nodal density array (value
        MPI-reduced, gradient the local nodal slice)."""
        rho = np.asarray(rho)
        self._rho.p[...] = self.h.to_device(rho)
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._Lrho)  # L rho
        Lrho = self.h.to_host(self._Lrho.p)

        # rhoᵀ L rho = ∫|∇rho|² (L already includes the physical volume
        # weighting); multiplied by eta below.
        grad_pen = self.h.comm.sum(float(np.sum(rho * Lrho)))

        if self._dwell is not None:
            # Fully consistent Galerkin: exact ∫ W(rho(x)) dx of the nodal
            # interpolant, with its exact nodal gradient.
            dwell, dwell_grad = self._dwell.value_and_gradient(rho)
        else:
            # Lumped nodal quadrature (pointwise).
            dwell = self.h.comm.sum(
                float(np.sum(rho**2 * (1.0 - rho) ** 2))
            ) * self.vol_pixel
            dwell_grad = (
                2.0 * rho * (1.0 - rho) * (1.0 - 2.0 * rho) * self.vol_pixel
            )

        # Raw Modica-Mortola bracket eta * ∫|∇rho|^2 + (1/eta) * ∫ W; the
        # normalization weight / (c_W V^((D-1)/D)) is applied by `scale`.
        f = self.eta * grad_pen + dwell / self.eta
        g = self.eta * 2.0 * Lrho + dwell_grad / self.eta
        scale = self.scale
        return scale * f, scale * g

    def hessian_vector_product(self, rho, v):
        """Return ``H_reg v`` for a nodal density direction ``v``.

        The gradient-penalty Hessian is the constant FE-Laplacian ``2η L``;
        the double-well Hessian uses the *lumped* pointwise diagonal
        ``h^d W''(ρ)/η`` with ``W''(ρ) = 2 (1 − 6ρ + 6ρ²)`` even when the
        value and gradient use the consistent quadrature, so the product is
        exact for ``dwell='lumped'`` and an approximation otherwise. A
        trust-region method only requires a bounded model Hessian, and the
        lumped diagonal retains the indefiniteness in the interface
        (``W''(1/2) < 0``)."""
        rho = np.asarray(rho)
        v = np.asarray(v)
        self._rho.p[...] = self.h.to_device(v)
        self.h.engine.communicate_ghosts(self._rho)
        self.laplace.apply(self._rho, self._Lrho)  # L v
        Lv = self.h.to_host(self._Lrho.p)

        d2well = 2.0 * (1.0 - 6.0 * rho + 6.0 * rho**2)
        hv = (
            self.eta * 2.0 * Lv
            + d2well * v * self.vol_pixel / self.eta
        )
        return self.scale * hv
