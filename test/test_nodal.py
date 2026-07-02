"""
Nodal phase-field variant: the fused scalar FE-Laplacian regularization.

Two levels of validation:
  1. the extracted FE-Laplacian stencil (applied matrix-free via
     GenericLinearOperator) reproduces the direct BᵀWB action of muGrid's
     FEMGradientOperator, for P1 and Q1 in 2D and 3D; and
  2. the full topology-optimization gradient (stress adjoint + nodal FE
     regularization) matches finite differences.
"""

import numpy as np
import pytest

import muGrid
from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
    fe_laplacian_stencil,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases

ELEMENTS = [muGrid.FEMElement.p1, muGrid.FEMElement.q1]
ELEMENT_NAMES = {muGrid.FEMElement.p1: "p1", muGrid.FEMElement.q1: "q1"}


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_fe_laplacian_energy_identity(dim, element, comm):
    """Independent check that the fused stencil L equals BᵀWB: the quadratic
    form rhoᵀ L rho (via the GenericLinearOperator stencil) must equal the
    quadrature energy Σ_q w_q |∇rho_q|² (via FEMGradientOperator.apply only —
    no transpose, so this is not circular with the extraction)."""
    n = 6
    spacing = [0.7, 1.3, 1.1][:dim]
    grad = muGrid.FEMGradientOperator(dim, list(spacing), element)
    nq = grad.nb_quad_pts
    eng = muGrid.FFTEngine(
        (n,) * dim, comm, nb_ghosts_left=(1,) * dim,
        nb_ghosts_right=(1,) * dim, nb_sub_pts={"quad": nq},
    )
    fc = eng.real_space_collection
    rho = fc.real_field("rho", (1,))
    g = fc.real_field("g", (grad.nb_output_components,), "quad")
    Lrho = fc.real_field("Lrho", (1,))

    rng = np.random.default_rng(0)
    rho.p[...] = rng.standard_normal(np.asarray(rho.p).shape)
    eng.communicate_ghosts(rho)

    # Direct quadrature energy from the gradient (apply only).
    grad.apply(rho, g)
    w = np.asarray(grad.quadrature_weights)
    gs = np.asarray(g.s)  # (dim, nq, *grid)
    # Σ_q w_q Σ_c (∂_c rho_q)^2, summed over the grid.
    energy_direct = comm.sum(float(
        np.sum(gs**2 * w.reshape((1, nq) + (1,) * dim))))

    # Quadratic form via the fused stencil.
    offset, stencil = fe_laplacian_stencil(dim, spacing, element)
    assert stencil.shape == (3,) * dim
    np.testing.assert_allclose(
        stencil, np.flip(stencil), atol=1e-12,
        err_msg="FE-Laplacian stencil must be symmetric",
    )
    glo = muGrid.GenericLinearOperator(offset, stencil)
    glo.apply(rho, Lrho)
    energy_stencil = comm.sum(float(
        np.sum(np.asarray(rho.p) * np.asarray(Lrho.p))))

    np.testing.assert_allclose(energy_stencil, energy_direct, rtol=1e-9)


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_fe_laplacian_annihilates_constant(dim, element, comm):
    """A constant density has zero gradient energy: L·1 = 0."""
    n = 6
    spacing = [0.7, 1.3, 1.1][:dim]
    offset, stencil = fe_laplacian_stencil(dim, spacing, element)
    # Sum of stencil weights is L applied to a constant field -> must vanish.
    assert abs(float(stencil.sum())) < 1e-10


def _make_nodal_problem(dim, n, element, comm):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization(
        (n,) * dim, material, comm=comm,
        element=ELEMENT_NAMES[element], cg_tol=1e-12, cg_maxiter=5000,
    )
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, K=0.15, G=0.08), magnitude=0.01
    )
    reg = NodalPhaseFieldRegularization(homog)
    return StressTargetProblem(homog, cases, regularization=reg)


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_nodal(dim, element, comm):
    n = 6 if dim == 2 else 4
    problem = _make_nodal_problem(dim, n, element, comm)
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, (n,) * dim)
    _, g = problem.objective_and_gradient(rho)

    d = 1e-6
    sample = [tuple([0] * dim), tuple([n - 1] * dim),
              tuple([1] + [2] * (dim - 1)), tuple([n // 2] * dim)]
    for idx in sample:
        rp = rho.copy()
        rp[idx] += d
        fp, _ = problem.objective_and_gradient(rp)
        rm = rho.copy()
        rm[idx] -= d
        fm, _ = problem.objective_and_gradient(rm)
        fd = (fp - fm) / (2 * d)
        assert np.isclose(g[idx], fd, rtol=2e-4, atol=1e-7), (
            f"{ELEMENT_NAMES[element]} {dim}D pixel {idx}: "
            f"analytic {g[idx]:.6e} vs FD {fd:.6e}"
        )
