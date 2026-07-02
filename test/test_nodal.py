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


# ---------------------------------------------------------------------------
# Nodal DESIGN space (design="nodal"): SIMP on the element average of the
# nodal interpolant, with the fully consistent Galerkin double-well.
# ---------------------------------------------------------------------------

from muTopOpt.loadcases import isotropic_stiffness_from_E_nu, unit_strains  # noqa: E402
from muTopOpt.nodal import ConsistentDoubleWell, NodalElementMap  # noqa: E402


def _homog(dim, n, element, comm):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    return Homogenization(
        (n,) * dim, material, comm=comm, element=ELEMENT_NAMES[element],
        cg_tol=1e-12, cg_maxiter=5000,
    )


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_gather_scatter_adjoint(comm, dim, element):
    """<scatter_mean(s), r> must equal <s, gather_mean(r)> -- the scatter is
    the exact transpose of the gather (also across the periodic boundary)."""
    n = 6 if dim == 2 else 4
    h = _homog(dim, n, element, comm)
    m = NodalElementMap(h)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(h.nb_pixels)
    s = rng.standard_normal(h.nb_pixels)
    lhs = comm.sum(float(np.sum(m.scatter_mean(s) * r)))
    rhs = comm.sum(float(np.sum(s * m.gather_mean(r))))
    assert np.isclose(lhs, rhs, rtol=1e-12)


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_gather_mean_of_constant(comm, dim, element):
    """The element average of a constant nodal field is that constant (the
    averaging weights sum to one)."""
    n = 6 if dim == 2 else 4
    h = _homog(dim, n, element, comm)
    m = NodalElementMap(h)
    np.testing.assert_allclose(m.gather_mean(np.full(h.nb_pixels, 0.37)),
                               0.37, rtol=1e-14)


def _dwell_reference(rho, element_name, dim, vol_pixel, nsub=48):
    """Brute-force reference: sample the FE interpolant of the (periodic)
    nodal field on an nsub^dim subgrid of every element and integrate W by the
    midpoint rule. Serial only."""
    from muTopOpt.nodal import _P1_SIMPLICES, _node_offset

    n = rho.shape
    total = 0.0
    pts = (np.arange(nsub) + 0.5) / nsub
    grids = np.meshgrid(*([pts] * dim), indexing="ij")
    xi = np.stack([g.ravel() for g in grids], axis=1)  # (nsub^dim, dim)

    if element_name == "q1":
        N = np.ones((xi.shape[0], 2 ** dim))
        for c in range(2 ** dim):
            for d in range(dim):
                N[:, c] *= xi[:, d] if _node_offset(c, d) else 1.0 - xi[:, d]
    else:
        corners = np.array([[_node_offset(c, d) for d in range(dim)]
                            for c in range(2 ** dim)], dtype=float)
        N = np.zeros((xi.shape[0], 2 ** dim))
        assigned = np.zeros(xi.shape[0], dtype=bool)
        for nodes, _frac in _P1_SIMPLICES[dim]:
            verts = corners[list(nodes)]  # (dim+1, dim)
            T = (verts[1:] - verts[0]).T  # (dim, dim)
            lam_rest = np.linalg.solve(T, (xi - verts[0]).T).T
            lam = np.concatenate(
                [1.0 - lam_rest.sum(axis=1, keepdims=True), lam_rest], axis=1)
            inside = np.all(lam >= -1e-12, axis=1) & ~assigned
            for j, node in enumerate(nodes):
                N[inside, node] += lam[inside, j]
            assigned |= inside
        assert assigned.all()

    for idx in np.ndindex(*n):
        vals = np.array([
            rho[tuple((idx[d] + _node_offset(c, d)) % n[d]
                      for d in range(dim))]
            for c in range(2 ** dim)
        ])
        rho_pts = N @ vals
        W = rho_pts**2 * (1.0 - rho_pts) ** 2
        total += W.mean() * vol_pixel
    return total


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_consistent_dwell_value(comm, dim, element):
    """The consistent double-well matches a brute-force integration of the
    interpolant (the Galerkin quadrature is exact; the midpoint reference
    carries an O(nsub^-2) error)."""
    if comm.size > 1:
        pytest.skip("brute-force reference is serial")
    n = 4
    h = _homog(dim, n, element, comm)
    m = NodalElementMap(h)
    dw = ConsistentDoubleWell(m)
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.0, 1.0, h.nb_pixels)
    f, _ = dw.value_and_gradient(rho)
    nsub = 48 if dim == 2 else 16
    ref = _dwell_reference(rho, ELEMENT_NAMES[element], dim, m.vol_pixel,
                           nsub=nsub)
    assert np.isclose(f, ref, rtol=5e-3), f"{f} vs reference {ref}"


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
@pytest.mark.parametrize("dim", [2, 3])
def test_consistent_dwell_gradient(comm, dim, element):
    """FD check of the consistent double-well gradient."""
    n = 4
    h = _homog(dim, n, element, comm)
    dw = ConsistentDoubleWell(NodalElementMap(h))
    rng = np.random.default_rng(4)
    rho = rng.uniform(0.2, 0.8, h.nb_pixels)
    _, g = dw.value_and_gradient(rho)
    d = 1e-6
    for idx in [(0,) * dim, (1,) * dim, tuple(range(1, dim + 1))]:
        rp = rho.copy()
        rp[idx] += d
        fp, _ = dw.value_and_gradient(rp)
        rm = rho.copy()
        rm[idx] -= d
        fm, _ = dw.value_and_gradient(rm)
        fd = (fp - fm) / (2 * d)
        assert np.isclose(g[idx], fd, rtol=1e-5, atol=1e-10), (
            f"node {idx}: analytic {g[idx]:.6e} vs FD {fd:.6e}")


def _nodal_design_problem(dim, n, element, comm):
    h = _homog(dim, n, element, comm)
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, K=0.15, G=0.08), magnitude=0.01
    )
    reg = NodalPhaseFieldRegularization(h)
    return StressTargetProblem(h, cases, regularization=reg, design="nodal")


@pytest.mark.parametrize("element", ELEMENTS, ids=ELEMENT_NAMES.get)
def test_nodal_design_gradient_2d(comm, element):
    """FD check of the FULL nodal-design objective gradient (forward + adjoint
    + SIMP chain rule through the element average + consistent Galerkin
    regularization)."""
    n = 6
    problem = _nodal_design_problem(2, n, element, comm)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, (n, n))
    _, g = problem.objective_and_gradient(rho)
    d = 1e-6
    for idx in [(0, 0), (0, 3), (3, 0), (2, 2), (n - 1, n - 1)]:
        rp = rho.copy()
        rp[idx] += d
        fp, _ = problem.objective_and_gradient(rp)
        rm = rho.copy()
        rm[idx] -= d
        fm, _ = problem.objective_and_gradient(rm)
        fd = (fp - fm) / (2 * d)
        assert np.isclose(g[idx], fd, rtol=2e-4, atol=1e-7), (
            f"node {idx}: analytic {g[idx]:.6e} vs FD {fd:.6e}")


def test_nodal_design_gradient_3d(comm):
    n = 4
    problem = _nodal_design_problem(3, n, ELEMENTS[0], comm)  # p1
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, (n, n, n))
    _, g = problem.objective_and_gradient(rho)
    d = 1e-6
    for idx in [(0, 0, 0), (1, 2, 3), (n - 1, n - 1, n - 1)]:
        rp = rho.copy()
        rp[idx] += d
        fp, _ = problem.objective_and_gradient(rp)
        rm = rho.copy()
        rm[idx] -= d
        fm, _ = problem.objective_and_gradient(rm)
        fd = (fp - fm) / (2 * d)
        assert np.isclose(g[idx], fd, rtol=2e-4, atol=1e-7), (
            f"node {idx}: analytic {g[idx]:.6e} vs FD {fd:.6e}")


@pytest.mark.parametrize("dim", [2, 3])
def test_target_from_E_nu_matches_solid(comm, dim):
    """A target built from the solid's own (E, nu) must equal the homogenized
    response of the fully dense cell."""
    n = 4
    E_s, nu_s = 1.3, 0.28
    material = SimpMaterial(E_solid=E_s, nu=nu_s, penalty=2.0, void_ratio=1e-3)
    h = Homogenization((n,) * dim, material, comm=comm, cg_tol=1e-12)
    target = isotropic_stiffness_from_E_nu(dim, E_s, nu_s)
    u = h.vector_field("test_u")
    h.set_density(np.ones(h.nb_pixels))
    for E_macro in unit_strains(dim, magnitude=0.01):
        h.solve_macro(E_macro, u)
        sigma = h.homogenized_stress(u, E_macro)
        np.testing.assert_allclose(sigma, target(E_macro),
                                   rtol=1e-8, atol=1e-12)
