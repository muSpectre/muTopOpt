"""
Tests for the energy objective (load cases with ``energy_terms``) and the
auxetic target of :mod:`muTopOpt.loadcases`.
"""
import numpy as np
import pytest

from muTopOpt import Homogenization, SimpMaterial, StressTargetProblem
from muTopOpt.loadcases import (
    auxetic_target,
    energy_load_cases,
    isotropic_stiffness_tensor,
    target_load_cases,
    unit_strains,
)
from muTopOpt.problem import LoadCase

from mpi_helpers import check_fd_gradient, local


def make_problem(dim, n, comm, hessian=False, magnitude=0.01):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homogenization = Homogenization((n,) * dim, material, comm=comm,
                                    cg_tol=1e-12, cg_maxiter=5000)
    cases = energy_load_cases(
        dim, auxetic_target(dim, E_solid=1.0, nu=-0.5, scale=0.3),
        magnitude=magnitude)
    return StressTargetProblem(homogenization, cases, hessian=hessian)


@pytest.mark.parametrize("dim", [2, 3])
def test_number_of_energies(dim):
    """An isotropic target constrains 4 entries of the effective stiffness
    in 2D and 9 in 3D."""
    cases = energy_load_cases(dim, isotropic_stiffness_tensor(dim, 0.2, 0.1))
    assert len(cases) == dim * (dim + 1) // 2
    assert sum(len(case.energy_terms) for case in cases) == {2: 4, 3: 9}[dim]
    # the target energies are the entries of the target stiffness
    strains = unit_strains(dim)
    first = cases[0]
    for (left, target, weight), strain in zip(first.energy_terms, strains):
        assert target == pytest.approx(np.sum(strain * first.target_stress))
        assert weight == 1.0


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu", [-0.5, 0.0, 0.3])
def test_auxetic_target_has_the_requested_poisson_ratio(dim, nu):
    """Uniaxial stress along x: the lateral strain is -nu times the axial
    one. (Plane strain in 2D, where the in-plane ratio is nu / (1 - nu).)"""
    target = auxetic_target(dim, E_solid=2.0, nu=nu, scale=0.5)
    shear = 0.5 * 0.15 * 2.0
    E = np.zeros((dim, dim))
    E[0, 1] = E[1, 0] = 0.5
    assert target(E)[0, 1] == pytest.approx(shear)
    # stiffness in Voigt form for the normal directions, then solve for the
    # strain under a uniaxial stress
    C = np.array([[target(strain)[j, j] for strain in unit_strains(dim)[:dim]]
                  for j in range(dim)])
    strain = np.linalg.solve(C, np.eye(dim)[0])
    expected = nu if dim == 3 else nu / (1 - nu)
    assert -strain[1] / strain[0] == pytest.approx(expected)


def test_misfit_of_a_load_case():
    """Value, derivative and second derivative of the energy misfit against
    finite differences in the stress."""
    rng = np.random.default_rng(0)
    case = energy_load_cases(3, isotropic_stiffness_tensor(3, 0.2, 0.1))[0]
    case.weight = 1.7
    stress = rng.standard_normal((3, 3))
    stress = stress + stress.T
    value, derivative = case.misfit(stress)
    expected = 1.7 * sum(
        c * ((np.sum(left * stress) - target) / target) ** 2
        for left, target, c in case.energy_terms)
    assert value == pytest.approx(expected)
    direction = rng.standard_normal((3, 3))
    d = 1e-6
    plus, d_plus = case.misfit(stress + d * direction)
    minus, d_minus = case.misfit(stress - d * direction)
    assert np.sum(derivative * direction) == pytest.approx(
        (plus - minus) / (2 * d), rel=1e-6)
    np.testing.assert_allclose(
        case.misfit_derivative_increment(direction),
        (d_plus - d_minus) / (2 * d), rtol=1e-6, atol=1e-12)


def test_stress_objective_is_unchanged():
    """A load case without energy terms gives the stress misfit as before."""
    target = np.array([[1.0, 0.2], [0.2, -0.5]])
    stress = np.array([[0.7, 0.1], [0.1, 0.3]])
    case = LoadCase(np.eye(2), target, weight=2.0)
    value, derivative = case.misfit(stress)
    norm = np.sum(target**2)
    assert value == pytest.approx(2.0 * np.sum((stress - target) ** 2) / norm)
    np.testing.assert_allclose(derivative, 4.0 * (stress - target) / norm)
    np.testing.assert_allclose(case.misfit_derivative_increment(stress),
                               4.0 * stress / norm)


def test_objective_vanishes_at_the_target(comm):
    """A uniform solid cell has the stiffness of the solid: with that
    stiffness as the target, the energy objective is zero."""
    from muTopOpt.loadcases import isotropic_stiffness_from_E_nu

    n = 8
    material = SimpMaterial(E_solid=1.0, nu=0.3)
    homogenization = Homogenization((n, n), material, comm=comm)
    cases = energy_load_cases(2, isotropic_stiffness_from_E_nu(2, 1.0, 0.3))
    problem = StressTargetProblem(homogenization, cases)
    f, g = problem.objective_and_gradient(
        local(homogenization, np.ones((n, n))))
    assert abs(f) < 1e-20


@pytest.mark.parametrize("dim,n", [(2, 6), (3, 4)])
def test_gradient(comm, dim, n):
    problem = make_problem(dim, n, comm)
    rng = np.random.default_rng(0)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n,) * dim))
    sample = [(0,) * dim, (n - 1,) * dim, (1, 2, 3)[:dim], (2, 0, 1)[:dim]]
    f, g = problem.objective_and_gradient(rho)
    assert f > 0
    check_fd_gradient(
        problem.h, lambda r: problem.objective_and_gradient(r)[0], rho, g,
        sample, d=1e-6, rtol=2e-4, atol=1e-7 * abs(g).max())


def test_hessian_vector_product(comm):
    """H v against the central difference of the gradient along v."""
    n = 6
    problem = make_problem(2, n, comm, hessian=True)
    rng = np.random.default_rng(1)
    rho = local(problem.h, rng.uniform(0.3, 0.7, (n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))
    problem.objective_and_gradient(rho)
    hv = np.asarray(problem.hessian_vector_product(v)).copy()
    d = 1e-5
    _, g_plus = problem.objective_and_gradient(rho + d * v)
    g_plus = np.asarray(g_plus).copy()
    _, g_minus = problem.objective_and_gradient(rho - d * v)
    fd = (g_plus - np.asarray(g_minus)) / (2 * d)
    np.testing.assert_allclose(hv, fd, rtol=1e-4, atol=1e-6 * abs(fd).max())


def test_energy_objective_weighs_every_entry_by_its_own_target(comm):
    """Scaling one target energy leaves the relative misfit of the others
    alone: the objective is a sum of squared *relative* misfits."""
    n = 8
    material = SimpMaterial(E_solid=1.0, nu=0.3)
    homogenization = Homogenization((n, n), material, comm=comm)
    rho = local(homogenization, np.full((n, n), 0.6))
    target = isotropic_stiffness_tensor(2, K=0.15, G=0.08)
    cases = energy_load_cases(2, target)
    problem = StressTargetProblem(homogenization, cases)
    f, _ = problem.objective_and_gradient(rho)
    stresses = problem.last["stresses"]
    expected = sum(
        c * ((np.sum(left * stress) - goal) / goal) ** 2
        for case, stress in zip(cases, stresses)
        for left, goal, c in case.energy_terms)
    assert f == pytest.approx(expected, rel=1e-8)
    # and it differs from the stress objective of the same target
    stress_problem = StressTargetProblem(homogenization,
                                         target_load_cases(2, target))
    f_stress, _ = stress_problem.objective_and_gradient(rho)
    assert f_stress != pytest.approx(f, rel=1e-3)
