"""
Finite-difference check of the exact (second-order-adjoint) Hessian-vector
product -- the correctness gate for the whole Hv assembly: forward-sensitivity
and second-adjoint solves, the material-derivative operator applies, the
sensitivity-kernel chain rule, the SIMP second derivatives, and the
regularization Hessian.

No optimizer involvement (and no NuMPI requirement): the reference is the
central finite difference of the analytic gradient,
``H v ~ (g(rho + h v) - g(rho - h v)) / 2h`` at tight CG tolerance.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases


def _make_problem(dim, n, comm, design="element", with_reg=True,
                  dwell="lumped"):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization(
        (n,) * dim, material, comm=comm, cg_tol=1e-12, cg_maxiter=5000)
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, K=0.15, G=0.08), magnitude=0.01)
    if not with_reg:
        reg = None
    elif design == "nodal":
        # The consistent double-well's Hv uses a lumped approximation, so the
        # FD identity is only exact for the lumped quadrature.
        reg = NodalPhaseFieldRegularization(homog, dwell=dwell)
    else:
        reg = PhaseFieldRegularization(homog)
    return StressTargetProblem(homog, cases, regularization=reg,
                               design=design, hessian=True)


def _check_hv(problem, rho, v, d=1e-6, rtol=1e-5):
    problem.ensure_state(rho)
    hv = problem.hessian_vector_product(v)
    _, gp = problem.objective_and_gradient(rho + d * v)
    _, gm = problem.objective_and_gradient(rho - d * v)
    fd = (gp - gm) / (2.0 * d)
    scale = max(float(np.max(np.abs(fd))), 1e-30)
    assert np.allclose(hv, fd, rtol=rtol, atol=rtol * scale), (
        f"max rel err {np.max(np.abs(hv - fd)) / scale:.3e}")


@pytest.mark.parametrize("with_reg", [False, True])
def test_hv_matches_fd_2d(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, comm, with_reg=with_reg)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, (n, n))
    v = rng.standard_normal((n, n))
    _check_hv(problem, rho, v)


def test_hv_matches_fd_3d(comm):
    n = 4
    problem = _make_problem(3, n, comm)
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, (n, n, n))
    v = rng.standard_normal((n, n, n))
    _check_hv(problem, rho, v)


@pytest.mark.parametrize("with_reg", [False, True])
def test_hv_matches_fd_nodal(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, comm, design="nodal", with_reg=with_reg)
    rng = np.random.default_rng(2)
    rho = rng.uniform(0.2, 0.8, (n, n))
    v = rng.standard_normal((n, n))
    _check_hv(problem, rho, v)


def test_hv_symmetry(comm):
    """The reduced Hessian is symmetric: u.(H v) == v.(H u)."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.2, 0.8, (n, n))
    u = rng.standard_normal((n, n))
    v = rng.standard_normal((n, n))
    problem.ensure_state(rho)
    hu = problem.hessian_vector_product(u)
    hv = problem.hessian_vector_product(v)
    s1 = problem.h.comm.sum(float(np.sum(u * hv)))
    s2 = problem.h.comm.sum(float(np.sum(v * hu)))
    assert np.isclose(s1, s2, rtol=1e-10)


def test_hv_linear(comm):
    """H (a u + b v) == a H u + b H v (the product is linear in v)."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(4)
    rho = rng.uniform(0.2, 0.8, (n, n))
    u = rng.standard_normal((n, n))
    v = rng.standard_normal((n, n))
    problem.ensure_state(rho)
    h_lin = problem.hessian_vector_product(2.0 * u - 0.5 * v)
    hu = problem.hessian_vector_product(u)
    hv = problem.hessian_vector_product(v)
    np.testing.assert_allclose(h_lin, 2.0 * hu - 0.5 * hv,
                               rtol=1e-8, atol=1e-12)


def test_ensure_state_reprimes_after_other_evaluation(comm):
    """After evaluating a *different* iterate (a rejected trial point in a
    trust-region run), ensure_state must re-prime the cache so the Hv is
    taken around the requested point."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(5)
    rho_a = rng.uniform(0.2, 0.8, (n, n))
    rho_b = rng.uniform(0.2, 0.8, (n, n))
    v = rng.standard_normal((n, n))

    problem.ensure_state(rho_a)
    hv_a = problem.hessian_vector_product(v)

    # Evaluate elsewhere (stale cache), then ask for the Hv at rho_a again.
    problem.objective_and_gradient(rho_b)
    problem.ensure_state(rho_a)
    hv_a2 = problem.hessian_vector_product(v)

    np.testing.assert_allclose(hv_a2, hv_a, rtol=1e-12, atol=1e-14)


def test_hessian_off_raises(comm):
    n = 6
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization((n, n), material, comm=comm)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.15, G=0.08), magnitude=0.01)
    problem = StressTargetProblem(homog, cases)  # hessian=False
    with pytest.raises(RuntimeError, match="hessian=True"):
        problem.ensure_state(np.full((n, n), 0.5))
