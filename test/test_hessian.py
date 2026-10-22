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

from mpi_helpers import (
    assert_allclose_on_all_ranks,
    assert_on_all_ranks,
    local,
)


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


# TODO: rationale for tolerance rtol=1e-5 (relative to max|fd|)?
def _check_hv(problem, rho, v, d=1e-6, rtol=1e-5):
    problem.ensure_state(rho)
    hv = problem.hessian_vector_product(v)
    _, gp = problem.objective_and_gradient(rho + d * v)
    _, gm = problem.objective_and_gradient(rho - d * v)
    fd = (gp - gm) / (2.0 * d)
    comm = problem.h.comm
    scale = max(comm.max(float(np.max(np.abs(fd), initial=0.0))), 1e-30)
    ok = np.allclose(hv, fd, rtol=rtol, atol=rtol * scale)
    assert_on_all_ranks(comm, [] if ok else [
        f"max rel err {np.max(np.abs(hv - fd)) / scale:.3e}"])


@pytest.mark.parametrize("with_reg", [False, True])
def test_hv_matches_fd_2d(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, comm, with_reg=with_reg)
    rng = np.random.default_rng(0)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))
    _check_hv(problem, rho, v)


def test_hv_matches_fd_3d(comm):
    n = 4
    problem = _make_problem(3, n, comm)
    rng = np.random.default_rng(1)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n, n)))
    v = local(problem.h, rng.standard_normal((n, n, n)))
    _check_hv(problem, rho, v)


@pytest.mark.parametrize("with_reg", [False, True])
def test_hv_matches_fd_nodal(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, comm, design="nodal", with_reg=with_reg)
    rng = np.random.default_rng(2)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))
    _check_hv(problem, rho, v)


def test_hv_symmetry(comm):
    """The reduced Hessian is symmetric: <u, H v> = <v, H u>."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(3)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    u = local(problem.h, rng.standard_normal((n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))
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
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    u = local(problem.h, rng.standard_normal((n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))
    problem.ensure_state(rho)
    h_lin = problem.hessian_vector_product(2.0 * u - 0.5 * v)
    hu = problem.hessian_vector_product(u)
    hv = problem.hessian_vector_product(v)
    assert_allclose_on_all_ranks(comm, h_lin, 2.0 * hu - 0.5 * hv,
                                 rtol=1e-8, atol=1e-12)


def test_ensure_state_reprimes_after_other_evaluation(comm):
    """After evaluating a *different* iterate (a rejected trial point in a
    trust-region run), ensure_state must re-prime the cache so the Hv is
    taken around the requested point."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(5)
    rho_a = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    rho_b = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    v = local(problem.h, rng.standard_normal((n, n)))

    problem.ensure_state(rho_a)
    hv_a = problem.hessian_vector_product(v)

    # Evaluate elsewhere (stale cache), then ask for the Hv at rho_a again.
    problem.objective_and_gradient(rho_b)
    problem.ensure_state(rho_a)
    hv_a2 = problem.hessian_vector_product(v)

    assert_allclose_on_all_ranks(comm, hv_a2, hv_a, rtol=1e-12, atol=1e-14)


def test_warm_start_is_result_invariant(comm):
    """Warm-starting the Hv solves (reusing the previous product's solution as
    the CG initial guess) must not change the converged result: a warm-started
    sequence of products must match cold-started ones at tight tolerance, for
    each of several distinct directions."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(6)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    problem.ensure_state(rho)
    directions = [local(problem.h, rng.standard_normal((n, n)))
                  for _ in range(4)]

    # Cold reference: zero the per-case warm fields before each product.
    cold = []
    for v in directions:
        for f in problem._du_cases + problem._dadj_cases:
            f.set_zero()
        cold.append(problem.hessian_vector_product(v).copy())

    # Warm: let the fields persist across the sequence (the production path).
    for f in problem._du_cases + problem._dadj_cases:
        f.set_zero()
    warm = [problem.hessian_vector_product(v).copy() for v in directions]

    for c, w in zip(cold, warm):
        assert_allclose_on_all_ranks(comm, w, c, rtol=1e-9, atol=1e-11)


def test_warm_start_guard_survives_bad_guess(comm):
    """A stale/garbage warm-start guess (e.g. left over from a previous outer
    iterate with different, less ill-conditioned material) must not derail the
    Hv solves: the guard rejects any guess that does not beat a cold start, so
    the product still converges to the correct value in a bounded iteration
    count. Without the guard, CG started from such a guess on a high-contrast
    design stalls."""
    n = 6
    problem = _make_problem(2, n, comm)
    rng = np.random.default_rng(7)
    # A high-contrast (near-binary) design -- the regime where K^{-1} is large
    # and a bad initial guess blows the CG residual up.
    rho = np.clip(np.round(local(problem.h, rng.uniform(0, 1, (n, n)))),
                  1e-3, 1.0)
    problem.ensure_state(rho)
    v = local(problem.h, rng.standard_normal((n, n)))

    ref = problem.hessian_vector_product(v).copy()

    # Poison every warm-start field with a large stale guess, then repeat.
    for f in problem._du_cases + problem._dadj_cases:
        f.p[...] = problem.h.to_device(
            1e2 * rng.standard_normal(f.p.shape))
    poisoned = problem.hessian_vector_product(v).copy()

    assert_allclose_on_all_ranks(comm, poisoned, ref, rtol=1e-7, atol=1e-9)


def test_hessian_off_raises(comm):
    n = 6
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization((n, n), material, comm=comm)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.15, G=0.08), magnitude=0.01)
    problem = StressTargetProblem(homog, cases)  # hessian=False
    with pytest.raises(RuntimeError, match="hessian=True"):
        problem.ensure_state(np.full(homog.nb_pixels, 0.5))
