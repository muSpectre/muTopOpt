"""Adaptive inner (CG) / outer (L-BFGS) tolerance coupling.

The forward/adjoint CG solves can be run coarsely while the outer optimizer is
far from stationary and tightened as its projected gradient shrinks (an
Eisenstat-Walker forcing term; see :class:`AdaptiveInnerTolerance`). These tests
check the controller logic, that the coupling reaches the same optimum as a
fixed tight tolerance with fewer total CG iterations, and that the tolerance
floor still yields an accurate gradient (the norm condition).
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import (
    AdaptiveInnerTolerance,
    initial_density,
    optimize_bounded_lbfgs,
)


def _mpi_comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        return None


def test_controller_clips_and_freezes():
    """The forcing term clips to [rtol_min, rtol_start] and only moves on
    advance() -- observe() alone must not change the live tolerance."""
    c = AdaptiveInnerTolerance(rtol_start=1e-1, rtol_min=1e-6, c=1.0, alpha=1.0)
    assert c.current == 1e-1

    # advance() before any observe() is a no-op.
    assert c.advance() == 1e-1

    # A large gradient clamps to the coarse start (never oversolve early).
    c.observe(0.5)
    assert c.advance() == pytest.approx(1e-1)

    # In-range gradient maps straight through (alpha = 1, c = 1).
    c.observe(1e-3)
    assert c.advance() == pytest.approx(1e-3)

    # A tiny gradient clamps to the floor.
    c.observe(1e-9)
    assert c.advance() == pytest.approx(1e-6)

    # observe() is frozen between advance() calls.
    c.observe(1e-2)
    assert c.current == pytest.approx(1e-6)
    assert c.advance() == pytest.approx(1e-2)


def _build(comm, cg_tol):
    n = 16
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((n, n), material, comm=comm, cg_tol=cg_tol,
                           cg_maxiter=1000)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01
    )
    reg = PhaseFieldRegularization(homog)
    problem = StressTargetProblem(homog, cases, regularization=reg)
    # A heterogeneous start: a uniform density stays uniform (symmetric
    # gradient) and its fluctuation rhs is negligible, so CG would be skipped
    # entirely -- a random field gives genuine forward/adjoint solves.
    rho0 = initial_density(homog.nb_pixels, kind="random", seed=0)
    return problem, rho0


def _run(problem, rho0, maxiter, **cg_kwargs):
    """Run the optimizer, counting *every* inner CG iteration (line-search
    trials included) by wrapping the objective."""
    total = {"cg": 0}
    orig = problem.objective_and_gradient

    def counting(rho):
        f, g = orig(rho)
        total["cg"] += int(sum(problem.last["cg_iters"]))
        return f, g

    problem.objective_and_gradient = counting
    rho, info = optimize_bounded_lbfgs(
        problem, rho0, comm=_mpi_comm(), maxiter=maxiter, **cg_kwargs)
    return rho, info, total["cg"]


def test_adaptive_matches_fixed_with_fewer_cg_iters(comm):
    """Coarse-to-fine inner tolerance reaches the same optimum as a fixed tight
    tolerance, but with materially fewer total CG iterations."""
    maxiter = 30

    prob_fixed, rho0 = _build(comm, cg_tol=1e-6)
    _, info_fixed, cg_fixed = _run(prob_fixed, rho0, maxiter)

    # Same floor (1e-6) as the fixed run, but a coarse 1e-1 start.
    prob_adapt, rho0b = _build(comm, cg_tol=1e-6)
    _, info_adapt, cg_adapt = _run(
        prob_adapt, rho0b, maxiter, cg_tol_start=1e-1, cg_tol_min=1e-6)

    # Both make real progress, and the adaptive run (same tight floor) lands in
    # the same basin -- its objective is no worse than the fixed run to within
    # the difference the coarse early path can introduce.
    f0, _ = _build(comm, cg_tol=1e-6)[0].objective_and_gradient(rho0)
    assert info_fixed["objective"] < f0
    assert info_adapt["objective"] < f0
    assert info_adapt["objective"] == pytest.approx(
        info_fixed["objective"], rel=0.15, abs=1e-8)

    # Cheaper: early iterates were solved coarsely.
    assert cg_adapt < cg_fixed

    # The controller actually tightened over the run and stayed in range.
    hist = info_adapt["cg_rtol_history"]
    assert hist, "adaptive run recorded no tolerance history"
    rtols = [r for _, r in hist]
    assert all(1e-6 <= r <= 1e-1 for r in rtols)
    assert rtols[-1] <= rtols[0]


def test_gradient_accurate_at_floor(comm):
    """With the controller frozen at its floor, the analytic gradient still
    matches finite differences -- the floor must satisfy the norm condition."""
    n = 6
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization((n, n), material, comm=comm, cg_tol=1e-4,
                           cg_maxiter=5000)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.15, G=0.08), magnitude=0.01)
    problem = StressTargetProblem(
        homog, cases, regularization=PhaseFieldRegularization(homog))

    # Attach a controller pinned at a tight floor (never advanced, so `current`
    # stays at rtol_start = the floor for every evaluation below).
    problem.inner_tolerance = AdaptiveInnerTolerance(
        rtol_start=1e-10, rtol_min=1e-10)

    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, (n, n))
    _, g = problem.objective_and_gradient(rho)
    for idx in [(0, 0), (0, 3), (3, 0), (2, 2), (n - 1, n - 1)]:
        rp = rho.copy(); rp[idx] += 1e-6
        fp, _ = problem.objective_and_gradient(rp)
        rm = rho.copy(); rm[idx] -= 1e-6
        fm, _ = problem.objective_and_gradient(rm)
        fd = (fp - fm) / 2e-6
        assert np.isclose(g[idx], fd, rtol=2e-4, atol=1e-7), (
            f"pixel {idx}: analytic {g[idx]:.6e} vs FD {fd:.6e}")
