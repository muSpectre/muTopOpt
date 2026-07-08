"""
Trust-region Newton-CG outer optimizer (NuMPI ``tr_newton_bounded`` wired to
the second-order-adjoint Hessian-vector product).

These tests are gated on the installed NuMPI providing ``tr_newton_bounded``:
muTopOpt's CI installs NuMPI from PyPI, which may predate it. The Hessian
itself is tested without NuMPI in ``test_hessian.py``.
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
    initial_density,
    optimize_bounded_lbfgs,
    optimize_trust_region,
)

try:
    from NuMPI.Optimization import tr_newton_bounded  # noqa: F401

    HAS_TR = True
except ImportError:
    HAS_TR = False

pytestmark = pytest.mark.skipif(
    not HAS_TR, reason="NuMPI without tr_newton_bounded")


def _mpi_comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        return None


def _build(comm, cg_tol, hessian):
    n = 16
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((n, n), material, comm=comm, cg_tol=cg_tol,
                           cg_maxiter=1000)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    reg = PhaseFieldRegularization(homog)
    problem = StressTargetProblem(homog, cases, regularization=reg,
                                  hessian=hessian)
    rho0 = initial_density(homog.nb_pixels, kind="random", seed=0)
    return problem, rho0


def test_tr_reduces_objective_and_respects_bounds(comm):
    problem, rho0 = _build(comm, cg_tol=1e-6, hessian=True)
    f0, _ = problem.objective_and_gradient(rho0)

    rho, info = optimize_trust_region(
        problem, rho0, comm=_mpi_comm(), maxiter=30)

    assert info["objective"] < f0
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0)
    assert info["nb_hessp"] > 0


def test_tr_comparable_to_lbfgs(comm):
    """From the same start, the TR run must reach an objective at least as
    good as (comparable to) the L-BFGS run."""
    maxiter = 30
    prob_lb, rho0 = _build(comm, cg_tol=1e-6, hessian=False)
    _, info_lb = optimize_bounded_lbfgs(
        prob_lb, rho0, comm=_mpi_comm(), maxiter=maxiter)

    prob_tr, rho0b = _build(comm, cg_tol=1e-6, hessian=True)
    _, info_tr = optimize_trust_region(
        prob_tr, rho0b, comm=_mpi_comm(), maxiter=maxiter,
        cg_tol_start=1e-2, cg_tol_min=1e-8)

    assert info_tr["objective"] <= info_lb["objective"] * 1.05


def test_tr_robust_at_loose_tolerance(comm):
    """The line-search stall scenario: start at a deliberately loose state
    tolerance. The TR run must keep decreasing the objective (accuracy
    control kicks in when the evaluation noise would corrupt acceptance)
    and converge or at least make monotone progress."""
    problem, rho0 = _build(comm, cg_tol=1e-6, hessian=True)
    f0, _ = problem.objective_and_gradient(rho0)

    rho, info = optimize_trust_region(
        problem, rho0, comm=_mpi_comm(), maxiter=40,
        cg_tol_start=1e-2, cg_tol_min=1e-8)

    # Accepted-iterate objective history is monotone non-increasing (each
    # accepted step passed the rho-test) and clearly below the start.
    hist = [f for f in info["history"] if f is not None]
    assert hist[-1] < 0.5 * f0
    assert all(b <= a + 1e-10 for a, b in zip(hist, hist[1:]))
    # The accuracy control leaves a tolerance no looser than the start.
    assert info["final_cg_rtol"] <= 1e-2 + 1e-15


def test_tr_requires_hessian_flag(comm):
    problem, rho0 = _build(comm, cg_tol=1e-6, hessian=False)
    with pytest.raises(ValueError, match="hessian=True"):
        optimize_trust_region(problem, rho0, maxiter=2)
