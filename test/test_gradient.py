"""
Finite-difference gradient check -- the decisive correctness gate for the whole
adjoint sensitivity assembly (forward solve + homogenized stress + adjoint solve
+ compute_sensitivity + SIMP chain rule + phase-field regularization).
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


def _make_problem(dim, n, with_reg, comm):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization(
        (n,) * dim, material, comm=comm, cg_tol=1e-12, cg_maxiter=5000
    )
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, K=0.15, G=0.08), magnitude=0.01
    )
    reg = (
        PhaseFieldRegularization(homog)
        if with_reg
        else None
    )
    return StressTargetProblem(homog, cases, regularization=reg)


def _check_gradient(problem, rho, sample, d=1e-6, rtol=2e-4, atol=1e-7):
    _, g = problem.objective_and_gradient(rho)
    for idx in sample:
        rp = rho.copy()
        rp[idx] += d
        fp, _ = problem.objective_and_gradient(rp)
        rm = rho.copy()
        rm[idx] -= d
        fm, _ = problem.objective_and_gradient(rm)
        fd = (fp - fm) / (2 * d)
        assert np.isclose(g[idx], fd, rtol=rtol, atol=atol), (
            f"pixel {idx}: analytic {g[idx]:.6e} vs FD {fd:.6e}"
        )


@pytest.mark.parametrize("with_reg", [False, True])
def test_gradient_2d(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, with_reg, comm)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, (n, n))
    # Sample interior and boundary pixels (the element-wise sensitivity is
    # gather-based, so boundary pixels must match too).
    sample = [(0, 0), (0, 3), (3, 0), (2, 2), (n - 1, n - 1), (n - 1, 2)]
    _check_gradient(problem, rho, sample)


def test_gradient_3d(comm):
    n = 4
    problem = _make_problem(3, n, True, comm)
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, (n, n, n))
    sample = [(0, 0, 0), (1, 2, 3), (2, 2, 2), (n - 1, n - 1, n - 1)]
    _check_gradient(problem, rho, sample)
