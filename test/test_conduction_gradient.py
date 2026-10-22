"""
Finite-difference gradient check for the conductivity (flux-matching) problem
-- the decisive correctness gate for the whole adjoint sensitivity assembly
(forward solve + homogenized flux + adjoint solve + compute_sensitivity +
SIMP chain rule + phase-field regularization). Mirrors test_gradient.py.

For sampled pixels i the analytic gradient must match the central difference

    df/drho_i ~ (f(rho + d e_i) - f(rho - d e_i)) / (2 d),  d = 1e-6,

at tight inner CG tolerance (cg_tol = 1e-12).
"""

import numpy as np
import pytest

from muTopOpt import (
    FluxTargetProblem,
    HomogenizationConductivity,
    PhaseFieldRegularization,
    SimpConductivity,
)
from muTopOpt.loadcases_conduction import (
    isotropic_conductivity_tensor,
    target_load_cases,
)

from mpi_helpers import check_fd_gradient, local


def _make_problem(dim, n, with_reg, comm):
    material = SimpConductivity(kappa_solid=1.0, penalty=3.0, void_ratio=1e-2)
    homog = HomogenizationConductivity(
        (n,) * dim, material, comm=comm, cg_tol=1e-12, cg_maxiter=5000
    )
    cases = target_load_cases(
        dim, isotropic_conductivity_tensor(dim, kappa=0.3), magnitude=0.01
    )
    reg = PhaseFieldRegularization(homog) if with_reg else None
    return FluxTargetProblem(homog, cases, regularization=reg)


# TODO: rationale for tolerance rtol=2e-4, atol=1e-7?
def _check_gradient(problem, rho, sample, d=1e-6, rtol=2e-4, atol=1e-7):
    """``sample`` holds global pixel indices; ``rho`` is the local slab."""
    _, g = problem.objective_and_gradient(rho)
    check_fd_gradient(
        problem.h, lambda r: problem.objective_and_gradient(r)[0], rho, g,
        sample, d=d, rtol=rtol, atol=atol)
@pytest.mark.parametrize("with_reg", [False, True])
def test_gradient_2d(comm, with_reg):
    n = 6
    problem = _make_problem(2, n, with_reg, comm)
    rng = np.random.default_rng(0)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n)))
    sample = [(0, 0), (0, 3), (3, 0), (2, 2), (n - 1, n - 1), (n - 1, 2)]
    _check_gradient(problem, rho, sample)


def test_gradient_3d(comm):
    n = 4
    problem = _make_problem(3, n, True, comm)
    rng = np.random.default_rng(1)
    rho = local(problem.h, rng.uniform(0.2, 0.8, (n, n, n)))
    sample = [(0, 0, 0), (1, 2, 3), (2, 2, 2), (n - 1, n - 1, n - 1)]
    _check_gradient(problem, rho, sample)
