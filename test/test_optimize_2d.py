"""End-to-end smoke tests: a few L-BFGS iterations reduce the objective.

Runs with NuMPI's bound-constrained, MPI-distributed L-BFGS, so the same test
exercises the serial and (under mpirun) the parallel optimizer.
"""

import numpy as np

from muTopOpt import Homogenization, PhaseFieldRegularization, SimpMaterial, \
    StressTargetProblem
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs


def _mpi_comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        return None


def test_lbfgs_reduces_objective_2d(comm):
    n = 16
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((n, n), material, comm=comm, cg_tol=1e-8)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01
    )
    reg = PhaseFieldRegularization(homog, eta=1.0, well_weight=1e-3)
    problem = StressTargetProblem(homog, cases, regularization=reg)

    rho0 = initial_density(homog.nb_pixels, kind="uniform", volume_fraction=0.5)
    f0, _ = problem.objective_and_gradient(rho0)

    rho, info = optimize_bounded_lbfgs(
        problem, rho0, comm=_mpi_comm(), maxiter=15
    )

    assert info["objective"] < f0
    # Box bounds are respected exactly (projection, not penalty).
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0)
