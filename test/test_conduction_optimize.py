"""
Known-answer check on the forward conductivity homogenization, and an
end-to-end smoke test that a few L-BFGS iterations reduce the flux-matching
objective. Mirrors test_effective_properties.py / test_optimize_2d.py.
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
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs


@pytest.mark.parametrize("element", ["p1", "q1"])
@pytest.mark.parametrize("c", [1.0, 0.6, 0.0])
def test_uniform_density_recovers_material_conductivity(comm, element, c):
    """A spatially uniform density needs no fluctuation, so the homogenized
    flux must equal kappa(c) * E_macro exactly -- validating the solve +
    homogenized_flux measurement, including the fully-void endpoint."""
    material = SimpConductivity(kappa_solid=2.0, penalty=3.0, void_ratio=1e-3)
    homog = HomogenizationConductivity(
        (16, 16), material, comm=comm, element=element, cg_tol=1e-12)
    homog.set_density(np.full(homog.nb_pixels, c))

    u = homog.temperature_field("test_u")
    E_macro = np.array([1.0, 0.4])
    homog.solve_macro(E_macro, u)
    np.testing.assert_allclose(homog.to_host(u.p), 0.0, atol=1e-10)

    q = homog.homogenized_flux(u, E_macro)
    kappa = material.kappa(np.array(c))
    np.testing.assert_allclose(q, kappa * E_macro, rtol=1e-8, atol=1e-12)


def test_objective_zero_when_target_is_realized(comm):
    """The flux-matching objective (without regularization) is exactly zero
    when the design is spatially uniform and the target flux is exactly the
    solid material's response to the load cases."""
    material = SimpConductivity(kappa_solid=1.0, penalty=3.0, void_ratio=1e-3)
    homog = HomogenizationConductivity((8, 8), material, comm=comm,
                                       cg_tol=1e-12)
    cases = target_load_cases(
        2, isotropic_conductivity_tensor(2, kappa=material.kappa_solid))
    problem = FluxTargetProblem(homog, cases, regularization=None)

    f, g = problem.objective_and_gradient(np.ones(homog.nb_pixels))
    assert f == pytest.approx(0.0, abs=1e-12)


def test_lbfgs_reduces_objective_2d(comm):
    n = 16
    material = SimpConductivity(kappa_solid=1.0, penalty=3.0, void_ratio=1e-3)
    homog = HomogenizationConductivity((n, n), material, comm=comm,
                                       cg_tol=1e-8)
    cases = target_load_cases(
        2, isotropic_conductivity_tensor(2, kappa=0.25), magnitude=0.01)
    reg = PhaseFieldRegularization(homog)
    problem = FluxTargetProblem(homog, cases, regularization=reg)

    rho0 = initial_density(homog.nb_pixels, kind="uniform", volume_fraction=0.5)
    f0, _ = problem.objective_and_gradient(rho0)

    rho, info = optimize_bounded_lbfgs(problem, rho0, maxiter=15)

    assert info["objective"] < f0
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0)
