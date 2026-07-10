"""
Validation of the forward homogenization: known-answer checks that the measured
effective stiffness is physically correct, and that a target-stiffness objective
is zero exactly when the design already realizes the target.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    LoadCase,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
    effective_stiffness,
    isotropic_moduli_2d,
    lame_from_E_nu,
)
from muTopOpt.loadcases import (
    isotropic_stiffness_tensor,
    target_load_cases,
    unit_strains,
)
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs


@pytest.mark.parametrize("element", ["p1", "q1"])
@pytest.mark.parametrize("c", [1.0, 0.6])
def test_uniform_density_recovers_material_stiffness_2d(comm, element, c):
    """A spatially uniform density needs no fluctuation, so C_eff must equal the
    (SIMP-interpolated) base material stiffness C(c) exactly — validating the
    solve + average_stress + effective-stiffness measurement."""
    E, nu = 1.0, 0.3
    material = SimpMaterial(E, nu, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((16, 16), material, comm=comm, element=element,
                           cg_tol=1e-12)
    C = effective_stiffness(homog, np.full(homog.nb_pixels, c))

    lam, mu = material.lame(np.array(c))
    lam, mu = float(lam), float(mu)
    C_expected = np.array([[lam + 2 * mu, lam, 0.0],
                           [lam, lam + 2 * mu, 0.0],
                           [0.0, 0.0, mu]])
    np.testing.assert_allclose(C, C_expected, rtol=1e-9, atol=1e-9)

    # Poisson ratio of the solid (c=1) matches the plane-strain value nu/(1-nu).
    if c == 1.0:
        _, _, poisson, _, zener = isotropic_moduli_2d(C)
        np.testing.assert_allclose(poisson, nu / (1.0 - nu), rtol=1e-9)
        np.testing.assert_allclose(zener, 1.0, rtol=1e-9)  # isotropic


def test_objective_zero_when_target_is_realized(comm):
    """If the target stresses are exactly those produced by the current design,
    the stress part of the objective is zero and its gradient (stress part) too
    — an end-to-end consistency check of objective + adjoint wiring."""
    material = SimpMaterial(1.0, 0.3, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((12, 12), material, comm=comm, element="q1",
                           cg_tol=1e-12)
    rho = np.full(homog.nb_pixels, 0.7)

    # Targets = the design's own homogenized stresses at the unit strains.
    C = effective_stiffness(homog, rho)
    lam, mu = material.lame(np.array(0.7))
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=float(lam + mu), G=float(mu)),
        magnitude=0.01,
    )
    problem = StressTargetProblem(homog, cases, regularization=None)
    f, g = problem.objective_and_gradient(rho)
    assert f < 1e-16
    assert np.max(np.abs(g)) < 1e-8


def test_optimizer_recovers_reachable_effective_stiffness(comm):
    """The end-to-end validation: given a *reachable* target (the effective
    stiffness of a reference design), the optimizer -- from a different random
    start -- must drive the objective toward zero and recover that effective
    stiffness tensor. This exercises SIMP + forward + adjoint + sensitivity +
    the bounded L-BFGS together and checks the *design*, not just the gradient."""
    n = 32
    material = SimpMaterial(1.0, 0.3, penalty=3.0, void_ratio=1e-3)
    h = Homogenization((n, n), material, comm=comm, element="q1", cg_tol=1e-9)

    # Reference: smooth, gray (well-conditioned), hence reachable.
    rng = np.random.default_rng(7)
    r = rng.random((n, n))
    for _ in range(3):
        for ax in range(2):
            r = (r + np.roll(r, 1, ax) + np.roll(r, -1, ax)) / 3
    r = (r - r.min()) / (r.max() - r.min())
    ref = 0.2 + 0.6 * r
    C_ref = effective_stiffness(h, ref)

    # Targets = the reference design's homogenized stresses at the unit strains.
    m = 0.01
    strains = [E * m for E in unit_strains(2, 1.0)]
    u = h.vector_field("uu")
    targets = [h.homogenized_stress(h.solve_macro(E, u), E) for E in strains]
    cases = [LoadCase(E, t, 1.0) for E, t in zip(strains, targets)]

    reg = PhaseFieldRegularization(h, weight=1e-2)
    problem = StressTargetProblem(h, cases, regularization=reg)
    rho0 = initial_density(h.nb_pixels, kind="random", seed=99,
                           volume_fraction=0.5)
    f0, _ = problem.objective_and_gradient(rho0)
    # Reachable target: drive to near-complete stationarity (the default
    # mesh-invariant gtol is calibrated for production stopping, not for
    # this collapse-to-zero check).
    rho, info = optimize_bounded_lbfgs(problem, rho0, maxiter=300, gtol=0.01)

    C_got = effective_stiffness(h, rho)
    rel = np.linalg.norm(C_got - C_ref) / np.linalg.norm(C_ref)
    assert info["objective"] < f0 / 100.0          # objective collapses
    assert rel < 0.05                               # effective stiffness recovered
