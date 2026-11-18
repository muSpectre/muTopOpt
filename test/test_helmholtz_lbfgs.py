"""
Tests for the Helmholtz smoother and the field-based L-BFGS driver
(:func:`muTopOpt.optimize.optimize_helmholtz_lbfgs`).
"""
import numpy as np
import pytest

import muGrid

from muTopOpt import (
    Homogenization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import (
    first_step_scaled,
    helmholtz_preconditioner,
    initial_density,
    optimize_bounded_lbfgs,
    optimize_helmholtz_lbfgs,
)

pytest.importorskip("muGrid.Optimize")

serial_only = pytest.mark.skipif(
    muGrid.Communicator.world().size > 1, reason="serial test")


def make_problem(n=16, device=None, comm=None, **options):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homogenization = Homogenization((n, n), material, comm=comm,
                                    device=device, **options)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    regularization = PhaseFieldRegularization(homogenization)
    return StressTargetProblem(homogenization, cases,
                               regularization=regularization)


@serial_only
def test_smoother_inverts_the_helmholtz_stencil():
    """(I − c ℓ² Δ_h) M v = v, with Δ_h the five-point Laplacian."""
    n, length, c = 16, 0.07, 4.0
    problem = make_problem(n)
    h = problem.h
    smoother = helmholtz_preconditioner(h, length, c)
    v, Mv = h.scalar_field("v"), h.scalar_field("Mv")
    values = np.random.default_rng(0).standard_normal((n, n))
    v.p[...] = values
    smoother(v, Mv)
    z = np.array(Mv.p).reshape(n, n)
    spacing = h.grid_spacing
    laplace = sum(
        (np.roll(z, 1, d) + np.roll(z, -1, d) - 2 * z) / spacing[d] ** 2
        for d in range(2))
    np.testing.assert_allclose(z - c * length**2 * laplace, values,
                               atol=1e-12)
    # a smoother: the mean is kept, the variation is damped
    assert z.mean() == pytest.approx(values.mean())
    assert z.std() < 0.5 * values.std()


@serial_only
def test_first_step_scaled_fixes_the_scale_at_the_first_call():
    """σ M: the first result has the largest entry first_step, and every
    later one is the same σ times M v."""
    h = make_problem().h
    smoother = helmholtz_preconditioner(h, 0.07)
    scaled = first_step_scaled(smoother, h.comm, 0.05)
    v, out, reference = (h.scalar_field(name)
                         for name in ("v", "out", "reference"))
    rng = np.random.default_rng(1)
    v.p[...] = rng.standard_normal(v.p.shape)
    scaled(v, out)
    assert np.abs(out.p).max() == pytest.approx(0.05, rel=1e-12)
    (sigma,) = scaled.sigma
    v.p[...] = 10.0 * rng.standard_normal(v.p.shape)
    scaled(v, out)
    smoother(v, reference)
    np.testing.assert_allclose(out.p, sigma * reference.p, rtol=1e-12)
    assert scaled.sigma == [sigma]


@serial_only
def test_first_step_bounds_the_first_change():
    """With first_step the first accepted step changes no density by more
    than first_step, the run still descends, and σ is reported."""
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    problem = make_problem()
    f0, _ = problem.objective_and_gradient(rho0)
    iterates = []
    _, info = optimize_helmholtz_lbfgs(
        problem, rho0, maxiter=8, first_step=0.05,
        callback=lambda it, x, last: iterates.append(np.array(x)))
    assert np.abs(iterates[0] - rho0).max() <= 0.05 + 1e-12
    assert info["objective"] < f0
    assert info["first_step_sigma"] > 0
    with pytest.raises(ValueError, match="first_step"):
        optimize_helmholtz_lbfgs(make_problem(), rho0, precond=None,
                                 first_step=0.05)


@serial_only
def test_without_smoother_follows_the_numpi_driver():
    """precond=None is NuMPI's algorithm on fields: the same objectives,
    iterate by iterate, to the accuracy of the inner solves."""
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    _, reference = optimize_bounded_lbfgs(make_problem(), rho0, maxiter=8,
                                          maxcor=20)
    _, info = optimize_helmholtz_lbfgs(make_problem(), rho0, maxiter=8,
                                       precond=None)
    np.testing.assert_allclose(info["history"], reference["history"],
                               rtol=1e-6)
    assert info["nit"] == reference["nit"]


def test_reduces_the_objective_within_the_bounds(comm):
    problem = make_problem(comm=comm)
    rho0 = initial_density(problem.h.nb_pixels, kind="uniform",
                           volume_fraction=0.5)
    f0, _ = problem.objective_and_gradient(rho0)
    seen = []
    rho, info = optimize_helmholtz_lbfgs(
        problem, rho0, maxiter=15,
        callback=lambda it, x, last: seen.append((it, last["objective"])))
    assert info["objective"] < f0
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0)
    assert rho.shape == rho0.shape
    assert info["nfev"] >= info["nit"] + 1
    # one callback per accepted iterate, with decreasing objectives
    assert [it for it, _ in seen] == list(range(1, info["nit"] + 1))
    assert seen[-1][1] == pytest.approx(info["objective"])
    assert all(a[1] >= b[1] for a, b in zip(seen, seen[1:]))


@serial_only
def test_smoother_length_defaults_to_the_interface_width():
    """The default length is the regularization's eta: passing it
    explicitly changes nothing."""
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    problem = make_problem()
    _, default = optimize_helmholtz_lbfgs(problem, rho0, maxiter=5)
    other = make_problem()
    _, explicit = optimize_helmholtz_lbfgs(
        other, rho0, maxiter=5, length=other.regularization.eta)
    np.testing.assert_allclose(explicit["history"], default["history"],
                               rtol=1e-9)
    with pytest.raises(ValueError, match="preconditioner"):
        optimize_helmholtz_lbfgs(problem, rho0, precond="sobolev")


@serial_only
def test_with_multigrid_energy_stop_and_warm_starts():
    """The fast configuration end to end."""
    pytest.importorskip("muGrid.Multigrid")
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homogenization = Homogenization(
        (16, 16), material, preconditioner="multigrid", cg_stop="energy",
        cg_tol=1e-6, multigrid_precision="single")
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    problem = StressTargetProblem(
        homogenization, cases, warm_start="device",
        regularization=PhaseFieldRegularization(homogenization))
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    f0, _ = problem.objective_and_gradient(rho0)
    _, info = optimize_helmholtz_lbfgs(problem, rho0, maxiter=10)
    assert info["objective"] < f0


@pytest.mark.gpu
def test_gpu_matches_cpu():
    pytest.importorskip("cupy")
    if not muGrid.has_gpu:
        pytest.skip("muGrid without GPU support")
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    _, cpu = optimize_helmholtz_lbfgs(make_problem(), rho0, maxiter=6)
    _, gpu = optimize_helmholtz_lbfgs(make_problem(device="gpu"), rho0,
                                      maxiter=6)
    np.testing.assert_allclose(gpu["history"], cpu["history"], rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("maxcor", [2, 20])
def test_host_pairs_give_the_same_run(maxcor):
    """The L-BFGS pairs in host memory: the same objectives and design bit
    for bit as with the pairs on the device (maxcor 2 drops pairs every
    step). Needs a muGrid whose L-BFGS has pair_store."""
    pytest.importorskip("cupy")
    if not muGrid.has_gpu:
        pytest.skip("muGrid without GPU support")
    import inspect

    from muGrid.Optimize import l_bfgs_bounded

    if "pair_store" not in inspect.signature(l_bfgs_bounded).parameters:
        pytest.skip("muGrid's L-BFGS has no pair_store")
    rho0 = initial_density((16, 16), kind="uniform", volume_fraction=0.5)
    runs = [optimize_helmholtz_lbfgs(make_problem(device="gpu"), rho0,
                                     maxiter=8, maxcor=maxcor,
                                     pair_store=store)
            for store in ("device", "host")]
    (rho_device, device), (rho_host, host) = runs
    assert host["history"] == device["history"]
    assert host["nfev"] == device["nfev"]
    np.testing.assert_array_equal(rho_host, rho_device)
