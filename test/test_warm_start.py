"""
Tests for warm starts of the forward and adjoint solves between evaluations
(:mod:`muTopOpt.warmstart`).
"""
import numpy as np
import pytest

import muGrid

from muTopOpt import Homogenization, SimpMaterial, StressTargetProblem
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.warmstart import STORES, WarmStarts

N = 16


def designs():
    """Two nearby designs, as two consecutive iterates late in an
    optimisation: they differ by 1e-4."""
    rng = np.random.default_rng(0)
    first = rng.uniform(0.3, 0.7, (N, N))
    return first, np.clip(first + 1e-4 * rng.standard_normal((N, N)), 0, 1)


def make_problem(warm_start, device=None, **options):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homogenization = Homogenization((N, N), material, cg_tol=1e-7,
                                    device=device, **options)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.15, G=0.08), magnitude=0.01)
    return StressTargetProblem(homogenization, cases, warm_start=warm_start)


def evaluate_twice(problem):
    """Objective, gradient and CG iterations at the second design, after an
    evaluation at the first."""
    first, second = designs()
    problem.objective_and_gradient(first)
    f, g = problem.objective_and_gradient(second)
    return f, np.asarray(g), sum(problem.last["cg_iters"])


@pytest.mark.parametrize("store", STORES)
def test_same_result_in_fewer_iterations(store):
    f_cold, g_cold, iterations_cold = evaluate_twice(make_problem(None))
    f, g, iterations = evaluate_twice(make_problem(store))
    assert f == pytest.approx(f_cold, rel=1e-7)
    np.testing.assert_allclose(g, g_cold, atol=1e-5 * np.abs(g_cold).max())
    assert iterations < 0.7 * iterations_cold


@pytest.mark.parametrize("store", ["device", "host32"])
def test_with_multigrid_and_energy_stop(store):
    pytest.importorskip("muGrid.Multigrid")
    options = dict(preconditioner="multigrid", cg_stop="energy")
    f_cold, g_cold, iterations_cold = evaluate_twice(
        make_problem(None, **options))
    f, g, iterations = evaluate_twice(make_problem(store, **options))
    assert f == pytest.approx(f_cold, rel=1e-7)
    np.testing.assert_allclose(g, g_cold, atol=1e-5 * np.abs(g_cold).max())
    assert iterations < 0.7 * iterations_cold


def test_first_evaluation_starts_cold():
    """Nothing is kept before the first evaluation: it equals a cold one."""
    first, _ = designs()
    cold, warm = make_problem(None), make_problem("device")
    f_cold, g_cold = cold.objective_and_gradient(first)
    f_warm, g_warm = warm.objective_and_gradient(first)
    assert f_warm == f_cold
    np.testing.assert_array_equal(g_warm, g_cold)
    assert warm.last["cg_iters"] == cold.last["cg_iters"]


def test_store_keeps_one_vector_per_key():
    problem = make_problem(None)
    h = problem.h
    for store in STORES:
        warm = WarmStarts(h, store)
        field, other = h.vector_field("a"), h.vector_field("b")
        field.p[...] = np.random.default_rng(1).standard_normal(field.p.shape)
        assert ("forward", 0) not in warm
        assert not warm.load(("forward", 0), other)
        warm.store(("forward", 0), field)
        assert warm.load(("forward", 0), other)
        tolerance = 1e-6 if store.endswith("32") else 0.0
        np.testing.assert_allclose(other.p, field.p, rtol=tolerance)
        warm.clear()
        assert ("forward", 0) not in warm
    with pytest.raises(ValueError, match="store"):
        WarmStarts(h, "disk")


@pytest.mark.gpu
@pytest.mark.parametrize("store", STORES)
def test_gpu(store):
    pytest.importorskip("cupy")
    if not muGrid.has_gpu:
        pytest.skip("muGrid without GPU support")
    f_cold, g_cold, iterations_cold = evaluate_twice(
        make_problem(None, device="gpu"))
    f, g, iterations = evaluate_twice(make_problem(store, device="gpu"))
    assert f == pytest.approx(f_cold, rel=1e-7)
    np.testing.assert_allclose(g, g_cold, atol=1e-5 * np.abs(g_cold).max())
    assert iterations < 0.7 * iterations_cold
