#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The same problems on the GPU as on the CPU.

The element-wise work of the gradient, the Hessian-vector product and the
regularization runs in the fields' array module -- cupy on a GPU -- and only
the gradient and the product come back to the host. On the CPU that module
is numpy, so the other tests cannot catch a cupy-specific mistake (such as an
implicit device-to-host conversion, which cupy refuses); these do. They are
skipped without cupy and a GPU build of muGrid.
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

from mpi_helpers import assert_allclose_on_all_ranks

pytestmark = pytest.mark.gpu

DESIGNS = ["element", "nodal"]


def _make_problem(comm, device, design, n=16):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    homog = Homogenization((n, n), material, comm=comm, cg_tol=1e-12,
                           cg_maxiter=5000, device=device)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.15, G=0.08), magnitude=0.01)
    reg = (NodalPhaseFieldRegularization(homog, dwell="consistent")
           if design == "nodal" else PhaseFieldRegularization(homog))
    return StressTargetProblem(homog, cases, regularization=reg,
                               design=design, hessian=True)


@pytest.fixture
def gpu_problem(comm, request):
    """The GPU problem of the test's design, or a skip without a GPU."""
    try:
        import cupy

        cupy.zeros(1)
        problem = _make_problem(comm, "gpu", request.param)
    except Exception as err:  # no cupy, no GPU, or a CPU-only muGrid
        pytest.skip(f"needs cupy and a GPU build of muGrid ({err})")
    if not problem.h.device.is_device:
        pytest.skip("muGrid did not place the fields on a GPU")
    return problem


def _inputs(problem, seed=0):
    rng = np.random.default_rng(seed + problem.h.comm.rank)
    shape = problem.h.nb_pixels
    return rng.uniform(0.2, 0.8, shape), rng.standard_normal(shape)


@pytest.mark.parametrize("gpu_problem", DESIGNS, indirect=True)
def test_gradient_and_hessian_match_the_cpu(comm, gpu_problem):
    cpu_problem = _make_problem(comm, "cpu", gpu_problem.design)
    rho, v = _inputs(gpu_problem)
    f_gpu, g_gpu = gpu_problem.objective_and_gradient(rho)
    f_cpu, g_cpu = cpu_problem.objective_and_gradient(rho)
    # The gradient comes back to the host for the optimizer
    assert isinstance(g_gpu, np.ndarray)
    assert f_gpu == pytest.approx(f_cpu, rel=1e-8)
    assert_allclose_on_all_ranks(comm, g_gpu, g_cpu, rtol=1e-7,
                                 atol=1e-7 * np.abs(g_cpu).max())
    hv_gpu = gpu_problem.hessian_vector_product(v)
    hv_cpu = cpu_problem.hessian_vector_product(v)
    assert isinstance(hv_gpu, np.ndarray)
    assert_allclose_on_all_ranks(comm, hv_gpu, hv_cpu, rtol=1e-6,
                                 atol=1e-6 * np.abs(hv_cpu).max())


@pytest.mark.parametrize("gpu_problem", DESIGNS, indirect=True)
def test_regularization_keeps_device_arrays_on_the_device(comm, gpu_problem):
    import cupy

    reg = gpu_problem.regularization
    cpu_reg = _make_problem(comm, "cpu", gpu_problem.design).regularization
    rho, v = _inputs(gpu_problem)
    f_cpu, g_cpu = cpu_reg.value_and_gradient(rho)
    hv_cpu = cpu_reg.hessian_vector_product(rho, v)
    # Device in, device out
    f, g = reg.value_and_gradient(cupy.asarray(rho))
    hv = reg.hessian_vector_product(cupy.asarray(rho), cupy.asarray(v))
    assert isinstance(g, cupy.ndarray) and isinstance(hv, cupy.ndarray)
    assert f == pytest.approx(f_cpu, rel=1e-10)
    assert_allclose_on_all_ranks(comm, g.get(), g_cpu, rtol=1e-10, atol=0)
    assert_allclose_on_all_ranks(comm, hv.get(), hv_cpu, rtol=1e-10,
                                 atol=1e-12 * np.abs(hv_cpu).max())
    # Host in, host out
    _, g_host = reg.value_and_gradient(rho)
    assert isinstance(g_host, np.ndarray)
    assert_allclose_on_all_ranks(comm, g_host, g_cpu, rtol=1e-10, atol=0)


@pytest.mark.parametrize("gpu_problem", DESIGNS, indirect=True)
def test_device_arrays_stay_on_the_device(comm, gpu_problem):
    """Gradient and Hessian product of device input are device arrays (the
    trust region's path); those of host input are host arrays."""
    import cupy

    rho, v = _inputs(gpu_problem)
    f_dev, g_dev = gpu_problem.objective_and_gradient(cupy.asarray(rho))
    assert isinstance(g_dev, cupy.ndarray)
    hv_dev = gpu_problem.hessian_vector_product(cupy.asarray(v))
    assert isinstance(hv_dev, cupy.ndarray)
    f_host, g_host = gpu_problem.objective_and_gradient(rho)
    hv_host = gpu_problem.hessian_vector_product(v)
    assert f_dev == pytest.approx(f_host, rel=1e-12)
    assert_allclose_on_all_ranks(comm, g_dev.get(), g_host, rtol=1e-12,
                                 atol=0)
    assert_allclose_on_all_ranks(comm, hv_dev.get(), hv_host, rtol=1e-10,
                                 atol=1e-12 * np.abs(hv_host).max())


@pytest.mark.parametrize("gpu_problem", DESIGNS, indirect=True)
def test_trust_region_runs_on_the_device(comm, gpu_problem):
    """The trust region iterates on device arrays -- the callback gets them
    -- and follows the CPU run."""
    import cupy

    from muTopOpt.optimize import optimize_trust_region

    rho0, _ = _inputs(gpu_problem)
    kinds = []

    def callback(it, rho, last):
        kinds.append(isinstance(rho, cupy.ndarray))

    rho_gpu, info_gpu = optimize_trust_region(
        gpu_problem, rho0, comm=gpu_problem.h.comm, maxiter=5,
        callback=callback)
    cpu_problem = _make_problem(comm, "cpu", gpu_problem.design)
    rho_cpu, info_cpu = optimize_trust_region(
        cpu_problem, rho0, comm=cpu_problem.h.comm, maxiter=5)
    assert kinds and all(kinds)
    assert isinstance(rho_gpu, np.ndarray)
    np.testing.assert_allclose(info_gpu["history"], info_cpu["history"],
                               rtol=1e-6)
    assert_allclose_on_all_ranks(comm, rho_gpu, rho_cpu, rtol=0, atol=1e-6)
