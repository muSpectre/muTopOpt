#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The element-wise phase-field regularization on anisotropic voxels.

On a grid with unequal spacing ``h_d`` per axis, the gradient penalty of
``PhaseFieldRegularization`` must be the finite-difference
``eta * sum_d |d rho / d x_d|^2`` of the actual grid, with each axis scaled by
its own ``h_d``; a Laplacian built with ``grid_spacing[0]`` on every axis
mis-scales the penalty along the other axes. Value, gradient and
Hessian-vector product must also agree with each other. The tests need a
muGrid whose ``LaplaceOperator`` takes the spacing per axis and are skipped
otherwise.
"""

import numpy as np
import pytest

from muTopOpt import Homogenization, SimpMaterial
from muTopOpt.regularization import (
    PhaseFieldRegularization,
    _laplace_takes_grid_spacing,
)

needs_anisotropic_laplacian = pytest.mark.skipif(
    not _laplace_takes_grid_spacing(),
    reason="muGrid's LaplaceOperator has no per-axis grid spacing")


def _homogenization(comm, nb_grid_pts, domain_lengths):
    return Homogenization(
        nb_grid_pts,
        SimpMaterial(E_solid=1.0, nu=0.3, penalty=2.0, void_ratio=1e-3),
        comm=comm, domain_lengths=domain_lengths, preconditioner=None,
    )


def _gradient_penalty_reference(rho, grid_spacing):
    """eta-free FD gradient penalty on the periodic grid: sum over pixels of
    sum_d ((rho[i + e_d] - rho[i]) / h_d)^2 times the pixel volume, which is
    what rho . (-Laplace rho) * vol_pixel equals for the per-axis stencil."""
    total = 0.0
    for axis, h in enumerate(grid_spacing):
        total += np.sum(((np.roll(rho, -1, axis=axis) - rho) / h) ** 2)
    return total * float(np.prod(grid_spacing))


@needs_anisotropic_laplacian
@pytest.mark.parametrize("nb_grid_pts,domain_lengths", [
    ((12, 6), (1.0, 2.0)), ((8, 6, 5), (1.0, 3.0, 0.5))])
def test_gradient_penalty_uses_every_axis_spacing(comm, nb_grid_pts,
                                                   domain_lengths):
    """The regularization value equals the per-axis reference

        f = scale * [eta * vol_pixel * sum_i sum_d ((rho[i + e_d] - rho[i]) / h_d)^2
                     + (1/eta) * vol_pixel * sum_i rho_i^2 (1 - rho_i)^2],

    and differs from what an isotropic stencil with h_0 on every axis gives."""
    if comm.size > 1:
        pytest.skip("serial reference")
    hom = _homogenization(comm, nb_grid_pts, domain_lengths)
    assert not np.allclose(hom.grid_spacing, hom.grid_spacing[0])
    eta = 0.3
    reg = PhaseFieldRegularization(hom, eta=eta, weight=1.0)
    rng = np.random.default_rng(1)
    rho = rng.random(nb_grid_pts)
    f, _ = reg.value_and_gradient(rho)
    grad_pen = _gradient_penalty_reference(rho, hom.grid_spacing)
    dwell = np.sum(rho**2 * (1 - rho) ** 2) * reg.vol_pixel
    expected = reg.scale * (eta * grad_pen + dwell / eta)
    assert np.isclose(f, expected, rtol=1e-12)
    # and it is not what the isotropic stencil would give
    wrong = _gradient_penalty_reference(
        rho, [hom.grid_spacing[0]] * len(nb_grid_pts))
    assert not np.isclose(grad_pen, wrong, rtol=1e-3)


@needs_anisotropic_laplacian
@pytest.mark.parametrize("nb_grid_pts,domain_lengths", [
    ((12, 6), (1.0, 2.0)), ((6, 5, 4), (1.0, 3.0, 0.5))])
def test_gradient_and_hessian_are_consistent(comm, nb_grid_pts,
                                             domain_lengths):
    """Central finite differences of the value give the gradient, and of
    the gradient the Hessian-vector product, on an anisotropic grid:

        (f(rho + eps v) - f(rho - eps v)) / (2 eps) = <g(rho), v>
        <(g(rho + eps v) - g(rho - eps v)) / (2 eps), w> = <H(rho) v, w>

    The functional is a quartic polynomial in rho, so the O(eps^2) truncation
    error of the central difference is negligible at eps = 1e-6."""
    hom = _homogenization(comm, nb_grid_pts, domain_lengths)
    reg = PhaseFieldRegularization(hom, eta=0.4, weight=2.0)
    rng = np.random.default_rng(2)
    local = tuple(hom.engine.nb_subdomain_grid_pts)
    rho = rng.random(local)
    v = rng.standard_normal(local)
    w = rng.standard_normal(local)
    eps = 1e-6

    f_plus, _ = reg.value_and_gradient(rho + eps * v)
    f_minus, _ = reg.value_and_gradient(rho - eps * v)
    _, g = reg.value_and_gradient(rho)
    g_dot_v = hom.comm.sum(float(np.sum(g * v)))
    assert np.isclose((f_plus - f_minus) / (2 * eps), g_dot_v, rtol=1e-6)

    _, g_plus = reg.value_and_gradient(rho + eps * v)
    _, g_minus = reg.value_and_gradient(rho - eps * v)
    hv = reg.hessian_vector_product(rho, v)
    fd = (g_plus - g_minus) / (2 * eps)
    lhs = hom.comm.sum(float(np.sum(fd * w)))
    rhs = hom.comm.sum(float(np.sum(hv * w)))
    assert np.isclose(lhs, rhs, rtol=1e-6)
