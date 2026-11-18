"""
Quadrature of the gradient term of the nodal phase-field regularization.

``NodalPhaseFieldRegularization(gradient_quadrature="one_point")`` evaluates
``∫ |∇ρ_h|²`` with one point per voxel, the voxel centre, as the muFFTTO
large-grid studies do. Validated here:

1. the value equals ``h³ Σ_e |∇ρ_h(x_e)|²`` computed directly with numpy
   from the eight corner values of every voxel (independent of muGrid);
2. a density hourglass mode costs no gradient energy with one point, but
   does with the full quadrature;
3. the gradient matches finite differences and the Hessian-vector product
   matches differences of the gradient;
4. elements without a one-point variant are rejected.
"""

import numpy as np
import pytest

import muGrid
from muTopOpt import Homogenization, NodalPhaseFieldRegularization, SimpMaterial

from mpi_helpers import assert_on_all_ranks, check_fd_gradient, local

pytestmark = pytest.mark.skipif(
    not hasattr(muGrid.FEMElement, "q1_reduced"),
    reason="needs a muGrid with FEMElement.q1_reduced")


def _homogenization(comm, n=4, dim=3, element="q1"):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=2.0, void_ratio=1e-3)
    return Homogenization((n,) * dim, material, comm=comm, element=element)


def _regularization(h, quadrature, eta=0.3):
    # lumped double well: its value is a plain nodal sum, and the Hessian
    # product is exact (see NodalPhaseFieldRegularization)
    return NodalPhaseFieldRegularization(
        h, eta=eta, weight=1.0, dwell="lumped",
        gradient_quadrature=quadrature)


def _voxel_centre_gradient_energy(rho, h):
    """``h³ Σ_e |∇ρ_h(x_e)|²`` of the periodic nodal field ``rho`` (global
    array; node (i, j, k) is the lower corner of voxel (i, j, k)). The
    gradient of the trilinear interpolant at the voxel centre along x is the
    mean of the four differences along the x edges, divided by h."""
    energy = 0.0
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        derivative = 0.0
        for shift_1 in (0, 1):
            for shift_2 in (0, 1):
                corner = np.roll(rho, (-shift_1, -shift_2), axis=others)
                derivative = derivative + (
                    np.roll(corner, -1, axis=axis) - corner) / (4 * h)
        energy += np.sum(derivative ** 2)
    return h ** 3 * energy


def test_one_point_value_is_voxel_centre_gradient(comm):
    """The regularization's value with one point per voxel equals the
    voxel-centre gradient energy plus the lumped double well."""
    n, eta = 4, 0.3
    h = _homogenization(comm, n)
    reg = _regularization(h, "one_point", eta)
    rho = np.random.default_rng(0).uniform(0.0, 1.0, (n, n, n))
    f, _ = reg.value_and_gradient(local(h, rho))

    spacing = 1.0 / n
    gradient_energy = _voxel_centre_gradient_energy(rho, spacing)
    double_well = spacing ** 3 * np.sum(rho ** 2 * (1 - rho) ** 2)
    expected = reg.scale * (eta * gradient_energy + double_well / eta)
    assert np.isclose(f, expected, rtol=1e-12), (f, expected)


def test_one_point_misses_hourglass_mode(comm):
    """``0.5 + a (−1)^(i+j+k)`` has zero gradient at every voxel centre: with
    one point only the double well is left. The full (2×2×2 Gauss)
    quadrature sees the mode: at a Gauss point each derivative is
    ``±2a / (3h)`` (the factor 1/3 is ``(1 − 2ξ)(1 − 2ζ)`` at the Gauss
    coordinates), so ``∫ |∇ρ|² = 3 (2a / (3h))²`` on the unit cell."""
    n, eta, a = 4, 0.3, 0.1
    h = _homogenization(comm, n)
    i, j, k = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    rho = local(h, 0.5 + a * (-1.0) ** (i + j + k))

    one_point = _regularization(h, "one_point", eta)
    full = _regularization(h, "full", eta)
    # lumped double well on the unit cell, W(0.5 - a) = W(0.5 + a)
    double_well = one_point.scale * (0.25 - a ** 2) ** 2 / eta
    gradient_term = full.scale * eta * 3 * (2 * a * n / 3) ** 2
    f_one_point, _ = one_point.value_and_gradient(rho)
    f_full, _ = full.value_and_gradient(rho)
    assert np.isclose(f_one_point, double_well, rtol=1e-12)
    assert np.isclose(f_full, double_well + gradient_term, rtol=1e-12)


def test_one_point_gradient_fd(comm):
    """Finite differences of the value against the gradient."""
    n = 4
    h = _homogenization(comm, n)
    reg = _regularization(h, "one_point")
    rho = local(h, np.random.default_rng(1).uniform(0.2, 0.8, (n, n, n)))
    _, g = reg.value_and_gradient(rho)
    check_fd_gradient(
        h, lambda r: reg.value_and_gradient(r)[0], rho, g,
        [(0, 0, 0), (1, 2, 3), (n - 1, n - 1, n - 1)],
        rtol=1e-6, atol=1e-10, label="node")


def test_one_point_hessian_vector_product(comm):
    """``H v`` against central differences of the gradient along ``v``. The
    regularization is a quartic polynomial, so the central difference is
    exact up to a term in ε² and round-off."""
    n, eps = 4, 1e-4
    h = _homogenization(comm, n)
    reg = _regularization(h, "one_point")
    rng = np.random.default_rng(2)
    rho = local(h, rng.uniform(0.2, 0.8, (n, n, n)))
    v = local(h, rng.standard_normal((n, n, n)))
    hv = reg.hessian_vector_product(rho, v)
    _, g_plus = reg.value_and_gradient(rho + eps * v)
    _, g_minus = reg.value_and_gradient(rho - eps * v)
    difference = (g_plus - g_minus) / (2 * eps)
    scale = comm.max(float(np.abs(hv).max()))
    failures = []
    if not np.allclose(hv, difference, rtol=0.0, atol=1e-6 * scale):
        failures.append(
            f"largest deviation {np.abs(hv - difference).max():.3e} "
            f"of {scale:.3e}")
    assert_on_all_ranks(comm, failures)


@pytest.mark.parametrize("dim, element", [(2, "q1"), (3, "p1")])
def test_one_point_needs_trilinear_hexahedra(comm, dim, element):
    h = _homogenization(comm, 4, dim, element)
    with pytest.raises(ValueError, match="3D trilinear hexahedra"):
        _regularization(h, "one_point")


def test_unknown_gradient_quadrature(comm):
    h = _homogenization(comm, 4)
    with pytest.raises(ValueError, match="unknown gradient quadrature"):
        _regularization(h, "two_point")
