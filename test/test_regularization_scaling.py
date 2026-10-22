"""
Normalization of the phase-field regularization.

``f_reg / weight`` is the interfacial area in units of ``L^(D-1)`` with ``L``
the linear cell size (see :mod:`muTopOpt.regularization`). Two consequences
are pinned down here:

* a flat slab with the analytic logistic profile on a unit-length cell
  (grid spacing ``h = 1/NX``) carries ``2 * (NY * h) / L``, i.e. two
  interfaces of length ``NY * h`` on a cell of volume ``NY * h`` with
  ``L = sqrt(V)`` (the raw Modica-Mortola bracket gives ``c_W = 1/3`` times
  the length);
* scaling the cell by ``lambda`` together with ``eta``,
  ``f_reg(lambda x; lambda eta) = f_reg(x; eta)``, leaves ``f_reg`` and the
  density gradient unchanged, so the optimized design does not depend on the
  absolute size of the cell.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
)
from muTopOpt.regularization import (
    C_W,
    PhaseFieldRegularization,
    perimeter_prefactor,
)

from mpi_helpers import assert_allclose_on_all_ranks, local

NX, NY = 64, 8
X1, X2 = 0.375, 0.625

VARIANTS = ["consistent", "lumped", "element-FD"]


def _homogenization(scale, comm, element="p1"):
    h = scale / NX
    return Homogenization(
        (NX, NY),
        SimpMaterial(E_solid=1.0, nu=0.3, penalty=2.0, void_ratio=1e-3),
        comm=comm,
        domain_lengths=[scale, NY * h],
        element=element,
        preconditioner=None,
    )


def _regularization(hom, variant, eta):
    if variant == "element-FD":
        return PhaseFieldRegularization(hom, eta=eta, weight=1.0)
    return NodalPhaseFieldRegularization(hom, eta=eta, weight=1.0, dwell=variant)


def _slab(scale, eta):
    """Periodic slab with the logistic equilibrium profile on a cell of length
    ``scale`` (summed over periodic images so the field is exactly periodic)."""
    x = np.arange(NX) * scale / NX
    rho = np.zeros_like(x)
    for image in (-2, -1, 0, 1, 2):
        xi = x + image * scale
        rho += 1.0 / (
            (1.0 + np.exp(-(xi - X1 * scale) / eta))
            * (1.0 + np.exp(-(X2 * scale - xi) / eta))
        )
    return np.broadcast_to(rho[:, None], (NX, NY)).copy()


def test_c_w_is_interface_energy_of_quartic_well():
    """c_W = 2 int_0^1 sqrt(W(s)) ds = 2 int_0^1 s (1 - s) ds = 1/3 for
    W(s) = s^2 (1 - s)^2 (trapezoid rule on 20001 points)."""
    s = np.linspace(0.0, 1.0, 20001)
    assert C_W == pytest.approx(2.0 * np.trapezoid(s * (1.0 - s), s), rel=1e-8)


def test_prefactor(comm):
    """perimeter_prefactor = 1 / (c_W V^((D-1)/D)) = 1 / (c_W V^(1/2)) in 2D."""
    hom = _homogenization(2.0, comm)
    V = hom.domain_volume
    assert perimeter_prefactor(hom) == pytest.approx(1.0 / (C_W * V**0.5))


@pytest.mark.parametrize("variant", VARIANTS)
def test_slab_energy_is_relative_interface_length(comm, variant):
    """f_reg / weight = (interfacial length) / L = 2 (NY h) / sqrt(V) at
    eta = 1.5 h, to 5%: a coarse check, since a malformed (e.g. non-periodic)
    profile is off by orders of magnitude."""
    scale = 1.0
    hom = _homogenization(scale, comm)
    eta = 1.5 * scale / NX
    f, _ = _regularization(hom, variant, eta).value_and_gradient(
            local(hom, _slab(scale, eta)))
    L = hom.domain_volume**0.5
    assert f == pytest.approx(2.0 * (NY * scale / NX) / L, rel=0.05)


@pytest.mark.parametrize("variant", VARIANTS)
def test_invariant_under_cell_rescaling(comm, variant):
    """Same design on cells of size 1 and 2.5 with eta/L fixed: identical
    f_reg and identical density gradient."""
    results = []
    for scale in (1.0, 2.5):
        hom = _homogenization(scale, comm)
        eta = 1.5 * scale / NX
        results.append(
            _regularization(hom, variant, eta).value_and_gradient(
            local(hom, _slab(scale, eta))))
    (f1, g1), (f2, g2) = results
    assert f1 == pytest.approx(f2, rel=1e-10)
    assert_allclose_on_all_ranks(comm, g1, g2, rtol=1e-10, atol=1e-14)
