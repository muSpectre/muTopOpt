"""
Lattice (Peierls-Nabarro) pinning of the diffuse phase-field interface.

The Modica-Mortola energy of a flat interface is translation invariant in the
continuum. On a grid that invariance is broken: translating the interface by a
fraction of a pixel changes the *discrete* energy by a small amount that is
periodic in the sub-pixel offset. The amplitude of that ripple is a barrier the
interface must overcome to move, so below a critical driving force it cannot
move at all -- features freeze where the initial condition put them and the
optimizer converges to a grid-locked design instead of the continuum optimum.
This is "discretization trapping", and it is the reason phase-field models need
several grid points across the interface.

In the Modica-Mortola normalization used by :mod:`muTopOpt.regularization`,
``eta`` *is* the interface width: the equilibrium profile is the logistic
``rho(x) = 1/(1 + exp(-x/eta))``, whose 10%-90% width is ``4.39 eta``. The
barrier decays exponentially in ``eta/h`` -- roughly two decades per ``0.25``
-- so the interesting question is where the default ``eta = h`` sits on that
curve. Measured here, relative barrier ``dE/E`` over one pixel of translation:

    eta/h    0.50      0.75      1.00      1.50      2.00
    dE/E     1.7e-2    3.7e-4    6.1e-6    1.0e-9    1.3e-13

``eta = h`` is ~6 orders below the interface energy. Two driving forces set the
scale it has to be compared against, both in energy per unit volume:

* curvature, ``sigma/R`` with ``sigma = 1/3`` the Modica-Mortola interface
  energy per unit area -- between 0.3 (a feature as wide as the cell) and ~20
  (a two-pixel feature) on a 64^2 grid;
* the gradient tolerance the optimizer stops at. ``--bfgs-gtol`` is measured on
  the mesh-invariant volume-fraction derivative ``(V/V_e) df/drho``, which for
  a unit cell is exactly this quantity, and defaults to 2.5.

The critical depinning force at ``eta = h`` is ``g_c ~ 2e-4`` (measured
separately by bisection, and scaling like ``1/h``), so the lattice barrier sits
3-4 orders of magnitude below the weakest force in play -- including at the
very end of a run, which is the tightest point. At ``eta = h/2`` it does not,
which :func:`test_under_resolved_interface_is_pinned` pins down, so that this
file demonstrably *can* detect trapping rather than passing vacuously.

The bounds below are the worst case over the element x quadrature matrix, with
roughly an order of magnitude of headroom. They were cross-checked against a
fully relaxed landscape (each sub-pixel position re-minimized under fixed
volume and fixed first moment, so nothing depends on the analytic profile being
the discrete minimizer); relaxed and analytic agree to 2% at ``eta >= h``
(9.0e-7 vs 9.2e-7), which is what licenses the cheap analytic profile used
here. That relaxed computation needs a constrained solve per sample and is too
slow for this suite.

Note that P1 and Q1 give identical numbers: the fixture varies along one axis
only, where both elements reduce to the same 1-D stencil. Both are kept in the
matrix so that a regression in either code path is caught.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
)
from muTopOpt.regularization import PhaseFieldRegularization

# Grid: long in x (the direction the interface moves), thin in y (the field is
# constant there, and dE/E is independent of the interface length).
NX, NY = 128, 8
H = 1.0 / NX
# Slab edges. 48 px to the nearest periodic image, so at eta <= 2h the two
# interfaces do not interact at double precision.
X1, X2 = 0.375, 0.625
# Samples across one pixel of translation. The ripple is resolved to three
# digits already at 9; 17 and 33 give the same answer.
NB_SHIFTS = 9

#: Modica-Mortola energy of one interface per unit area,
#: ``int[eta rho'^2 + W/eta] = 2 int_0^1 sqrt(W) drho = 1/3``.
SIGMA = 1.0 / 3.0

#: Relative translation barrier ``dE/E``: upper bounds for resolved interfaces,
#: and a *lower* bound at eta = h/2, where pinning is real.
BARRIER_BOUNDS = {
    0.5: ("gt", 1e-4),   # positive control: measured 3.1e-3 .. 1.7e-2
    1.0: ("lt", 5e-5),   # measured 9.2e-7 .. 6.1e-6
    1.5: ("lt", 1e-7),   # measured 1.5e-10 .. 1.0e-9
    2.0: ("lt", 1e-9),   # measured ~1e-13, i.e. at roundoff
}

ELEMENTS = ["p1", "q1"]
#: Which regularization/quadrature to exercise. The lumped nodal double well is
#: the classic pinning culprit, so it must stay in the matrix.
VARIANTS = ["consistent", "lumped", "element-FD"]

_HOMOGENIZATIONS = {}


def _homogenization(element, comm):
    """One Homogenization per element, reused across the parametrization."""
    if element not in _HOMOGENIZATIONS:
        _HOMOGENIZATIONS[element] = Homogenization(
            (NX, NY),
            SimpMaterial(E_solid=1.0, nu=0.3, penalty=2.0, void_ratio=1e-3),
            comm=comm,
            domain_lengths=[1.0, NY * H],
            element=element,
            preconditioner=None,
        )
    return _HOMOGENIZATIONS[element]


def _regularization(element, variant, eta, comm):
    hom = _homogenization(element, comm)
    if variant == "element-FD":
        return PhaseFieldRegularization(hom, eta=eta, weight=1.0)
    return NodalPhaseFieldRegularization(
        hom, eta=eta, weight=1.0, dwell=variant)


def _slab(eta, shift):
    """Periodic slab carrying the analytic equilibrium profile, translated by
    ``shift`` pixels.

    The two edges are logistic steps; summing over periodic images makes the
    field exactly cell-periodic (the naive single-image form wraps into a
    discontinuity and drives rho negative).
    """
    x = np.arange(NX) * H
    rho = np.zeros_like(x)
    for image in (-2, -1, 0, 1, 2):
        xi = x + image
        rho += 1.0 / (
            (1.0 + np.exp(-(xi - X1 - shift * H) / eta))
            * (1.0 + np.exp(-(X2 + shift * H - xi) / eta))
        )
    return np.broadcast_to(rho[:, None], (NX, NY)).copy()


def _energy_vs_shift(reg, eta):
    """Regularization energy at ``NB_SHIFTS`` sub-pixel offsets over one pixel."""
    return np.array([
        reg.value_and_gradient(_slab(eta, s))[0]
        for s in np.linspace(0.0, 1.0, NB_SHIFTS)
    ])


def _relative_barrier(reg, eta):
    e = _energy_vs_shift(reg, eta)
    return (e.max() - e.min()) / e.mean()


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fixture_reproduces_continuum_interface_energy(comm, element, variant):
    """Guard on the fixture itself: the slab must carry the analytic interface
    energy ``2 * sigma`` per unit length. A malformed profile (the easy mistake
    is a non-periodic one) inflates this by orders of magnitude, and would make
    every barrier number below meaningless."""
    eta = 1.0 * H
    reg = _regularization(element, variant, eta, comm)
    expected = 2.0 * SIGMA * (NY * H)  # two interfaces, cell height NY*H
    energy = _energy_vs_shift(reg, eta).mean()
    assert energy == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("eta_over_h", sorted(BARRIER_BOUNDS))
def test_translation_barrier(comm, element, variant, eta_over_h):
    """The sub-pixel translation barrier stays where it belongs: negligible for
    a resolved interface (``eta >= h``), and detectable when it is not."""
    sense, bound = BARRIER_BOUNDS[eta_over_h]
    barrier = _relative_barrier(
        _regularization(element, variant, eta_over_h * H, comm), eta_over_h * H)
    if sense == "lt":
        assert barrier < bound, (
            f"{element}/{variant}: interface at eta = {eta_over_h} h is pinned "
            f"more strongly than expected (dE/E = {barrier:.2e} >= {bound:.0e})"
        )
    else:
        assert barrier > bound, (
            f"{element}/{variant}: the eta = {eta_over_h} h control no longer "
            f"shows pinning (dE/E = {barrier:.2e} <= {bound:.0e}); this test "
            "can no longer detect discretization trapping"
        )


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_under_resolved_interface_is_pinned(comm, element, variant):
    """Positive control, stated on its own because it is what gives the upper
    bounds above their meaning: at half a grid spacing the interface *is*
    lattice-pinned, and the barrier is orders of magnitude above the bound
    asserted at ``eta = h``."""
    pinned = _relative_barrier(
        _regularization(element, variant, 0.5 * H, comm), 0.5 * H)
    resolved = _relative_barrier(
        _regularization(element, variant, 1.0 * H, comm), 1.0 * H)
    assert pinned > 100.0 * resolved


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_barrier_decays_with_interface_width(comm, element, variant):
    """The barrier falls monotonically (and steeply) as the interface is
    resolved by more grid points -- the signature of a Peierls barrier, and a
    structural check that the ripple is lattice pinning rather than noise."""
    widths = [0.5, 0.75, 1.0, 1.25, 1.5]
    barriers = [
        _relative_barrier(_regularization(element, variant, w * H, comm), w * H)
        for w in widths
    ]
    assert all(b > n for b, n in zip(barriers, barriers[1:])), barriers
    # Steeply: at least two decades per half grid spacing.
    assert barriers[0] > 1e4 * barriers[-1]
