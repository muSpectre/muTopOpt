"""Helpers for tests that must give the same verdict serially and under
``mpirun``.

Every muTopOpt density, gradient and Hessian-vector product is the rank-local
slab of the global grid (``homog.nb_pixels`` starting at
``homog.engine.subdomain_locations``). The tests therefore build their fields
on the *global* grid, identically on every rank (a seeded generator), hand each
rank its slab with :func:`local`, and address single pixels by their *global*
index: a finite-difference perturbation of pixel ``i`` is applied on the rank
that owns it only, while every rank takes part in the (collective) evaluation.

Assertions on rank-local data go through :func:`assert_on_all_ranks`, which
fails the test on every rank when it fails on one. A test that failed on
some ranks only would let the others run ahead into the next test's collectives
and hang the run.
"""

import numpy as np


def owned(homog):
    """Slices of the global grid that this rank owns."""
    loc = homog.engine.subdomain_locations
    return tuple(slice(lo, lo + m) for lo, m in zip(loc, homog.nb_pixels))


def local(homog, global_array):
    """This rank's slab of ``global_array`` (grid axes last)."""
    return np.ascontiguousarray(
        np.asarray(global_array)[(Ellipsis,) + owned(homog)])


def local_index(homog, global_index):
    """Local index of the global pixel ``global_index``, or None if another
    rank owns it."""
    loc = homog.engine.subdomain_locations
    idx = tuple(g - lo for g, lo in zip(global_index, loc))
    if all(0 <= i < m for i, m in zip(idx, homog.nb_pixels)):
        return idx
    return None


def perturbed(homog, x, global_index, delta):
    """Copy of the local field ``x`` with ``delta`` added at a global pixel
    (on the owning rank; elsewhere an unchanged copy)."""
    xp = np.array(x, copy=True)
    idx = local_index(homog, global_index)
    if idx is not None:
        xp[idx] += delta
    return xp


def value_at(homog, x, global_index):
    """Value of the local field ``x`` at a global pixel, on every rank."""
    idx = local_index(homog, global_index)
    return homog.comm.sum(float(x[idx]) if idx is not None else 0.0)


def assert_on_all_ranks(comm, failures):
    """Fail on every rank if any rank collected a failure message."""
    nb_failed = comm.sum(len(failures))
    assert nb_failed == 0, (
        "\n".join(failures) if failures
        else f"{nb_failed} check(s) failed on other ranks")


def check_fd_gradient(homog, f, x, g, global_indices, d=1e-6, rtol=2e-4,
                      atol=1e-7, label="pixel"):
    """Central finite differences of the scalar ``f`` (collective: every rank
    calls it with its slab) at ``x``, perturbing one global pixel at a time,
    against the local gradient ``g``."""
    failures = []
    for gi in global_indices:
        fp = f(perturbed(homog, x, gi, d))
        fm = f(perturbed(homog, x, gi, -d))
        fd = (fp - fm) / (2 * d)
        idx = local_index(homog, gi)
        if idx is not None and not np.isclose(g[idx], fd, rtol=rtol,
                                              atol=atol):
            failures.append(
                f"{label} {gi}: analytic {g[idx]:.6e} vs FD {fd:.6e}")
    assert_on_all_ranks(homog.comm, failures)


def assert_allclose_on_all_ranks(comm, actual, desired, **kwargs):
    """``np.testing.assert_allclose`` on rank-local arrays, failing on every
    rank if it fails on one."""
    try:
        np.testing.assert_allclose(actual, desired, **kwargs)
        failures = []
    except AssertionError as err:
        failures = [str(err)]
    assert_on_all_ranks(comm, failures)
