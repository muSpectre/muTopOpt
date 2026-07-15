"""Hardening of the inner CG solves against non-convergence.

In finite precision the *true* residual of a CG solve has a floor (float32:
``|r|/|b| ~ 1e-6`` for a healthy rhs, far worse for a round-off-level rhs on a
semi-definite operator); past it the recursive residual stagnates or diverges.
``Homogenization.solve_rhs`` therefore (a) snapshots the best iterate and cuts
the solve short once the residual stops improving (stagnation safeguard), and
(b) returns that best iterate -- never raises -- when the iteration cap is hit,
recomputing the true residual it hands back. A solve that is merely less
accurate than requested is fine for the callers (consistent objective,
trust-region model); a raised ConvergenceError kills the whole optimization.
"""

import numpy as np
import pytest

from muTopOpt import Homogenization, SimpMaterial


@pytest.fixture
def homog(comm):
    mat = SimpMaterial(1.0, 0.3, 3.0, 1e-3)
    h = Homogenization((16, 16), mat, comm=comm, cg_tol=1e-10)
    h.set_density(np.random.default_rng(0).uniform(0.1, 0.9, h.nb_pixels))
    return h


def _relative_residual(h, b, x):
    Kx = h.vector_field("test_Kx")
    h._hessp(x, Kx)
    r = h.to_host(b.p) - h.to_host(Kx.p)
    return np.linalg.norm(r) / np.linalg.norm(h.to_host(b.p))


def test_maxiter_returns_best_iterate(homog):
    """Hitting the iteration cap returns the best iterate (with its true
    residual in the out-field) instead of raising ConvergenceError."""
    h = homog
    E = np.array([[0.01, 0.0], [0.0, 0.01]])
    u = h.vector_field("test_u")
    res = h.vector_field("test_res")
    # Far too few iterations to reach rtol: must not raise.
    h.solve_macro(E, u, rtol=1e-10, maxiter=5, residual=res)
    rel = _relative_residual(h, h._rhs, u)
    # The 5 iterations still made progress over the zero initial guess...
    assert rel < 1.0
    # ...and the reported residual is the true b - K u of the returned u.
    Ku = h.vector_field("test_Ku")
    h._hessp(u, Ku)
    np.testing.assert_allclose(
        h.to_host(res.p), h.to_host(h._rhs.p) - h.to_host(Ku.p),
        rtol=1e-5, atol=1e-12)


def test_stagnation_cuts_solve_short(homog):
    """An unreachable tolerance stops at the precision floor after the
    stagnation patience instead of burning the full iteration budget."""
    h = homog
    h.cg_stagnation_patience = 20
    E = np.array([[0.01, 0.0], [0.0, 0.01]])
    u = h.vector_field("test_u")
    # rtol far below the float64 floor: can only end by stagnation.
    h.solve_macro(E, u, rtol=1e-300, maxiter=2000)
    assert h.last_cg_iters < 2000
    # The returned iterate is fully converged (the floor is ~1e-15).
    assert _relative_residual(h, h._rhs, u) < 1e-10


def test_converging_solve_unaffected(homog):
    """A healthy solve converges to its tolerance as before."""
    h = homog
    E = np.array([[0.01, 0.005], [0.005, -0.01]])
    u = h.vector_field("test_u")
    h.solve_macro(E, u, rtol=1e-8)
    assert _relative_residual(h, h._rhs, u) < 1e-7
