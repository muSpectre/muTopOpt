"""The single-precision inner-CG tolerance floor.

float32 eps is 1.19e-7, so the *true* residual ``b - Kx`` stagnates around
1e-6 relative even while the recursively updated CG residual keeps shrinking
below it. A tolerance under that floor cannot be met: the solve burns
``cg_maxiter`` iterations and is rescued by the stagnation guard, returning a
worse iterate than the reachable tolerance would have.

The floor used to live only in ``simulate.py``, which is how it came to be
applied on some of that driver's paths, none of ``simulate_conduction.py``'s,
and neither library default (``cg_tol=1e-8``, ``cg_tol_min=1e-10``). These
tests pin it at every point that consumes a tolerance.
"""

import warnings

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import _make_inner_tolerance, _problem_rtol_floor
from muTopOpt.precision import FLOAT32_RTOL_FLOOR, cg_rtol_floor


def _homog(comm, dtype, **kwargs):
    mat = SimpMaterial(1.0, 0.3, 3.0, 1e-3)
    return Homogenization((16, 16), mat, comm=comm, dtype=dtype, **kwargs)


def _problem(comm, dtype, **kwargs):
    h = _homog(comm, dtype, **kwargs)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    return StressTargetProblem(
        h, cases, regularization=PhaseFieldRegularization(h))


def test_cg_rtol_floor_is_precision_dependent():
    assert cg_rtol_floor(np.float32) == FLOAT32_RTOL_FLOOR
    assert cg_rtol_floor(np.float64) == 0.0
    # np.dtype instances, not just the scalar types.
    assert cg_rtol_floor(np.dtype("float32")) == FLOAT32_RTOL_FLOOR


def test_float32_default_cg_tol_is_reachable(comm):
    """The 1e-8 default is meaningless in float32; the default must adapt --
    and silently, because the caller never chose it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        h = _homog(comm, np.float32)
    assert h.cg_tol >= FLOAT32_RTOL_FLOOR


def test_float64_default_cg_tol_is_unchanged(comm):
    assert _homog(comm, np.float64).cg_tol == 1e-8


def test_explicit_unreachable_cg_tol_warns_and_clamps(comm):
    """An explicit request the arithmetic cannot satisfy is the caller's
    mistake and should be reported, unlike the default."""
    with pytest.warns(RuntimeWarning, match="accuracy floor"):
        h = _homog(comm, np.float32, cg_tol=1e-9)
    assert h.cg_tol == FLOAT32_RTOL_FLOOR


def test_explicit_tight_cg_tol_is_fine_in_double(comm):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        h = _homog(comm, np.float64, cg_tol=1e-12)
    assert h.cg_tol == 1e-12


def test_per_solve_rtol_is_clamped(comm):
    """The clamp sits where the tolerance is *consumed*, so a per-solve rtol
    -- an adaptive controller's current value, a Hessian-vector solve's own
    tolerance -- cannot route around it either."""
    h = _homog(comm, np.float32, cg_tol=1e-4)
    with pytest.warns(RuntimeWarning, match="accuracy floor"):
        assert h._clamp_rtol(1e-10) == FLOAT32_RTOL_FLOOR
    # Warns once per instance, not once per solve.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert h._clamp_rtol(1e-10) == FLOAT32_RTOL_FLOOR
    # Reachable tolerances pass through untouched.
    assert h._clamp_rtol(1e-4) == 1e-4
    assert h._clamp_rtol(None) is None


@pytest.mark.parametrize("dtype,expected",
                         [(np.float32, FLOAT32_RTOL_FLOOR), (np.float64, 0.0)])
def test_problem_rtol_floor(comm, dtype, expected):
    assert _problem_rtol_floor(_problem(comm, dtype)) == expected


def test_adaptive_controller_floor_is_raised_for_float32(comm):
    """``_make_inner_tolerance`` is where the clamp belongs -- not inside
    AdaptiveInnerTolerance, which stays a pure dtype-agnostic controller."""
    p = _problem(comm, np.float32, cg_tol=1e-4)
    controller = _make_inner_tolerance(
        p, cg_tol_start=1e-2, cg_tol_min=1e-10, cg_forcing_c=0.5,
        cg_forcing_exp=1.0, cg_stall_rel=0.1, cg_stall_shrink=0.3, bounds=None)
    assert controller.rtol_min >= FLOAT32_RTOL_FLOOR


def test_adaptive_controller_floor_untouched_for_float64(comm):
    p = _problem(comm, np.float64, cg_tol=1e-4)
    controller = _make_inner_tolerance(
        p, cg_tol_start=1e-2, cg_tol_min=1e-10, cg_forcing_c=0.5,
        cg_forcing_exp=1.0, cg_stall_rel=0.1, cg_stall_shrink=0.3, bounds=None)
    assert controller.rtol_min == 1e-10
