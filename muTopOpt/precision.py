"""Precision-dependent accuracy limits.

Single precision puts a hard floor under the inner CG tolerance: float32
machine epsilon is 1.19e-7, so the *true* residual ``r = b − K u`` stagnates
around 1e-6 relative to ``‖b‖`` even while the recursively updated CG residual
keeps shrinking below it. A tolerance under that floor cannot be met in that
arithmetic; the solve then runs to ``cg_maxiter`` and is salvaged by the
stagnation guard of :meth:`muTopOpt.homogenization.Homogenization.solve_rhs`
-- slower, and with a worse iterate than a reachable tolerance gives.

The floor is therefore defined once, here, and every consumer of an inner
tolerance (the :class:`~muTopOpt.homogenization.Homogenization` solves, the
adaptive tolerance controller, the command-line drivers) clamps against
:func:`cg_rtol_floor`, so no code path and no library default (``cg_tol``,
``cg_tol_min``) can request an unreachable tolerance.
"""

import numpy as np

# Single source of truth, shared with muGrid's CG, which warns when it is
# handed a tolerance below this.
from muGrid.Solvers import FLOAT32_RTOL_FLOOR

__all__ = ["FLOAT32_RTOL_FLOOR", "cg_rtol_floor"]


def cg_rtol_floor(dtype):
    """Smallest inner-CG relative tolerance ``dtype`` can actually reach.

    Returns ``FLOAT32_RTOL_FLOOR`` (1e-6) for float32 and 0.0 for float64,
    so ``max(rtol, cg_rtol_floor(dtype))`` is a no-op in double precision.
    """
    if np.dtype(dtype) == np.dtype(np.float32):
        return FLOAT32_RTOL_FLOOR
    return 0.0
