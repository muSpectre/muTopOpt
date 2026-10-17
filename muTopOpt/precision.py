"""Precision-dependent accuracy limits.

Single precision does not merely make the solve less accurate, it puts a hard
floor under the inner CG tolerance: float32 eps is 1.19e-7, so the *true*
residual ``b - Kx`` stagnates around 1e-6 relative even while the recursively
updated CG residual keeps shrinking below it. A tolerance under that floor
cannot be met by arithmetic, so the solve runs to ``cg_maxiter`` and is
salvaged by the stagnation guard -- slow, and with a worse iterate than a
tolerance the solve could actually reach.

The floor used to be spelled out in ``simulate.py`` alone, which is how it
came to be applied on three of that driver's five code paths, on none of
``simulate_conduction.py``'s, and on neither library default (``cg_tol=1e-8``
and ``cg_tol_min=1e-10``, both unreachable in float32). Everything that
consumes a tolerance now clamps it here instead.
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
