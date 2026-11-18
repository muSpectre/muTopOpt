#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Warm starts of the forward and adjoint solves between evaluations.

Between two evaluations of an optimisation the design changes little, so the
solution ``u^Γ`` of a load case, and its adjoint ``v^Γ``, are good start
vectors for the next evaluation: with them a solve takes a few CG iterations
instead of a dozen (multigrid) or a hundred (Green-Jacobi).

The price is memory: two vector fields per load case, 12 in 3D, where the
solver itself needs about 10. :class:`WarmStarts` therefore offers four
places to keep them:

==============  ==========================================================
``"device"``    fields next to the solver's, in its precision (fastest)
``"device32"``  the same in single precision: half the memory
``"host"``      host memory (pinned for a GPU run), copied to the device
                one load case at a time
``"host32"``    host memory in single precision
==============  ==========================================================

A start vector in single precision has a relative error of about 1e-7. That
is far below the change of the solution between two designs, so it costs no
iterations. With the ``"host"`` stores a GPU holds the fields of one load
case only, at the cost of two transfers per solve.
"""

import numpy as np

STORES = ("device", "device32", "host", "host32")


class WarmStarts:
    """Keeps one vector field per key between evaluations.

    Parameters
    ----------
    homogenization : Homogenization
        Provides the field collection and tells whether the fields are on a
        device.
    store : str
        One of :data:`STORES`; see the module docstring.
    """

    def __init__(self, homogenization, store="device"):
        if store not in STORES:
            raise ValueError(
                f"unknown warm-start store {store!r}; choose from {STORES}")
        self.h = homogenization
        self.store_kind = store
        self.on_host = store.startswith("host")
        self.dtype = (np.dtype(np.float32) if store.endswith("32")
                      else np.dtype(homogenization.dtype))
        self._kept = {}

    def __contains__(self, key):
        return key in self._kept

    def store(self, key, field):
        """Keep the content of ``field`` under ``key``."""
        kept = self._kept.get(key)
        if self.on_host:
            # only the interior: the solver exchanges the ghosts itself
            values = field.s
            if kept is None:
                kept = self._kept[key] = self._host_buffer(values.shape)
            if self.h.on_device:
                # straight into the kept buffer when no conversion is needed
                if values.dtype == self.dtype:
                    values.get(out=kept)
                else:
                    kept[...] = values.astype(self.dtype).get()
            else:
                kept[...] = values
        else:
            if kept is None:
                name = ("to_warm_start_" + "_".join(str(k) for k in key)
                        + f"_{self.dtype.name}")
                kept = self._kept[key] = self.h.fc.real_field(
                    name, tuple(field.components_shape), dtype=self.dtype)
            kept.s[...] = field.s

    def load(self, key, field):
        """Copy the vector kept under ``key`` into ``field``. Returns False,
        and leaves ``field`` alone, if nothing is kept under that key."""
        kept = self._kept.get(key)
        if kept is None:
            return False
        if self.on_host:
            field.s[...] = self.h.to_device(kept)
        else:
            field.s[...] = kept.s
        return True

    def clear(self):
        """Forget all vectors. Device fields stay registered in the collection
        and are found again by name."""
        self._kept.clear()

    def _host_buffer(self, shape):
        if self.h.on_device:
            # Pinned (page-locked) memory: transfers to and from the device
            # need no intermediate copy.
            import cupyx

            return cupyx.empty_pinned(shape, dtype=self.dtype)
        return np.empty(shape, dtype=self.dtype)
