#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Shared hierarchical timer for all muTopOpt components.

Every module records into this single process-global :class:`Timer`
instance through its context manager, e.g. ``with timer("forward solve"): ...``.
Nesting follows the call stack, so a driver that wraps the optimization in
``with timer("optimization")`` gets the solves, sensitivity kernels and file
output grouped under it automatically. At the end of a run, print the full
breakdown with ``timer.print_summary(comm=...)`` -- passing a communicator
(muGrid or mpi4py style) gathers the per-rank timings and prints a single
min/mean/max summary on rank 0.
"""

from contextlib import contextmanager

import muTimer


class Timer(muTimer.Timer):
    """:class:`muTimer.Timer` that can wait for the GPU around every region.

    Kernel launches return before the kernel has run, so on a GPU a region
    otherwise measures only the cost of queueing its work; the wait for that
    work is charged to whichever later region first reads a result back to
    the host or hands a buffer to MPI. Setting :attr:`synchronize` to a
    callable that blocks until the device is idle (e.g.
    ``cupy.cuda.Device().synchronize``) calls it on entry and on exit of
    every region, so each region is charged with exactly its own kernels.
    That serializes host and device at every region boundary -- the launch
    of the next kernels can no longer overlap the running ones -- so the
    synchronized total is an upper bound on the unsynchronized run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Callable that blocks until the device is idle, or None.
        self.synchronize = None

    @contextmanager
    def __call__(self, name, max_depth=None):
        synchronize = self.synchronize
        if synchronize is None:
            with super().__call__(name, max_depth):
                yield
            return
        # Work queued before the region belongs to the enclosing one.
        synchronize()
        with super().__call__(name, max_depth):
            try:
                yield
            finally:
                synchronize()


#: Process-global hierarchical timer shared by all muTopOpt components.
timer = Timer()
