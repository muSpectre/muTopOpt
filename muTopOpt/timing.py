#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Shared hierarchical timer for all muTopOpt components.

Every module records into this single process-global :class:`muTimer.Timer`
instance through its context manager, e.g. ``with timer("forward solve"): ...``.
Nesting follows the call stack, so a driver that wraps the optimization in
``with timer("optimization")`` gets the solves, sensitivity kernels and file
output grouped under it automatically. At the end of a run, print the full
breakdown with ``timer.print_summary(comm=...)`` -- passing a communicator
(muGrid or mpi4py style) gathers the per-rank timings and prints a single
min/mean/max summary on rank 0.
"""

from muTimer import Timer

#: Process-global hierarchical timer shared by all muTopOpt components.
timer = Timer()
