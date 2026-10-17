#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""Provenance of a run: which muTopOpt and which muGrid produced it.

muGrid reports a git-derived version (``1.1.1-5-g493f675d``) because its build
system stamps one in; muTopOpt's ``__version__`` is a hand-written string that
does not move between commits, so on its own it identifies nothing. The commit
is what distinguishes two runs, and muGrid's is what explains a change in
solver behaviour -- the numerics live there -- so both belong next to every
result.

Everything here degrades to ``"unknown"`` rather than raising: a missing git,
a source tree shipped without ``.git``, or an installed wheel must not stop a
simulation from running.
"""

import functools
import subprocess
from pathlib import Path

import muGrid

from . import __version__

__all__ = ["git_revision", "mugrid_version", "mutopopt_version",
           "provenance_attributes", "version_string"]


@functools.lru_cache(maxsize=None)
def git_revision(path=None):
    """Short git revision of the checkout containing `path`, else ``None``.

    ``--dirty`` is included deliberately: a run from a modified working tree is
    not reproducible from the commit alone, and that is exactly what one wants
    to know when two runs of "the same version" disagree.
    """
    path = Path(__file__).resolve().parent if path is None else Path(path)
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # No git binary, or it could not be executed.
        return None
    rev = out.stdout.strip()
    return rev if out.returncode == 0 and rev else None


def mutopopt_version():
    """``muTopOpt <version>`` plus the git revision when there is one."""
    rev = git_revision()
    return f"muTopOpt {__version__}" + (f" (git {rev})" if rev else "")


def mugrid_version(communicator=None, device=None):
    """muGrid's own banner: version, build options and the linked libraries.

    `communicator` and `device` let muGrid report the decomposition and the GPU
    it will actually run on, so pass the ones the run uses.
    """
    kwargs = {}
    if communicator is not None:
        kwargs["communicator"] = communicator
    if device is not None:
        kwargs["device"] = device
    try:
        return muGrid.version_string(**kwargs)
    except TypeError:
        # An older muGrid without the keyword arguments still reports a version.
        return muGrid.version_string()


def version_string(communicator=None, device=None):
    """Both banners, one per line, muGrid first (it carries the build options)."""
    return (mugrid_version(communicator, device) + "\n" + mutopopt_version())


def provenance_attributes():
    """Version strings as ``{name: value}``, for an output file's attributes.

    Keeps a finished ``.nc`` self-describing: ``command_line`` already records
    *what* was asked for, and these record *what ran it*.

    Deliberately takes no communicator or device, unlike the printed banner:
    those make muGrid's string rank-dependent (``rank 2/4``, ``device cuda:2``),
    and a global attribute is written collectively, so every rank must offer
    the same bytes. What the file needs is the build, which is not per-rank.
    """
    return {
        "mugrid_version": mugrid_version(),
        "mutopopt_version": mutopopt_version(),
    }
