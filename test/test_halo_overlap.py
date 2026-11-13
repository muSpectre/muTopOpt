#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The overlapped halo exchange in Homogenization._hessp.

With more than one rank, _hessp starts the exchange of the distributed axis,
applies the stiffness to the planes that do not read its ghosts, completes the
exchange and only then does the two boundary planes. The product must be the
one the blocking exchange gives, at any rank count, on the CPU and the GPU.
"""

import numpy as np
import pytest

import muGrid
from muTopOpt import Homogenization, SimpMaterial

from mpi_helpers import assert_allclose_on_all_ranks, local


def _homogenization(comm, dim, n, device):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-2)
    try:
        homog = Homogenization((n,) * dim, material, comm=comm, device=device)
    except Exception as err:  # a CPU-only muGrid asked for a GPU
        pytest.skip(f"cannot place the fields on '{device}' ({err})")
    if device == "gpu" and not homog.on_device:
        pytest.skip("muGrid did not place the fields on a GPU")
    rng = np.random.default_rng(3)
    homog.set_density(local(homog, rng.uniform(0.05, 1.0, (n,) * dim)))
    return homog


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("dim,n", [(2, 24), (3, 12)])
def test_overlapped_hessp_matches_blocking(comm, device, dim, n):
    if device == "gpu":
        pytest.importorskip("cupy")
    if n % comm.size or n // comm.size < 3:
        pytest.skip(f"{n} planes do not split into slabs of >= 3 for "
                    f"{comm.size} ranks")
    homog = _homogenization(comm, dim, n, device)
    overlaps = (getattr(muGrid, "has_mpi", False) and comm.size > 1
                and hasattr(homog.op, "apply_planes"))
    assert (homog._overlap_nz is not None) == overlaps

    rng = np.random.default_rng(7)
    u_global = rng.standard_normal((dim,) + (n,) * dim)
    u = homog.fc.real_field(f"ov-u-{dim}", (dim,), dtype=homog.dtype)
    Au = homog.fc.real_field(f"ov-Au-{dim}", (dim,), dtype=homog.dtype)
    u.p[...] = homog.to_device(local(homog, u_global))

    homog._hessp(u, Au)
    overlapped = np.array(homog.to_host(Au.p), copy=True)

    nz = homog._overlap_nz
    homog._overlap_nz = None  # the blocking path
    try:
        homog._hessp(u, Au)
    finally:
        homog._overlap_nz = nz
    blocking = np.array(homog.to_host(Au.p), copy=True)

    assert_allclose_on_all_ranks(comm, overlapped, blocking, rtol=1e-13,
                                 atol=1e-13 * float(np.abs(blocking).max()))
