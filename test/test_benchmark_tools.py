#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
The helpers of the large-3D benchmark: the Cahn-Hilliard start
(``benchmarks/cahn_hilliard_start.py``) and the snapshots of
``benchmarks/large_3d.py --save-every``.
"""
import os
import sys

import numpy as np
import pytest

import muGrid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "benchmarks"))
import cahn_hilliard_start  # noqa: E402

netCDF4 = pytest.importorskip("netCDF4")

# Serial: the files go to pytest's tmp_path, which differs between MPI ranks
# (a collective open of different files hangs), and the benchmark scripts
# themselves are run serially here.
pytestmark = pytest.mark.skipif(muGrid.Communicator.world().size > 1,
                                reason="serial test")


def test_cahn_hilliard_separates_and_conserves_the_mean():
    # a short run with a thin interface (on 16³ the default η = 2h is too
    # wide for the cell to be unstable): the noise grows into two phases,
    # the mean of c stays that of the noise (the scheme conserves it)
    noise = 0.05 * np.random.default_rng(3).standard_normal((16, 16, 16))
    c = cahn_hilliard_start.cahn_hilliard(16, seed=3, eta_h=0.5,
                                          t_eps2=100.0, log=None)
    assert c.shape == (16, 16, 16)
    assert abs(c.mean() - noise.mean()) < 1e-12
    assert np.abs(c).max() > 0.5  # separated, far beyond the noise


def test_start_file_round_trip(tmp_path):
    import large_3d

    rho = np.random.default_rng(0).uniform(0.0, 1.0, (8, 8, 8))
    path = str(tmp_path / "start.nc")
    cahn_hilliard_start.write_start(path, rho, "test")
    np.testing.assert_array_equal(large_3d.read_start(path), rho)


def test_snapshots_are_the_start_and_the_iterates(tmp_path, monkeypatch):
    import large_3d
    from muTopOpt.restart import read_final_density

    start = str(tmp_path / "start.nc")
    rho = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(16) / 16)
    cahn_hilliard_start.write_start(
        start, np.broadcast_to(rho[:, None, None], (16, 16, 16)), "test")
    prefix = str(tmp_path / "run")
    monkeypatch.setattr(sys, "argv", [
        "large_3d.py", "--nb-grid-pts", "16", "--maxiter", "2",
        "--save-every", "1", "--output", prefix, "--start", start])
    large_3d.main()

    first = read_final_density(f"{prefix}_N16_it0000.nc")[0]
    np.testing.assert_allclose(first, large_3d.read_start(start))
    last = sorted(f for f in os.listdir(tmp_path)
                  if f.startswith("run_N16_it") and f.endswith(".nc"))[-1]
    np.testing.assert_array_equal(
        read_final_density(str(tmp_path / last))[0],
        read_final_density(f"{prefix}_N16.nc")[0])
    pytest.importorskip("matplotlib")
    for name in ("run_N16_it0000", "run_N16"):
        assert (tmp_path / f"{name}.png").exists()
