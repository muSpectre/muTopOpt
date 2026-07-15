"""Unit tests for restarting from a previous run (muTopOpt.restart):
Fourier resampling between grids and reading the density back through
muGrid's FileIONetCDF."""

import shlex

import muGrid
import numpy as np
import pytest

from muTopOpt.restart import (
    fourier_resample,
    read_final_density,
    restart_density,
)


def _harmonic_field(nb_grid_pts):
    """Smooth periodic test field from a few low harmonics: band-limited well
    below the Nyquist of an 8-point grid, so trigonometric interpolation is
    exact on every grid used in the tests."""
    x = np.meshgrid(
        *[np.arange(n) / n for n in nb_grid_pts], indexing="ij")
    f = 0.5 * np.ones(tuple(nb_grid_pts))
    for ax, xi in enumerate(x):
        f += 0.2 * np.cos(2 * np.pi * xi + 0.3 * ax)
        f += 0.1 * np.sin(2 * np.pi * (2 * xi + 0.1 * ax))
    return f


@pytest.mark.parametrize("shape_new", [(12, 12), (16, 10), (9, 15), (6, 8)])
def test_fourier_resample_band_limited_exact(shape_new):
    """Up/downsampling a band-limited field equals evaluating its harmonics
    on the new grid, and the mean is preserved exactly."""
    shape_old = (8, 10)
    a = _harmonic_field(shape_old)
    b = fourier_resample(a, shape_new)
    np.testing.assert_allclose(b, _harmonic_field(shape_new), atol=1e-12)
    np.testing.assert_allclose(b.mean(), a.mean(), rtol=1e-13)


def test_fourier_resample_constant_and_roundtrip_3d():
    rng = np.random.default_rng(0)
    # A constant field stays constant on any grid.
    c = np.full((6, 7, 8), 0.42)
    np.testing.assert_allclose(
        fourier_resample(c, (9, 5, 12)), 0.42, rtol=1e-13)
    # Upsampling then downsampling back is the identity (band-limited).
    a = rng.random((6, 7, 8))
    up = fourier_resample(a, (12, 13, 16))
    np.testing.assert_allclose(fourier_resample(up, a.shape), a, atol=1e-12)


def test_fourier_resample_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        fourier_resample(np.zeros((4, 4)), (4, 4, 4))


def _write_run_output(filename, frames, comm, dtype=np.float64,
                      explicit_attributes=True, command_line=None):
    """Write a NetCDF file the way simulate.py does (through muGrid), with the
    density frames in ``frames``."""
    nb_grid_pts = frames[0].shape
    fc = muGrid.GlobalFieldCollection(list(nb_grid_pts))
    field = fc.real_field("density", dtype=dtype)
    fio = muGrid.FileIONetCDF(
        str(filename), muGrid.FileIONetCDF.OpenMode.Overwrite, comm)
    fio.register_field_collection(fc, field_names=["density"])
    fio.write_global_attribute(
        "domain_lengths", [1.0] * len(nb_grid_pts))
    if command_line is not None:
        fio.write_global_attribute("command_line", command_line)
    if explicit_attributes:
        fio.write_global_attribute(
            "nb_grid_pts", [int(n) for n in nb_grid_pts])
        fio.write_global_attribute(
            "precision",
            "single" if dtype == np.float32 else "double")
    for rho in frames:
        field.p[...] = rho
        fio.append_frame().write(["density"])
    fio.close()


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_read_final_density(tmp_path, comm, dtype):
    """The last frame comes back bit-exact (up to the stored precision), with
    the grid/precision metadata from the explicit global attributes."""
    rng = np.random.default_rng(1)
    frames = [rng.random((6, 5)).astype(dtype) for _ in range(3)]
    fn = tmp_path / "run.nc"
    _write_run_output(fn, frames, comm, dtype=dtype)

    rho, meta = read_final_density(str(fn))
    assert meta["nb_grid_pts"] == (6, 5)
    assert meta["precision"] == ("single" if dtype == np.float32 else "double")
    assert meta["domain_lengths"] == [1.0, 1.0]
    assert meta["frame"] == 2
    assert rho.dtype == np.float64
    np.testing.assert_array_equal(rho, frames[-1].astype(np.float64))


def test_read_final_density_command_line_fallback(tmp_path, comm):
    """Files from before the explicit nb_grid_pts/precision attributes: the
    metadata is recovered from the recorded command line."""
    rng = np.random.default_rng(2)
    frames = [rng.random((4, 6, 5)).astype(np.float32)]
    fn = tmp_path / "old.nc"
    _write_run_output(
        fn, frames, comm, dtype=np.float32, explicit_attributes=False,
        command_line=shlex.join(
            ["simulate.py", "--device", "cuda:0", "--precision", "single",
             "-n", "4", "6", "5", "--reg-weight", "0.1", "--output", "old.nc"]
        ))

    rho, meta = read_final_density(str(fn))
    assert meta["nb_grid_pts"] == (4, 6, 5)
    assert meta["precision"] == "single"
    np.testing.assert_array_equal(rho, frames[-1].astype(np.float64))


def test_read_final_density_2d_command_line_fallback(tmp_path, comm):
    """The -n fallback also parses a 2-value grid followed by more options."""
    frames = [np.linspace(0.0, 1.0, 30).reshape(6, 5)]
    fn = tmp_path / "old2d.nc"
    _write_run_output(
        fn, frames, comm, explicit_attributes=False,
        command_line="simulate.py -n 6 5 --target-E 0.5 --target-nu -0.3")
    rho, meta = read_final_density(str(fn))
    assert meta["nb_grid_pts"] == (6, 5)
    assert meta["precision"] == "double"
    np.testing.assert_array_equal(rho, frames[-1])


def test_read_final_density_no_metadata(tmp_path, comm):
    frames = [np.zeros((4, 4))]
    fn = tmp_path / "bare.nc"
    _write_run_output(fn, frames, comm, explicit_attributes=False)
    with pytest.raises(ValueError, match="cannot determine the grid"):
        read_final_density(str(fn))


@pytest.mark.parametrize("shape_new", [(6, 5), (12, 10), (5, 4)])
def test_restart_density(tmp_path, comm, shape_new):
    """restart_density returns the last frame on the requested grid, clipped
    to the admissible [0, 1] range, with the volume fraction preserved."""
    rho_old = np.clip(_harmonic_field((6, 5)), 0.0, 1.0)
    fn = tmp_path / "run.nc"
    _write_run_output(fn, [np.zeros((6, 5)), rho_old], comm)

    rho, meta = restart_density(str(fn), shape_new)
    assert rho.shape == tuple(shape_new)
    assert meta["resampled"] == (tuple(shape_new) != (6, 5))
    assert rho.min() >= 0.0 and rho.max() <= 1.0
    if not meta["resampled"]:
        np.testing.assert_array_equal(rho, rho_old)
    else:
        # Fourier resampling preserves the mean; clipping can only move it
        # slightly where the interpolant overshoots.
        np.testing.assert_allclose(rho.mean(), rho_old.mean(), atol=1e-2)
