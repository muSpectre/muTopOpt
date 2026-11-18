"""
Grid continuation: moving a nodal design to another grid by d-linear
interpolation (``muTopOpt.restart.linear_resample``) and writing it in the
format the restart functions read (``muTopOpt.restart.write_density``).

Validated here:

1. refining by two keeps the coarse nodes and puts the mean of the two
   neighbours on every new edge node; coarsening by two samples every
   second node;
2. on any grid, the result is the d-linear interpolant (checked on a
   separable field, whose interpolant is the product of 1D interpolants),
   and it stays within the range of the input;
3. a part of the target grid (an MPI subdomain) equals that part of the
   whole result;
4. ``write_density`` from a distributed homogenization reads back as the
   global design, and ``restart_density(method="linear")`` resamples it.
"""

import numpy as np
import pytest

from muTopOpt import Homogenization, SimpMaterial
from muTopOpt.restart import (
    linear_resample,
    read_final_density,
    restart_density,
    write_density,
)

from mpi_helpers import local


def test_refine_by_two_keeps_nodes_and_halves_edges():
    a = np.random.default_rng(0).uniform(0.0, 1.0, (4, 5, 6))
    b = linear_resample(a, (8, 10, 12))
    np.testing.assert_array_equal(b[::2, ::2, ::2], a)
    np.testing.assert_allclose(b[1::2, ::2, ::2],
                               0.5 * (a + np.roll(a, -1, axis=0)), rtol=1e-15)
    np.testing.assert_allclose(
        b[1::2, 1::2, 1::2],
        np.mean([np.roll(a, (-i, -j, -k), axis=(0, 1, 2))
                 for i in (0, 1) for j in (0, 1) for k in (0, 1)], axis=0),
        rtol=1e-14)


def test_coarsen_by_two_samples():
    a = np.random.default_rng(1).uniform(0.0, 1.0, (8, 6))
    np.testing.assert_array_equal(linear_resample(a, (4, 3)), a[::2, ::2])


def _periodic_linear_1d(values, n_new):
    """1D periodic linear interpolant of ``values`` at ``n_new`` equally
    spaced nodes (an independent implementation through np.interp)."""
    n = len(values)
    x_new = np.arange(n_new) * n / n_new
    return np.interp(x_new, np.arange(n + 1), np.append(values, values[0]))


@pytest.mark.parametrize("shape_new", [(10, 7, 9), (3, 12, 5)])
def test_separable_field_gives_product_of_1d_interpolants(shape_new):
    """For ``a = f(x) g(y) h(z)`` the trilinear interpolant is the product of
    the 1D interpolants; non-integer factors, refining and coarsening."""
    rng = np.random.default_rng(2)
    f, g, h = (rng.uniform(0.0, 1.0, n) for n in (6, 5, 8))
    a = np.einsum("i,j,k->ijk", f, g, h)
    expected = np.einsum("i,j,k->ijk",
                         *[_periodic_linear_1d(v, n)
                           for v, n in zip((f, g, h), shape_new)])
    np.testing.assert_allclose(linear_resample(a, shape_new), expected,
                               rtol=1e-13, atol=1e-15)


def test_result_stays_within_range():
    a = np.random.default_rng(3).uniform(-0.1, 1.2, (5, 7, 6))
    b = linear_resample(a, (13, 11, 17))
    assert b.min() >= a.min() and b.max() <= a.max()


def test_subdomain_is_part_of_the_whole():
    a = np.random.default_rng(4).uniform(0.0, 1.0, (6, 6, 6))
    whole = linear_resample(a, (16, 16, 16))
    for location, size in [((0, 0, 0), (16, 16, 16)), ((5, 0, 0), (6, 16, 16)),
                           ((3, 9, 1), (4, 7, 15))]:
        part = linear_resample(a, (16, 16, 16), location, size)
        region = tuple(slice(o, o + s) for o, s in zip(location, size))
        np.testing.assert_array_equal(part, whole[region])


def test_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        linear_resample(np.zeros((4, 4)), (8, 8, 8))


def _shared_file(tmp_path, comm, name):
    """A file name that is the same on all ranks (pytest's tmp_path is per
    process): rank 0's, broadcast."""
    path = str(tmp_path / name)
    if comm.size > 1:
        from mpi4py import MPI

        path = MPI.COMM_WORLD.bcast(path, root=0)
    return path


def test_write_density_round_trip(tmp_path, comm):
    """A design written from a (distributed) homogenization reads back as
    the global design, with the grid metadata and further attributes; the
    linear restart resamples it."""
    n = 6
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=2.0, void_ratio=1e-3)
    h = Homogenization((n, n, n), material, comm=comm)
    design = np.random.default_rng(5).uniform(0.0, 1.0, (n, n, n))
    filename = _shared_file(tmp_path, comm, "design.nc")
    write_density(filename, h, local(h, design), {"objective": [1.25]})
    comm.barrier()

    rho, metadata = read_final_density(filename)
    np.testing.assert_array_equal(rho, design)
    assert metadata["nb_grid_pts"] == (n, n, n)
    assert metadata["precision"] == "double"

    import muGrid

    fio = muGrid.FileIONetCDF(filename, muGrid.FileIONetCDF.OpenMode.Read,
                              muGrid.Communicator())
    try:
        assert list(fio.read_global_attribute("objective")) == [1.25]
    finally:
        fio.close()

    rho_fine, metadata = restart_density(filename, (12, 12, 12),
                                         method="linear")
    assert metadata["resampled"]
    np.testing.assert_allclose(rho_fine, linear_resample(design, (12,) * 3),
                               rtol=1e-15)
    comm.barrier()  # nobody deletes tmp_path while others still read


def test_unknown_resampling_method(tmp_path):
    with pytest.raises(ValueError, match="unknown resampling method"):
        restart_density(str(tmp_path / "missing.nc"), (4, 4, 4), method="cubic")
