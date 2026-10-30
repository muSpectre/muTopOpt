import muGrid
import pytest


@pytest.fixture
def comm():
    """muGrid communicator over all ranks: MPI_COMM_WORLD under an MPI
    launcher, serial otherwise. muGrid starts MPI itself, so mpi4py is not
    needed (muGrid.Communicator() alone would always be serial)."""
    return muGrid.Communicator.world()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: compares a GPU run with the CPU; skipped without "
        "cupy and a GPU build of muGrid")
