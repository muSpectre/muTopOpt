import muGrid
import pytest


@pytest.fixture
def comm():
    """muGrid communicator over all ranks: MPI.COMM_WORLD under an MPI
    launcher, serial without mpi4py (or with a single rank).

    muGrid.Communicator() alone is always serial, so under `mpirun pytest`
    every rank would solve the whole problem by itself while the optimizers,
    which reduce over MPI.COMM_WORLD, treat those identical copies as slices
    of one distributed vector.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return muGrid.Communicator()
    return muGrid.Communicator(MPI.COMM_WORLD)
