import muGrid
import pytest


@pytest.fixture
def comm():
    """muGrid communicator over all ranks (a serial stub without MPI)."""
    return muGrid.Communicator()
