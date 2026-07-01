import muGrid
import pytest


@pytest.fixture
def comm():
    return muGrid.Communicator()
