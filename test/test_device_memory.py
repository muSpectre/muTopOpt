#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The choice between managed and plain device memory.

The rules and the measurements behind them are in the notes above
``muTopOpt.homogenization._device_has_unified_memory``. The functions are pure
-- device kind, rank count and environment go in as arguments -- so every case
is tested here without a GPU.
"""

import pytest

from muTopOpt.homogenization import (
    _device_memory_policy,
    _local_rank,
    _requested_managed_memory,
)


@pytest.mark.parametrize(
    "on_device, unified, nb_ranks, managed",
    [
        (False, False, 1, False),  # CPU: no device memory at all
        (False, False, 4, False),
        (True, False, 1, False),   # discrete GPU: managed pages migrate
        (True, False, 4, False),
        (True, True, 1, True),     # APU, one rank: full memory, no cost
        (True, True, 2, False),    # APU with MPI: MPI is slow on managed
        (True, True, 4, False),
    ],
)
def test_automatic_choice(on_device, unified, nb_ranks, managed):
    got, reason = _device_memory_policy(on_device, unified, nb_ranks)
    assert got is managed
    assert reason  # every decision says why


@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.parametrize("unified, nb_ranks",
                         [(False, 1), (True, 1), (True, 4)])
def test_explicit_request_wins_on_any_gpu(requested, unified, nb_ranks):
    got, reason = _device_memory_policy(True, unified, nb_ranks, requested)
    assert got is requested
    assert "requested" in reason


def test_cpu_ignores_a_request():
    assert _device_memory_policy(False, False, 1, True)[0] is False


@pytest.mark.parametrize(
    "argument, environ, expected",
    [
        (None, {}, None),
        (None, {"MUTOPOPT_MANAGED": "0"}, False),
        (None, {"MUTOPOPT_MANAGED": "1"}, True),
        (True, {"MUTOPOPT_MANAGED": "0"}, True),   # the argument wins
        (False, {"MUTOPOPT_MANAGED": "1"}, False),
    ],
)
def test_request_from_argument_or_environment(argument, environ, expected):
    assert _requested_managed_memory(argument, environ) is expected


def test_local_rank_without_communicator():
    assert _local_rank(None) == 0


def test_local_rank_is_node_rank(comm):
    """GPU placement uses muGrid's node-local rank: one GPU per rank on the
    node, whatever the MPI library and launcher."""
    assert _local_rank(comm) == comm.node_rank
    assert 0 <= comm.node_rank < comm.node_size
    if comm.node_size == comm.size:  # all ranks on one node
        assert sorted(comm.allgather(comm.node_rank)) == list(range(comm.size))
