#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The choice between managed and plain device memory.

The rule -- plain device memory unless managed is asked for -- and the
measurements behind it are in the device-memory notes in
``muTopOpt/homogenization.py``. The functions are pure -- device and
environment go in as arguments -- so every case is tested here without a GPU.
"""

import pytest

from muTopOpt.homogenization import (
    _device_memory_policy,
    _local_rank,
    _requested_managed_memory,
)


@pytest.mark.parametrize("on_device", [False, True])
def test_default_is_plain(on_device):
    """Plain device memory on every GPU, whatever its kind and the number of
    ranks: managed pages migrate on discrete GPUs, and MPI is slow on managed
    buffers on unified-memory APUs."""
    got, reason = _device_memory_policy(on_device)
    assert got is False
    assert reason  # every decision says why
    if on_device:
        assert reason.startswith("plain device memory")


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_request_wins_on_a_gpu(requested):
    got, reason = _device_memory_policy(True, requested)
    assert got is requested
    assert "requested" in reason


def test_cpu_ignores_a_request():
    assert _device_memory_policy(False, True)[0] is False


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
