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
    _mugrid_settings_for_plain_memory,
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


def test_mugrid_settings_only_for_plain_memory_on_an_apu():
    for unified, managed in [(False, False), (False, True), (True, True)]:
        env = {"MPICH_GPU_SUPPORT_ENABLED": "1"}
        assert _mugrid_settings_for_plain_memory(unified, managed, 4, env) == {}
        assert env == {"MPICH_GPU_SUPPORT_ENABLED": "1"}


def test_mugrid_settings_for_plain_memory_on_an_apu():
    # One rank: only the host-readability correction.
    env = {}
    assert _mugrid_settings_for_plain_memory(True, False, 1, env) == {
        "MUGRID_UNIFIED_MEMORY": "0"}
    # Several ranks under Cray MPICH with GPU support: device pointers to MPI.
    env = {"MPICH_GPU_SUPPORT_ENABLED": "1"}
    assert _mugrid_settings_for_plain_memory(True, False, 4, env) == {
        "MUGRID_UNIFIED_MEMORY": "0", "MUGRID_GPU_AWARE_MPI": "1"}
    assert env["MUGRID_GPU_AWARE_MPI"] == "1"
    # Several ranks, but no sign of a GPU-aware MPI: muGrid stages instead.
    env = {}
    assert "MUGRID_GPU_AWARE_MPI" not in _mugrid_settings_for_plain_memory(
        True, False, 4, env)


def test_mugrid_settings_never_override_the_user():
    env = {"MUGRID_UNIFIED_MEMORY": "1", "MUGRID_GPU_AWARE_MPI": "0",
           "MPICH_GPU_SUPPORT_ENABLED": "1"}
    assert _mugrid_settings_for_plain_memory(True, False, 4, env) == {}
    assert env["MUGRID_UNIFIED_MEMORY"] == "1"
    assert env["MUGRID_GPU_AWARE_MPI"] == "0"
