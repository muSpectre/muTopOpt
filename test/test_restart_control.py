#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The restart control file: the control of the run that continues a run
from its output file, into a new, numbered output file."""

import json

import pytest

from muTopOpt.control import defaults, dumps, validate
from muTopOpt.restart import next_output_name, restart_control
from muTopOpt.simulate import _write_restart_control


@pytest.mark.parametrize("path, expected", [
    ("design.nc", "design.1.nc"),
    ("design.1.nc", "design.2.nc"),
    ("design.9.nc", "design.10.nc"),
    ("run_009.nc", "run_010.nc"),
    ("run-99.nc", "run-100.nc"),
    ("out/design.nc", "out/design.1.nc"),
    ("design2.nc", "design2.1.nc"),
    ("design", "design.1"),
])
def test_next_output_name(path, expected):
    assert next_output_name(path) == expected


def test_restart_control_continues_from_the_output():
    control = defaults()
    control["output"]["file"] = "run.3.nc"
    restart = restart_control(control)
    assert restart["initial_density"]["source"] == "run.3.nc"
    assert restart["output"]["file"] == "run.4.nc"
    # Nothing else changes, and the original is left alone.
    restart["initial_density"]["source"] = control["initial_density"]["source"]
    restart["output"]["file"] = control["output"]["file"]
    assert restart == control
    assert control["output"]["file"] == "run.3.nc"
    # It is a valid, complete control file.
    restart = restart_control(control)
    assert validate(json.loads(dumps(restart))) == restart


def test_write_restart_control(tmp_path):
    path = tmp_path / "restart.json"
    control = defaults()
    control["output"]["file"] = "design.nc"
    started_from = tmp_path / "run.json"
    started_from.write_text(dumps(control))
    assert _write_restart_control(str(path), control, str(started_from))
    written = json.loads(path.read_text())
    assert written["initial_density"]["source"] == "design.nc"
    assert written["output"]["file"] == "design.1.nc"

    # Running the restart control replaces it with the next link of the chain.
    assert _write_restart_control(str(path), written, str(path))
    chained = json.loads(path.read_text())
    assert chained["initial_density"]["source"] == "design.1.nc"
    assert chained["output"]["file"] == "design.2.nc"


def test_hand_written_control_is_not_replaced(tmp_path):
    path = tmp_path / "restart.json"
    control = defaults()  # starts from a generated density
    path.write_text(dumps(control))
    assert not _write_restart_control(str(path), control, str(path))
    assert json.loads(path.read_text()) == control
