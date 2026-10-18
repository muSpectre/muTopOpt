#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""JSON control files of simulate.py (muTopOpt.control).

The schema is the single definition of every key, so the defaults file, the
JSON Schema and validation must agree with it and with each other: the
defaults file is valid, round-trips to the defaults, is readable (one key per
line), and validation rejects what the schema forbids with a message that
names the key.
"""

import json
import os
import subprocess
import sys

import pytest

from muTopOpt.control import (
    ELASTICITY,
    ControlError,
    Key,
    Section,
    defaults,
    describe,
    dumps,
    json_schema,
    load,
    validate,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_simulate(*args, cwd):
    """Run simulate.py in a subprocess that imports this checkout's package
    and the caller's muGrid, whatever the working directory."""
    path = [ROOT] + [os.path.abspath(p) for p in
                     os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path))
    return subprocess.run([sys.executable, os.path.join(ROOT, "simulate.py"),
                           *args], cwd=cwd, env=env, capture_output=True,
                          text=True)


def _keys(section, prefix=""):
    for name, item in section.keys.items():
        if isinstance(item, Section):
            yield from _keys(item, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}", item


def test_defaults_round_trip_and_validate():
    """validate(defaults()) == defaults(), and dumps/json.loads is the
    identity on it."""
    d = defaults()
    assert validate(d) == d
    assert json.loads(dumps(d)) == d


def test_defaults_file_is_readable():
    """One key per line, lists of numbers on a single line."""
    text = dumps(defaults())
    lines = text.splitlines()
    n_keys = sum(1 for _ in _keys(ELASTICITY))
    assert len(lines) > n_keys
    assert '"nb_grid_pts": [64, 64, 64],' in text
    assert all(len(line) < 100 for line in lines)


def test_null_only_where_a_heuristic_exists():
    """A null default is only admissible for keys whose help says how the
    value is determined."""
    for path, key in _keys(ELASTICITY):
        if key.default is None:
            assert key.nullable, path
            assert "null:" in key.help, path


def test_partial_file_takes_defaults(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"grid": {"nb_grid_pts": [8, 6]},'
                    ' "optimizer": {"trust_region": {"max_radius": 0.25}}}')
    c = load(path)
    assert c["grid"]["nb_grid_pts"] == [8, 6]
    assert c["optimizer"]["trust_region"]["max_radius"] == 0.25
    assert c["optimizer"]["trust_region"]["initial_radius"] == 0.05
    assert c["material"] == defaults()["material"]


@pytest.mark.parametrize("given, match", [
    ({"grid": {"nb_grid_pt": [8, 8]}}, r"grid: unknown key\(s\) 'nb_grid_pt'"),
    ({"solver": {}}, r"unknown key\(s\) 'solver'"),
    ({"grid": {"nb_grid_pts": [8]}}, r"grid.nb_grid_pts: expected 2 to 3"),
    ({"grid": {"nb_grid_pts": [8, 8.5]}}, r"nb_grid_pts\[1\]: expected an "
                                          "integer"),
    ({"material": {"void_ratio": 0}}, r"material.void_ratio: must be > 0"),
    ({"material": {"young_modulus": None}}, r"null is not allowed"),
    ({"compute": {"precision": "half"}}, r"compute.precision: 'half' is not"),
    ({"output": {"flush_frames": 1}}, r"expected true or false"),
    ({"inner_solver": {"adaptive": {"enabled": "yes"}}},
     r"inner_solver.adaptive.enabled: expected true or false"),
])
def test_invalid_files_are_rejected(given, match):
    with pytest.raises(ControlError, match=match):
        validate(given)


def test_target_alternatives():
    """Setting one parameterization nulls the other; mixing them or leaving
    one incomplete is an error."""
    c = validate({"target": {"young_modulus": 0.2, "poisson_ratio": -0.3}})
    assert c["target"] == {"bulk_modulus": None, "shear_modulus": None,
                           "young_modulus": 0.2, "poisson_ratio": -0.3}
    c = validate({"target": {"bulk_modulus": 0.3}})
    assert (c["target"]["bulk_modulus"], c["target"]["shear_modulus"]) == \
        (0.3, 0.05)
    assert c["target"]["young_modulus"] is None
    with pytest.raises(ControlError, match="set only one of"):
        validate({"target": {"bulk_modulus": 0.3, "young_modulus": 0.2}})
    with pytest.raises(ControlError, match="missing poisson_ratio"):
        validate({"target": {"young_modulus": 0.2}})


def test_json_schema_describes_every_key():
    schema = json_schema()
    for path, key in _keys(ELASTICITY):
        node = schema
        for part in path.split("."):
            node = node["properties"][part]
        assert node["default"] == key.default, path
        assert node["description"] == key.help, path
    assert schema["additionalProperties"] is False


def test_json_schema_accepts_the_defaults_file():
    jsonschema = pytest.importorskip("jsonschema")
    control = {"$schema": "simulate.schema.json", **defaults()}
    jsonschema.validate(control, json_schema())
    bad = defaults()
    bad["grid"]["typo"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, json_schema())


def test_describe_lists_every_key():
    text = describe()
    for path, _ in _keys(ELASTICITY):
        assert f"  {path} = " in text, path


def test_key_rejects_bool_as_number():
    with pytest.raises(ControlError, match="expected a number"):
        validate({"material": {"poisson_ratio": True}})


def test_write_defaults_cli(tmp_path):
    """simulate.py --write-defaults writes a valid, complete control file and
    the schema next to it."""
    out = tmp_path / "run.json"
    result = _run_simulate("--write-defaults", str(out), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    written = json.loads(out.read_text())
    assert written.pop("$schema") == "simulate.schema.json"
    assert written == defaults()
    schema = json.loads((tmp_path / "simulate.schema.json").read_text())
    assert schema == json_schema()


def test_simulate_rejects_a_bad_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"grid": {"nb_grid_pt": [8, 8]}}')
    result = _run_simulate(str(path), cwd=tmp_path)
    assert result.returncode != 0
    assert "unknown key(s) 'nb_grid_pt'" in result.stderr


def test_key_dataclass_is_frozen():
    with pytest.raises(Exception):
        Key("int", 0, "x").default = 1
