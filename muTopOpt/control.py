#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""JSON control files for ``simulate.py``.

A run is described by one JSON object with a fixed set of sections and keys.
:data:`ELASTICITY` defines every key once -- type, default, admissible values
and help text -- and everything else is derived from it: the defaults file
(:func:`defaults`, :func:`dumps`), the JSON Schema (:func:`json_schema`),
validation (:func:`load`, :func:`validate`) and the key reference printed by
``--help`` (:func:`describe`).

A control file may set any subset of the keys; the others take their
defaults. Unknown sections or keys are an error, so a misspelt key cannot
silently fall back to its default. ``null`` means *determined automatically*
by the heuristic described in the key's help text; it is only admissible for
keys that have such a heuristic.
"""

import json
import re
from dataclasses import dataclass, field

class ControlError(ValueError):
    """A control file that does not satisfy the schema."""


@dataclass(frozen=True)
class Key:
    """One key of a control file.

    ``kind`` is one of ``"int"``, ``"number"``, ``"string"``, ``"bool"``,
    ``"int_list"`` and ``"number_list"``. ``nullable`` marks a key whose value
    ``null`` means "determined automatically". ``choices`` restricts a string
    key, ``minimum``/``exclusive_minimum`` a numeric one, and
    ``min_items``/``max_items`` a list.
    """

    kind: str
    default: object
    help: str
    nullable: bool = False
    choices: tuple = ()
    minimum: float = None
    exclusive_minimum: float = None
    min_items: int = None
    max_items: int = None


@dataclass(frozen=True)
class Section:
    """A named group of keys (and nested sections).

    ``alternatives`` lists groups of keys that describe the same quantity in
    different parameterizations, the first group being the default. A file
    that sets keys of exactly one group gets the keys of the other groups set
    to ``null``; setting keys of two groups is an error, and the chosen group
    must be complete.
    """

    help: str
    keys: dict = field(default_factory=dict)
    alternatives: tuple = ()


_POS = dict(exclusive_minimum=0)

#: The control-file schema of the elasticity driver ``simulate.py``.
ELASTICITY = Section(
    "Topology optimization of a periodic unit cell towards a target "
    "isotropic stiffness.",
    {
        "grid": Section("Computational grid.", {
            "nb_grid_pts": Key(
                "int_list", [64, 64, 64],
                "Grid points per axis (2 or 3 values).",
                min_items=2, max_items=3),
            "domain_lengths": Key(
                "number_list", None,
                "Physical edge length of the unit cell per axis, one value "
                "per axis. null: unit length on every axis.",
                nullable=True, min_items=2, max_items=3),
        }),
        "discretization": Section("Finite-element and design discretization.", {
            "element": Key(
                "string", "q1",
                "Finite element: 'q1' (bilinear/trilinear, one element per "
                "pixel) or 'p1' (linear simplices).",
                choices=("q1", "p1")),
            "density": Key(
                "string", "nodal",
                "Design variable: 'nodal' (nodal finite-element field; SIMP "
                "acts on the element average of its interpolant and the "
                "phase-field regularization is the consistent Galerkin one) "
                "or 'element' (one density per pixel, finite-difference "
                "Laplacian penalty).",
                choices=("nodal", "element")),
        }),
        "material": Section("Solid phase and SIMP interpolation.", {
            "young_modulus": Key("number", 1.0,
                                 "Young's modulus of the solid phase.", **_POS),
            "poisson_ratio": Key("number", 0.3,
                                 "Poisson's ratio of the solid phase."),
            "simp_exponent": Key(
                "number", 2.0,
                "SIMP exponent p in C(rho) = C_void + rho^p (C_solid - "
                "C_void).", **_POS),
            "void_ratio": Key(
                "number", 1e-3,
                "Void/solid stiffness ratio. Must be positive: with 0 the "
                "stiffness of void pixels vanishes and the system is only "
                "positive semidefinite.", **_POS),
        }),
        "target": Section(
            "Target isotropic stiffness, either as bulk_modulus and "
            "shear_modulus or as young_modulus and poisson_ratio; setting one "
            "pair nulls the other.", {
                "bulk_modulus": Key(
                    "number", 0.1,
                    "Target bulk modulus K = lambda + 2 mu / d. null: derived "
                    "from young_modulus and poisson_ratio.",
                    nullable=True, **_POS),
                "shear_modulus": Key(
                    "number", 0.05,
                    "Target shear modulus G. null: derived from young_modulus "
                    "and poisson_ratio.", nullable=True, **_POS),
                "young_modulus": Key(
                    "number", None,
                    "Target Young's modulus. null: derived from bulk_modulus "
                    "and shear_modulus.", nullable=True, **_POS),
                "poisson_ratio": Key(
                    "number", None,
                    "Target Poisson's ratio. null: derived from bulk_modulus "
                    "and shear_modulus.", nullable=True),
            },
            alternatives=(("bulk_modulus", "shear_modulus"),
                          ("young_modulus", "poisson_ratio"))),
        "regularization": Section("Modica-Mortola phase-field regularization.", {
            "interface_width": Key(
                "number", None,
                "Interface width eta, in physical length units. null: one grid "
                "spacing (the largest if they differ).",
                nullable=True, **_POS),
            "weight": Key(
                "number", 1.0,
                "Penalty per unit relative interfacial area: the functional is "
                "normalized by 1/(c_W V^((d-1)/d)) with c_W = 1/3 and V the "
                "cell volume.", minimum=0),
        }),
        "initial_density": Section("Starting design.", {
            "source": Key(
                "string", "filtered_random",
                "'uniform' (constant), 'random' (noise lightly smoothed by two "
                "box-filter sweeps), 'filtered_random' (noise smoothed to a "
                "correlation length; least prone to locking the initial "
                "topology), or the path of a NetCDF output of a previous run, "
                "restarted from its last frame and Fourier-resampled if its "
                "grid differs."),
            "volume_fraction": Key(
                "number", 0.5,
                "Volume fraction of a generated density field.",
                minimum=0),
            "correlation_length": Key(
                "number", None,
                "Correlation length of 'filtered_random', in physical length "
                "units. null: three interface widths.",
                nullable=True, **_POS),
            "seed": Key("int", 0,
                        "Random seed of 'random' and 'filtered_random'.",
                        minimum=0),
        }),
        "optimizer": Section("Outer optimizer.", {
            "method": Key(
                "string", None,
                "'trust_region' (bound-constrained trust-region Newton-CG with "
                "exact second-order-adjoint Hessian-vector products; robust "
                "against loose inner tolerances, at about two extra solves "
                "per load case per Hessian product) or 'lbfgs' "
                "(bound-constrained L-BFGS). null: 'trust_region' when the "
                "installed NuMPI provides it, else 'lbfgs'.",
                nullable=True, choices=("trust_region", "lbfgs")),
            "max_iterations": Key("int", 200,
                                  "Maximum number of outer iterations.",
                                  minimum=1),
            "gradient_tolerance": Key(
                "number", None,
                "Convergence tolerance on the projected gradient, measured on "
                "the mesh-invariant volume-fraction derivative "
                "(V/V_pixel) df/drho, so that one value means the same "
                "stationarity at every resolution (initial designs start "
                "around 100). null: 2.5 in double precision, 25 in single "
                "precision.", nullable=True, **_POS),
            "step_tolerance": Key(
                "number", 0.0,
                "L-BFGS only: stop when the maximum absolute change of the "
                "density between iterates falls below this; 0 disables the "
                "criterion.", minimum=0),
            "trust_region": Section("Trust-region Newton-CG.", {
                "initial_radius": Key(
                    "number", 0.05,
                    "Initial radius, in RMS density change per pixel.", **_POS),
                "max_radius": Key(
                    "number", 0.5,
                    "Maximum radius, in RMS density change per pixel.", **_POS),
                "acceptance_threshold": Key(
                    "number", 0.1,
                    "A step is accepted when actual/predicted reduction "
                    "exceeds this.", minimum=0),
                "hessian_tolerance": Key(
                    "number", 1e-3,
                    "Relative CG tolerance of the two extra solves per "
                    "Hessian-vector product; may be loose, the Hessian only "
                    "shapes the model.", **_POS),
                "hessian_warm_start": Key(
                    "bool", False,
                    "Warm-start each Hessian-vector solve from the same load "
                    "case's previous product. Costs two extra vector fields "
                    "per load case; measured to save 2-4% of inner "
                    "iterations and no wall time."),
            }),
        }),
        "inner_solver": Section("Conjugate-gradient solves of the cell problem.", {
            "tolerance": Key(
                "number", 1e-4,
                "Relative tolerance when the adaptive tolerance is disabled, "
                "and the cap of its floor for L-BFGS. Loose values are safe: "
                "the reported objective is adjoint-corrected and thus "
                "second-order accurate in the solve error.", **_POS),
            "max_iterations": Key("int", 2000,
                                  "Maximum CG iterations per solve.",
                                  minimum=1),
            "preconditioner": Key(
                "string", "green-jacobi",
                "'green-jacobi' (J-FFT: reference stiffness times a per-node "
                "Jacobi scaling), 'green' (reference-stiffness Green "
                "operator), or 'hybrid-jacobi' / 'hybrid', the same two with "
                "the reference stiffness inverted by FFT in the rank-local "
                "axes and a tridiagonal solve in the distributed one (no "
                "all-to-all).",
                choices=("green-jacobi", "green", "hybrid-jacobi", "hybrid")),
            "print_iterations": Key(
                "bool", False,
                "Print one line per CG iteration of every forward and adjoint "
                "solve."),
            "adaptive": Section(
                "Adaptive relative tolerance: solves start coarse and tighten "
                "as the outer iteration converges.", {
                    "enabled": Key("bool", True,
                                   "Adapt the tolerance; false uses "
                                   "inner_solver.tolerance throughout."),
                    "initial_tolerance": Key(
                        "number", 1e-2, "Relative tolerance of the first "
                        "solves.", **_POS),
                    "minimum_tolerance": Key(
                        "number", None,
                        "Floor of the adaptive tolerance. null: for L-BFGS "
                        "gradient_tolerance / 1e4 capped at "
                        "inner_solver.tolerance, for the trust region 1e-10; "
                        "raised to 1e-6 in single precision, where the true "
                        "residual of a float32 solve stagnates.",
                        nullable=True, **_POS),
                    "forcing_exponent": Key(
                        "number", 1.0,
                        "Exponent alpha of the L-BFGS forcing term "
                        "rtol = initial_tolerance (|g| / |g_0|)^alpha.",
                        minimum=0),
                    "stall_shrink": Key(
                        "number", 0.3,
                        "Factor applied to the tolerance when the projected "
                        "gradient stops decreasing.", **_POS),
                    "stall_relative_decrease": Key(
                        "number", 1e-2,
                        "Minimum relative decrease of the projected gradient "
                        "counted as progress.", minimum=0),
                }),
        }),
        "compute": Section("Where and in which precision to compute.", {
            "device": Key(
                "string", "cpu",
                "'cpu', or 'gpu' (also 'cuda'/'rocm') for the accelerator of "
                "each rank; append ':N' to pin device N. The outer optimizer "
                "always runs on the host."),
            "precision": Key(
                "string", "double",
                "Scalar precision of the fields and solves; the outer "
                "optimizer stays in double.",
                choices=("double", "single")),
        }),
        "output": Section("NetCDF output.", {
            "file": Key(
                "string", "design.nc",
                "NetCDF file for the density, the homogenized stresses and the "
                "effective moduli. It can be used as initial_density.source of "
                "a later run."),
            "dump_every": Key(
                "int", 0,
                "Also write the initial design and every N-th iterate as "
                "frames; 0 writes only the final design.", minimum=0),
            "flush_frames": Key(
                "bool", True,
                "Flush every frame to disk as it is written, so the file can "
                "be inspected during the run."),
        }),
    },
)


# --------------------------------------------------------------------------- #
# Defaults and serialization

def defaults(schema=ELASTICITY):
    """The complete control object with every key at its default."""
    out = {}
    for name, item in schema.keys.items():
        out[name] = (defaults(item) if isinstance(item, Section)
                     else _copy(item.default))
    return out


def _copy(value):
    return list(value) if isinstance(value, list) else value


def dumps(control):
    """Human-readable JSON: two-space indentation, one key per line, and
    lists of scalars kept on one line."""
    text = json.dumps(control, indent=2)

    def inline(match):
        items = [item.strip() for item in match.group(2).split(",")]
        return f"{match.group(1)}[{', '.join(items)}]"

    return re.sub(r'(:\s|^\s*)\[\s*((?:[^\[\]{}"]|"[^"]*")*?)\s*\]',
                  inline, text, flags=re.M) + "\n"


def json_schema(schema=ELASTICITY, title="muTopOpt simulate.py control file"):
    """JSON Schema (draft 2020-12) of a control file."""
    out = _section_schema(schema)
    out["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    out["title"] = title
    out["properties"]["$schema"] = {
        "type": "string",
        "description": "Path or URL of this schema (ignored by simulate.py).",
    }
    return out


def _section_schema(section):
    properties = {}
    for name, item in section.keys.items():
        properties[name] = (_section_schema(item) if isinstance(item, Section)
                            else _key_schema(item))
    return {"type": "object", "description": section.help,
            "properties": properties, "additionalProperties": False}


_JSON_TYPES = {"int": "integer", "number": "number", "string": "string",
               "bool": "boolean"}


def _key_schema(key):
    if key.kind.endswith("_list"):
        item = {"type": _JSON_TYPES[key.kind[:-5]]}
        _bounds(item, key)
        out = {"type": "array", "items": item}
        if key.min_items is not None:
            out["minItems"] = key.min_items
        if key.max_items is not None:
            out["maxItems"] = key.max_items
    else:
        out = {"type": _JSON_TYPES[key.kind]}
        _bounds(out, key)
        if key.choices:
            out["enum"] = list(key.choices)
    if key.nullable:
        out = {"anyOf": [out, {"type": "null"}]}
        if key.choices:
            out["anyOf"][0]["enum"] = list(key.choices)
    out["description"] = key.help
    out["default"] = _copy(key.default)
    return out


def _bounds(target, key):
    if key.minimum is not None:
        target["minimum"] = key.minimum
    if key.exclusive_minimum is not None:
        target["exclusiveMinimum"] = key.exclusive_minimum


# --------------------------------------------------------------------------- #
# Loading and validation

def load(path, schema=ELASTICITY):
    """Read, validate and complete a control file: keys it does not set take
    their defaults. Raises :class:`ControlError` on any violation."""
    try:
        with open(path) as f:
            given = json.load(f)
    except json.JSONDecodeError as err:
        raise ControlError(f"{path}: not valid JSON: {err}") from err
    return validate(given, schema)


def validate(given, schema=ELASTICITY, where=""):
    """Validate a (possibly partial) control object and return it completed
    with defaults."""
    if not isinstance(given, dict):
        raise ControlError(f"{where or 'control file'}: expected an object")
    unknown = sorted(set(given) - set(schema.keys) - ({"$schema"} if not where
                                                       else set()))
    if unknown:
        raise ControlError(
            f"{where or 'control file'}: unknown key(s) "
            f"{', '.join(repr(u) for u in unknown)}; allowed: "
            f"{', '.join(schema.keys)}")
    out = {}
    for name, item in schema.keys.items():
        path = f"{where}.{name}" if where else name
        if isinstance(item, Section):
            out[name] = validate(given.get(name, {}), item, path)
        else:
            out[name] = (_check(path, item, given[name]) if name in given
                         else _copy(item.default))
    if schema.alternatives:
        _resolve_alternatives(schema, given, out, where)
    return out


def _resolve_alternatives(schema, given, out, where):
    """Apply the either-or rule of ``schema.alternatives`` (see
    :class:`Section`) to the completed section ``out``."""
    chosen = [group for group in schema.alternatives
              if any(given.get(k) is not None for k in group)]
    if len(chosen) > 1:
        raise ControlError(
            f"{where}: set only one of "
            + " or ".join("(" + ", ".join(g) + ")" for g in chosen))
    group = chosen[0] if chosen else schema.alternatives[0]
    for other in schema.alternatives:
        if other is not group:
            for k in other:
                out[k] = None
    missing = [k for k in group if out[k] is None]
    if missing:
        raise ControlError(f"{where}: {', '.join(group)} must all be set "
                           f"(missing {', '.join(missing)})")


def _check(path, key, value):
    if value is None:
        if key.nullable:
            return None
        raise ControlError(f"{path}: null is not allowed (this key has no "
                           "automatic value)")
    if key.kind.endswith("_list"):
        if not isinstance(value, list):
            raise ControlError(f"{path}: expected a list, got {value!r}")
        n = len(value)
        if key.min_items is not None and n < key.min_items or \
                key.max_items is not None and n > key.max_items:
            raise ControlError(
                f"{path}: expected {key.min_items} to {key.max_items} values, "
                f"got {n}")
        return [_scalar(f"{path}[{i}]", key, key.kind[:-5], v)
                for i, v in enumerate(value)]
    value = _scalar(path, key, key.kind, value)
    if key.choices and value not in key.choices:
        raise ControlError(
            f"{path}: {value!r} is not one of "
            f"{', '.join(repr(c) for c in key.choices)}")
    return value


def _scalar(path, key, kind, value):
    if kind == "bool":
        if not isinstance(value, bool):
            raise ControlError(f"{path}: expected true or false, got {value!r}")
        return value
    if kind == "string":
        if not isinstance(value, str):
            raise ControlError(f"{path}: expected a string, got {value!r}")
        return value
    # JSON has one number type; bool is a subclass of int in Python
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError(f"{path}: expected a number, got {value!r}")
    if kind == "int":
        if float(value) != int(value):
            raise ControlError(f"{path}: expected an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if key.minimum is not None and value < key.minimum:
        raise ControlError(f"{path}: must be >= {key.minimum}, got {value}")
    if key.exclusive_minimum is not None and value <= key.exclusive_minimum:
        raise ControlError(
            f"{path}: must be > {key.exclusive_minimum}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# Reference text

def describe(schema=ELASTICITY, prefix=""):
    """Plain-text reference of every key with its default, for ``--help``."""
    lines = []
    for name, item in schema.keys.items():
        path = f"{prefix}{name}"
        if isinstance(item, Section):
            lines.append(f"\n{path}: {item.help}")
            lines.extend(describe(item, path + ".").splitlines())
        else:
            lines.append(f"  {path} = {json.dumps(item.default)}")
            lines.extend(_wrap(item.help, "      "))
    return "\n".join(lines)


def _wrap(text, indent, width=79):
    out, line = [], indent
    for word in text.split():
        if len(line) + len(word) + 1 > width and line.strip():
            out.append(line)
            line = indent
        line += ("" if line == indent else " ") + word
    if line.strip():
        out.append(line)
    return out
