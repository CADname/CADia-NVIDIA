from __future__ import annotations

"""PromptCAD-style structured CAD program for NVIDIA Nemotron.

Design goals:
- ONE declarative model-facing operation registry drives schema + prompt reference + validation.
- Nemotron never sees CADia raw tool names, document activation, file paths, runtime binds,
  or backend topology bookkeeping for ordinary generation.
- Deterministic generators are first-class operations and are preferred when they exactly
  match a requested standard mechanical component.
- A deterministic interpreter owns CADia document lifecycle and translates the small IR
  into the existing CadEngine commands.  The 118-tool backend remains untouched for
  Codex/Copilot/MCP and for the interpreter's underlying implementation.
- Assembly relations are discriminated operations, so impossible combinations (for
  example angular limits on a slider) are unrepresentable in the model-facing schema.

This module contains no prompt-, product-, benchmark-, or dimension-specific routing.
"""

from dataclasses import dataclass
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from standalonecad.planning import direct_plan
from standalonecad.core.topology import edge_records, face_records


NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "minimum": 0.001}
NONNEG = {"type": "number", "minimum": 0.0}
INTEGER = {"type": "integer"}
STRING = {"type": "string", "minLength": 1}
BOOL = {"type": "boolean"}
VEC3 = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
POINT2 = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
PROFILE2 = {"type": "array", "minItems": 3, "items": POINT2}
MODE = {"type": "string", "enum": ["new", "join", "cut", "intersect"]}
ORIGIN_REF = {"type": "string", "enum": ["X Axis", "Y Axis", "Z Axis", "XY Plane", "XZ Plane", "YZ Plane"]}



def _validate_schema(schema: dict[str, Any], value: Any, path: str = "$") -> None:
    """Small JSON-Schema validator for the exact subset emitted by this module.

    Kept local so the web image does not gain a new dependency and, like GPT4FreeCAD,
    the same declarative schema drives both constrained generation and validation.
    """
    if "oneOf" in schema:
        errors = []
        matches = 0
        for branch in schema.get("oneOf") or []:
            try:
                _validate_schema(branch, value, path)
                matches += 1
            except ValueError as exc:
                errors.append(str(exc))
        if matches != 1:
            raise ValueError(errors[0] if errors else f"{path}: expected exactly one schema branch")
        return
    if "anyOf" in schema:
        errors = []
        for branch in schema.get("anyOf") or []:
            try:
                _validate_schema(branch, value, path)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(errors[0] if errors else f"{path}: does not match any allowed schema")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: must equal {schema['const']!r}")
    typ = schema.get("type")
    if typ == "null":
        if value is not None:
            raise ValueError(f"{path}: expected null")
        return
    if typ == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                raise ValueError(f"{path}.{req}: required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                raise ValueError(f"{path}: unknown field(s): {extra}")
        for key, item in value.items():
            if key in props:
                _validate_schema(props[key], item, f"{path}.{key}")
        return
    if typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path}: requires at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path}: allows at most {schema['maxItems']} items")
        item_schema = schema.get("items") or {}
        for i, item in enumerate(value):
            _validate_schema(item_schema, item, f"{path}[{i}]")
        return
    if typ == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValueError(f"{path}: string is too short")
    elif typ == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected integer")
    elif typ == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number")
    elif typ == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean")
    elif typ is not None:
        raise ValueError(f"{path}: unsupported schema type {typ}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: minimum is {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: maximum is {schema['maximum']}")


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [copy.deepcopy(schema), {"type": "null"}]}


def _obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _op_schema(name: str, properties: dict[str, Any], required: tuple[str, ...], description: str) -> dict[str, Any]:
    props = {"op": {"type": "string", "const": name}, "name": STRING, **copy.deepcopy(properties)}
    out = _obj(props, ("op", "name", *required))
    out["description"] = description
    return out


REVOLVE_LOOP = {
    "oneOf": [
        _obj(
            {"kind": {"type": "string", "const": "polyline"}, "points": PROFILE2},
            ("kind", "points"),
        ),
        _obj(
            {"kind": {"type": "string", "const": "circle"}, "center": POINT2, "radius_mm": POSITIVE},
            ("kind", "center", "radius_mm"),
        ),
    ]
}


@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    schema: dict[str, Any]
    compiler: str


# ---------------------------------------------------------------------------
# Single declarative operation registry
# ---------------------------------------------------------------------------
# This is the model-facing source of truth.  program_schema(), operations_reference(),
# validate_program(), and CadiaProgramInterpreter all use these exact names/specs.
_OPERATION_ROWS: list[OperationSpec] = []


def _register(name: str, description: str, properties: dict[str, Any], required: tuple[str, ...], compiler: str) -> None:
    _OPERATION_ROWS.append(OperationSpec(name, description, _op_schema(name, properties, required, description), compiler))


# Small PromptCAD-like generic vocabulary.
_register("box", "Rectangular solid aligned to XYZ.", {
    "length_mm": POSITIVE, "width_mm": POSITIVE, "height_mm": POSITIVE,
    "origin_mm": _nullable(VEC3), "centered": _nullable(BOOL), "mode": _nullable(MODE),
}, ("length_mm", "width_mm", "height_mm"), "primitive")
_register("cylinder", "Cylinder; diameter and height are in mm.", {
    "diameter_mm": POSITIVE, "height_mm": POSITIVE, "origin_mm": _nullable(VEC3),
    "axis": _nullable(VEC3), "mode": _nullable(MODE),
}, ("diameter_mm", "height_mm"), "primitive")
_register("cone", "Cone or frustum.", {
    "diameter1_mm": POSITIVE, "diameter2_mm": NONNEG, "height_mm": POSITIVE,
    "origin_mm": _nullable(VEC3), "axis": _nullable(VEC3), "mode": _nullable(MODE),
}, ("diameter1_mm", "height_mm"), "primitive")
_register("sphere", "Sphere.", {
    "diameter_mm": POSITIVE, "center_mm": _nullable(VEC3), "mode": _nullable(MODE),
}, ("diameter_mm",), "primitive")
_register("torus", "Torus; minor radius must be smaller than major radius.", {
    "major_radius_mm": POSITIVE, "minor_radius_mm": POSITIVE, "center_mm": _nullable(VEC3),
    "axis": _nullable(VEC3), "mode": _nullable(MODE),
}, ("major_radius_mm", "minor_radius_mm"), "primitive")
_register("extrude_profile", "Extrude a closed 2D polygon. Interpreter creates/closes the sketch internally.", {
    "profile": PROFILE2,
    "distance_mm": POSITIVE,
    "plane": {"type": "string", "enum": ["XY Plane", "XZ Plane", "YZ Plane"]},
    "direction": {"type": "string", "enum": ["positive", "negative", "symmetric"]},
    "taper_deg": {"type": "number", "minimum": -89, "maximum": 89},
    "mode": _nullable(MODE),
}, ("profile", "distance_mm"), "extrude_profile")
_register("revolve_profile", "Revolve one or more closed 2D loops around the profile Y axis. Supports exact circle loops and polyline loops; nested loops create holes. Use angle_deg < 360 for partial revolutions.", {
    "loops": {"type": "array", "minItems": 1, "items": REVOLVE_LOOP},
    "angle_deg": {"type": "number", "minimum": 0.001, "maximum": 360.0},
    "mode": _nullable(MODE),
}, ("loops", "angle_deg"), "revolve_profile")
_register("hole", "Cylindrical material removal. Use center_mm for a true WORLD entry point. For a hole located relative to an earlier flange/boss/feature, prefer support_name + offset_mm so the interpreter resolves the real feature frame instead of guessing world coordinates.", {
    "center_mm": _nullable(VEC3),
    "support_name": _nullable(STRING),
    "offset_mm": _nullable(POINT2),
    "diameter_mm": POSITIVE, "depth_mm": _nullable(POSITIVE),
    "axis": {"type": "string", "enum": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]},
    "through": BOOL,
}, ("diameter_mm", "axis", "through"), "hole")
_register("fillet", "Round all edges or selected 1-based edge indices of the current body.", {
    "radius_mm": POSITIVE, "edge_indices": _nullable({"type": "array", "items": {"type": "integer", "minimum": 1}}),
}, ("radius_mm",), "finish")
_register("chamfer", "Chamfer all edges or selected 1-based edge indices of the current body.", {
    "distance_mm": POSITIVE, "edge_indices": _nullable({"type": "array", "items": {"type": "integer", "minimum": 1}}),
}, ("distance_mm",), "finish")
_register("shell", "Hollow the current body. Optional removed_face_indices are 1-based topology face indices.", {
    "thickness_mm": POSITIVE,
    "removed_face_indices": _nullable({"type": "array", "items": {"type": "integer", "minimum": 1}}),
    "kind": {"type": "string", "enum": ["arc", "intersection"]},
}, ("thickness_mm",), "shell")
_register("mirror", "Mirror the current body about a principal plane.", {
    "plane": {"type": "string", "enum": ["XY", "XZ", "YZ"]},
    "base_point_mm": _nullable(VEC3), "keep_original": BOOL,
}, ("plane", "keep_original"), "mirror")
_register("transform", "Translate/rotate/scale the current body.", {
    "translate_mm": _nullable(VEC3), "rotate_axis": _nullable(VEC3),
    "rotate_deg": _nullable(NUMBER), "scale": _nullable({"type": "number", "minimum": 0.000001}),
}, (), "transform")
_register("linear_pattern", "Pattern one or more earlier named features in one/two directions.", {
    "source_names": {"type": "array", "minItems": 1, "items": STRING},
    "direction1": ORIGIN_REF, "count1": {"type": "integer", "minimum": 1}, "spacing_mm1": NUMBER,
    "direction2": _nullable(ORIGIN_REF), "count2": _nullable({"type": "integer", "minimum": 1}),
    "spacing_mm2": _nullable(NUMBER),
}, ("source_names", "direction1", "count1", "spacing_mm1"), "pattern_linear")
_register("polar_pattern", "Pattern one or more earlier named features around an axis.", {
    "source_names": {"type": "array", "minItems": 1, "items": STRING},
    "axis": ORIGIN_REF, "count": {"type": "integer", "minimum": 2}, "angle_deg": NUMBER,
}, ("source_names", "axis", "count"), "pattern_polar")

# CADia deterministic mechanical generators.  Nemotron chooses parameters; existing
# CADia generator code creates the engineering geometry.
_register("gear", "CADia deterministic gear-family generator. Prefer this over constructing gear teeth manually.", {
    "kind": {"type": "string", "enum": ["spur", "helical", "herringbone", "ring", "bevel", "rack", "worm", "worm_pair", "planetary"]},
    "module": POSITIVE, "teeth": _nullable({"type": "integer", "minimum": 6}),
    "width_mm": _nullable(POSITIVE), "bore_diameter_mm": NONNEG,
    "pressure_angle_deg": {"type": "number", "minimum": 5, "maximum": 35},
    "helix_angle_deg": {"type": "number", "minimum": -55, "maximum": 55},
    "backlash_mm": NONNEG, "clearance_mm": NONNEG,
    "rim_width_mm": _nullable(POSITIVE), "cone_angle_deg": _nullable({"type": "number", "minimum": 0.01, "maximum": 89.99}),
    "length_mm": _nullable(POSITIVE), "height_mm": _nullable(POSITIVE),
    "lead_angle_deg": _nullable({"type": "number", "minimum": -79.99, "maximum": 79.99}),
    "thread_starts": _nullable({"type": "integer", "minimum": 1}),
    "sun_teeth": _nullable({"type": "integer", "minimum": 6}), "planet_teeth": _nullable({"type": "integer", "minimum": 6}),
    "planet_count": _nullable({"type": "integer", "minimum": 2}), "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("kind", "module"), "generator")
_register("shaft", "CADia stepped-shaft generator with optional bore/keyway.", {
    "sections": {"type": "array", "minItems": 1, "items": _obj({"length_mm": POSITIVE, "diameter_mm": POSITIVE}, ("length_mm", "diameter_mm"))},
    "bore_diameter_mm": NONNEG, "keyway_width_mm": _nullable(POSITIVE), "keyway_depth_mm": _nullable(POSITIVE),
    "keyway_length_mm": _nullable(POSITIVE), "keyway_start_mm": NONNEG,
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("sections",), "generator")
_register("parallel_key", "CADia parallel-key generator.", {
    "width_mm": POSITIVE, "height_mm": POSITIVE, "length_mm": POSITIVE,
    "end_style": {"type": "string", "enum": ["square", "round"]},
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("width_mm", "height_mm", "length_mm"), "generator")
_register("bearing", "CADia bearing generator.", {
    "kind": {"type": "string", "enum": ["deep_groove_ball", "cylindrical_roller", "envelope"]},
    "bore_diameter_mm": POSITIVE, "outer_diameter_mm": POSITIVE, "width_mm": POSITIVE,
    "rolling_elements": {"type": "integer", "minimum": 3}, "detailed": BOOL,
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("kind", "bore_diameter_mm", "outer_diameter_mm", "width_mm"), "generator")
_register("spring", "CADia spring generator.", {
    "kind": {"type": "string", "enum": ["compression", "belleville"]},
    "wire_diameter_mm": _nullable(POSITIVE), "mean_diameter_mm": _nullable(POSITIVE),
    "free_length_mm": _nullable(POSITIVE), "active_turns": _nullable(POSITIVE), "right_handed": BOOL,
    "outer_diameter_mm": _nullable(POSITIVE), "inner_diameter_mm": _nullable(POSITIVE),
    "thickness_mm": _nullable(POSITIVE), "free_height_mm": _nullable(POSITIVE),
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("kind",), "generator")
_register("v_pulley", "CADia V-belt pulley generator.", {
    "pitch_diameter_mm": POSITIVE, "width_mm": POSITIVE, "groove_count": {"type": "integer", "minimum": 1},
    "groove_angle_deg": {"type": "number", "minimum": 10, "maximum": 80}, "bore_diameter_mm": NONNEG,
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("pitch_diameter_mm", "width_mm"), "generator")
_register("coupling", "CADia flange-coupling generator.", {
    "bore_diameter_mm": POSITIVE, "hub_diameter_mm": POSITIVE, "flange_diameter_mm": POSITIVE,
    "hub_length_mm": POSITIVE, "flange_thickness_mm": POSITIVE,
    "bolt_circle_diameter_mm": _nullable(POSITIVE), "bolt_count": {"type": "integer", "minimum": 0},
    "bolt_hole_diameter_mm": _nullable(POSITIVE), "keyway_width_mm": _nullable(POSITIVE), "keyway_depth_mm": _nullable(POSITIVE),
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("bore_diameter_mm", "hub_diameter_mm", "flange_diameter_mm", "hub_length_mm", "flange_thickness_mm"), "generator")
_register("lead_screw_nut", "CADia lead-screw and nut generator.", {
    "major_diameter_mm": POSITIVE, "pitch_mm": POSITIVE, "screw_length_mm": POSITIVE, "nut_length_mm": POSITIVE,
    "starts": {"type": "integer", "minimum": 1}, "thread_angle_deg": {"type": "number", "minimum": 20, "maximum": 60},
    "nut_outer_diameter_mm": _nullable(POSITIVE), "clearance_mm": NONNEG, "right_handed": BOOL,
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("major_diameter_mm", "pitch_mm", "screw_length_mm", "nut_length_mm"), "generator")
_register("hex_bolt", "CADia metric hex-bolt generator.", {
    "diameter_mm": {"type": "number", "minimum": 1}, "length_mm": POSITIVE,
    "pitch_mm": _nullable(POSITIVE), "head_across_flats_mm": _nullable(POSITIVE), "head_height_mm": _nullable(POSITIVE),
    "thread_length_mm": _nullable(NONNEG), "right_handed": BOOL, "modeled_thread": BOOL,
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("diameter_mm", "length_mm"), "generator")
_register("hex_nut", "CADia metric hex-nut generator.", {
    "diameter_mm": {"type": "number", "minimum": 1}, "pitch_mm": _nullable(POSITIVE),
    "across_flats_mm": _nullable(POSITIVE), "height_mm": _nullable(POSITIVE),
    "right_handed": BOOL, "modeled_thread": BOOL, "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("diameter_mm",), "generator")
_register("washer", "CADia metric washer generator.", {
    "diameter_mm": {"type": "number", "minimum": 1}, "inner_diameter_mm": _nullable(POSITIVE),
    "outer_diameter_mm": _nullable(POSITIVE), "thickness_mm": _nullable(POSITIVE),
    "mode": _nullable({"type": "string", "enum": ["new", "join"]}),
}, ("diameter_mm",), "generator")
_register("external_thread", "CADia modeled external thread generator.", {
    "major_diameter_mm": POSITIVE, "pitch_mm": POSITIVE, "length_mm": POSITIVE, "z0_mm": NUMBER,
    "right_handed": BOOL, "mode": _nullable(MODE),
}, ("major_diameter_mm", "pitch_mm", "length_mm"), "generator")
_register("coil", "CADia general helical coil primitive.", {
    "pitch_mm": POSITIVE, "height_mm": POSITIVE, "helix_radius_mm": POSITIVE, "section_diameter_mm": POSITIVE,
    "z0_mm": NUMBER, "right_handed": BOOL, "mode": _nullable(MODE),
}, ("pitch_mm", "height_mm", "helix_radius_mm", "section_diameter_mm"), "generator")
_register("sheet_metal_base", "Create a CADia sheet-metal base.", {
    "width_mm": POSITIVE, "height_mm": POSITIVE, "thickness_mm": POSITIVE,
    "bend_radius_mm": NONNEG, "k_factor": {"type": "number", "minimum": 0, "maximum": 1},
}, ("width_mm", "height_mm", "thickness_mm"), "generator")
_register("sheet_metal_flange", "Add a flange to an existing CADia sheet-metal base.", {
    "edge": {"type": "string", "enum": ["top", "bottom", "left", "right"]},
    "length_mm": POSITIVE, "angle_deg": {"type": "number", "minimum": -179.999, "maximum": 179.999},
    "bend_radius_mm": _nullable(NONNEG),
}, ("edge", "length_mm"), "generator")

OPERATIONS: dict[str, OperationSpec] = {row.name: row for row in _OPERATION_ROWS}


# ---------------------------------------------------------------------------
# Assembly IR: high-level component relations only.  No paths/occurrence names/binds.
# ---------------------------------------------------------------------------

def _relation_schema(kind: str, props: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    base = {
        "type": {"type": "string", "const": kind},
        "a": STRING,
        "b": STRING,
        "a_ref": _nullable(ORIGIN_REF),
        "b_ref": _nullable(ORIGIN_REF),
        "name": _nullable(STRING),
    }
    return _obj({**base, **props}, ("type", "a", "b", *required))


RELATION_SCHEMAS: dict[str, dict[str, Any]] = {
    "mate": _relation_schema("mate", {"offset_mm": NUMBER}, ()),
    "flush": _relation_schema("flush", {"offset_mm": NUMBER}, ()),
    "insert": _relation_schema("insert", {"offset_mm": NUMBER, "opposed": BOOL}, ()),
    "angle": _relation_schema("angle", {"angle_deg": NUMBER}, ("angle_deg",)),
    "rigid": _relation_schema("rigid", {}, ()),
    "slider": _relation_schema("slider", {
        "position_mm": _nullable(NUMBER), "start_mm": _nullable(NUMBER), "end_mm": _nullable(NUMBER),
    }, ()),
    "rotational": _relation_schema("rotational", {
        "position_deg": _nullable(NUMBER), "start_deg": _nullable(NUMBER), "end_deg": _nullable(NUMBER),
    }, ()),
    "cylindrical": _relation_schema("cylindrical", {
        "linear_position_mm": _nullable(NUMBER), "linear_start_mm": _nullable(NUMBER), "linear_end_mm": _nullable(NUMBER),
        "angular_position_deg": _nullable(NUMBER), "angular_start_deg": _nullable(NUMBER), "angular_end_deg": _nullable(NUMBER),
    }, ()),
    "planar": _relation_schema("planar", {
        "angular_position_deg": _nullable(NUMBER), "angular_start_deg": _nullable(NUMBER), "angular_end_deg": _nullable(NUMBER),
    }, ()),
    "ball": _relation_schema("ball", {}, ()),
}


def operation_schema() -> dict[str, Any]:
    return {"oneOf": [copy.deepcopy(OPERATIONS[name].schema) for name in OPERATIONS]}


def program_schema() -> dict[str, Any]:
    component = _obj({
        "name": STRING,
        "operations": {"type": "array", "minItems": 1, "maxItems": 40, "items": operation_schema()},
        "grounded": BOOL,
        "position_mm": VEC3,
        "rotation_deg_xyz": VEC3,
    }, ("name", "operations", "grounded", "position_mm", "rotation_deg_xyz"))
    assembly = _obj({
        "name": STRING,
        "relations": {"type": "array", "maxItems": 64, "items": {"oneOf": [copy.deepcopy(x) for x in RELATION_SCHEMAS.values()]}},
    }, ("name", "relations"))
    return _obj({
        "components": {"type": "array", "minItems": 1, "maxItems": 16, "items": component},
        "assembly": {"anyOf": [assembly, {"type": "null"}]},
        "note": {"type": "string"},
    }, ("components", "assembly", "note"))


def operations_reference() -> str:
    return "\n".join(f"- {name}: {OPERATIONS[name].description}" for name in OPERATIONS)


# ---------------------------------------------------------------------------
# Validation beyond JSON schema: references + geometry semantics.
# ---------------------------------------------------------------------------

def _same_point(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _signed_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _turn(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _sign(v: float) -> int:
    return 0 if abs(v) <= 1e-12 else (1 if v > 0 else -1)


def _crossing(points: list[tuple[float, float]]) -> tuple[int, int] | None:
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = points[j], points[(j + 1) % n]
            if (_sign(_turn(c, d, a)) * _sign(_turn(c, d, b)) < 0
                    and _sign(_turn(a, b, c)) * _sign(_turn(a, b, d)) < 0):
                return i + 1, j + 1
    return None


def _normalized_profile(profile: Any) -> list[list[float]]:
    if not isinstance(profile, list):
        return profile
    points = [[float(x), float(y)] for x, y in profile]
    if len(points) > 3 and _same_point(tuple(points[0]), tuple(points[-1])):
        points = points[:-1]
    return points


def _profile_problem(profile: Any) -> str | None:
    if not isinstance(profile, list) or len(profile) < 3:
        return "profile needs at least 3 points"
    try:
        pts = [tuple(map(float, raw)) for raw in profile]
    except Exception:
        return "each profile point must be [x,y]"
    if any(len(raw) != 2 for raw in profile):
        return "each profile point must be [x,y]"
    if len(pts) > 3 and _same_point(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return "profile needs at least 3 distinct points"
    for i, point in enumerate(pts):
        nxt = (i + 1) % len(pts)
        if _same_point(point, pts[nxt]):
            return f"profile repeats consecutive point {i + 1}/{nxt + 1}"
    crossed = _crossing(pts)
    if crossed:
        return f"profile self-intersects: edge {crossed[0]} crosses edge {crossed[1]}"
    if abs(_signed_area(pts)) < 1e-9:
        return "profile area is zero/collinear"
    return None


def _nonzero_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    return math.sqrt(sum(float(x) ** 2 for x in value)) > 1e-12


def _require(op: dict[str, Any], cname: str, oname: str, *keys: str) -> None:
    missing = [k for k in keys if op.get(k) is None]
    if missing:
        raise ValueError(f"component {cname!r} operation {oname!r}: requires {', '.join(missing)}")


def _validate_generator_semantics(op: dict[str, Any], cname: str, oname: str) -> None:
    kind = str(op.get("op"))
    if kind == "gear":
        gear_kind = str(op.get("kind"))
        if gear_kind == "spur":
            _require(op, cname, oname, "teeth", "width_mm")
        elif gear_kind in {"helical", "herringbone"}:
            _require(op, cname, oname, "teeth", "width_mm")
            if gear_kind == "helical" and abs(float(op.get("helix_angle_deg") or 0.0)) < 1e-12:
                raise ValueError(f"component {cname!r} operation {oname!r}: helical gear requires non-zero helix_angle_deg")
        elif gear_kind == "ring":
            _require(op, cname, oname, "teeth", "width_mm", "rim_width_mm")
        elif gear_kind == "bevel":
            _require(op, cname, oname, "teeth", "width_mm", "cone_angle_deg")
        elif gear_kind == "rack":
            _require(op, cname, oname, "length_mm", "width_mm", "height_mm")
        elif gear_kind == "worm":
            _require(op, cname, oname, "length_mm", "lead_angle_deg", "thread_starts")
            if abs(float(op.get("lead_angle_deg") or 0.0)) < 1e-12:
                raise ValueError(f"component {cname!r} operation {oname!r}: worm requires non-zero lead_angle_deg")
        elif gear_kind == "worm_pair":
            _require(op, cname, oname, "teeth", "width_mm", "length_mm", "lead_angle_deg", "thread_starts")
        elif gear_kind == "planetary":
            _require(op, cname, oname, "sun_teeth", "planet_teeth", "width_mm", "rim_width_mm", "planet_count")
    elif kind == "bearing":
        if float(op["outer_diameter_mm"]) <= float(op["bore_diameter_mm"]):
            raise ValueError(f"component {cname!r} operation {oname!r}: bearing outer_diameter_mm must exceed bore_diameter_mm")
    elif kind == "coupling":
        bore, hub, flange = map(float, (op["bore_diameter_mm"], op["hub_diameter_mm"], op["flange_diameter_mm"]))
        if not (bore < hub < flange):
            raise ValueError(f"component {cname!r} operation {oname!r}: coupling requires bore < hub < flange diameter")
        if (op.get("keyway_width_mm") is None) != (op.get("keyway_depth_mm") is None):
            raise ValueError(f"component {cname!r} operation {oname!r}: keyway width/depth must be supplied together")
    elif kind == "lead_screw_nut":
        if float(op["nut_length_mm"]) >= float(op["screw_length_mm"]):
            raise ValueError(f"component {cname!r} operation {oname!r}: nut_length_mm must be smaller than screw_length_mm")
    elif kind == "spring":
        sk = op.get("kind")
        if sk == "belleville":
            _require(op, cname, oname, "outer_diameter_mm", "inner_diameter_mm", "thickness_mm", "free_height_mm")
            if float(op["outer_diameter_mm"]) <= float(op["inner_diameter_mm"]):
                raise ValueError(f"component {cname!r} operation {oname!r}: belleville outer diameter must exceed inner diameter")
        else:
            _require(op, cname, oname, "wire_diameter_mm", "mean_diameter_mm", "free_length_mm", "active_turns")


def validate_program(program: dict[str, Any]) -> dict[str, Any]:
    _validate_schema(program_schema(), program, "program")
    program = copy.deepcopy(program)
    names: set[str] = set()
    for ci, component in enumerate(program.get("components") or []):
        cname = str(component["name"])
        if cname in names:
            raise ValueError(f"components[{ci}].name: duplicate component name {cname!r}")
        names.add(cname)
        op_names: set[str] = set()
        have_body = False
        for oi, op in enumerate(component.get("operations") or []):
            oname = str(op["name"])
            if oname in op_names:
                raise ValueError(f"component {cname!r}: duplicate operation name {oname!r}")
            kind = str(op["op"])
            if kind == "extrude_profile":
                op["profile"] = _normalized_profile(op.get("profile"))
                problem = _profile_problem(op.get("profile"))
                if problem:
                    raise ValueError(f"component {cname!r} operation {oname!r}: {problem}")
            if kind == "revolve_profile":
                normalized_loops = []
                for li, loop in enumerate(op.get("loops") or []):
                    loop = copy.deepcopy(loop)
                    if loop.get("kind") == "polyline":
                        loop["points"] = _normalized_profile(loop.get("points"))
                        problem = _profile_problem(loop.get("points"))
                        if problem:
                            raise ValueError(
                                f"component {cname!r} operation {oname!r} revolve loop #{li + 1}: {problem}"
                            )
                        for point in loop["points"]:
                            if float(point[0]) < -1e-9:
                                raise ValueError(
                                    f"component {cname!r} operation {oname!r} revolve loop #{li + 1}: "
                                    "profile radial coordinate x must be >= 0 relative to the revolve axis"
                                )
                    elif loop.get("kind") == "circle":
                        cx = float(loop["center"][0])
                        radius = float(loop["radius_mm"])
                        if cx - radius < -1e-9:
                            raise ValueError(
                                f"component {cname!r} operation {oname!r} revolve loop #{li + 1}: "
                                "circle crosses the revolve axis; center.x must be >= radius_mm"
                            )
                    normalized_loops.append(loop)
                op["loops"] = normalized_loops
            if kind == "torus" and float(op["minor_radius_mm"]) >= float(op["major_radius_mm"]):
                raise ValueError(f"component {cname!r} operation {oname!r}: minor_radius_mm must be < major_radius_mm")
            if kind in {"cylinder", "cone"} and op.get("axis") is not None and not _nonzero_vector(op.get("axis")):
                raise ValueError(f"component {cname!r} operation {oname!r}: axis must be non-zero")
            if kind == "transform" and op.get("rotate_deg") not in (None, 0, 0.0):
                axis = op.get("rotate_axis") or [0, 0, 1]
                if not _nonzero_vector(axis):
                    raise ValueError(f"component {cname!r} operation {oname!r}: rotate_axis must be non-zero")
            if kind == "shell" and abs(float(op["thickness_mm"])) <= 1e-12:
                raise ValueError(f"component {cname!r} operation {oname!r}: shell thickness must be non-zero")
            if kind == "hole":
                support_name = op.get("support_name")
                offset_mm = op.get("offset_mm")
                center_mm = op.get("center_mm")
                if support_name is not None:
                    if support_name not in op_names:
                        raise ValueError(
                            f"component {cname!r} operation {oname!r}: support_name {support_name!r} "
                            "must reference an earlier operation in the same component"
                        )
                    if offset_mm is None:
                        raise ValueError(
                            f"component {cname!r} operation {oname!r}: offset_mm is required when support_name is used"
                        )
                elif center_mm is None:
                    raise ValueError(
                        f"component {cname!r} operation {oname!r}: provide either center_mm or support_name + offset_mm"
                    )
                if offset_mm is not None and support_name is None:
                    raise ValueError(
                        f"component {cname!r} operation {oname!r}: offset_mm requires support_name"
                    )
                if not bool(op.get("through")) and op.get("depth_mm") is None:
                    raise ValueError(f"component {cname!r} operation {oname!r}: depth_mm is required when through=false")
            if kind in {"linear_pattern", "polar_pattern"}:
                for source in op.get("source_names") or []:
                    if source not in op_names:
                        raise ValueError(f"component {cname!r} operation {oname!r}: source {source!r} must reference an earlier operation name")
            if kind in _GENERATOR_COMMANDS:
                _validate_generator_semantics(op, cname, oname)
            explicit_mode = op.get("mode")
            if explicit_mode in {"join", "cut", "intersect"} and not have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: mode={explicit_mode!r} requires an existing body")
            if explicit_mode == "new" and have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: mode='new' after a body exists would discard prior geometry; use join/cut/intersect or split into another component")
            op_names.add(oname)
            # Every current operation except patterns/finishes still leaves a current body on success.
            have_body = True
    assembly = program.get("assembly")
    if assembly is not None:
        if len(names) < 2:
            raise ValueError("assembly requires at least two components")
        for ri, rel in enumerate(assembly.get("relations") or []):
            if rel.get("a") not in names or rel.get("b") not in names:
                raise ValueError(f"assembly.relations[{ri}]: a/b must name declared components")
            if rel.get("a") == rel.get("b"):
                raise ValueError(f"assembly.relations[{ri}]: a and b must be different components")
            typ = rel.get("type")
            if typ == "slider" and rel.get("start_mm") is not None and rel.get("end_mm") is not None and float(rel["start_mm"]) > float(rel["end_mm"]):
                raise ValueError(f"assembly.relations[{ri}]: slider start_mm must be <= end_mm")
            if typ == "rotational" and rel.get("start_deg") is not None and rel.get("end_deg") is not None and float(rel["start_deg"]) > float(rel["end_deg"]):
                raise ValueError(f"assembly.relations[{ri}]: rotational start_deg must be <= end_deg")
    return program


PROGRAM_SYSTEM_PROMPT = f"""You are the NVIDIA Nemotron mechanical CAD planner for CADia.
Convert the user's request into ONE strict JSON high-level CAD program. CADia's deterministic
interpreter executes it with the existing native CAD engine and mechanical generators.

RULES:
- Output only the JSON object required by the supplied schema; no prose or markdown.
- Understand the user's language directly, including Korean mechanical terminology. Preserve every
  explicit dimension, unit, count, component relationship, and requested motion limit.
- Never emit raw CADia/Inventor tool names, file paths, document switching, occurrence ids, $bind,
  $result, persistent topology ids, or other backend bookkeeping. The interpreter owns all of that.
- Prefer a CADia deterministic standard-component operation whenever it exactly matches the requested
  part: gear, shaft, parallel key, bearing, spring, V-pulley, coupling, lead screw/nut, metric
  bolt/nut/washer, thread/coil, or sheet metal.
- For custom geometry, use the small generic vocabulary: box/cylinder/cone/sphere/torus,
  extrude_profile/revolve_profile, hole, fillet/chamfer/shell, mirror/transform, and patterns.
- torus is a FULL closed 360-degree torus. It has no planar end faces. For a partial revolution,
  an annular/bored revolved body, or geometry that needs open end faces, use revolve_profile.
  revolve_profile loops may be nested: an outer loop plus inner loop(s) creates a hollow region
  before the native CADia revolve is executed.
- Give every operation a short unique name. Pattern source_names may refer only to earlier operations
  in the same component.
- The first solid-forming operation of a component should use mode=new (or omit mode). Later material
  additions/removals use join/cut/intersect. Never start a new unrelated body inside an existing
  component; make it a separate component instead.
- hole coordinates are not all world-space by default. If a hole belongs to a previously-created flange,
  boss, pad, or other feature, set support_name to that earlier operation and give offset_mm in the
  cross-section plane. offset_mm uses the two GLOBAL transverse axes: for ±Z drilling it is [X,Y],
  for ±X it is [Y,Z], and for ±Y it is [X,Z]. The interpreter derives the actual support-face origin
  from the executed feature. Use center_mm only when the requested point is genuinely a world coordinate.
- fillet/chamfer edge_indices are optional. If omitted, the interpreter filters seam/degenerate edges
  and deterministically reduces an oversized radius/size before asking for any model repair.
- shell thickness_mm is a positive wall-thickness magnitude; the interpreter hollows inward.
  removed_face_indices must refer to faces that actually exist on the current body. Do not invent
  multiple end faces on one-face primitives such as a full sphere or full torus.
- Use fillet/chamfer only when requested or mechanically meaningful; do not add decorative finishing
  that the user did not ask for.
- Components are independent manufactured parts. position_mm/rotation_deg_xyz/grounded describe their
  initial assembly placement. If there is only one independent part, set assembly=null.
- Assembly relations refer only to COMPONENT NAMES and principal origin references (X/Y/Z Axis,
  XY/XZ/YZ Plane). The interpreter resolves real occurrence names and document lifecycle.
- Relation schemas are type-specific. A slider has only linear travel fields; a rotational joint has
  angular fields; a rigid joint has no limits. Do not mix relation-type fields.
- Keep the program minimal and dependency-correct. Do not model details that were not requested.

AVAILABLE OPERATIONS (same registry used by schema validation and native execution):
{operations_reference()}
""".strip()



def repair_prompt(user_request: str, failed_program: dict[str, Any], error: str) -> str:
    return (
        f"ORIGINAL_REQUEST:\n{user_request}\n\n"
        "The previous structured CAD program was rejected or failed during deterministic execution.\n"
        f"FAILED_PROGRAM:\n{json.dumps(failed_program, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"ERROR:\n{error}\n\n"
        "Return one corrected COMPLETE high-level program. The diagnostic names the exact component/operation "
        "that failed and the state before it. Change the causal operation or its dependencies; do not return the "
        "same program unchanged. Do not expose backend calls, paths, bindings, or document bookkeeping. "
        "Preserve explicit user dimensions/counts unless the diagnostic proves that a derived finishing value must be reduced."
    )


# ---------------------------------------------------------------------------
# Deterministic interpreter
# ---------------------------------------------------------------------------

_GENERATOR_COMMANDS = {
    "gear": "create_gear",
    "shaft": "create_shaft",
    "parallel_key": "create_parallel_key",
    "bearing": "create_bearing",
    "spring": "create_spring",
    "v_pulley": "create_v_pulley",
    "coupling": "create_coupling",
    "lead_screw_nut": "create_lead_screw_nut",
    "hex_bolt": "create_metric_hex_bolt",
    "hex_nut": "create_metric_hex_nut",
    "washer": "create_metric_washer",
    "external_thread": "create_external_thread",
    "coil": "create_coil",
    "sheet_metal_base": "create_sheet_metal_base",
    "sheet_metal_flange": "add_sheet_metal_flange",
}

_PRIMITIVE_COMMANDS = {
    "box": "create_box", "cylinder": "create_cylinder", "cone": "create_cone",
    "sphere": "create_sphere", "torus": "create_torus",
}

_AXIS_VECTORS = {
    "+X": [1.0, 0.0, 0.0], "-X": [-1.0, 0.0, 0.0],
    "+Y": [0.0, 1.0, 0.0], "-Y": [0.0, -1.0, 0.0],
    "+Z": [0.0, 0.0, 1.0], "-Z": [0.0, 0.0, -1.0],
}


def _slug(value: str, fallback: str = "component") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", str(value or "").strip()).strip(".-")
    return (cleaned or fallback)[:80]


class ProgramBuildError(RuntimeError):
    """A precise structured-program execution error suitable for model repair."""


class CadiaProgramInterpreter:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.engine = runtime.engine
        self.document_dir = (Path(runtime.root).resolve() / "documents")
        self.document_dir.mkdir(parents=True, exist_ok=True)

    def _mode(self, op: dict[str, Any]) -> str:
        explicit = op.get("mode")
        if explicit:
            return str(explicit)
        doc = self.engine.doc
        return "new" if doc is None or doc.shape is None else "join"

    @staticmethod
    def _without(op: dict[str, Any], *keys: str) -> dict[str, Any]:
        banned = set(keys)
        return {k: copy.deepcopy(v) for k, v in op.items() if k not in banned and v is not None}

    def _shape_metrics(self) -> dict[str, Any]:
        doc = self.engine.doc
        shape = None if doc is None else doc.shape
        if shape is None:
            return {"has_shape": False, "valid": False, "volume_mm3": 0.0, "solids": 0, "bbox_mm": [0.0, 0.0, 0.0]}
        try:
            valid = bool(shape.isValid())
        except Exception:
            valid = False
        try:
            volume = float(shape.Volume())
        except Exception:
            volume = float("nan")
        try:
            solids = len(shape.Solids())
        except Exception:
            solids = 0
        try:
            bb = shape.BoundingBox()
            bbox = [float(bb.xlen), float(bb.ylen), float(bb.zlen)]
        except Exception:
            bbox = [0.0, 0.0, 0.0]
        return {"has_shape": True, "valid": valid, "volume_mm3": volume, "solids": solids, "bbox_mm": bbox}

    @staticmethod
    def _metrics_text(metrics: dict[str, Any]) -> str:
        if not metrics.get("has_shape"):
            return "no body"
        v = metrics.get("volume_mm3")
        bb = metrics.get("bbox_mm") or [0.0, 0.0, 0.0]
        return (f"valid={metrics.get('valid')} solids={metrics.get('solids')} "
                f"volume={v:.6g}mm3 bbox={bb[0]:.4g}x{bb[1]:.4g}x{bb[2]:.4g}mm")

    def _assert_current_shape(self, *, label: str) -> dict[str, Any]:
        m = self._shape_metrics()
        if not m.get("has_shape"):
            raise ValueError(f"{label}: produced no body")
        if not m.get("valid"):
            raise ValueError(f"{label}: resulting B-Rep is invalid")
        volume = float(m.get("volume_mm3") or 0.0)
        if not math.isfinite(volume) or volume <= 1e-9:
            raise ValueError(f"{label}: resulting body has zero/non-finite volume")
        return m

    def _feature_edges(self, indices: list[int] | None, *, op_name: str) -> list[str]:
        """PromptCAD-style deterministic safe-edge selection on CADia native topology.

        The model may use 1-based edge indices, but it never sees persistent CADia ids.
        If no indices are supplied, only non-degenerate edges shared by at least two
        faces are selected, which excludes cylindrical/conical seam edges that trigger
        OCCT ChFi3d failures.  Invalid explicit indices are dropped rather than turning
        an otherwise valid part into an LLM-repair round.
        """
        doc = self.engine.doc
        if doc is None or doc.shape is None:
            raise ValueError("edge operation requires an existing body")
        rows = edge_records(doc.shape)
        if not rows:
            raise ValueError("Current body has no selectable edges")
        safe_indices = {
            int(row["index"]) + 1
            for row in rows
            if float(row.get("length") or 0.0) > 1e-9 and int(row.get("adjacent_face_count") or 0) >= 2
        }
        requested = [int(x) for x in (indices or [])]
        if requested:
            in_range = [i for i in requested if 1 <= i <= len(rows)]
            usable = [i for i in in_range if i in safe_indices]
            if usable:
                return [str(rows[i - 1]["id"]) for i in usable]
        usable = sorted(safe_indices)
        if not usable:
            raise ValueError(f"{op_name}: current body has no edges safe for fillet/chamfer")
        return [str(rows[i - 1]["id"]) for i in usable]

    def _safe_finish(self, op: dict[str, Any], *, kind: str) -> dict[str, Any]:
        key = "radius_mm" if kind == "fillet" else "distance_mm"
        asked = float(op[key])
        current = asked
        base_snapshot = self.runtime.executor._snapshot()
        last_error = "geometry kernel rejected the edge operation"
        for _ in range(6):
            try:
                self.runtime.executor._rollback(base_snapshot)
                ids = self._feature_edges(op.get("edge_indices"), op_name=str(op["name"]))
                args = {"edge_ids": ids, key: current}
                result = self.engine.execute(kind, args)
                self._assert_current_shape(label=f"{kind} {op['name']!r}")
                if current != asked and isinstance(result, dict):
                    result = dict(result)
                    result["requested_" + key] = asked
                    result["used_" + key] = current
                    result["deterministic_fallback"] = True
                return result if isinstance(result, dict) else {"result": result}
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                current = round(current / 2.0, 8)
        self.runtime.executor._rollback(base_snapshot)
        raise ValueError(
            f"{kind} {op['name']!r} failed after six deterministic size reductions "
            f"(requested {asked:g} mm); last kernel diagnostic: {last_error}"
        )

    def _safe_shell(self, op: dict[str, Any]) -> dict[str, Any]:
        asked = abs(float(op["thickness_mm"]))
        current = asked
        base_snapshot = self.runtime.executor._snapshot()
        # Face selection is topological, not dimensional. Resolve it once before
        # the thickness fallback loop so an impossible face index never burns six
        # pointless shell retries.
        remove_face_ids = self._feature_faces(
            op.get("removed_face_indices"), op_name=f"shell {op['name']!r}"
        )
        last_error = "geometry kernel rejected shell"
        for _ in range(6):
            try:
                self.runtime.executor._rollback(base_snapshot)
                before = self._shape_metrics()
                result = self.engine.execute("shell", {
                    "thickness_mm": -current,
                    "remove_face_ids": remove_face_ids,
                    "kind": op.get("kind") or "arc", "name": op["name"],
                })
                after = self._assert_current_shape(label=f"shell {op['name']!r}")
                # CADia high-level shell semantics are an inward hollow, matching PromptCAD.
                bv = float(before.get("volume_mm3") or 0.0)
                av = float(after.get("volume_mm3") or 0.0)
                if av >= bv - max(1e-7, abs(bv) * 1e-9):
                    raise ValueError("inward shell removed no measurable material")
                if current != abs(asked) and isinstance(result, dict):
                    result = dict(result)
                    result["requested_thickness_mm"] = asked
                    result["used_thickness_mm"] = current
                    result["deterministic_fallback"] = True
                return result if isinstance(result, dict) else {"result": result}
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                current = round(current / 2.0, 8)
        self.runtime.executor._rollback(base_snapshot)
        raise ValueError(
            f"shell {op['name']!r} failed after six deterministic thickness reductions; "
            f"last kernel diagnostic: {last_error}"
        )

    def _feature_faces(self, indices: list[int] | None, *, op_name: str = "shell") -> list[str]:
        topo = self.engine.execute("get_topology", {})
        rows = topo.get("faces") or []
        if not indices:
            return []
        requested = [int(x) for x in indices]
        invalid = [x for x in requested if x < 1 or x > len(rows)]
        if invalid:
            raise ValueError(
                f"{op_name}: removed_face_indices {requested} reference nonexistent face(s) {invalid}; "
                f"the current body has {len(rows)} face(s). This is a topology-selection error; "
                "reducing shell thickness cannot repair it."
            )
        return [str(rows[i - 1]["id"]) for i in requested]

    def _through_depth(self, axis: str) -> float:
        doc = self.engine.doc
        if doc is None or doc.shape is None:
            raise ValueError("through-hole requires an existing body")
        bb = doc.shape.BoundingBox()
        span = {"X": float(bb.xlen), "Y": float(bb.ylen), "Z": float(bb.zlen)}[axis[-1]]
        return max(1.0, span * 2.0 + 20.0)

    @staticmethod
    def _opposite_axis(axis: str) -> str:
        table = {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y", "+Z": "-Z", "-Z": "+Z"}
        try:
            return table[str(axis)]
        except KeyError as exc:
            raise ValueError(f"unsupported hole axis {axis!r}") from exc

    @staticmethod
    def _axis_index(axis: str) -> int:
        return {"X": 0, "Y": 1, "Z": 2}[str(axis)[-1]]

    @staticmethod
    def _axis_unit(axis: str) -> list[float]:
        return list(_AXIS_VECTORS[str(axis)])

    @staticmethod
    def _dot(a, b) -> float:
        return sum(float(a[i]) * float(b[i]) for i in range(3))

    @classmethod
    def _parallel_principal_axis(cls, vector, axis: str, tol: float = 0.999) -> bool:
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            return False
        norm = math.sqrt(sum(float(x) ** 2 for x in vector))
        if norm <= 1e-12:
            return False
        unit = [float(x) / norm for x in vector]
        return abs(cls._dot(unit, cls._axis_unit(axis))) >= tol

    @staticmethod
    def _transverse_offset(axis: str, raw3: list[float] | None = None, raw2: list[float] | None = None) -> list[float]:
        if raw2 is not None:
            u, v = map(float, raw2)
        elif raw3 is not None:
            vals = [float(x) for x in raw3]
            if str(axis)[-1] == "Z":
                u, v = vals[0], vals[1]
            elif str(axis)[-1] == "X":
                u, v = vals[1], vals[2]
            else:
                u, v = vals[0], vals[2]
        else:
            raise ValueError("hole local offset is missing")
        if str(axis)[-1] == "Z":
            return [u, v, 0.0]
        if str(axis)[-1] == "X":
            return [0.0, u, v]
        return [u, 0.0, v]

    def _feature_record(self, feature_names: dict[str, str], ir_name: str):
        actual = feature_names.get(str(ir_name))
        if not actual or self.engine.doc is None:
            return None
        return next((f for f in self.engine.doc.features if f.name == actual), None)

    def _support_centers(self, feature_names: dict[str, str], axis: str, *, support_name: str | None = None) -> list[tuple[str, list[float], str]]:
        """Candidate feature-local face origins for a hole.

        A model should not have to recompute world coordinates after it creates an
        offset/rotated flange.  Primitive cylinders/cones expose an exact centerline
        frame; other additive features fall back to planar faces of their replayed
        B-Rep effect.  No name heuristic is used.
        """
        doc = self.engine.doc
        if doc is None:
            return []
        wanted = [support_name] if support_name else list(feature_names.keys())[::-1]
        drill = self._axis_unit(axis)
        desired = [-x for x in drill]
        out: list[tuple[str, list[float], str]] = []
        seen: set[tuple[float, float, float, str]] = set()

        def add(label: str, center, normal_label: str) -> None:
            c = [float(center[0]), float(center[1]), float(center[2])]
            key = (round(c[0], 7), round(c[1], 7), round(c[2], 7), normal_label)
            if key not in seen:
                seen.add(key); out.append((label, c, normal_label))

        for ir_name in wanted:
            rec = self._feature_record(feature_names, ir_name)
            if rec is None:
                continue
            p = rec.params or {}
            if rec.kind in {"cylinder", "cone"}:
                raw_axis = p.get("axis", [0, 0, 1])
                if not self._parallel_principal_axis(raw_axis, axis):
                    continue
                norm = math.sqrt(sum(float(x) ** 2 for x in raw_axis))
                a = [float(x) / norm for x in raw_axis]
                origin = [float(x) for x in p.get("origin_mm", [0, 0, 0])]
                try:
                    height = float(doc.value(p.get("height_mm")))
                except Exception:
                    height = float(p.get("height_mm") or 0.0)
                end = [origin[i] + a[i] * height for i in range(3)]
                start_n = [-x for x in a]; end_n = list(a)
                for tag, c, n in (("start", origin, start_n), ("end", end, end_n)):
                    dot = self._dot(n, desired)
                    if dot >= 0.999:
                        normal_label = self._opposite_axis(axis)
                    elif bool(dot <= -0.999):
                        normal_label = axis
                    else:
                        continue
                    add(f"{ir_name}:{tag}", c, normal_label)
                continue

            # Generic fallback: replay this one feature and use planar faces from
            # its additive B-Rep delta. This preserves CADia's native feature history.
            try:
                additive, _ = doc._feature_effect(rec)
                if additive is None:
                    continue
                for row in face_records(additive):
                    if row.get("geom") != "PLANE":
                        continue
                    direction = row.get("direction")
                    if direction not in {self._opposite_axis(axis), axis}:
                        continue
                    add(f"{ir_name}:effect", row.get("center") or [0, 0, 0], str(direction))
            except Exception:
                continue
        return out

    def _hole_face_rows(self, axis: str, point: list[float], *, through: bool) -> list[dict[str, Any]]:
        """All axis-compatible planar faces ranked by the requested entry plane.

        v9.2 used selector extreme=max before near_mm, which discarded non-extreme
        flange faces before proximity could help.  Enumerating exact face ids keeps
        the model away from topology bookkeeping while letting the interpreter try
        the geometrically nearest valid support face first.
        """
        rows = (self.engine.execute("get_topology", {}) or {}).get("faces") or []
        idx = self._axis_index(axis)
        normals = [self._opposite_axis(axis)] + ([axis] if through else [])
        ranked: list[tuple[tuple[int, float, float], dict[str, Any]]] = []
        for row in rows:
            if row.get("geom") != "PLANE" or row.get("direction") not in normals:
                continue
            c = [float(x) for x in (row.get("center") or [0, 0, 0])]
            normal_rank = normals.index(row.get("direction"))
            plane_distance = abs(c[idx] - float(point[idx]))
            transverse = math.sqrt(sum((c[i] - float(point[i])) ** 2 for i in range(3) if i != idx))
            ranked.append(((normal_rank, plane_distance, transverse), row))
        ranked.sort(key=lambda item: item[0])
        return [row for _, row in ranked]

    def _safe_hole(self, op: dict[str, Any], feature_names: dict[str, str]) -> dict[str, Any]:
        """Create a native CADia hole with world and feature-local coordinate resolution.

        The high-level planner is allowed to express a true world point, or to anchor a
        hole to an earlier feature.  World coordinates are attempted exactly first. If
        they miss, the deterministic interpreter may reinterpret the transverse values
        in the legacy center_mm as a feature-local offset around an earlier compatible
        cylinder/cone/feature frame. This is a compatibility fallback, not a name-based
        special case.
        """
        axis = str(op["axis"])
        through = bool(op.get("through"))
        depth = None if through else float(op.get("depth_mm") or 0.0)
        if not through and (depth is None or depth <= 0):
            raise ValueError(f"hole {op['name']!r}: depth_mm is required when through=false")

        center_raw = op.get("center_mm")
        center = None if center_raw is None else [float(x) for x in center_raw]
        support_name = op.get("support_name")
        offset_raw = op.get("offset_mm")
        base_snapshot = self.runtime.executor._snapshot()
        before = self._shape_metrics()
        before_v = float(before.get("volume_mm3") or 0.0)
        diagnostics: list[str] = []
        attempted_points: set[tuple[float, float, float]] = set()

        def try_native(point: list[float], source: str):
            key = tuple(round(float(x), 7) for x in point)
            if key in attempted_points:
                return None
            attempted_points.add(key)
            face_rows = self._hole_face_rows(axis, point, through=through)
            if not face_rows:
                diagnostics.append(f"{source}: no axis-compatible planar face")
                return None
            for row in face_rows[:16]:
                try:
                    self.runtime.executor._rollback(base_snapshot)
                    args = {
                        "face": str(row["id"]),
                        "points_mm": [point],
                        "diameter_mm": float(op["diameter_mm"]),
                        "kind": "drilled",
                        "through": through,
                    }
                    if not through:
                        args["depth_mm"] = depth
                    result = self.engine.execute("hole", args)
                    after = self._assert_current_shape(label=f"hole {op['name']!r}")
                    after_v = float(after.get("volume_mm3") or 0.0)
                    removed = before_v - after_v
                    if removed <= max(1e-7, abs(before_v) * 1e-9):
                        raise ValueError(f"removed no measurable material; before={before_v:.6g} after={after_v:.6g}")
                    out = dict(result) if isinstance(result, dict) else {"result": result}
                    out["resolved_face_id"] = str(row["id"])
                    out["resolved_point_mm"] = list(point)
                    out["coordinate_resolution"] = source
                    out["native_face_attached_hole"] = True
                    return out
                except Exception as exc:
                    diagnostics.append(
                        f"{source} face={row.get('id')} center={row.get('center')}: "
                        f"{str(exc) or type(exc).__name__}"
                    )
            return None

        def try_axis_cut(point: list[float], source: str):
            try:
                self.runtime.executor._rollback(base_snapshot)
                direction = _AXIS_VECTORS[axis]
                cutter_depth = self._through_depth(axis) if through else float(depth or 0.0)
                origin = list(point)
                if through:
                    origin = [origin[i] - 0.5 * cutter_depth * direction[i] for i in range(3)]
                result = self.engine.execute("create_cylinder", {
                    "diameter_mm": float(op["diameter_mm"]),
                    "height_mm": cutter_depth,
                    "origin_mm": origin,
                    "axis": direction,
                    "operation": "cut",
                    "replace": False,
                    "name": op["name"],
                })
                after = self._assert_current_shape(label=f"hole {op['name']!r}")
                after_v = float(after.get("volume_mm3") or 0.0)
                removed = before_v - after_v
                if removed <= max(1e-7, abs(before_v) * 1e-9):
                    raise ValueError(f"removed no measurable material; before={before_v:.6g} after={after_v:.6g}")
                out = dict(result) if isinstance(result, dict) else {"result": result}
                out["resolved_point_mm"] = list(point)
                out["coordinate_resolution"] = source
                out["native_face_attached_hole"] = False
                out["axis_cylinder_fallback"] = True
                return out
            except Exception as exc:
                diagnostics.append(f"{source} axis-cylinder: {str(exc) or type(exc).__name__}")
                return None

        # 1) Explicit feature-local placement: the preferred representation for
        # holes on flanges/bosses that were created away from the global origin.
        if support_name is not None:
            offset = self._transverse_offset(axis, raw2=[float(x) for x in offset_raw])
            centers = self._support_centers(feature_names, axis, support_name=str(support_name))
            if not centers:
                diagnostics.append(f"support {support_name!r}: no compatible executed feature frame")
            for label, base, _normal in centers:
                point = [base[i] + offset[i] for i in range(3)]
                result = try_native(point, f"support-local {label}")
                if result is not None:
                    return result
                result = try_axis_cut(point, f"support-local {label}")
                if result is not None:
                    return result
        else:
            # 2) True world coordinate semantics first. Never move a valid world
            # point merely because another feature frame also exists.
            if center is not None:
                result = try_native(center, "world")
                if result is not None:
                    return result
                result = try_axis_cut(center, "world")
                if result is not None:
                    return result

                # 3) Legacy compatibility: older v9 planners used center_mm for
                # flange-local offsets. Only after the exact world line failed, try
                # executed feature frames. Rank frames by axial proximity to the
                # supplied point so unrelated remote flanges are not preferred.
                offset = self._transverse_offset(axis, raw3=center)
                idx = self._axis_index(axis)
                frames = self._support_centers(feature_names, axis)
                frames.sort(key=lambda item: abs(float(item[1][idx]) - float(center[idx])))
                for label, base, _normal in frames[:12]:
                    point = [base[i] + offset[i] for i in range(3)]
                    result = try_native(point, f"legacy-feature-local {label}")
                    if result is not None:
                        return result
                    result = try_axis_cut(point, f"legacy-feature-local {label}")
                    if result is not None:
                        return result

        self.runtime.executor._rollback(base_snapshot)
        summary = " | ".join(diagnostics[-24:])
        raise ValueError(
            f"hole {op['name']!r} could not be attached to the requested body after world and "
            f"feature-local resolution; {summary}"
        )

    def _execute_operation(self, component_name: str, op: dict[str, Any], feature_names: dict[str, str]) -> dict[str, Any]:
        kind = str(op["op"])
        ir_name = str(op["name"])
        before = self._shape_metrics()
        effective_mode = self._mode(op) if kind not in {"fillet", "chamfer", "shell", "mirror", "transform", "linear_pattern", "polar_pattern", "hole", "sheet_metal_flange"} else None

        if kind in _PRIMITIVE_COMMANDS:
            args = self._without(op, "op", "mode")
            args["operation"] = effective_mode
            args["replace"] = False
            result = self.engine.execute(_PRIMITIVE_COMMANDS[kind], args)
        elif kind == "extrude_profile":
            profile = op["profile"]
            plane = op.get("plane") or "XY Plane"
            sketch = self.engine.execute("create_sketch", {
                "name": f"{_slug(component_name)}_{_slug(ir_name)}_sk", "plane": plane
            })["sketch_name"]
            for i, a in enumerate(profile):
                b = profile[(i + 1) % len(profile)]
                self.engine.execute("draw_line", {
                    "sketch_name": sketch, "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]
                })
            self.engine.execute("close_sketch", {"sketch_name": sketch})
            result = self.engine.execute("extrude_advanced", {
                "sketch_name": sketch,
                "distance_mm": op["distance_mm"],
                "taper_deg": op.get("taper_deg", 0.0),
                "operation": effective_mode,
                "direction": op.get("direction") or "positive",
                "name": ir_name,
            })
        elif kind == "revolve_profile":
            # PromptCAD-style revolve mapped onto CADia's existing native sketch +
            # revolve feature.  Multiple loops are built in one sketch, so nested
            # loops become holes before the revolution; no second geometry engine.
            sketch = self.engine.execute("create_sketch", {
                "name": f"{_slug(component_name)}_{_slug(ir_name)}_rev_sk", "plane": "XY Plane"
            })["sketch_name"]
            for loop in op["loops"]:
                if loop["kind"] == "circle":
                    self.engine.execute("draw_circle", {
                        "sketch_name": sketch,
                        "cx": float(loop["center"][0]), "cy": float(loop["center"][1]),
                        "radius": float(loop["radius_mm"]),
                    })
                else:
                    points = loop["points"]
                    for i, a in enumerate(points):
                        b = points[(i + 1) % len(points)]
                        self.engine.execute("draw_line", {
                            "sketch_name": sketch,
                            "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1],
                        })
            self.engine.execute("close_sketch", {"sketch_name": sketch})
            # CADia's native Inventor-compatible revolve command accepts join/cut/
            # intersect.  On an empty part, join is the native equivalent of the
            # high-level IR's first-body mode=new.
            native_mode = "join" if effective_mode == "new" else effective_mode
            result = self.engine.execute("revolve", {
                "sketch_name": sketch, "axis_id": "Y Axis",
                "angle_deg": float(op["angle_deg"]), "operation": native_mode,
            })
        elif kind == "hole":
            result = self._safe_hole(op, feature_names)
        elif kind == "fillet":
            result = self._safe_finish(op, kind="fillet")
        elif kind == "chamfer":
            result = self._safe_finish(op, kind="chamfer")
        elif kind == "shell":
            result = self._safe_shell(op)
        elif kind == "mirror":
            result = self.engine.execute("mirror_body", {
                "plane": op["plane"], "base_point_mm": op.get("base_point_mm") or [0, 0, 0],
                "keep_original": bool(op["keep_original"]), "name": ir_name,
            })
        elif kind == "transform":
            result = self.engine.execute("transform_body", {
                "translate_mm": op.get("translate_mm") or [0, 0, 0],
                "rotate_axis": op.get("rotate_axis") or [0, 0, 1],
                "rotate_deg": op.get("rotate_deg") or 0.0,
                "scale": op.get("scale") or 1.0,
                "name": ir_name,
            })
        elif kind == "linear_pattern":
            actual = []
            for source in op["source_names"]:
                if source not in feature_names:
                    raise ValueError(f"linear_pattern source {source!r} has no executed feature")
                actual.append(feature_names[source])
            result = self.engine.execute("rectangular_pattern", {
                "feature_names": actual, "dir1": op["direction1"], "count1": op["count1"],
                "spacing_mm1": op["spacing_mm1"], "dir2": op.get("direction2"),
                "count2": op.get("count2"), "spacing_mm2": op.get("spacing_mm2"),
                "natural_direction1": True, "natural_direction2": True,
            })
        elif kind == "polar_pattern":
            actual = []
            for source in op["source_names"]:
                if source not in feature_names:
                    raise ValueError(f"polar_pattern source {source!r} has no executed feature")
                actual.append(feature_names[source])
            result = self.engine.execute("circular_pattern", {
                "feature_names": actual, "axis": op["axis"], "count": op["count"],
                "angle_deg": op["angle_deg"], "natural_direction": True,
            })
        elif kind in _GENERATOR_COMMANDS:
            args = self._without(op, "op", "mode")
            command = _GENERATOR_COMMANDS[kind]
            if kind != "sheet_metal_flange":
                args["replace"] = False
            if kind not in {"sheet_metal_base", "sheet_metal_flange"}:
                args["operation"] = effective_mode
            result = self.engine.execute(command, args)
        else:
            raise ValueError(f"Unsupported CAD program operation: {kind}")

        after = self._assert_current_shape(label=f"operation {kind!r} ({ir_name!r})")
        before_v = float(before.get("volume_mm3") or 0.0) if before.get("has_shape") else None
        after_v = float(after.get("volume_mm3") or 0.0)
        cut_like = kind == "hole" or effective_mode == "cut"
        if cut_like and before_v is not None:
            removed = before_v - after_v
            if removed <= max(1e-7, abs(before_v) * 1e-9):
                raise ValueError(
                    f"operation {kind!r} ({ir_name!r}) removed no measurable material; "
                    f"before={before_v:.6g} mm3 after={after_v:.6g} mm3"
                )

        feature_name = result.get("feature_name") if isinstance(result, dict) else None
        if not feature_name and isinstance(result, dict):
            names = result.get("feature_names") or []
            if names:
                feature_name = names[0]
        if feature_name:
            feature_names[ir_name] = str(feature_name)
        row = result if isinstance(result, dict) else {"result": result}
        if isinstance(row, dict):
            row = dict(row)
            row.setdefault("geometry", after)
        return row

    def _verify_part(self, component_name: str) -> dict[str, Any]:
        doc = self.engine.doc
        if doc is None or doc.doc_type != "part":
            raise ValueError(f"component {component_name!r}: interpreter did not end on a part document")
        if doc.shape is None:
            raise ValueError(f"component {component_name!r}: program produced no body")
        if not doc.shape.isValid():
            raise ValueError(f"component {component_name!r}: resulting B-Rep is invalid")
        volume = float(doc.shape.Volume())
        if not math.isfinite(volume) or volume <= 1e-9:
            raise ValueError(f"component {component_name!r}: resulting body has zero/non-finite volume")
        bb = doc.shape.BoundingBox()
        return {
            "volume_mm3": volume,
            "bbox_mm": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
            "feature_count": len(doc.features),
        }

    def _save_component(self, name: str) -> str:
        session = self.engine._active_session()
        suffix = str(getattr(session, "id", ""))[:8] or "part"
        path = self.document_dir / f"nemotron-{_slug(name)}-{suffix}.scad.json"
        result = self.engine.execute("save_document", {"path": str(path)})
        return str(result.get("path") or path)

    @staticmethod
    def _default_ref(rel_type: str) -> str:
        return "XY Plane" if rel_type in {"mate", "flush", "angle"} else "Z Axis"

    def _execute_relation(self, rel: dict[str, Any], occurrence_names: dict[str, str]) -> dict[str, Any]:
        typ = str(rel["type"])
        a = occurrence_names[str(rel["a"])]
        b = occurrence_names[str(rel["b"])]
        a_ref = rel.get("a_ref") or self._default_ref(typ)
        b_ref = rel.get("b_ref") or self._default_ref(typ)
        if typ in {"mate", "flush", "insert", "angle"}:
            return self.engine.execute("add_constraint", {
                "type": typ, "a_occurrence": a, "a_ref": a_ref,
                "b_occurrence": b, "b_ref": b_ref,
                "offset_mm": rel.get("offset_mm", 0.0),
                "angle_deg": rel.get("angle_deg"),
                "insert_opposed": rel.get("opposed", True),
            })
        args: dict[str, Any] = {
            "type": typ, "a_occurrence": a, "a_ref": a_ref,
            "b_occurrence": b, "b_ref": b_ref, "name": rel.get("name"),
            "a_intent": None, "b_intent": None,
            "flip_origin_direction": False, "flip_alignment_direction": False,
        }
        if typ == "slider":
            args.update(linear_position_mm=rel.get("position_mm"), linear_start_mm=rel.get("start_mm"), linear_end_mm=rel.get("end_mm"))
        elif typ == "rotational":
            args.update(angular_position_deg=rel.get("position_deg"), angular_start_deg=rel.get("start_deg"), angular_end_deg=rel.get("end_deg"))
        elif typ == "cylindrical":
            for key in ("linear_position_mm", "linear_start_mm", "linear_end_mm", "angular_position_deg", "angular_start_deg", "angular_end_deg"):
                args[key] = rel.get(key)
        elif typ == "planar":
            args.update(angular_position_deg=rel.get("angular_position_deg"), angular_start_deg=rel.get("angular_start_deg"), angular_end_deg=rel.get("angular_end_deg"))
        return self.engine.execute("add_joint", args)

    def execute(self, program: dict[str, Any], on_event=None) -> dict[str, Any]:
        program = validate_program(program)
        snapshot = self.runtime.executor._snapshot()
        results: list[dict[str, Any]] = []
        try:
            component_paths: dict[str, str] = {}
            metrics: dict[str, Any] = {}
            components = program["components"]
            for ci, component in enumerate(components, 1):
                if on_event:
                    on_event("progress", {"message": f"Building component {ci}/{len(components)}…", "percent": 30 + int(35 * (ci - 1) / max(1, len(components)))})
                self.engine.execute("new_part", {"name": component["name"]})
                feature_names: dict[str, str] = {}
                for oi, op in enumerate(component["operations"], 1):
                    before = self._shape_metrics()
                    try:
                        row = self._execute_operation(component["name"], op, feature_names)
                    except Exception as exc:
                        raise ProgramBuildError(
                            f"component {component['name']!r} operation #{oi} {op['op']!r} "
                            f"({op['name']!r}) failed: {exc}; "
                            f"state before operation: {self._metrics_text(before)}; "
                            f"operation={json.dumps(op, ensure_ascii=False, separators=(',', ':'))}"
                        ) from exc
                    results.append({"component": component["name"], "operation": op["name"], "op": op["op"], "result": row})
                try:
                    metrics[component["name"]] = self._verify_part(component["name"])
                except Exception as exc:
                    raise ProgramBuildError(
                        f"component {component['name']!r} final verification failed: {exc}; "
                        f"final state: {self._metrics_text(self._shape_metrics())}"
                    ) from exc
                component_paths[component["name"]] = self._save_component(component["name"])

            assembly = program.get("assembly")
            if assembly is not None:
                if on_event:
                    on_event("progress", {"message": "Assembling components…", "percent": 68})
                self.engine.execute("new_assembly", {"name": assembly["name"]})
                occurrence_names: dict[str, str] = {}
                for component in components:
                    placed = self.engine.execute("place_occurrence", {
                        "path": component_paths[component["name"]],
                        "grounded": bool(component["grounded"]),
                        "position_mm": component["position_mm"],
                        "rotation_deg_xyz": component["rotation_deg_xyz"],
                    })
                    occurrence_names[component["name"]] = str(placed["occurrence_name"])
                for ri, rel in enumerate(assembly.get("relations") or [], 1):
                    try:
                        rel_result = self._execute_relation(rel, occurrence_names)
                    except Exception as exc:
                        raise ProgramBuildError(
                            f"assembly relation #{ri} {rel['type']!r} ({rel.get('name') or ri!r}) failed: {exc}; "
                            f"relation={json.dumps(rel, ensure_ascii=False, separators=(',', ':'))}"
                        ) from exc
                    results.append({"relation": rel.get("name") or rel["type"], "result": rel_result})
                health = self.engine.execute("get_assembly_bom", {})
                if len(health.get("occurrences") or []) < len(components):
                    raise ValueError("assembly verification failed: one or more components were not placed")
                session = self.engine._active_session()
                suffix = str(getattr(session, "id", ""))[:8] or "assembly"
                self.engine.execute("save_document", {"path": str(self.document_dir / f"nemotron-{_slug(assembly['name'])}-{suffix}.scad.json")})
                summary = f"{assembly['name']} complete · {len(components)} components · {len(assembly.get('relations') or [])} relations"
            else:
                summary = f"{components[-1]['name']} complete · {metrics[components[-1]['name']]['feature_count']} features · {metrics[components[-1]['name']]['volume_mm3']:,.3f} mm³"

            self.runtime.persist()
            return {"ok": True, "message": summary, "program": program, "results": results, "metrics": metrics}
        except Exception:
            try:
                self.runtime.executor._rollback(snapshot)
            finally:
                self.runtime.persist()
            raise


class RepairStalledError(RuntimeError):
    pass


class NemotronProgramAgent:
    """Nemotron high-level planner. Deterministic CAD fallbacks run before model repair."""

    def __init__(self, *, provider, model: str, reasoning: str = "medium") -> None:
        self.provider = provider
        self.model = model
        self.reasoning = reasoning
        self.last_program: dict[str, Any] | None = None

    def cancel(self) -> None:
        self.provider.cancel()

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        state_summary = {
            "document_type": current_state.get("document_type") or current_state.get("doc_type"),
            "selection": current_state.get("selection") or {},
        }
        prompt = (
            PROGRAM_SYSTEM_PROMPT
            + "\n\nCURRENT_STATE_SUMMARY:\n" + json.dumps(state_summary, ensure_ascii=False, separators=(",", ":"))
            + "\n\nUSER_REQUEST:\n" + str(user_prompt or "").strip()
        )
        program = self.provider.complete_cad_program(
            prompt,
            schema=program_schema(),
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            repair=False,
        )
        self.last_program = validate_program(program)
        return copy.deepcopy(self.last_program)

    def repair(self, user_prompt: str, error: str, on_event=None) -> dict[str, Any]:
        if self.last_program is None:
            raise ValueError("No Nemotron CAD program is available to repair")
        failed = copy.deepcopy(self.last_program)
        prompt = PROGRAM_SYSTEM_PROMPT + "\n\n" + repair_prompt(user_prompt, failed, error)
        repaired = self.provider.complete_cad_program(
            prompt,
            schema=program_schema(),
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            repair=True,
        )
        repaired = validate_program(repaired)
        if repaired == failed:
            raise RepairStalledError("Nemotron returned the same high-level CAD program")
        self.last_program = copy.deepcopy(repaired)
        return repaired


def _try_direct_cadia_route(*, runtime, user_prompt: str, current_state: dict[str, Any], on_event=None):
    """Use CADia's proven deterministic language/generator router when it is certain.

    This is not Codex or another model. It preserves the existing Korean/English parser,
    selection edits, primitive shortcuts, and standard mechanical generators. If the
    deterministic planner has no high-confidence plan, Nemotron handles the request.
    """
    plan = direct_plan(user_prompt, current_state)
    if plan is None:
        return None
    if on_event:
        on_event("progress", {"message": "Using CADia deterministic intent/generator route…", "percent": 20})
    executed = runtime.executor.execute(plan, on_event=on_event, progress_span=(22, 72))
    if not executed.ok:
        return None
    runtime.persist()
    return {
        "ok": True,
        "message": executed.message,
        "program": {"route": "cadia_deterministic", "note": plan.get("note", "")},
        "results": executed.results,
        "metrics": {},
    }


def execute_nemotron_program_with_repair(*, runtime, agent: NemotronProgramAgent, user_prompt: str,
                                         current_state: dict[str, Any], on_event=None,
                                         max_repairs: int = 1) -> dict[str, Any]:
    # Preserve the strongest parts of the pre-v8 system before involving the model.
    direct = _try_direct_cadia_route(
        runtime=runtime, user_prompt=user_prompt, current_state=current_state, on_event=on_event
    )
    if direct is not None:
        return direct

    program = agent.plan(user_prompt, current_state, on_event=on_event)
    interpreter = CadiaProgramInterpreter(runtime)
    first_error: Exception | None = None
    last_error: Exception | None = None
    for attempt in range(max(0, int(max_repairs)) + 1):
        try:
            return interpreter.execute(program, on_event=on_event)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            last_error = exc
            if attempt >= max_repairs:
                break
            if on_event:
                on_event("progress", {
                    "message": "Native CAD verification failed · asking Nemotron for one structured correction…",
                    "percent": 86,
                })
            try:
                program = agent.repair(user_prompt, str(exc), on_event=on_event)
            except RepairStalledError:
                # Do not replace the useful native CAD diagnostic with a secondary loop-guard error.
                break

    err = last_error or first_error or RuntimeError("unknown structured CAD failure")
    first = first_error or err
    suffix = ""
    if last_error is first_error:
        suffix = " Nemotron did not produce a different executable program."
    raise RuntimeError(f"Modeling failed: {first}.{suffix}") from err

