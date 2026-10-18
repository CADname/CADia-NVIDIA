from __future__ import annotations

"""Audited model-facing contract for NVIDIA Nemotron Direct.

v10 deliberately leaves the existing CADia geometry engine, mechanical generators,
assembly executor, Codex/Copilot paths, and 118-tool backend untouched.  Only the
Nemotron Direct abstraction boundary is replaced.

The contract is intentionally small and structurally discriminated:
model schema -> semantic validator -> deterministic compiler -> CADia native engine.
Field combinations that caused the v9.x debug loop are made unrepresentable wherever
JSON Schema can express the dependency (hole location/extent, gear/spring variants,
shaft keyway, coupling bolt/keyway groups, second pattern direction).
"""

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any

from . import nemotron_program as v9
from standalonecad.planning import direct_plan
from standalonecad.core.topology import face_records


# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "minimum": 0.001}
NONNEG = {"type": "number", "minimum": 0.0}
STRING = {"type": "string", "minLength": 1}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
VEC3 = {"type": "array", "minItems": 3, "maxItems": 3, "items": NUMBER}
POINT2 = {"type": "array", "minItems": 2, "maxItems": 2, "items": NUMBER}
PROFILE2 = {"type": "array", "minItems": 3, "items": POINT2}
AXIS_REF = {"type": "string", "enum": ["X Axis", "Y Axis", "Z Axis"]}
ORIGIN_REF = {"type": "string", "enum": ["X Axis", "Y Axis", "Z Axis", "XY Plane", "XZ Plane", "YZ Plane"]}
PRINCIPAL = {"type": "string", "enum": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]}
MODE = {"type": "string", "enum": ["new", "join", "cut", "intersect"]}
GEN_MODE = {"type": "string", "enum": ["new", "join"]}
STANDARD_FASTENER_DIAMETER = {"type": "number", "enum": [3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30]}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [copy.deepcopy(schema), {"type": "null"}]}


def _obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _op_obj(op: str, properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return _obj({"op": {"type": "string", "const": op}, "name": STRING, **properties}, ("op", "name", *required))


def _disc_obj(field: str, value: str, properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return _obj({field: {"type": "string", "const": value}, **properties}, (field, *required))


def _enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _validate_schema(schema: dict[str, Any], value: Any, path: str = "$") -> None:
    """Validate the exact schema subset used here, with discriminator-aware oneOf errors.

    v9's validator tried oneOf branches in declaration order and often surfaced an
    unrelated branch error.  v10 first narrows by every available const discriminator
    (typically op/kind), so a missing field error names the branch the model actually chose.
    """
    if "oneOf" in schema:
        branches = list(schema.get("oneOf") or [])
        narrowed = branches
        if isinstance(value, dict):
            const_pairs: list[tuple[str, Any]] = []
            for key, raw in value.items():
                candidates = []
                for branch in branches:
                    props = branch.get("properties") or {}
                    item = props.get(key) or {}
                    if "const" in item:
                        candidates.append(item["const"])
                if candidates:
                    const_pairs.append((key, raw))
            if const_pairs:
                exact = []
                for branch in branches:
                    props = branch.get("properties") or {}
                    okay = True
                    matched_any = False
                    for key, raw in const_pairs:
                        item = props.get(key) or {}
                        if "const" in item:
                            matched_any = True
                            if raw != item["const"]:
                                okay = False
                                break
                    if okay and matched_any:
                        exact.append(branch)
                if exact:
                    narrowed = exact
        errors: list[str] = []
        matches = 0
        for branch in narrowed:
            try:
                _validate_schema(branch, value, path)
                matches += 1
            except ValueError as exc:
                errors.append(str(exc))
        if matches == 1:
            return
        if matches > 1:
            raise ValueError(f"{path}: matches multiple mutually-exclusive schema branches")
        if errors:
            raise ValueError(errors[0])
        disc = []
        if isinstance(value, dict):
            for key in ("op", "kind", "type"):
                if key in value:
                    disc.append(f"{key}={value[key]!r}")
        suffix = f" ({', '.join(disc)})" if disc else ""
        raise ValueError(f"{path}: no allowed schema branch{suffix}")

    if "anyOf" in schema:
        errors: list[str] = []
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
        if schema.get("uniqueItems"):
            keys = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in value]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{path}: items must be unique")
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
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: minimum is {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: maximum is {schema['maximum']}")


# ---------------------------------------------------------------------------
# Single source of truth: operation registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    schema: dict[str, Any]


REVOLVE_LOOP = {
    "oneOf": [
        _disc_obj("kind", "polyline", {"points": PROFILE2}, ("points",)),
        _disc_obj("kind", "circle", {"center": POINT2, "radius_mm": POSITIVE}, ("center", "radius_mm")),
    ]
}

HOLE_LOCATION = {
    "oneOf": [
        _disc_obj("kind", "world", {"point_mm": VEC3}, ("point_mm",)),
        _disc_obj("kind", "feature", {"support": STRING, "offset_mm": POINT2}, ("support", "offset_mm")),
    ]
}
HOLE_EXTENT = {
    "oneOf": [
        _disc_obj("kind", "through", {}, ()),
        _disc_obj("kind", "blind", {"depth_mm": POSITIVE}, ("depth_mm",)),
    ]
}

KEYWAY = _obj({"width_mm": POSITIVE, "depth_mm": POSITIVE, "length_mm": POSITIVE, "start_mm": NONNEG},
              ("width_mm", "depth_mm", "length_mm", "start_mm"))
COUPLING_KEYWAY = _obj({"width_mm": POSITIVE, "depth_mm": POSITIVE}, ("width_mm", "depth_mm"))
BOLT_PATTERN = _obj({
    "circle_diameter_mm": POSITIVE,
    "count": {"type": "integer", "minimum": 1, "maximum": 24},
    "hole_diameter_mm": POSITIVE,
}, ("circle_diameter_mm", "count", "hole_diameter_mm"))


def _gear_branch(kind: str, props: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    common = {
        "kind": {"type": "string", "const": kind},
        "module": {"type": "number", "minimum": 0.01},
        "bore_diameter_mm": _nullable(NONNEG),
        "pressure_angle_deg": _nullable({"type": "number", "minimum": 5, "maximum": 35}),
        "backlash_mm": _nullable(NONNEG),
        "clearance_mm": _nullable(NONNEG),
        "mode": _nullable(GEN_MODE),
    }
    return _op_obj("gear", {**common, **props}, ("kind", "module", *required))


GEAR_SCHEMA = {
    "oneOf": [
        _gear_branch("spur", {"teeth": {"type": "integer", "minimum": 6}, "width_mm": POSITIVE}, ("teeth", "width_mm")),
        _gear_branch("helical", {"teeth": {"type": "integer", "minimum": 6}, "width_mm": POSITIVE,
                                 "helix_angle_deg": {"type": "number", "minimum": -55, "maximum": 55}},
                     ("teeth", "width_mm", "helix_angle_deg")),
        _gear_branch("herringbone", {"teeth": {"type": "integer", "minimum": 6}, "width_mm": POSITIVE,
                                     "helix_angle_deg": _nullable({"type": "number", "minimum": -55, "maximum": 55})},
                     ("teeth", "width_mm")),
        _gear_branch("ring", {"teeth": {"type": "integer", "minimum": 10}, "width_mm": POSITIVE,
                              "rim_width_mm": POSITIVE,
                              "helix_angle_deg": _nullable({"type": "number", "minimum": -55, "maximum": 55})},
                     ("teeth", "width_mm", "rim_width_mm")),
        _gear_branch("bevel", {"teeth": {"type": "integer", "minimum": 6}, "width_mm": POSITIVE,
                               "cone_angle_deg": {"type": "number", "minimum": 0.01, "maximum": 89.99},
                               "helix_angle_deg": _nullable({"type": "number", "minimum": -55, "maximum": 55})},
                     ("teeth", "width_mm", "cone_angle_deg")),
        _gear_branch("rack", {"length_mm": POSITIVE, "width_mm": POSITIVE, "height_mm": {"type": "number", "minimum": 0.01},
                              "helix_angle_deg": _nullable({"type": "number", "minimum": -55, "maximum": 55})},
                     ("length_mm", "width_mm", "height_mm")),
        _gear_branch("worm", {"length_mm": POSITIVE,
                              "lead_angle_deg": {"type": "number", "minimum": -79.99, "maximum": 79.99},
                              "thread_starts": {"type": "integer", "minimum": 1, "maximum": 8}},
                     ("length_mm", "lead_angle_deg", "thread_starts")),
        _gear_branch("worm_pair", {"teeth": {"type": "integer", "minimum": 8}, "width_mm": POSITIVE,
                                   "length_mm": POSITIVE,
                                   "lead_angle_deg": {"type": "number", "minimum": -79.99, "maximum": 79.99},
                                   "thread_starts": {"type": "integer", "minimum": 1, "maximum": 8}},
                     ("teeth", "width_mm", "length_mm", "lead_angle_deg", "thread_starts")),
        _gear_branch("planetary", {"sun_teeth": {"type": "integer", "minimum": 6},
                                   "planet_teeth": {"type": "integer", "minimum": 6},
                                   "width_mm": POSITIVE, "rim_width_mm": POSITIVE,
                                   "planet_count": {"type": "integer", "minimum": 2, "maximum": 12},
                                   "helix_angle_deg": _nullable({"type": "number", "minimum": -55, "maximum": 55})},
                     ("sun_teeth", "planet_teeth", "width_mm", "rim_width_mm", "planet_count")),
    ]
}

SPRING_SCHEMA = {
    "oneOf": [
        _op_obj("spring", {
            "kind": {"type": "string", "const": "compression"},
            "wire_diameter_mm": POSITIVE, "mean_diameter_mm": POSITIVE,
            "free_length_mm": POSITIVE, "active_turns": POSITIVE,
            "right_handed": _nullable(BOOL), "mode": _nullable(GEN_MODE),
        }, ("kind", "wire_diameter_mm", "mean_diameter_mm", "free_length_mm", "active_turns")),
        _op_obj("spring", {
            "kind": {"type": "string", "const": "belleville"},
            "outer_diameter_mm": POSITIVE, "inner_diameter_mm": POSITIVE,
            "thickness_mm": POSITIVE, "free_height_mm": POSITIVE,
            "mode": _nullable(GEN_MODE),
        }, ("kind", "outer_diameter_mm", "inner_diameter_mm", "thickness_mm", "free_height_mm")),
    ]
}


_OPERATION_LIST: list[OperationSpec] = [
    OperationSpec("box", "Rectangular solid aligned to XYZ.", _op_obj("box", {
        "length_mm": POSITIVE, "width_mm": POSITIVE, "height_mm": POSITIVE,
        "origin_mm": _nullable(VEC3), "centered": _nullable(BOOL), "mode": _nullable(MODE),
    }, ("length_mm", "width_mm", "height_mm"))),
    OperationSpec("cylinder", "Cylinder with explicit diameter/height.", _op_obj("cylinder", {
        "diameter_mm": POSITIVE, "height_mm": POSITIVE, "origin_mm": _nullable(VEC3),
        "axis": _nullable(VEC3), "mode": _nullable(MODE),
    }, ("diameter_mm", "height_mm"))),
    OperationSpec("cone", "Cone/frustum.", _op_obj("cone", {
        "diameter1_mm": POSITIVE, "diameter2_mm": NONNEG, "height_mm": POSITIVE,
        "origin_mm": _nullable(VEC3), "axis": _nullable(VEC3), "mode": _nullable(MODE),
    }, ("diameter1_mm", "height_mm"))),
    OperationSpec("sphere", "Full sphere.", _op_obj("sphere", {
        "diameter_mm": POSITIVE, "center_mm": _nullable(VEC3), "mode": _nullable(MODE),
    }, ("diameter_mm",))),
    OperationSpec("torus", "Full closed 360-degree torus.", _op_obj("torus", {
        "major_radius_mm": POSITIVE, "minor_radius_mm": POSITIVE,
        "center_mm": _nullable(VEC3), "axis": _nullable(VEC3), "mode": _nullable(MODE),
    }, ("major_radius_mm", "minor_radius_mm"))),
    OperationSpec("extrude_profile", "Extrude a simple closed polygon.", _op_obj("extrude_profile", {
        "profile": PROFILE2, "distance_mm": POSITIVE,
        "plane": _nullable(_enum("XY Plane", "XZ Plane", "YZ Plane")),
        "direction": _nullable(_enum("positive", "negative", "symmetric")),
        "taper_deg": _nullable({"type": "number", "minimum": -89, "maximum": 89}),
        "mode": _nullable(MODE),
    }, ("profile", "distance_mm"))),
    OperationSpec("revolve_profile", "Revolve closed polyline/circle loops about profile Y axis; nested loops create bores.", _op_obj("revolve_profile", {
        "loops": {"type": "array", "minItems": 1, "items": REVOLVE_LOOP},
        "angle_deg": {"type": "number", "minimum": 0.001, "maximum": 360},
        "mode": _nullable(MODE),
    }, ("loops", "angle_deg"))),
    OperationSpec("hole", "Cylindrical hole with structurally-discriminated location and extent.", _op_obj("hole", {
        "location": HOLE_LOCATION, "extent": HOLE_EXTENT,
        "diameter_mm": POSITIVE, "axis": PRINCIPAL,
    }, ("location", "extent", "diameter_mm", "axis"))),
    OperationSpec("fillet", "Round current-body edges; omitted indices use safe deterministic edge filtering.", _op_obj("fillet", {
        "radius_mm": POSITIVE,
        "edge_indices": _nullable({"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 1}}),
    }, ("radius_mm",))),
    OperationSpec("chamfer", "Chamfer current-body edges; omitted indices use safe deterministic edge filtering.", _op_obj("chamfer", {
        "distance_mm": POSITIVE,
        "edge_indices": _nullable({"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 1}}),
    }, ("distance_mm",))),
    OperationSpec("shell", "Inward shell; openings are semantic principal face normals, never guessed face indices.", _op_obj("shell", {
        "thickness_mm": POSITIVE,
        "open_normals": {"type": "array", "minItems": 1, "maxItems": 6, "uniqueItems": True, "items": PRINCIPAL},
        "kind": _nullable(_enum("arc", "intersection")),
    }, ("thickness_mm", "open_normals"))),
    OperationSpec("mirror", "Mirror current body about a principal plane.", _op_obj("mirror", {
        "plane": _enum("XY", "XZ", "YZ"), "base_point_mm": _nullable(VEC3),
        "keep_original": _nullable(BOOL),
    }, ("plane",))),
    OperationSpec("translate", "Translate current body.", _op_obj("translate", {"vector_mm": VEC3}, ("vector_mm",))),
    OperationSpec("rotate", "Rotate current body about a principal/arbitrary axis through origin.", _op_obj("rotate", {
        "axis": VEC3, "angle_deg": NUMBER,
    }, ("axis", "angle_deg"))),
    OperationSpec("scale", "Uniformly scale current body about origin.", _op_obj("scale", {
        "factor": {"type": "number", "minimum": 0.000001},
    }, ("factor",))),
    OperationSpec("linear_pattern", "Rectangular feature pattern with atomic optional second direction.", _op_obj("linear_pattern", {
        "source_names": {"type": "array", "minItems": 1, "items": STRING},
        "direction1": AXIS_REF, "count1": {"type": "integer", "minimum": 2}, "spacing_mm1": POSITIVE,
        "direction2": _nullable(_obj({"axis": AXIS_REF, "count": {"type": "integer", "minimum": 2}, "spacing_mm": POSITIVE},
                                     ("axis", "count", "spacing_mm"))),
    }, ("source_names", "direction1", "count1", "spacing_mm1"))),
    OperationSpec("polar_pattern", "Circular feature pattern about a principal axis.", _op_obj("polar_pattern", {
        "source_names": {"type": "array", "minItems": 1, "items": STRING},
        "axis": AXIS_REF, "count": {"type": "integer", "minimum": 2},
        "angle_deg": {"type": "number", "minimum": 0.001, "maximum": 360},
    }, ("source_names", "axis", "count", "angle_deg"))),
    OperationSpec("gear", "Deterministic CADia gear-family generator; kind-specific fields are structurally discriminated.", GEAR_SCHEMA),
    OperationSpec("shaft", "Stepped shaft with atomic optional keyway group.", _op_obj("shaft", {
        "sections": {"type": "array", "minItems": 1, "items": _obj({"length_mm": POSITIVE, "diameter_mm": POSITIVE}, ("length_mm", "diameter_mm"))},
        "bore_diameter_mm": _nullable(NONNEG), "keyway": _nullable(KEYWAY), "mode": _nullable(GEN_MODE),
    }, ("sections",))),
    OperationSpec("parallel_key", "Parallel key generator.", _op_obj("parallel_key", {
        "width_mm": POSITIVE, "height_mm": POSITIVE, "length_mm": POSITIVE,
        "end_style": _nullable(_enum("square", "round")), "mode": _nullable(GEN_MODE),
    }, ("width_mm", "height_mm", "length_mm"))),
    OperationSpec("bearing", "Rolling bearing/envelope generator.", _op_obj("bearing", {
        "kind": _enum("deep_groove_ball", "cylindrical_roller", "envelope"),
        "bore_diameter_mm": POSITIVE, "outer_diameter_mm": POSITIVE, "width_mm": POSITIVE,
        "rolling_elements": _nullable({"type": "integer", "minimum": 3, "maximum": 200}),
        "detailed": _nullable(BOOL), "mode": _nullable(GEN_MODE),
    }, ("kind", "bore_diameter_mm", "outer_diameter_mm", "width_mm"))),
    OperationSpec("spring", "Compression or Belleville spring with kind-specific required fields.", SPRING_SCHEMA),
    OperationSpec("v_pulley", "V-belt pulley generator.", _op_obj("v_pulley", {
        "pitch_diameter_mm": POSITIVE, "width_mm": POSITIVE,
        "groove_count": _nullable({"type": "integer", "minimum": 1, "maximum": 24}),
        "groove_angle_deg": _nullable({"type": "number", "minimum": 20, "maximum": 80}),
        "bore_diameter_mm": _nullable(NONNEG), "mode": _nullable(GEN_MODE),
    }, ("pitch_diameter_mm", "width_mm"))),
    OperationSpec("coupling", "Flange coupling with atomic bolt-pattern/keyway groups.", _op_obj("coupling", {
        "bore_diameter_mm": POSITIVE, "hub_diameter_mm": POSITIVE, "flange_diameter_mm": POSITIVE,
        "hub_length_mm": POSITIVE, "flange_thickness_mm": POSITIVE,
        "bolt_pattern": _nullable(BOLT_PATTERN), "keyway": _nullable(COUPLING_KEYWAY),
        "mode": _nullable(GEN_MODE),
    }, ("bore_diameter_mm", "hub_diameter_mm", "flange_diameter_mm", "hub_length_mm", "flange_thickness_mm"))),
    OperationSpec("lead_screw_nut", "Lead screw and nut generator.", _op_obj("lead_screw_nut", {
        "major_diameter_mm": POSITIVE, "pitch_mm": POSITIVE, "screw_length_mm": POSITIVE, "nut_length_mm": POSITIVE,
        "starts": _nullable({"type": "integer", "minimum": 1, "maximum": 8}),
        "thread_angle_deg": _nullable({"type": "number", "minimum": 20, "maximum": 60}),
        "nut_outer_diameter_mm": _nullable(POSITIVE), "clearance_mm": _nullable(NONNEG),
        "right_handed": _nullable(BOOL), "mode": _nullable(GEN_MODE),
    }, ("major_diameter_mm", "pitch_mm", "screw_length_mm", "nut_length_mm"))),
    OperationSpec("hex_bolt", "Metric hex bolt generator.", _op_obj("hex_bolt", {
        "diameter_mm": STANDARD_FASTENER_DIAMETER, "length_mm": POSITIVE,
        "pitch_mm": _nullable(POSITIVE), "head_across_flats_mm": _nullable(POSITIVE),
        "head_height_mm": _nullable(POSITIVE), "thread_length_mm": _nullable(NONNEG),
        "right_handed": _nullable(BOOL), "modeled_thread": _nullable(BOOL), "mode": _nullable(GEN_MODE),
    }, ("diameter_mm", "length_mm"))),
    OperationSpec("hex_nut", "Metric hex nut generator.", _op_obj("hex_nut", {
        "diameter_mm": STANDARD_FASTENER_DIAMETER, "pitch_mm": _nullable(POSITIVE),
        "across_flats_mm": _nullable(POSITIVE), "height_mm": _nullable(POSITIVE),
        "right_handed": _nullable(BOOL), "modeled_thread": _nullable(BOOL), "mode": _nullable(GEN_MODE),
    }, ("diameter_mm",))),
    OperationSpec("washer", "Metric washer generator.", _op_obj("washer", {
        "diameter_mm": STANDARD_FASTENER_DIAMETER, "inner_diameter_mm": _nullable(POSITIVE),
        "outer_diameter_mm": _nullable(POSITIVE), "thickness_mm": _nullable(POSITIVE),
        "mode": _nullable(GEN_MODE),
    }, ("diameter_mm",))),
    OperationSpec("external_thread", "Modeled external thread generator.", _op_obj("external_thread", {
        "major_diameter_mm": POSITIVE, "pitch_mm": POSITIVE, "length_mm": POSITIVE,
        "z0_mm": _nullable(NUMBER), "right_handed": _nullable(BOOL), "mode": _nullable(MODE),
    }, ("major_diameter_mm", "pitch_mm", "length_mm"))),
    OperationSpec("coil", "General helical coil primitive.", _op_obj("coil", {
        "pitch_mm": POSITIVE, "height_mm": POSITIVE, "helix_radius_mm": POSITIVE,
        "section_diameter_mm": POSITIVE, "z0_mm": _nullable(NUMBER),
        "right_handed": _nullable(BOOL), "mode": _nullable(MODE),
    }, ("pitch_mm", "height_mm", "helix_radius_mm", "section_diameter_mm"))),
    OperationSpec("sheet_metal_base", "Create CADia sheet-metal base; must be the first body operation.", _op_obj("sheet_metal_base", {
        "width_mm": POSITIVE, "height_mm": POSITIVE, "thickness_mm": POSITIVE,
        "bend_radius_mm": _nullable(NONNEG), "k_factor": _nullable({"type": "number", "minimum": 0, "maximum": 1}),
    }, ("width_mm", "height_mm", "thickness_mm"))),
    OperationSpec("sheet_metal_flange", "Add a flange to a preceding sheet-metal base.", _op_obj("sheet_metal_flange", {
        "edge": _enum("top", "bottom", "left", "right"), "length_mm": POSITIVE,
        "angle_deg": _nullable({"type": "number", "minimum": -179.999, "maximum": 179.999}),
        "bend_radius_mm": _nullable(NONNEG),
    }, ("edge", "length_mm"))),
]

OPERATIONS: dict[str, OperationSpec] = {row.name: row for row in _OPERATION_LIST}


# ---------------------------------------------------------------------------
# Assembly relation registry (same high-level semantics; no paths/binds)
# ---------------------------------------------------------------------------
def _relation(kind: str, props: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return _obj({
        "type": {"type": "string", "const": kind}, "a": STRING, "b": STRING,
        "a_ref": _nullable(ORIGIN_REF), "b_ref": _nullable(ORIGIN_REF), "name": _nullable(STRING),
        **props,
    }, ("type", "a", "b", *required))

RELATION_SCHEMAS: dict[str, dict[str, Any]] = {
    "mate": _relation("mate", {"offset_mm": _nullable(NUMBER)}),
    "flush": _relation("flush", {"offset_mm": _nullable(NUMBER)}),
    "insert": _relation("insert", {"offset_mm": _nullable(NUMBER), "opposed": _nullable(BOOL)}),
    "angle": _relation("angle", {"angle_deg": NUMBER}, ("angle_deg",)),
    "rigid": _relation("rigid", {}),
    "slider": _relation("slider", {"position_mm": _nullable(NUMBER), "start_mm": _nullable(NUMBER), "end_mm": _nullable(NUMBER)}),
    "rotational": _relation("rotational", {"position_deg": _nullable(NUMBER), "start_deg": _nullable(NUMBER), "end_deg": _nullable(NUMBER)}),
    "cylindrical": _relation("cylindrical", {
        "linear_position_mm": _nullable(NUMBER), "linear_start_mm": _nullable(NUMBER), "linear_end_mm": _nullable(NUMBER),
        "angular_position_deg": _nullable(NUMBER), "angular_start_deg": _nullable(NUMBER), "angular_end_deg": _nullable(NUMBER),
    }),
    "planar": _relation("planar", {"angular_position_deg": _nullable(NUMBER), "angular_start_deg": _nullable(NUMBER), "angular_end_deg": _nullable(NUMBER)}),
    "ball": _relation("ball", {}),
}


def _operation_branches() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _OPERATION_LIST:
        if "oneOf" in row.schema:
            out.extend(copy.deepcopy(row.schema["oneOf"]))
        else:
            out.append(copy.deepcopy(row.schema))
    return out


def operation_schema() -> dict[str, Any]:
    return {"oneOf": _operation_branches()}


def program_schema() -> dict[str, Any]:
    component = _obj({
        "name": STRING,
        "operations": {"type": "array", "minItems": 1, "maxItems": 48, "items": operation_schema()},
        "grounded": BOOL, "position_mm": VEC3, "rotation_deg_xyz": VEC3,
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


# CADIA_NEMOTRON_CONVERSATIONAL_TURN_V1
def conversational_turn_schema() -> dict[str, Any]:
    """Return one constrained turn: CAD action, blocking clarification, or grounded answer."""
    action = _obj({
        "response_type": {"type": "string", "const": "action"},
        "program": program_schema(),
        "message": {"type": "string"},
        "options": {"type": "array", "maxItems": 0, "items": STRING},
        "note": {"type": "string"},
    }, ("response_type", "program", "message", "options", "note"))
    clarify = _obj({
        "response_type": {"type": "string", "const": "clarify"},
        "program": {"type": "null"},
        "message": STRING,
        "options": {"type": "array", "maxItems": 6, "items": STRING},
        "note": {"type": "string"},
    }, ("response_type", "program", "message", "options", "note"))
    answer = _obj({
        "response_type": {"type": "string", "const": "answer"},
        "program": {"type": "null"},
        "message": STRING,
        "options": {"type": "array", "maxItems": 0, "items": STRING},
        "note": {"type": "string"},
    }, ("response_type", "program", "message", "options", "note"))
    return {"oneOf": [action, clarify, answer]}


def operations_reference() -> str:
    return "\n".join(f"- {r.name}: {r.description}" for r in _OPERATION_LIST)


# ---------------------------------------------------------------------------
# Semantic audit/validation beyond JSON Schema
# ---------------------------------------------------------------------------
_BODY_MODIFIERS = {"hole", "fillet", "chamfer", "shell", "mirror", "translate", "rotate", "scale", "linear_pattern", "polar_pattern", "sheet_metal_flange"}
_CREATOR_OPS = set(OPERATIONS) - _BODY_MODIFIERS
_PATTERN_OPS = {"linear_pattern", "polar_pattern"}


def _finite_vec3(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in v)


def _nonzero_vec3(v: Any) -> bool:
    return _finite_vec3(v) and math.sqrt(sum(float(x) ** 2 for x in v)) > 1e-12


def _generator_semantics(op: dict[str, Any], cname: str, oname: str) -> None:
    kind = op["op"]
    if kind == "gear":
        g = op["kind"]
        if g in {"helical", "worm", "worm_pair"}:
            key = "helix_angle_deg" if g == "helical" else "lead_angle_deg"
            if abs(float(op[key])) < 1e-9:
                raise ValueError(f"component {cname!r} operation {oname!r}: {g} requires non-zero {key}")
    elif kind == "shaft":
        sections = op["sections"]
        total = sum(float(x["length_mm"]) for x in sections)
        mind = min(float(x["diameter_mm"]) for x in sections)
        bore = float(op.get("bore_diameter_mm") or 0.0)
        if bore and bore >= mind:
            raise ValueError(f"component {cname!r} operation {oname!r}: bore_diameter_mm must be smaller than every shaft section diameter")
        kw = op.get("keyway")
        if kw is not None and float(kw["start_mm"]) + float(kw["length_mm"]) > total + 1e-9:
            raise ValueError(f"component {cname!r} operation {oname!r}: keyway extends beyond total shaft length")
    elif kind == "parallel_key":
        if (op.get("end_style") or "square") == "round" and float(op["length_mm"]) < float(op["width_mm"]):
            raise ValueError(f"component {cname!r} operation {oname!r}: round-end key requires length_mm >= width_mm")
    elif kind == "bearing":
        if float(op["outer_diameter_mm"]) <= float(op["bore_diameter_mm"]):
            raise ValueError(f"component {cname!r} operation {oname!r}: bearing outer_diameter_mm must exceed bore_diameter_mm")
    elif kind == "spring":
        if op["kind"] == "compression":
            wire = float(op["wire_diameter_mm"]); mean = float(op["mean_diameter_mm"]); length = float(op["free_length_mm"]); turns = float(op["active_turns"])
            if mean <= wire:
                raise ValueError(f"component {cname!r} operation {oname!r}: mean_diameter_mm must exceed wire_diameter_mm")
            if length <= wire:
                raise ValueError(f"component {cname!r} operation {oname!r}: free_length_mm must exceed wire_diameter_mm")
            pitch = (length - wire) / turns
            if pitch < wire * 0.55:
                raise ValueError(f"component {cname!r} operation {oname!r}: spring pitch is too small for wire diameter")
        else:
            if float(op["outer_diameter_mm"]) <= float(op["inner_diameter_mm"]):
                raise ValueError(f"component {cname!r} operation {oname!r}: Belleville outer diameter must exceed inner diameter")
    elif kind == "v_pulley":
        pd = float(op["pitch_diameter_mm"]); width = float(op["width_mm"])
        count = int(op.get("groove_count") or 1); angle = float(op.get("groove_angle_deg") or 40.0); bore = float(op.get("bore_diameter_mm") or 0.0)
        depth = max(1.0, pd * 0.05); half = depth * math.tan(math.radians(angle / 2.0)); pitch = max(2.2 * half, width / count)
        if (count - 1) * pitch + 2 * half > width + 1e-9:
            raise ValueError(f"component {cname!r} operation {oname!r}: requested grooves do not fit within width_mm")
        if bore and bore >= pd:
            raise ValueError(f"component {cname!r} operation {oname!r}: bore_diameter_mm must be smaller than pulley root diameter")
    elif kind == "coupling":
        bore = float(op["bore_diameter_mm"]); hub = float(op["hub_diameter_mm"]); flange = float(op["flange_diameter_mm"])
        if not (bore < hub < flange):
            raise ValueError(f"component {cname!r} operation {oname!r}: coupling requires bore < hub < flange diameter")
        bolts = op.get("bolt_pattern")
        if bolts is not None:
            bc = float(bolts["circle_diameter_mm"]); hole = float(bolts["hole_diameter_mm"])
            if not (hub < bc < flange):
                raise ValueError(f"component {cname!r} operation {oname!r}: bolt circle must lie between hub and flange diameters")
            if hole >= (flange - hub) / 2.0:
                raise ValueError(f"component {cname!r} operation {oname!r}: bolt hole diameter is too large for flange annulus")
        kw = op.get("keyway")
        if kw is not None and float(kw["depth_mm"]) >= (hub - bore) / 2.0:
            raise ValueError(f"component {cname!r} operation {oname!r}: keyway depth is too large for hub wall")
    elif kind == "lead_screw_nut":
        if float(op["nut_length_mm"]) >= float(op["screw_length_mm"]):
            raise ValueError(f"component {cname!r} operation {oname!r}: nut_length_mm must be smaller than screw_length_mm")
        od = op.get("nut_outer_diameter_mm")
        if od is not None:
            clear = float(op.get("clearance_mm") or 0.15)
            if float(od) <= float(op["major_diameter_mm"]) + 2 * clear:
                raise ValueError(f"component {cname!r} operation {oname!r}: nut outer diameter is too small for screw plus clearance")


def validate_program(program: dict[str, Any]) -> dict[str, Any]:
    _validate_schema(program_schema(), program, "program")
    program = copy.deepcopy(program)
    component_names: set[str] = set()
    for ci, component in enumerate(program["components"]):
        cname = str(component["name"])
        if cname in component_names:
            raise ValueError(f"program.components[{ci}].name: duplicate component name {cname!r}")
        component_names.add(cname)
        op_names: set[str] = set()
        op_kind_by_name: dict[str, str] = {}
        have_body = False
        have_sheet_base = False
        for oi, op in enumerate(component["operations"]):
            kind = str(op["op"]); oname = str(op["name"])
            if oname in op_names:
                raise ValueError(f"component {cname!r}: duplicate operation name {oname!r}")
            if kind not in OPERATIONS:
                raise ValueError(f"component {cname!r} operation {oname!r}: unsupported op {kind!r}")

            if kind in _BODY_MODIFIERS and not have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: {kind} requires an existing body")
            if kind == "sheet_metal_flange" and not have_sheet_base:
                raise ValueError(f"component {cname!r} operation {oname!r}: sheet_metal_flange requires an earlier sheet_metal_base")

            if kind == "extrude_profile":
                op["profile"] = v9._normalized_profile(op["profile"])
                problem = v9._profile_problem(op["profile"])
                if problem:
                    raise ValueError(f"component {cname!r} operation {oname!r}: {problem}")
            elif kind == "revolve_profile":
                loops = []
                for li, raw in enumerate(op["loops"], 1):
                    loop = copy.deepcopy(raw)
                    if loop["kind"] == "polyline":
                        loop["points"] = v9._normalized_profile(loop["points"])
                        problem = v9._profile_problem(loop["points"])
                        if problem:
                            raise ValueError(f"component {cname!r} operation {oname!r} loop #{li}: {problem}")
                        if any(float(p[0]) < -1e-9 for p in loop["points"]):
                            raise ValueError(f"component {cname!r} operation {oname!r} loop #{li}: radial coordinate must be >= 0")
                    else:
                        if float(loop["center"][0]) - float(loop["radius_mm"]) < -1e-9:
                            raise ValueError(f"component {cname!r} operation {oname!r} loop #{li}: circle crosses revolve axis")
                    loops.append(loop)
                op["loops"] = loops
            elif kind == "torus":
                if float(op["minor_radius_mm"]) >= float(op["major_radius_mm"]):
                    raise ValueError(f"component {cname!r} operation {oname!r}: minor_radius_mm must be < major_radius_mm")
                if op.get("axis") is not None and not _nonzero_vec3(op["axis"]):
                    raise ValueError(f"component {cname!r} operation {oname!r}: axis must be non-zero")
            elif kind in {"cylinder", "cone"}:
                if op.get("axis") is not None and not _nonzero_vec3(op["axis"]):
                    raise ValueError(f"component {cname!r} operation {oname!r}: axis must be non-zero")
            elif kind == "rotate":
                if not _nonzero_vec3(op["axis"]):
                    raise ValueError(f"component {cname!r} operation {oname!r}: axis must be non-zero")
            elif kind == "hole":
                loc = op["location"]
                if loc["kind"] == "feature" and loc["support"] not in op_names:
                    raise ValueError(f"component {cname!r} operation {oname!r}: hole support {loc['support']!r} must reference an earlier operation")
            elif kind in _PATTERN_OPS:
                for source in op["source_names"]:
                    if source not in op_names:
                        raise ValueError(f"component {cname!r} operation {oname!r}: source {source!r} must reference an earlier operation")
            if kind in {"gear", "shaft", "parallel_key", "bearing", "spring", "v_pulley", "coupling", "lead_screw_nut"}:
                _generator_semantics(op, cname, oname)

            explicit_mode = op.get("mode")
            if explicit_mode in {"join", "cut", "intersect"} and not have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: mode={explicit_mode!r} requires an existing body")
            if explicit_mode == "new" and have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: mode='new' after a body exists would discard prior geometry")
            if kind == "sheet_metal_base" and have_body:
                raise ValueError(f"component {cname!r} operation {oname!r}: sheet_metal_base must be the first body operation")

            op_names.add(oname); op_kind_by_name[oname] = kind
            have_body = True
            if kind == "sheet_metal_base":
                have_sheet_base = True

    assembly = program.get("assembly")
    if assembly is not None:
        if len(component_names) < 2:
            raise ValueError("assembly requires at least two components")
        for ri, rel in enumerate(assembly.get("relations") or []):
            if rel["a"] not in component_names or rel["b"] not in component_names:
                raise ValueError(f"assembly.relations[{ri}]: a/b must name declared components")
            if rel["a"] == rel["b"]:
                raise ValueError(f"assembly.relations[{ri}]: a and b must be different components")
            typ = rel["type"]
            if typ == "slider" and rel.get("start_mm") is not None and rel.get("end_mm") is not None and float(rel["start_mm"]) > float(rel["end_mm"]):
                raise ValueError(f"assembly.relations[{ri}]: slider start_mm must be <= end_mm")
            if typ == "rotational" and rel.get("start_deg") is not None and rel.get("end_deg") is not None and float(rel["start_deg"]) > float(rel["end_deg"]):
                raise ValueError(f"assembly.relations[{ri}]: rotational start_deg must be <= end_deg")
    return program


PROGRAM_SYSTEM_PROMPT = f"""You are the NVIDIA Nemotron mechanical CAD planner for CADia.
Return ONE strict JSON program matching the supplied schema. The schema, validator, and compiler are
one audited contract. CADia's deterministic interpreter owns backend tools, topology ids, documents,
files, runtime bindings, and execution bookkeeping.

RULES:
- Understand the user's language directly (including Korean) and preserve explicit dimensions, counts,
  component relationships, and motion limits.
- Never emit raw CADia/Inventor tools, file paths, occurrence names, topology ids, $bind, or $result.
- Prefer deterministic generator operations for standard parts (gear, shaft, key, bearing, spring,
  pulley, coupling, lead screw/nut, fasteners, thread/coil, sheet metal).
- The first body-forming operation uses mode=new or omits mode. Later additive/subtractive creators use
  join/cut/intersect. Body modifiers (hole/fillet/chamfer/shell/mirror/translate/rotate/scale/pattern)
  require an existing body and never start a new body.
- hole.location is EXACTLY one of: world(point_mm), or feature(support + offset_mm). A feature-local
  offset uses the two global transverse axes: ±Z drilling -> [X,Y], ±X -> [Y,Z], ±Y -> [X,Z].
  hole.extent is EXACTLY through, or blind(depth_mm). Never mix depth into a through hole.
- shell.open_normals uses semantic outward principal normals (+X/-X/+Y/-Y/+Z/-Z), never face indices.
- linear_pattern.direction2 is either null/omitted or one complete object {{axis,count,spacing_mm}}.
- gear and spring fields are kind-specific. Do not emit fields from another kind branch.
- shaft.keyway, coupling.bolt_pattern, and coupling.keyway are atomic nested groups; either emit the
  whole group or null/omit it.
- torus is a full closed torus. For partial/open/hollow revolved geometry use revolve_profile.
- Keep programs minimal and dependency-correct. Pattern source_names and feature-local hole supports
  may reference only earlier operation names in the same component.
- Components are independent manufactured parts. Assembly relations use component names and principal
  origin references only; CADia resolves actual document/occurrence identities.

AVAILABLE OPERATIONS:
{operations_reference()}
""".strip()


def repair_prompt(user_request: str, failed_program: dict[str, Any], error: str) -> str:
    return (
        f"ORIGINAL_REQUEST:\n{user_request}\n\nFAILED_PROGRAM:\n"
        f"{json.dumps(failed_program, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"DETERMINISTIC_DIAGNOSTIC:\n{error}\n\n"
        "Return one corrected COMPLETE program matching the same schema. Change the causal operation or its "
        "dependencies; do not repeat the same failed program. Preserve explicit user dimensions/counts unless "
        "the deterministic diagnostic proves a derived finishing value is geometrically impossible."
    )


# ---------------------------------------------------------------------------
# Deterministic compiler onto the EXISTING CADia engine
# ---------------------------------------------------------------------------
_GENERATORS = {
    "gear", "shaft", "parallel_key", "bearing", "spring", "v_pulley", "coupling",
    "lead_screw_nut", "hex_bolt", "hex_nut", "washer", "external_thread", "coil",
    "sheet_metal_base", "sheet_metal_flange",
}


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in d.items() if v is not None}


class CadiaProgramInterpreter(v9.CadiaProgramInterpreter):
    """v10 compiler; geometry execution remains CADia's existing native engine."""

    def _shell_indices_from_normals(self, normals: list[str]) -> list[int]:
        doc = self.engine.doc
        if doc is None or doc.shape is None:
            raise ValueError("shell requires an existing body")
        rows = face_records(doc.shape)
        result: list[int] = []
        axis_index = {"X": 0, "Y": 1, "Z": 2}
        for token in normals:
            candidates = [r for r in rows if str(r.get("geom") or "").upper() == "PLANE" and str(r.get("direction") or "").upper() == token.upper()]
            if not candidates:
                available = sorted({str(r.get("direction")) for r in rows if str(r.get("geom") or "").upper() == "PLANE"})
                raise ValueError(f"shell opening {token} matched no planar face; available principal planar normals={available}")
            idx = axis_index[token[-1].upper()]
            reverse = token.startswith("+")
            chosen = sorted(candidates, key=lambda r: float((r.get("center") or [0, 0, 0])[idx]), reverse=reverse)[0]
            result.append(int(chosen["index"]) + 1)
        if len(result) != len(set(result)):
            raise ValueError("shell open_normals resolved to duplicate faces")
        return result

    @staticmethod
    def _flatten_generator(op: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(op)
        kind = out["op"]
        if kind == "shaft":
            kw = out.pop("keyway", None)
            if kw is not None:
                out.update(keyway_width_mm=kw["width_mm"], keyway_depth_mm=kw["depth_mm"],
                           keyway_length_mm=kw["length_mm"], keyway_start_mm=kw["start_mm"])
        elif kind == "coupling":
            bolts = out.pop("bolt_pattern", None)
            keyway = out.pop("keyway", None)
            if bolts is None:
                out["bolt_count"] = 0
            else:
                out.update(bolt_circle_diameter_mm=bolts["circle_diameter_mm"], bolt_count=bolts["count"],
                           bolt_hole_diameter_mm=bolts["hole_diameter_mm"])
            if keyway is not None:
                out.update(keyway_width_mm=keyway["width_mm"], keyway_depth_mm=keyway["depth_mm"])
        return out

    def _normalize_for_v9_executor(self, op: dict[str, Any]) -> dict[str, Any]:
        kind = op["op"]
        out = copy.deepcopy(op)
        # Strict structured-output providers may materialize optional fields as null.
        # Normalize every default *before* handing the operation to the proven v9
        # native executor so Python None never changes CAD semantics.
        if kind == "extrude_profile":
            out["plane"] = out.get("plane") or "XY Plane"
            out["direction"] = out.get("direction") or "positive"
            out["taper_deg"] = 0.0 if out.get("taper_deg") is None else out["taper_deg"]
        elif kind == "mirror":
            out["base_point_mm"] = out.get("base_point_mm") or [0.0, 0.0, 0.0]
            out["keep_original"] = True if out.get("keep_original") is None else bool(out["keep_original"])
        if kind == "hole":
            loc = out.pop("location"); extent = out.pop("extent")
            if loc["kind"] == "world":
                out.update(center_mm=loc["point_mm"], support_name=None, offset_mm=None)
            else:
                out.update(center_mm=None, support_name=loc["support"], offset_mm=loc["offset_mm"])
            if extent["kind"] == "through":
                out.update(through=True, depth_mm=None)
            else:
                out.update(through=False, depth_mm=extent["depth_mm"])
            return out
        if kind == "shell":
            out["removed_face_indices"] = self._shell_indices_from_normals(out.pop("open_normals"))
            return out
        if kind == "translate":
            return {"op": "transform", "name": out["name"], "translate_mm": out["vector_mm"],
                    "rotate_axis": None, "rotate_deg": None, "scale": None}
        if kind == "rotate":
            return {"op": "transform", "name": out["name"], "translate_mm": None,
                    "rotate_axis": out["axis"], "rotate_deg": out["angle_deg"], "scale": None}
        if kind == "scale":
            return {"op": "transform", "name": out["name"], "translate_mm": None,
                    "rotate_axis": None, "rotate_deg": None, "scale": out["factor"]}
        if kind == "linear_pattern":
            second = out.pop("direction2", None)
            if second is None:
                out.update(direction2=None, count2=None, spacing_mm2=None)
            else:
                out.update(direction2=second["axis"], count2=second["count"], spacing_mm2=second["spacing_mm"])
            return out
        if kind in _GENERATORS:
            return self._flatten_generator(out)
        return out

    def _execute_operation(self, component_name: str, op: dict[str, Any], feature_names: dict[str, str]) -> dict[str, Any]:
        normalized = self._normalize_for_v9_executor(op)
        # CADia's historical create_sphere primitive is an upper hemisphere because
        # CadQuery Solid.makeSphere defaults to a 0..90 degree latitude sweep.  Do
        # not change that backend contract globally: Nemotron v10 compiles one high-
        # level full-sphere op into the existing hemisphere primitive plus a native
        # mirror/fuse across the equatorial XY plane.  This keeps Codex/Copilot and
        # the 118-tool backend untouched while making the v10 IR semantics truthful.
        if normalized["op"] == "sphere":
            first = super()._execute_operation(component_name, normalized, feature_names)
            center = normalized.get("center_mm") or [0.0, 0.0, 0.0]
            mirrored = self.engine.execute("mirror_body", {
                "plane": "XY", "base_point_mm": center, "keep_original": True,
                "name": str(op["name"]),
            })
            after = self._assert_current_shape(label=f"full sphere {op['name']!r}")
            expected = math.pi * float(op["diameter_mm"]) ** 3 / 6.0
            actual = float(after.get("volume_mm3") or 0.0)
            if abs(actual - expected) > max(1e-6, expected * 1e-6):
                raise ValueError(
                    f"sphere {op['name']!r}: native full-sphere compilation volume mismatch; "
                    f"expected={expected:.6g} mm3 actual={actual:.6g} mm3"
                )
            feature_name = mirrored.get("feature_name") if isinstance(mirrored, dict) else None
            if feature_name:
                feature_names[str(op["name"])] = str(feature_name)
            row = dict(mirrored) if isinstance(mirrored, dict) else {"result": mirrored}
            row["geometry"] = after
            row["compiled_from"] = "sphere:hemisphere+mirror"
            row["primitive_result"] = first
            return row
        row = super()._execute_operation(component_name, normalized, feature_names)
        if isinstance(row, dict) and normalized["op"] != op["op"]:
            row = dict(row); row["v10_compiled_op"] = normalized["op"]
        return row

    def _execute_relation(self, rel: dict[str, Any], occurrence_names: dict[str, str]) -> dict[str, Any]:
        # Same null-normalization rule for assembly: optional null means omitted.
        normalized = _strip_none(rel)
        if normalized.get("type") in {"mate", "flush", "insert"}:
            normalized.setdefault("offset_mm", 0.0)
        if normalized.get("type") == "insert":
            normalized.setdefault("opposed", True)
        return super()._execute_relation(normalized, occurrence_names)

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
                        raise v9.ProgramBuildError(
                            f"component {component['name']!r} operation #{oi} {op['op']!r} ({op['name']!r}) failed: {exc}; "
                            f"state before operation: {self._metrics_text(before)}; "
                            f"operation={json.dumps(op, ensure_ascii=False, separators=(',', ':'))}"
                        ) from exc
                    results.append({"component": component["name"], "operation": op["name"], "op": op["op"], "result": row})
                try:
                    metrics[component["name"]] = self._verify_part(component["name"])
                except Exception as exc:
                    raise v9.ProgramBuildError(
                        f"component {component['name']!r} final verification failed: {exc}; final state: {self._metrics_text(self._shape_metrics())}"
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
                        "path": component_paths[component["name"]], "grounded": bool(component["grounded"]),
                        "position_mm": component["position_mm"], "rotation_deg_xyz": component["rotation_deg_xyz"],
                    })
                    occurrence_names[component["name"]] = str(placed["occurrence_name"])
                for ri, rel in enumerate(assembly.get("relations") or [], 1):
                    try:
                        rel_result = self._execute_relation(rel, occurrence_names)
                    except Exception as exc:
                        raise v9.ProgramBuildError(
                            f"assembly relation #{ri} {rel['type']!r} ({rel.get('name') or ri!r}) failed: {exc}; "
                            f"relation={json.dumps(rel, ensure_ascii=False, separators=(',', ':'))}"
                        ) from exc
                    results.append({"relation": rel.get("name") or rel["type"], "result": rel_result})
                health = self.engine.execute("get_assembly_bom", {})
                if len(health.get("occurrences") or []) < len(components):
                    raise ValueError("assembly verification failed: one or more components were not placed")
                session = self.engine._active_session(); suffix = str(getattr(session, "id", ""))[:8] or "assembly"
                self.engine.execute("save_document", {"path": str(self.document_dir / f"nemotron-{v9._slug(assembly['name'])}-{suffix}.scad.json")})
                summary = f"{assembly['name']} complete · {len(components)} components · {len(assembly.get('relations') or [])} relations"
            else:
                last = components[-1]["name"]; summary = f"{last} complete · {metrics[last]['feature_count']} features · {metrics[last]['volume_mm3']:,.3f} mm³"

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
    def __init__(self, *, provider, model: str, reasoning: str = "medium") -> None:
        self.provider = provider; self.model = model; self.reasoning = reasoning
        self.last_program: dict[str, Any] | None = None

    def cancel(self) -> None:
        self.provider.cancel()

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        # Full feature/parameter/assembly/selection context is useful for conversation; raw topology arrays are too large.
        state_summary = {key: value for key, value in (current_state or {}).items() if key not in {"faces", "edges"}}
        interaction_rules = (
            "\n\nCONVERSATIONAL_TURN_RULES:\n"
            "Choose exactly one response_type: action, clarify, or answer. "
            "Use action when CAD should change. Ask clarify only when a missing or ambiguous fact is genuinely blocking and materially changes the result. "
            "Do not ask for confirmation merely because a change is complex, and do not ask for facts already resolved by current state, selection, prior conversation, or a safe conventional default. "
            "Ask at most 3 concise questions in one message and provide short options when useful. "
            "Use answer for design/model questions that require no CAD mutation. "
            "When the user answers a previous clarification, combine that answer with prior conversation and execute once enough information is available."
        )
        prompt = (
            PROGRAM_SYSTEM_PROMPT + interaction_rules
            + "\n\nCURRENT_STATE_SUMMARY:\n" + json.dumps(state_summary, ensure_ascii=False, separators=(",", ":"))
            + "\n\nUSER_REQUEST:\n" + str(user_prompt or "").strip()
        )
        turn = self.provider.complete_cad_program(prompt, schema=conversational_turn_schema(), model=self.model,
                                                  effort=self.reasoning, on_event=on_event, repair=False)
        _validate_schema(conversational_turn_schema(), turn, "turn")
        if str(turn.get("response_type") or "") == "action":
            program = validate_program(turn["program"])
            self.last_program = copy.deepcopy(program)
            turn = copy.deepcopy(turn)
            turn["program"] = program
        return copy.deepcopy(turn)

    def repair(self, user_prompt: str, error: str, on_event=None) -> dict[str, Any]:
        if self.last_program is None:
            raise ValueError("No Nemotron CAD program is available to repair")
        failed = copy.deepcopy(self.last_program)
        prompt = PROGRAM_SYSTEM_PROMPT + "\n\n" + repair_prompt(user_prompt, failed, error)
        repaired = self.provider.complete_cad_program(prompt, schema=program_schema(), model=self.model,
                                                      effort=self.reasoning, on_event=on_event, repair=True)
        repaired = validate_program(repaired)
        if repaired == failed:
            raise RepairStalledError("Nemotron returned the same high-level CAD program")
        self.last_program = copy.deepcopy(repaired)
        return repaired


def _try_direct_cadia_route(*, runtime, user_prompt: str, current_state: dict[str, Any], on_event=None):
    plan = direct_plan(user_prompt, current_state)
    if plan is None:
        return None
    if on_event:
        on_event("progress", {"message": "Using CADia deterministic intent/generator route…", "percent": 20})
    executed = runtime.executor.execute(plan, on_event=on_event, progress_span=(22, 72))
    if not executed.ok:
        return None
    runtime.persist()
    return {"ok": True, "message": executed.message, "program": {"route": "cadia_deterministic", "note": plan.get("note", "")},
            "results": executed.results, "metrics": {}}


def execute_nemotron_program_with_repair(*, runtime, agent: NemotronProgramAgent, user_prompt: str,
                                         current_state: dict[str, Any], on_event=None,
                                         max_repairs: int = 1) -> dict[str, Any]:
    direct = _try_direct_cadia_route(runtime=runtime, user_prompt=user_prompt, current_state=current_state, on_event=on_event)
    if direct is not None:
        return direct
    turn = agent.plan(user_prompt, current_state, on_event=on_event)
    response_type = str(turn.get("response_type") or "action")
    if response_type in {"clarify", "answer"}:
        message = str(turn.get("message") or "").strip()
        options = [str(x).strip() for x in (turn.get("options") or []) if str(x).strip()][:6]
        display = message
        if response_type == "clarify" and options:
            display += "\n" + "\n".join(f"{i + 1}. {value}" for i, value in enumerate(options))
        return {
            "ok": True,
            "message": display,
            "interaction": {"type": response_type, "message": display, "options": options},
            "program": {"route": "conversation", "note": str(turn.get("note") or "")},
            "results": [],
            "metrics": {},
        }
    program = turn.get("program")
    if not isinstance(program, dict):
        raise ValueError("Nemotron action turn did not contain a CAD program")
    interpreter = CadiaProgramInterpreter(runtime)
    first_error: Exception | None = None; last_error: Exception | None = None
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
                on_event("progress", {"message": "Deterministic CAD verification failed · requesting one schema-valid correction…", "percent": 86})
            try:
                program = agent.repair(user_prompt, str(exc), on_event=on_event)
            except RepairStalledError:
                break
    err = last_error or first_error or RuntimeError("unknown structured CAD failure")
    first = first_error or err
    suffix = " Nemotron did not produce a different executable program." if last_error is first_error else ""
    raise RuntimeError(f"Modeling failed: {first}.{suffix}") from err


__all__ = [
    "OPERATIONS", "RELATION_SCHEMAS", "PROGRAM_SYSTEM_PROMPT", "program_schema", "operation_schema",
    "validate_program", "CadiaProgramInterpreter", "NemotronProgramAgent", "execute_nemotron_program_with_repair",
]
