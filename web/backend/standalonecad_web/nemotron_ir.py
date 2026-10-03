from __future__ import annotations

"""Nemotron-facing CAD contract.

The direct Nemotron path deliberately does not expose CADia's entire raw backend
surface unchanged.  It follows the same reliability ideas used by public CAD-agent
projects: a closed/typed model-facing vocabulary, deterministic translation to the
real CAD backend, a smaller relevant tool set per request, and validation before any
kernel mutation.

This module is intentionally generic.  It contains no part-, prompt-, dimension-, or
benchmark-specific routing.
"""

from dataclasses import dataclass
import copy
import math
from pathlib import Path
import re
from typing import Any, Iterable

from standalonecad.planning import _validate_value


IR_PREFIX = "cadia_ir_"

# Always-visible operations are lifecycle, generic sketch/feature construction,
# assembly observation, and topology inspection.  The remaining live CAD tools are
# keyword-reranked from the current user request, following the same broad pattern as
# FreeCAD AI's public tool-reranking design.
PINNED_REAL_TOOLS = {
    "inventor_new_part",
    "inventor_new_assembly",
    "inventor_open_document",
    "inventor_save_document",
    "inventor_list_open_documents",
    "inventor_get_document_info",
    "inventor_create_sketch",
    "inventor_draw_line",
    "inventor_draw_circle",
    "inventor_draw_rectangle",
    "inventor_draw_arc",
    "inventor_add_sketch_dimension",
    "inventor_add_sketch_constraint",
    "inventor_close_sketch",
    "inventor_revolve",
    "inventor_hole",
    "inventor_fillet",
    "inventor_chamfer",
    "inventor_place_occurrence",
    "inventor_list_interfaces",
    "inventor_list_constraints",
    "inventor_list_joints",
    "inventor_check_interference",
    "inventor_get_assembly_bom",
    "cad_get_topology",
    "cad_get_selection",
    "cad_undo",
}

# Raw tools replaced by unambiguous model-facing IR operations.  The backend tools
# remain untouched and are still used after deterministic translation.
HIDDEN_RAW_TOOLS = {
    "inventor_extrude",
    "cad_extrude_advanced",
    "inventor_add_joint",
    "inventor_set_joint_limits",
    "inventor_drive_joint",
    "inventor_add_constraint",
}


# Model-visible result contracts.  CADia's MCP input schemas historically describe
# arguments only, but a multi-step structured planner also needs to know which exact
# values an earlier call returns.  These contracts document stable public result fields
# already produced by the existing backend; they do not change tool behavior.
#
# Runtime references use the executor's safe JSON-object path syntax:
#   $bind.<name>.field
#   $bind.<name>.items[0].field
# The deterministic executor resolves these after the producing tool actually runs.
RESULT_CONTRACTS: dict[str, dict[str, Any]] = {
    "inventor_new_part": {
        "outputSchema": {"type": "object", "properties": {"document_type": {"type": "string"}, "title": {"type": "string"}}},
        "referenceHint": "Bind only if you need a returned field; this call does not produce a file path. Save the document to obtain .path.",
    },
    "inventor_new_assembly": {
        "outputSchema": {"type": "object", "properties": {"document_type": {"type": "string"}, "title": {"type": "string"}}},
        "referenceHint": "This call does not produce a file path. Save the assembly to obtain .path.",
    },
    "inventor_create_sketch": {
        "outputSchema": {"type": "object", "properties": {"sketch_name": {"type": "string"}, "plane": {"type": "string"}}},
        "referenceHint": "Later sketch-name inputs may use $bind.<name>.sketch_name.",
    },
    "inventor_draw_line": {
        "outputSchema": {"type": "object", "properties": {"entity_id": {"type": "string"}}},
        "referenceHint": "inventor_add_sketch_dimension.entityId must use this .entity_id (not a face/edge topology id).",
    },
    "inventor_draw_circle": {
        "outputSchema": {"type": "object", "properties": {"entity_id": {"type": "string"}}},
        "referenceHint": "inventor_add_sketch_dimension.entityId must use this .entity_id (not a face/edge topology id).",
    },
    "inventor_draw_arc": {
        "outputSchema": {"type": "object", "properties": {"entity_id": {"type": "string"}}},
        "referenceHint": "inventor_add_sketch_dimension.entityId must use this .entity_id (not a face/edge topology id).",
    },
    "inventor_draw_rectangle": {
        "outputSchema": {"type": "object", "properties": {"entity_ids": {"type": "array", "items": {"type": "string"}}}},
        "referenceHint": "Rectangle edges are returned as .entity_ids[index].",
    },
    "inventor_close_sketch": {
        "outputSchema": {"type": "object", "properties": {"sketch_name": {"type": "string"}}},
        "referenceHint": "Later feature sketch-name inputs may use $bind.<name>.sketch_name.",
    },
    "inventor_save_document": {
        "outputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        "referenceHint": "For a newly created document, use path:null and bind this call; place/open later with $bind.<name>.path.",
    },
    "inventor_place_occurrence": {
        "outputSchema": {"type": "object", "properties": {"occurrence_name": {"type": "string"}}},
        "referenceHint": "Later assembly relations must use $bind.<name>.occurrence_name rather than guessing Part:1.",
    },
    "cad_get_topology": {
        "outputSchema": {
            "type": "object",
            "properties": {
                "faces": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "geom": {"type": "string"}, "radius_mm": {"type": "number"}}}},
                "edges": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "geom": {"type": "string"}}}},
                "vertices": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}},
            },
        },
        "referenceHint": "Use .faces[i].id only for face/topology inputs, .edges[i].id only for edge inputs; never use topology ids as sketch entityId values.",
    },
}


def _with_result_contract(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    name = str(out.get("name") or "")
    contract = RESULT_CONTRACTS.get(name)
    if contract:
        out.update(copy.deepcopy(contract))
    return out


@dataclass(frozen=True)
class IRMapping:
    ir_name: str
    real_name: str
    fixed_arguments: dict[str, Any]


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    text = text.replace("_", " ").replace("-", " ").lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) >= 2}


def _keyword_rank(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Pure-Python IDF-weighted lexical ranking over name + description.

    This is deterministic, costs no extra model call, and is deliberately simple so
    filtering can never mutate CAD behavior; it only changes which schemas Nemotron
    sees on that planning turn.
    """
    docs: list[tuple[dict[str, Any], set[str]]] = []
    df: dict[str, int] = {}
    for row in rows:
        tokens = _tokenize(str(row.get("name") or "") + " " + str(row.get("description") or ""))
        docs.append((row, tokens))
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    q = _tokenize(query)
    n = max(1, len(rows))
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row, tokens in docs:
        score = 0.0
        for t in q & tokens:
            score += math.log((n + 1.0) / (df.get(t, 0) + 1.0)) + 1.0
        name = str(row.get("name") or "")
        scored.append((score, name, row))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row for _, _, row in scored]


def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy((schema or {}).get("properties") or {})


def _required(schema: dict[str, Any]) -> list[str]:
    return [str(x) for x in ((schema or {}).get("required") or [])]


def _object_schema(properties: dict[str, Any], required: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _find(full_catalog: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in full_catalog:
        if row.get("name") == name:
            return row
    raise KeyError(name)


def _joint_base_properties(full_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    schema = _find(full_catalog, "inventor_add_joint").get("inputSchema") or {}
    props = _schema_properties(schema)
    return props


def _joint_schema(full_catalog: list[dict[str, Any]], joint_type: str) -> dict[str, Any]:
    props = _joint_base_properties(full_catalog)
    # Type is carried by the IR operation name, not by model arguments.  This is the
    # discriminated-union equivalent used by the translation layer.
    keep = {
        "a", "b", "name",
        "linear_position_mm", "angular_position_deg",
        "flip_origin_direction", "flip_alignment_direction",
    }
    if joint_type in {"slider", "cylindrical"}:
        keep.update({"linear_start_mm", "linear_end_mm"})
    if joint_type in {"rotational", "cylindrical", "planar"}:
        keep.update({"angular_start_deg", "angular_end_deg"})
    if joint_type == "ball":
        # Ball has three rotational DOF; scalar linear/angular positions are not a
        # meaningful coordinate in the backend joint contract.
        keep.discard("linear_position_mm")
        keep.discard("angular_position_deg")
    selected = {k: v for k, v in props.items() if k in keep}
    return _object_schema(selected, ("a", "b"))


def _canonical_rows(full_catalog: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, IRMapping]]:
    """Build the fixed IR vocabulary and deterministic backend mappings."""
    out: list[dict[str, Any]] = []
    mappings: dict[str, IRMapping] = {}

    # One extrusion contract covers new/join/cut/intersect, so the model never has to
    # choose between two overlapping backend extrusion APIs.
    adv = _find(full_catalog, "cad_extrude_advanced")
    name = IR_PREFIX + "extrude"
    out.append({
        "name": name,
        "description": "Create or boolean an extrusion from a closed sketch. operation may be new, join, cut, or intersect.",
        "inputSchema": copy.deepcopy(adv.get("inputSchema") or {}),
    })
    mappings[name] = IRMapping(name, "cad_extrude_advanced", {})

    # Kinematic joints are separate IR operations.  Invalid DOF/limit combinations are
    # therefore unrepresentable in the model-facing schema rather than repaired later.
    for typ in ("rigid", "rotational", "slider", "cylindrical", "planar", "ball"):
        ir_name = IR_PREFIX + "joint_" + typ
        dof = {
            "rigid": "0 translation / 0 rotation",
            "rotational": "one free rotation about the joint axis",
            "slider": "one free translation along the joint axis",
            "cylindrical": "free axial translation plus free axial rotation",
            "planar": "two in-plane translations plus rotation about the plane normal",
            "ball": "three free rotations about one origin",
        }[typ]
        out.append({
            "name": ir_name,
            "description": f"Create a {typ} assembly joint ({dof}). The schema exposes only positions/limits valid for this joint DOF.",
            "inputSchema": _joint_schema(full_catalog, typ),
        })
        mappings[ir_name] = IRMapping(ir_name, "inventor_add_joint", {"type": typ})

    # Limit/drive operations are split by free coordinate, again making the invalid
    # combinations unrepresentable before the plan reaches the kernel.
    limit_real = _find(full_catalog, "inventor_set_joint_limits").get("inputSchema") or {}
    limit_props = _schema_properties(limit_real)
    linear_limit = IR_PREFIX + "set_linear_joint_limits"
    angular_limit = IR_PREFIX + "set_angular_joint_limits"
    out.extend([
        {
            "name": linear_limit,
            "description": "Set min/max travel on a joint with a free linear coordinate (slider or cylindrical).",
            "inputSchema": _object_schema({k: limit_props[k] for k in ("name", "linear_start_mm", "linear_end_mm")}, ("name",)),
        },
        {
            "name": angular_limit,
            "description": "Set min/max travel on a joint with a free angular coordinate (rotational, cylindrical, or planar).",
            "inputSchema": _object_schema({k: limit_props[k] for k in ("name", "angular_start_deg", "angular_end_deg")}, ("name",)),
        },
    ])
    mappings[linear_limit] = IRMapping(linear_limit, "inventor_set_joint_limits", {})
    mappings[angular_limit] = IRMapping(angular_limit, "inventor_set_joint_limits", {})

    drive_real = _find(full_catalog, "inventor_drive_joint").get("inputSchema") or {}
    drive_props = _schema_properties(drive_real)
    drive_linear = IR_PREFIX + "drive_linear_joint"
    drive_angular = IR_PREFIX + "drive_angular_joint"
    out.extend([
        {
            "name": drive_linear,
            "description": "Drive the free linear coordinate of a slider or cylindrical joint to a position.",
            "inputSchema": _object_schema({k: drive_props[k] for k in ("name", "linear_position_mm")}, ("name", "linear_position_mm")),
        },
        {
            "name": drive_angular,
            "description": "Drive the free angular coordinate of a rotational, cylindrical, or planar joint to an angle.",
            "inputSchema": _object_schema({k: drive_props[k] for k in ("name", "angular_position_deg")}, ("name", "angular_position_deg")),
        },
    ])
    mappings[drive_linear] = IRMapping(drive_linear, "inventor_drive_joint", {})
    mappings[drive_angular] = IRMapping(drive_angular, "inventor_drive_joint", {})

    # Static assembly constraints are also a discriminated union in the backend.
    # Split them into typed IR operations so angle/insert-only fields cannot leak into
    # mate/flush calls merely because they coexist in one permissive raw schema.
    constraint_real = _find(full_catalog, "inventor_add_constraint").get("inputSchema") or {}
    constraint_props = _schema_properties(constraint_real)
    constraint_specs = {
        "mate": (("a", "b", "offset_mm"), ("a", "b")),
        "flush": (("a", "b", "offset_mm"), ("a", "b")),
        "insert": (("a", "b", "offset_mm", "insert_opposed"), ("a", "b")),
        "angle": (("a", "b", "angle_deg"), ("a", "b", "angle_deg")),
    }
    for typ, (keys, required) in constraint_specs.items():
        ir_name = IR_PREFIX + "constraint_" + typ
        out.append({
            "name": ir_name,
            "description": f"Create a typed {typ} static assembly constraint with only fields valid for that relation.",
            "inputSchema": _object_schema({k: constraint_props[k] for k in keys}, required),
        })
        mappings[ir_name] = IRMapping(ir_name, "inventor_add_constraint", {"type": typ})
    return out, mappings


def build_ir_catalog(
    full_catalog: list[dict[str, Any]],
    query: str,
    *,
    top_n: int = 28,
) -> tuple[list[dict[str, Any]], dict[str, IRMapping]]:
    """Return a request-relevant closed vocabulary plus IR-to-backend mappings.

    No backend capability is deleted.  Filtering is model visibility only: the
    deterministic executor still owns the full CAD tool set after IR translation.
    """
    canonical, mappings = _canonical_rows(full_catalog)
    canonical = [_with_result_contract(r) for r in canonical]
    raw = [_with_result_contract(r) for r in full_catalog if str(r.get("name") or "") not in HIDDEN_RAW_TOOLS]

    pinned: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for row in raw:
        if row.get("name") in PINNED_REAL_TOOLS:
            pinned.append(row)
        else:
            rest.append(row)

    ranked = _keyword_rank(rest, query)
    chosen = pinned + ranked[: max(0, int(top_n))]

    # Canonical IR operations are always available.  De-duplicate deterministically.
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in canonical + chosen:
        name = str(row.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(row)
    return result, mappings


_BIND_RUNTIME_RE = re.compile(r'^\$bind\.([A-Za-z_][A-Za-z0-9_-]{0,63})(.*)$')
_RESULT_RUNTIME_RE = re.compile(r'^\$result\[(\d+)\](.*)$')


def _iter_runtime_refs(value: Any, path: str):
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith('$bind.') or raw.startswith('$result['):
            yield path, raw
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from _iter_runtime_refs(item, f'{path}[{i}]')
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_runtime_refs(item, f'{path}.{key}')


def _reference_semantic(ref: str) -> str | None:
    """Infer the semantic namespace of a documented result path.

    The suffix vocabulary comes from stable backend result fields, not prompt-specific
    object names.  Unknown paths remain unknown rather than being guessed.
    """
    m = _BIND_RUNTIME_RE.fullmatch(ref) or _RESULT_RUNTIME_RE.fullmatch(ref)
    if not m:
        return None
    suffix = m.group(2) or ''
    if re.fullmatch(r'\.entity_id', suffix) or re.fullmatch(r'\.entity_ids\[\d+\]', suffix):
        return 'sketch_entity_id'
    if re.fullmatch(r'\.faces\[\d+\]\.id', suffix):
        return 'face_id'
    if re.fullmatch(r'\.edges\[\d+\]\.id', suffix):
        return 'edge_id'
    if re.fullmatch(r'\.vertices\[\d+\]\.id', suffix):
        return 'vertex_id'
    if suffix == '.occurrence_name':
        return 'occurrence_name'
    if suffix == '.path':
        return 'document_path'
    if suffix == '.sketch_name':
        return 'sketch_name'
    if suffix == '.feature_name':
        return 'feature_name'
    return None


def _expected_reference_semantic(tool: str, argument_path: str) -> str | None:
    leaf = argument_path.rsplit('.', 1)[-1]
    if leaf == 'entityId':
        return 'sketch_entity_id'
    if leaf in {'face_ref', 'target_face_ref'} or re.search(r'\.remove_face_ids\[\d+\]$', argument_path):
        return 'face_id'
    if leaf in {'edge_ref', 'edge_id'} or re.search(r'\.(?:edge_ids|edges)\[\d+\]$', argument_path):
        return 'edge_id'
    if leaf in {'occurrence', 'occurrence_name', 'a_occurrence', 'b_occurrence'}:
        return 'occurrence_name'
    if leaf in {'sketchName', 'sketch_name', 'profile_sketch_name'}:
        return 'sketch_name'
    if leaf == 'id' and '.selector.' in argument_path:
        return 'face_id'
    if tool in {'inventor_place_occurrence', 'inventor_open_document'} and leaf == 'path':
        return 'document_path'
    return None


def _parse_ir_runtime_ref(ref: str) -> tuple[str, str | int, list[str | int]]:
    raw = str(ref or '').strip()
    m = _BIND_RUNTIME_RE.fullmatch(raw)
    if m:
        source, root, suffix = 'bind', m.group(1), m.group(2) or ''
    else:
        m = _RESULT_RUNTIME_RE.fullmatch(raw)
        if not m:
            raise ValueError(f'unsupported runtime reference syntax: {ref}')
        source, root, suffix = 'result', int(m.group(1)), m.group(2) or ''
    tokens: list[str | int] = []
    pos = 0
    token_re = re.compile(r'(?:\.([A-Za-z_][A-Za-z0-9_-]*))|(?:\[(\d+)\])')
    while pos < len(suffix):
        token = token_re.match(suffix, pos)
        if token is None:
            raise ValueError(f'unsupported runtime reference syntax: {ref}')
        tokens.append(token.group(1) if token.group(1) is not None else int(token.group(2)))
        pos = token.end()
    return source, root, tokens


def _output_schema_accepts_path(schema: dict[str, Any] | None, tokens: list[str | int]) -> bool:
    node: Any = schema or {}
    for token in tokens:
        if not isinstance(node, dict):
            return False
        if isinstance(token, int):
            if node.get('type') != 'array':
                return False
            node = node.get('items') or {}
            continue
        props = node.get('properties') or {}
        if token not in props:
            return False
        node = props[token]
    return True


def _validate_ir_runtime_reference_semantics(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> None:
    """Reject unresolved/cross-namespace runtime references before CAD execution.

    The planner may only consume result fields explicitly documented by an earlier
    producer's ``outputSchema``.  This makes same-plan dataflow a typed contract rather
    than an arbitrary string convention, while keeping CADia's backend tools unchanged.
    """
    calls = plan.get('calls') or []
    rows = {str(row.get('name') or ''): row for row in catalog if row.get('name')}
    binds: dict[str, tuple[int, str]] = {}

    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        tool = str(call.get('tool') or '')
        for arg_path, ref in _iter_runtime_refs(call.get('arguments') or {}, 'arguments'):
            source, root, tokens = _parse_ir_runtime_ref(ref)
            if source == 'bind':
                producer = binds.get(str(root))
                if producer is None:
                    raise ValueError(
                        f'calls[{idx}].{tool}.{arg_path}: runtime binding {root!r} is not an earlier bound call.'
                    )
                producer_idx, producer_tool = producer
            else:
                producer_idx = int(root) - 1
                if producer_idx < 0 or producer_idx >= idx:
                    raise ValueError(
                        f'calls[{idx}].{tool}.{arg_path}: runtime result reference must point to an earlier call: {ref}'
                    )
                producer_call = calls[producer_idx]
                producer_tool = str(producer_call.get('tool') or '') if isinstance(producer_call, dict) else ''

            producer_row = rows.get(producer_tool) or {}
            output_schema = producer_row.get('outputSchema')
            if not isinstance(output_schema, dict) or not _output_schema_accepts_path(output_schema, tokens):
                raise ValueError(
                    f'calls[{idx}].{tool}.{arg_path}: {ref} is not a documented result path of '
                    f'calls[{producer_idx}].{producer_tool}. Use only outputSchema/referenceHint fields from TOOL_CATALOG.'
                )

            expected = _expected_reference_semantic(tool, arg_path)
            if expected is not None:
                actual = _reference_semantic(ref)
                if actual != expected:
                    raise ValueError(
                        f'calls[{idx}].{tool}.{arg_path}: typed runtime reference mismatch: '
                        f'expected {expected}, got {actual or "unknown"} from {ref}. '
                        'Use only documented outputSchema/referenceHint fields from an earlier bound call.'
                    )

        bind = call.get('bind')
        if isinstance(bind, str) and bind:
            if bind in binds:
                raise ValueError(f'calls[{idx}].bind duplicates earlier runtime binding {bind!r}.')
            binds[bind] = (idx, tool)


def validate_ir_plan(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("IR plan must be an object")
    calls = plan.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("IR plan must contain at least one call")
    schemas = {str(r.get("name")): r.get("inputSchema") or {} for r in catalog if r.get("name")}
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"IR call {idx} must be an object")
        tool = str(call.get("tool") or "")
        if tool not in schemas:
            raise ValueError(f"IR call {idx} uses a tool outside the current closed vocabulary: {tool!r}")
        args = call.get("arguments")
        if not isinstance(args, dict):
            raise ValueError(f"IR call {idx} arguments must be an object")
        _validate_value(schemas[tool], args, f"calls[{idx}].{tool}.arguments")
    _validate_ir_runtime_reference_semantics(plan, catalog)
    return plan


def normalize_ir_plan(plan: dict[str, Any], mappings: dict[str, IRMapping]) -> dict[str, Any]:
    """Translate model-facing IR calls to the existing CADia backend contract."""
    out = copy.deepcopy(plan)
    for call in out.get("calls") or []:
        name = str(call.get("tool") or "")
        mapping = mappings.get(name)
        if mapping is None:
            continue
        args = copy.deepcopy(call.get("arguments") or {})
        args.update(copy.deepcopy(mapping.fixed_arguments))
        call["tool"] = mapping.real_name
        call["arguments"] = args
    return out


def catalog_names(catalog: list[dict[str, Any]]) -> set[str]:
    return {str(r.get("name")) for r in catalog if r.get("name")}

# ---------------------------------------------------------------------------
# Deterministic multi-document lifecycle compiler
# ---------------------------------------------------------------------------
# Public CAD-agent projects that use a compact IR do not ask the language model to
# manually maintain application/UI document focus as if it were a human clicking tabs.
# CADia follows that pattern here: the model still describes CAD operations, while this
# compiler preserves the active-document invariant required by the existing backend.
# It is generic across all parts/assemblies and contains no component- or benchmark-
# specific branches.

ASSEMBLY_CONTEXT_TOOLS = {
    "inventor_place_occurrence",
    "inventor_add_constraint",
    "inventor_get_assembly_bom",
    "inventor_list_constraints",
    "inventor_check_interference",
    "inventor_add_joint",
    "inventor_edit_joint",
    "inventor_set_joint_limits",
    "inventor_drive_joint",
    "inventor_list_joints",
    "inventor_delete_joint",
    "inventor_transform_occurrence",
    "inventor_edit_constraint",
    "inventor_delete_constraint",
    "inventor_delete_occurrence",
}

_DOCUMENT_SWITCH_TOOLS = {
    "inventor_new_part",
    "inventor_new_assembly",
    "inventor_open_document",
    "inventor_close_document",
}


def _is_runtime_ref(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(("$bind.", "$result["))


def _is_readish_tool(name: str) -> bool:
    return (
        name.startswith("inventor_list_")
        or name.startswith("inventor_get_")
        or name.startswith("cad_get_")
        or name in {"inventor_health", "inventor_check_interference", "inventor_measure_min_distance"}
    )


def _initial_document_context(current_state: dict[str, Any] | None) -> tuple[str | None, str | None, dict[str, str], str | None]:
    state = current_state or {}
    active_type = None
    active_path = None
    known: dict[str, str] = {}
    last_assembly_path = None

    doc = state.get("document") or {}
    if isinstance(doc, dict):
        active_type = str(doc.get("type") or "") or None

    rows = state.get("open_documents") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            path = row.get("path")
            typ = str(row.get("type") or "") or None
            if isinstance(path, str) and path.strip() and typ:
                resolved = str(Path(path).expanduser())
                known[resolved] = typ
                if typ == "assembly":
                    last_assembly_path = resolved
                if row.get("active"):
                    active_path = resolved
                    active_type = typ
    return active_type, active_path, known, last_assembly_path


def _allocate_document_path(document_dir: Path, doc_type: str, reserved: set[str]) -> str:
    document_dir = Path(document_dir).resolve()
    stem = "assembly" if doc_type == "assembly" else "part"
    index = 1
    while True:
        candidate = str((document_dir / f"nemotron_auto_{stem}_{index:03d}.scad.json").resolve())
        if candidate not in reserved and not Path(candidate).exists():
            reserved.add(candidate)
            return candidate
        index += 1


def _rewrite_legacy_result_refs_to_bindings(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stabilize legacy $result[N] references before lifecycle calls are inserted.

    Lifecycle compilation may insert save/open calls.  Raw positional $result[N]
    references would then point at the wrong result object.  Convert every referenced
    original result to a named binding first, preserving exactly the same producer and
    path while making later lifecycle insertion index-independent.
    """
    out = copy.deepcopy(calls)
    used = {
        str(call.get("bind"))
        for call in out
        if isinstance(call, dict) and isinstance(call.get("bind"), str) and call.get("bind")
    }
    result_to_bind: dict[int, str] = {}

    def collect(value: Any) -> None:
        if isinstance(value, str):
            m = _RESULT_RUNTIME_RE.fullmatch(value.strip())
            if m:
                result_to_bind.setdefault(int(m.group(1)), "")
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)

    for call in out:
        if isinstance(call, dict):
            collect(call.get("arguments") or {})

    for result_index in sorted(result_to_bind):
        if result_index < 1 or result_index > len(out):
            raise ValueError(f"Legacy runtime result reference is out of range before lifecycle compilation: $result[{result_index}]")
        producer = out[result_index - 1]
        if not isinstance(producer, dict):
            raise ValueError(f"Legacy runtime result producer {result_index} is not a CAD call")
        existing = producer.get("bind")
        if isinstance(existing, str) and existing:
            bind = existing
        else:
            base = f"__cadia_result_{result_index}"
            bind = base
            suffix = 1
            while bind in used:
                suffix += 1
                bind = f"{base}_{suffix}"
            producer["bind"] = bind
            used.add(bind)
        result_to_bind[result_index] = bind

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            raw = value.strip()
            m = _RESULT_RUNTIME_RE.fullmatch(raw)
            if not m:
                return value
            idx = int(m.group(1))
            bind = result_to_bind.get(idx)
            if not bind:
                return value
            return f"$bind.{bind}{m.group(2) or ''}"
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        return value

    for call in out:
        if isinstance(call, dict):
            call["arguments"] = rewrite(call.get("arguments") or {})
    return out


def compile_document_lifecycle(
    plan: dict[str, Any],
    *,
    document_dir: Path,
    current_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile model-level work into a valid multi-document CADia execution order.

    This is a document-provenance compiler, not a feature repair pass.  It tracks which
    editable document produced every named runtime result, preserves documents before
    context switches, restores assembly context for assembly-only operations, and
    restores the producer document when a later operation consumes a document-bound
    runtime value such as a sketch entity, sketch name, face/edge id, feature name, or
    occurrence name.

    The compiler is generic across all parts/assemblies and contains no component,
    benchmark, dimension, feature, or joint-specific branch.
    """
    out = copy.deepcopy(plan)
    raw_calls = list(out.get("calls") or [])
    if not raw_calls:
        return out

    # Lifecycle insertion changes positional call numbers.  Named bindings are stable.
    calls = _rewrite_legacy_result_refs_to_bindings(raw_calls)

    needs_workspace = any(
        str(c.get("tool") or "") == "inventor_new_assembly"
        or str(c.get("tool") or "") in ASSEMBLY_CONTEXT_TOOLS
        for c in calls if isinstance(c, dict)
    )
    if not needs_workspace:
        out["calls"] = calls
        return out

    initial_type, initial_path, known_types, initial_last_assembly_path = _initial_document_context(current_state)
    reserved = set(known_types)
    compiled: list[dict[str, Any]] = []

    # Mutable context objects let bindings keep document provenance even when an
    # unsaved document later receives a path before a switch.
    next_doc_id = 1
    contexts_by_path: dict[str, dict[str, Any]] = {}

    def make_ctx(doc_type: str | None, path: str | None, *, created: bool, dirty: bool = False, initial_has_geometry: bool = False) -> dict[str, Any]:
        nonlocal next_doc_id
        ctx = {
            "id": next_doc_id,
            "type": doc_type,
            "path": path,
            "created": created,
            "dirty": dirty,
            "initial_has_geometry": initial_has_geometry,
        }
        next_doc_id += 1
        if isinstance(path, str) and path:
            contexts_by_path[path] = ctx
        return ctx

    # Register every already-open persisted document once.
    for path, typ in known_types.items():
        if path not in contexts_by_path:
            make_ctx(typ, path, created=False, dirty=False)

    state_doc = (current_state or {}).get("document")
    initial_has_geometry = bool(state_doc.get("has_geometry", False)) if isinstance(state_doc, dict) else False
    initial_dirty = bool(state_doc.get("dirty", False)) if isinstance(state_doc, dict) else False

    if initial_path and initial_path in contexts_by_path:
        active_ctx = contexts_by_path[initial_path]
        active_ctx["dirty"] = initial_dirty
        active_ctx["initial_has_geometry"] = initial_has_geometry
    elif initial_type in {"part", "assembly"}:
        active_ctx = make_ctx(initial_type, initial_path, created=False, dirty=initial_dirty, initial_has_geometry=initial_has_geometry)
    else:
        active_ctx = None

    last_assembly_ctx = None
    if initial_last_assembly_path:
        last_assembly_ctx = contexts_by_path.get(initial_last_assembly_path)
    if active_ctx is not None and active_ctx.get("type") == "assembly":
        last_assembly_ctx = active_ctx

    # Runtime result provenance.  Bindings point to the document that was active when
    # their producer executed.  This is the missing link between a typed IR and CADia's
    # active-document backend.
    bind_context: dict[str, dict[str, Any] | None] = {}

    def should_save_before_switch(ctx: dict[str, Any] | None) -> bool:
        if ctx is None or ctx.get("type") not in {"part", "assembly"}:
            return False
        if ctx.get("dirty"):
            return True
        if ctx.get("type") == "assembly" and not ctx.get("path"):
            return True
        if not ctx.get("created") and not ctx.get("path") and ctx.get("initial_has_geometry"):
            return True
        return False

    def ensure_saved(ctx: dict[str, Any] | None) -> None:
        nonlocal last_assembly_ctx
        if ctx is None or ctx.get("type") not in {"part", "assembly"}:
            return
        if ctx.get("path") is None:
            path = _allocate_document_path(Path(document_dir), str(ctx.get("type")), reserved)
            ctx["path"] = path
            known_types[path] = str(ctx.get("type"))
            contexts_by_path[path] = ctx
        compiled.append({"tool": "inventor_save_document", "arguments": {"path": ctx["path"]}})
        ctx["dirty"] = False
        if ctx.get("type") == "assembly":
            last_assembly_ctx = ctx

    def switch_to(ctx: dict[str, Any] | None, *, reason: str) -> None:
        nonlocal active_ctx, last_assembly_ctx
        if ctx is None or ctx is active_ctx:
            return
        if should_save_before_switch(active_ctx):
            ensure_saved(active_ctx)
        path = ctx.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"Document lifecycle cannot restore the document required by {reason}: "
                "the producing document has no persisted path. Save it before leaving that document."
            )
        compiled.append({"tool": "inventor_open_document", "arguments": {"path": path}})
        active_ctx = ctx
        active_ctx["dirty"] = False
        if active_ctx.get("type") == "assembly":
            last_assembly_ctx = active_ctx

    def context_from_path_ref(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        m = _BIND_RUNTIME_RE.fullmatch(raw)
        if not m or (m.group(2) or "") != ".path":
            return None
        return bind_context.get(m.group(1))

    doc_bound_semantics = {
        "sketch_entity_id", "face_id", "edge_id", "vertex_id",
        "occurrence_name", "sketch_name", "feature_name",
    }

    def required_context_from_runtime_refs(tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        candidates: dict[int, dict[str, Any]] = {}
        for arg_path, ref in _iter_runtime_refs(args, "arguments"):
            semantic = _expected_reference_semantic(tool, arg_path)
            if semantic not in doc_bound_semantics:
                continue
            m = _BIND_RUNTIME_RE.fullmatch(ref)
            if not m:
                continue
            ctx = bind_context.get(m.group(1))
            if ctx is not None:
                candidates[int(ctx["id"])] = ctx
        if len(candidates) > 1:
            detail = [f"{ctx.get('type')}:{ctx.get('path') or '<unsaved>'}" for ctx in candidates.values()]
            raise ValueError(
                f"Document lifecycle cannot execute {tool}: document-bound runtime references come from multiple documents {detail}. "
                "Split the operation or use references from one active document."
            )
        return next(iter(candidates.values()), None)

    def remember_bind(call: dict[str, Any], ctx: dict[str, Any] | None) -> None:
        bind = call.get("bind")
        if isinstance(bind, str) and bind:
            bind_context[bind] = ctx

    for call in calls:
        if not isinstance(call, dict):
            compiled.append(call)
            continue
        tool = str(call.get("tool") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}

        # Explicit document switches are honored, but the document being left is first
        # made reopenable. Runtime path bindings preserve the exact producer document.
        if tool in _DOCUMENT_SWITCH_TOOLS:
            if tool != "inventor_close_document" and should_save_before_switch(active_ctx):
                ensure_saved(active_ctx)

            compiled.append(call)
            if tool == "inventor_new_part":
                active_ctx = make_ctx("part", None, created=True)
            elif tool == "inventor_new_assembly":
                active_ctx = make_ctx("assembly", None, created=True)
                last_assembly_ctx = active_ctx
            elif tool == "inventor_open_document":
                path_value = args.get("path")
                ref_ctx = context_from_path_ref(path_value)
                if ref_ctx is not None:
                    active_ctx = ref_ctx
                elif isinstance(path_value, str) and path_value.strip() and not _is_runtime_ref(path_value):
                    resolved = str(Path(path_value).expanduser())
                    active_ctx = contexts_by_path.get(resolved)
                    if active_ctx is None:
                        active_ctx = make_ctx(known_types.get(resolved), resolved, created=False)
                else:
                    active_ctx = None
                if active_ctx is not None:
                    active_ctx["dirty"] = False
                    if active_ctx.get("type") == "assembly":
                        last_assembly_ctx = active_ctx
            elif tool == "inventor_close_document":
                active_ctx = None
            remember_bind(call, active_ctx)
            continue

        if tool == "inventor_save_document":
            fixed = copy.deepcopy(call)
            fargs = fixed.setdefault("arguments", {})
            path = fargs.get("path")
            if (path is None or path == "") and active_ctx is not None and active_ctx.get("type") in {"part", "assembly"}:
                if active_ctx.get("path") is None:
                    new_path = _allocate_document_path(Path(document_dir), str(active_ctx.get("type")), reserved)
                    active_ctx["path"] = new_path
                    known_types[new_path] = str(active_ctx.get("type"))
                    contexts_by_path[new_path] = active_ctx
                fargs["path"] = active_ctx.get("path")
                path = active_ctx.get("path")
            compiled.append(fixed)
            if active_ctx is not None and isinstance(path, str) and path.strip() and not _is_runtime_ref(path):
                resolved = str(Path(path).expanduser())
                active_ctx["path"] = resolved
                contexts_by_path[resolved] = active_ctx
                if active_ctx.get("type"):
                    known_types[resolved] = str(active_ctx.get("type"))
                if active_ctx.get("type") == "assembly":
                    last_assembly_ctx = active_ctx
            if active_ctx is not None:
                active_ctx["dirty"] = False
            remember_bind(fixed, active_ctx)
            continue

        # Assembly-only contracts always win over generic producer provenance.
        if tool in ASSEMBLY_CONTEXT_TOOLS:
            if active_ctx is None or active_ctx.get("type") != "assembly":
                target = last_assembly_ctx
                if target is None:
                    assembly_contexts = [ctx for ctx in contexts_by_path.values() if ctx.get("type") == "assembly"]
                    target = assembly_contexts[-1] if assembly_contexts else None
                if target is None:
                    raise ValueError(
                        f"Document lifecycle cannot execute {tool}: no assembly exists in CURRENT_STATE or this plan. "
                        "Create/open an assembly before assembly operations."
                    )
                switch_to(target, reason=tool)
        else:
            # A runtime value such as a sketch entity id or face id belongs to the
            # document that produced it.  Restore that exact document before dispatch.
            target = required_context_from_runtime_refs(tool, args)
            if target is not None and target is not active_ctx:
                switch_to(target, reason=f"{tool} runtime reference")

        compiled.append(call)
        remember_bind(call, active_ctx)
        if active_ctx is not None and not _is_readish_tool(tool):
            active_ctx["dirty"] = True

    out["calls"] = compiled
    return out
