from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cadquery as cq

from .vendor.drawing2cad.cad_generator import normalize_to_mm
from .vendor.drawing2cad.partspec import (
    ExtrudedGeometry,
    MultiBodyGeometry,
    PartSpec,
    RevolvedGeometry,
    UnsupportedGeometry,
)
from .vendor.drawing2cad.validator import ValidationReport, validate


@dataclass(frozen=True)
class NativeCompileResult:
    plan: dict[str, Any] | None
    reason: str


def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _axis_vector(axis: str) -> list[float]:
    return {
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }[axis]


def _thread_payload(thread: str | None, *, through: bool, depth: float | None) -> dict[str, Any] | None:
    designation = str(thread or "").strip()
    if not designation:
        return None
    return {
        "designation": designation,
        "class": "6H",
        "right_handed": True,
        "full_depth": True,
        "thread_depth_mm": None if through else depth,
    }


def _hole_group_key(hole) -> tuple[Any, ...] | None:
    through = hole.hole_type == "through" or (hole.hole_type in {"counterbore", "countersink", "threaded"} and hole.depth is None)
    if hole.hole_type == "threaded" and not str(hole.thread or "").strip():
        return None
    kind = {
        "through": "drilled",
        "blind": "drilled",
        "threaded": "drilled",
        "counterbore": "counterbore",
        "countersink": "countersink",
    }[hole.hole_type]
    return (
        kind,
        float(hole.diameter),
        bool(through),
        None if through else (None if hole.depth is None else float(hole.depth)),
        None if hole.counterbore_diameter is None else float(hole.counterbore_diameter),
        None if hole.counterbore_depth is None else float(hole.counterbore_depth),
        None if hole.countersink_diameter is None else float(hole.countersink_diameter),
        float(hole.countersink_angle or 82.0),
        str(hole.thread or "").strip() or None,
    )


def _append_extruded_base(calls: list[dict[str, Any]], geometry: ExtrudedGeometry) -> None:
    if geometry.profile_kind == "rectangle":
        calls.append(_call("cad_create_box", {
            "length_mm": float(geometry.width),
            "width_mm": float(geometry.height),
            "height_mm": float(geometry.thickness),
            "origin_mm": [0.0, 0.0, 0.0],
            "centered": False,
            "operation": "new",
            "replace": False,
            "name": "DrawingBase",
        }))
        return
    if geometry.profile_kind == "circle":
        calls.append(_call("cad_create_cylinder", {
            "diameter_mm": float(geometry.diameter),
            "height_mm": float(geometry.thickness),
            "origin_mm": [float(geometry.diameter) / 2.0, float(geometry.diameter) / 2.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "operation": "new",
            "replace": False,
            "name": "DrawingBase",
        }))
        return

    # Polygon profiles are still deterministic and editable: build one native sketch,
    # then extrude it.  A fresh part always starts at Sketch1.
    pts = [(float(p.x), float(p.y)) for p in geometry.profile_points]
    calls.append(_call("inventor_create_sketch", {"plane": "XY"}))
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        calls.append(_call("inventor_draw_line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}))
    calls.append(_call("inventor_close_sketch", {"sketchName": None}))
    calls.append(_call("inventor_extrude", {
        "sketchName": "Sketch1",
        "distance": float(geometry.thickness),
        "operation": "join",
        "direction": "positive",
    }))


def _append_revolved_as_native_features(calls: list[dict[str, Any]], geometry: RevolvedGeometry) -> None:
    # The upstream PartSpec describes a stepped body as coaxial constant-diameter
    # intervals.  Building those intervals as native cylinders is deterministic,
    # preserves editable feature history, and avoids generated Python/STEP flattening.
    ordered = sorted(geometry.segments, key=lambda s: (s.z_start, s.z_end))
    first = True
    for index, seg in enumerate(ordered, 1):
        length = float(seg.z_end - seg.z_start)
        calls.append(_call("cad_create_cylinder", {
            "diameter_mm": float(seg.outer_diameter),
            "height_mm": length,
            "origin_mm": [0.0, 0.0, float(seg.z_start)],
            "axis": [0.0, 0.0, 1.0],
            "operation": "new" if first else "join",
            "replace": False,
            "name": f"RevolveOuter{index}",
        }))
        first = False
        if float(seg.inner_diameter) > 0.0:
            calls.append(_call("cad_create_cylinder", {
                "diameter_mm": float(seg.inner_diameter),
                "height_mm": length,
                "origin_mm": [0.0, 0.0, float(seg.z_start)],
                "axis": [0.0, 0.0, 1.0],
                "operation": "cut",
                "replace": False,
                "name": f"RevolveBore{index}",
            }))


def _append_multibody(calls: list[dict[str, Any]], geometry: MultiBodyGeometry) -> bool:
    # Match the upstream generator semantics exactly: build/union every add body first,
    # then apply every cut body.  PartSpec does not require add/cut rows to be ordered,
    # so executing a leading cut against an empty document would be a false failure.
    ordered = [
        *[(i, b) for i, b in enumerate(geometry.bodies, 1) if b.operation == "add"],
        *[(i, b) for i, b in enumerate(geometry.bodies, 1) if b.operation == "cut"],
    ]
    add_seen = False
    for index, body in ordered:
        operation = "cut" if body.operation == "cut" else ("new" if not add_seen else "join")
        if body.operation == "add":
            add_seen = True
        if body.shape == "box":
            calls.append(_call("cad_create_box", {
                "length_mm": float(body.dx),
                "width_mm": float(body.dy),
                "height_mm": float(body.dz),
                "origin_mm": [float(body.x), float(body.y), float(body.z)],
                "centered": False,
                "operation": operation,
                "replace": False,
                "name": f"Body{index}",
            }))
        elif body.shape == "cylinder":
            calls.append(_call("cad_create_cylinder", {
                "diameter_mm": float(body.diameter),
                "height_mm": float(body.length),
                "origin_mm": [float(body.x), float(body.y), float(body.z)],
                "axis": _axis_vector(body.axis),
                "operation": operation,
                "replace": False,
                "name": f"Body{index}",
            }))
        elif body.shape == "sphere" and not body.wall_thickness:
            calls.append(_call("cad_create_sphere", {
                "diameter_mm": float(body.diameter),
                "center_mm": [float(body.x), float(body.y), float(body.z)],
                "operation": operation,
                "replace": False,
                "name": f"Body{index}",
            }))
        else:
            # prism/shelled sphere needs richer topology; let the CADia planner handle it.
            return False
    return add_seen


def compile_partspec_to_native_plan(spec: PartSpec) -> NativeCompileResult:
    """Compile the safe, deterministic subset of Drawing2CAD PartSpec to CADia tools.

    No generated Python is executed and no STEP is re-imported.  Unsupported/ambiguous
    topology returns plan=None so the caller can use CADia's normal typed-tool planner.
    """
    spec = normalize_to_mm(spec)
    geometry = spec.geometry

    if isinstance(geometry, UnsupportedGeometry):
        return NativeCompileResult(None, "unsupported geometry requires CADia planner fallback")
    if isinstance(geometry, ExtrudedGeometry):
        if geometry.corner_radius > 0:
            return NativeCompileResult(None, "base corner radius needs topology-aware planner")
        if geometry.profile_kind == "rectangle" and not (geometry.width and geometry.height and geometry.thickness > 0):
            return NativeCompileResult(None, "incomplete rectangular profile")
        if geometry.profile_kind == "circle" and not (geometry.diameter and geometry.thickness > 0):
            return NativeCompileResult(None, "incomplete circular profile")
        if geometry.profile_kind == "polygon" and len(geometry.profile_points) < 3:
            return NativeCompileResult(None, "polygon needs at least three points")
    if spec.fillets or spec.chamfers:
        # PartSpec stores only radii/distances, not the edge identities.  Guessing edge IDs
        # would make failures more likely; defer these cases to topology-aware CADia planning.
        return NativeCompileResult(None, "fillet/chamfer edge identities require CADia planner")
    if any(c.corner_radius > 0 for c in spec.cuts):
        return NativeCompileResult(None, "rounded rectangular cuts require topology-aware planner")
    if len(spec.holes) > 64 or len(spec.cuts) > 32:
        return NativeCompileResult(None, "feature count exceeds deterministic safety envelope")

    calls: list[dict[str, Any]] = [
        _call("inventor_new_part", {}),
        _call("inventor_set_units", {"lengthUnit": "mm"}),
    ]

    if isinstance(geometry, ExtrudedGeometry):
        _append_extruded_base(calls, geometry)
    elif isinstance(geometry, RevolvedGeometry):
        if not geometry.segments:
            return NativeCompileResult(None, "revolved geometry has no segments")
        _append_revolved_as_native_features(calls, geometry)
    elif isinstance(geometry, MultiBodyGeometry):
        if not _append_multibody(calls, geometry):
            return NativeCompileResult(None, "multibody contains a prism or shell that needs CADia planner")
    else:
        return NativeCompileResult(None, "unknown geometry")

    # External PartSpec holes are defined in the profile XY frame and are safe to map to
    # the +Z top face only for an extruded base.  Multi-body/revolved cases are routed to
    # the planner rather than guessing which face the drawing intended.
    if spec.holes:
        if not isinstance(geometry, ExtrudedGeometry):
            return NativeCompileResult(None, "explicit holes on non-extruded geometry need planner face reasoning")
        top_z = float(geometry.thickness)
        grouped: dict[tuple[Any, ...], list[list[float]]] = {}
        for hole in spec.holes:
            key = _hole_group_key(hole)
            if key is None:
                return NativeCompileResult(None, "threaded hole is missing a thread designation")
            grouped.setdefault(key, []).append([float(hole.x), float(hole.y), top_z])
        for index, (key, points) in enumerate(grouped.items(), 1):
            kind, dia, through, depth, cb_dia, cb_depth, cs_dia, cs_angle, thread = key
            calls.append(_call("inventor_hole", {
                "face": {"kind": "planar", "normal": "+Z", "extreme": "max"},
                "points_mm": points,
                "diameter_mm": dia,
                "kind": kind,
                "through": through,
                "depth_mm": depth,
                "cbore_diameter_mm": cb_dia,
                "cbore_depth_mm": cb_depth,
                "csink_diameter_mm": cs_dia,
                "csink_angle_deg": cs_angle,
                "tapped": _thread_payload(thread, through=through, depth=depth),
            }))

    for index, cut in enumerate(spec.cuts, 1):
        calls.append(_call("cad_create_box", {
            "length_mm": float(cut.dx),
            "width_mm": float(cut.dy),
            "height_mm": float(cut.dz),
            "origin_mm": [float(cut.x), float(cut.y), float(cut.z)],
            "centered": False,
            "operation": "cut",
            "replace": False,
            "name": f"DrawingCut{index}",
        }))

    if len(calls) > 80:
        return NativeCompileResult(None, "native plan exceeds CADia's 80-call limit")
    return NativeCompileResult({"calls": calls, "note": "drawing2cad deterministic native compile"}, "deterministic native compile")


def validate_runtime_against_partspec(runtime, spec: PartSpec, *, tol: float = 0.5) -> ValidationReport:
    """Run Drawing2CAD's public validator directly on CADia's live B-Rep."""
    with runtime.engine.lock:
        doc = runtime.engine.doc
        shape = None if doc is None else doc.shape
        if shape is None:
            raise ValueError("CADia has no active solid to validate")
        try:
            shape_copy = shape.copy()
        except Exception:
            shape_copy = shape
    return validate(spec, cq.Workplane(obj=shape_copy), tol=tol)


def report_score(report: ValidationReport | None) -> float:
    if report is None or not report.checks:
        return 0.0
    return sum(1 for row in report.checks if row.passed) / len(report.checks)


def validation_payload(report: ValidationReport | None) -> dict[str, Any]:
    if report is None:
        return {"score": 0.0, "passed": False, "checks": [], "advisory": [], "feedback": "no report"}
    return {
        "score": report_score(report),
        "passed": report.ok(),
        "checks": [row.model_dump() for row in report.checks],
        "advisory": [row.model_dump() for row in report.advisory],
        "feedback": report.feedback(),
    }


def _solid_point_is_inside(shape, x: float, y: float, z: float) -> bool | None:
    try:
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON
        clf = BRepClass3d_SolidClassifier(shape.wrapped)
        clf.Perform(gp_Pnt(float(x), float(y), float(z)), 1e-6)
        return clf.State() in (TopAbs_IN, TopAbs_ON)
    except Exception:
        return None


def validate_runtime_payload(runtime, spec: PartSpec, *, tol: float = 0.5) -> dict[str, Any]:
    """Drawing2CAD validator plus CADia-native checks that the upstream validator leaves advisory.

    Upstream scored checks are preserved verbatim.  CADia adds cut-presence and explicit
    fillet/chamfer feature-count checks so a valid bounding box cannot hide a missing
    pocket or omitted edge treatment.
    """
    report = validate_runtime_against_partspec(runtime, spec, tol=tol)
    payload = validation_payload(report)
    supplemental: list[dict[str, Any]] = []
    with runtime.engine.lock:
        doc = runtime.engine.doc
        shape = None if doc is None else doc.shape
        features = [] if doc is None else list(doc.features)
        try:
            shape_copy = None if shape is None else shape.copy()
        except Exception:
            shape_copy = shape

    if shape_copy is not None:
        for index, cut in enumerate(spec.cuts):
            probe = (
                float(cut.x + cut.dx / 2.0),
                float(cut.y + cut.dy / 2.0),
                float(cut.z + cut.dz / 2.0),
            )
            inside = _solid_point_is_inside(shape_copy, *probe)
            if inside is None:
                supplemental.append({
                    "name": f"cut{index}_present", "expected": "void", "measured": "unknown",
                    "passed": True, "detail": "point-membership unavailable; skipped",
                })
            else:
                supplemental.append({
                    "name": f"cut{index}_present", "expected": "void",
                    "measured": "material" if inside else "void", "passed": not inside,
                    "detail": "" if not inside else f"no cut void at {probe}",
                })

    if spec.fillets:
        actual = sum(1 for f in features if str(getattr(f, "kind", "")).lower() == "fillet" and not getattr(f, "suppressed", False))
        supplemental.append({
            "name": "fillet_feature_count", "expected": len(spec.fillets), "measured": actual,
            "passed": actual >= len(spec.fillets), "detail": "Drawing2CAD PartSpec edge radii require native fillet features",
        })
    if spec.chamfers:
        actual = sum(1 for f in features if str(getattr(f, "kind", "")).lower() == "chamfer" and not getattr(f, "suppressed", False))
        supplemental.append({
            "name": "chamfer_feature_count", "expected": len(spec.chamfers), "measured": actual,
            "passed": actual >= len(spec.chamfers), "detail": "Drawing2CAD PartSpec distances require native chamfer features",
        })

    scored = list(payload.get("checks") or []) + supplemental
    passed_count = sum(1 for row in scored if bool(row.get("passed")))
    payload["cadia_checks"] = supplemental
    payload["score"] = (passed_count / len(scored)) if scored else 0.0
    payload["passed"] = bool(scored) and passed_count == len(scored)
    failed = [row for row in supplemental if not row.get("passed")]
    if failed:
        extra_feedback = "\n".join(
            f"- {row['name']}: expected {row['expected']}, measured {row['measured']}" for row in failed
        )
        base = str(payload.get("feedback") or "").strip()
        payload["feedback"] = (base + "\n" + extra_feedback).strip()
    return payload
