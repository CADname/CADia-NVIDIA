from __future__ import annotations

"""General reliability gates for CADia Drawing -> CAD.

This module deliberately contains no part-family, feature-type, or operation-specific rescue.
It validates the same typed CAD plan the normal CADia executor will run and catches
parameter dependency errors before any document mutation.

The orchestration policy around this module is derived from public Drawing-to-CAD work:
- Ortho2CAD / cad-image-to-3d: execute, use concrete failure feedback, re-plan, render/compare.
- drawing2cad: sanity-check before build; repair the current attempt from validator/error feedback.
- cad-agent: keep tunable dimensions numeric and explicit rather than fragile symbolic chains.

Runtime CAD execution remains CADia's native typed tool executor so successful models keep
editable feature history and persistent topology rather than being flattened through STEP.
"""

import math
from dataclasses import dataclass
from typing import Any

from standalonecad.core.expressions import eval_expr


PUBLIC_PROTOCOL_ID = "cadia-drawing-public-protocol-v1"


@dataclass(frozen=True)
class PreflightReport:
    checked_calls: int
    resolved_parameters: dict[str, float]
    parameter_events: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": PUBLIC_PROTOCOL_ID,
            "checked_calls": self.checked_calls,
            "resolved_parameters": dict(self.resolved_parameters),
            "parameter_events": [dict(x) for x in self.parameter_events],
        }


def _resolve_existing_parameters(current_state: dict[str, Any] | None) -> dict[str, float]:
    """Resolve the current document's parameter expressions without mutating it."""
    raw = (current_state or {}).get("parameters") or {}
    if not isinstance(raw, dict):
        return {}
    pending: dict[str, str] = {}
    for name, row in raw.items():
        if not isinstance(name, str) or not isinstance(row, dict):
            continue
        expr = row.get("expression")
        if isinstance(expr, (str, int, float)):
            pending[name] = str(expr)

    values: dict[str, float] = {}
    for _ in range(len(pending) + 3):
        changed = False
        for name, expr in list(pending.items()):
            try:
                value = float(eval_expr(expr, values))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                continue
            if not math.isfinite(value):
                raise ValueError(f"Existing parameter {name!r} resolves to a non-finite value")
            values[name] = value
            del pending[name]
            changed = True
        if not changed:
            break
    if pending:
        names = ", ".join(sorted(pending))
        raise ValueError(f"Current document already contains unresolved/cyclic parameters: {names}")
    return values


def _eval_parameter_value(value: Any, values: dict[str, float], *, call_index: int, name: str) -> float:
    try:
        result = float(eval_expr(value, values))
    except Exception as exc:
        # Keep the native evaluator's concrete reason (unknown parameter, unsupported
        # expression, etc.) so the planner receives actionable feedback.
        raise ValueError(
            f"calls[{call_index}] parameter {name!r} cannot be resolved at this point: {exc}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"calls[{call_index}] parameter {name!r} resolves to a non-finite value")
    return result


def preflight_plan(executor, plan: dict[str, Any], current_state: dict[str, Any] | None) -> tuple[dict[str, Any], PreflightReport]:
    """Validate a plan exactly as CADia will execute it, plus parameter dependencies.

    `PlanExecutor.validate_plan` checks every public tool schema and runtime binding.  The
    extra sequential parameter pass catches a class that JSON-schema cannot: a parameter
    create/set expression that references itself, an unknown name, or a name that is only
    created later.  CADia rebuilds immediately on each parameter operation, so those plans
    are invalid even if the complete dependency graph would be resolvable eventually.
    """
    validated_calls = executor.validate_plan(plan)
    canonical = {"calls": validated_calls, "note": str((plan or {}).get("note") or "")}

    values = _resolve_existing_parameters(current_state)
    events: list[dict[str, Any]] = []

    for idx, call in enumerate(validated_calls):
        tool = str(call.get("tool") or "")
        args = call.get("arguments") or {}

        # Creating a new part/assembly establishes a fresh document, so parameters from an
        # older document are no longer legal dependencies after this point.
        if tool in {"inventor_new_part", "inventor_new_assembly"}:
            values = {}
            events.append({"call": idx, "tool": tool, "action": "reset-parameter-scope"})
            continue

        if tool == "inventor_create_parameter":
            name = str(args.get("name") or "").strip()
            if not name:
                raise ValueError(f"calls[{idx}] inventor_create_parameter requires a non-empty name")
            expression = args.get("expression")
            resolved = _eval_parameter_value(expression, values, call_index=idx, name=name)
            values[name] = resolved
            events.append({"call": idx, "tool": tool, "name": name, "value": resolved})
            continue

        if tool == "inventor_set_parameter":
            name = str(args.get("name") or "").strip()
            if name not in values:
                raise ValueError(f"calls[{idx}] cannot set unknown parameter {name!r}")
            resolved = _eval_parameter_value(args.get("value"), values, call_index=idx, name=name)
            values[name] = resolved
            events.append({"call": idx, "tool": tool, "name": name, "value": resolved})

    return canonical, PreflightReport(
        checked_calls=len(validated_calls),
        resolved_parameters=dict(values),
        parameter_events=tuple(events),
    )


def completed_shape_checks(runtime) -> dict[str, Any]:
    """Kernel-level completion checks that are independent of any specific part type."""
    with runtime.engine.lock:
        doc = runtime.engine.doc
        shape = doc.shape if doc is not None else None
        if shape is None:
            return {
                "solid_present": False,
                "brep_valid": False,
                "positive_volume": False,
                "finite_bbox": False,
                "solid_count": 0,
            }
        try:
            valid = bool(shape.isValid())
        except Exception:
            valid = False
        try:
            volume = float(shape.Volume())
        except Exception:
            volume = float("nan")
        positive_volume = math.isfinite(volume) and volume > 1e-9
        try:
            box = shape.BoundingBox()
            extents = [float(box.xlen), float(box.ylen), float(box.zlen)]
            finite_bbox = all(math.isfinite(x) and x >= 0 for x in extents) and max(extents) > 1e-9
        except Exception:
            extents = [float("nan")] * 3
            finite_bbox = False
        try:
            solid_count = len(shape.Solids())
        except Exception:
            solid_count = 1 if positive_volume else 0

    return {
        "solid_present": True,
        "brep_valid": valid,
        "positive_volume": positive_volume,
        "volume_mm3": volume if math.isfinite(volume) else None,
        "finite_bbox": finite_bbox,
        "bbox_mm": {"x": extents[0], "y": extents[1], "z": extents[2]} if finite_bbox else None,
        "solid_count": int(solid_count),
    }
