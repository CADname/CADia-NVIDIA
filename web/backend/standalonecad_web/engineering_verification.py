from __future__ import annotations

import copy
import math
import os
import re
from typing import Any


PATCH_MARKER = "CADIA_NEMOTRON_CODEX_VERIFIED_HYBRID_V4"


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def raw_user_request_text(user_prompt: str) -> str:
    """Strip route-added planner context while preserving the user's authored request."""
    return str(user_prompt or "").split("\n\nSelected CAD context:", 1)[0].strip()


def _quote_is_grounded(user_prompt: str, quote: Any) -> bool:
    q = _norm_text(quote)
    if not q:
        return False
    # The route may append a read-only "Selected CAD context" block for the planner.
    # It is evidence about the current document, not a user-authored requirement, so it
    # must never be accepted as the source quote for an automatic repair.
    return q in _norm_text(raw_user_request_text(user_prompt))


def _requirement_id(value: Any, index: int) -> str:
    raw = re.sub(r"[^A-Za-z0-9_.:-]+", "", str(value or "").strip())
    return raw[:48] if raw else f"R{index}"


def normalize_engineering_contract(raw: Any, user_prompt: str) -> dict[str, Any]:
    """Normalize Nemotron's requirement extraction and mark what is source-grounded.

    The raw user request always remains the source of truth. Requirements without a
    verbatim source quote are retained only as advisory context and are never eligible
    to trigger automatic post-execution repair.
    """
    data = raw if isinstance(raw, dict) else {}
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = data.get("requirements")
    if isinstance(rows, list):
        for i, item in enumerate(rows, 1):
            if not isinstance(item, dict):
                continue
            rid = _requirement_id(item.get("id"), i)
            if rid in seen:
                rid = f"{rid}_{i}"
            seen.add(rid)
            quote = str(item.get("source_quote") or "").strip()
            grounded = _quote_is_grounded(user_prompt, quote)
            expected = copy.deepcopy(item.get("expected"))
            req = {
                "id": rid,
                "kind": str(item.get("kind") or "other")[:48],
                "target": str(item.get("target") or "")[:240],
                "expected": expected,
                "unit": str(item.get("unit") or "")[:32],
                "hard": bool(item.get("hard", True)),
                "explicit": bool(item.get("explicit", True)),
                "source_quote": quote[:300],
                "source_supported": bool(grounded),
                "verification_hint": str(item.get("verification_hint") or "")[:160],
            }
            # An empty target/expected pair cannot be verified safely.
            if req["target"] or expected is not None:
                requirements.append(req)

    # Backward compatibility: old contracts are still useful to Codex/recovery, but
    # they cannot drive automatic repair because they have no verbatim source anchor.
    if not requirements:
        legacy_groups = (
            ("dimension", data.get("explicit_dimensions")),
            ("feature", data.get("required_features")),
            ("constraint", data.get("constraints")),
            ("preserve", data.get("preserve")),
        )
        idx = 1
        for kind, values in legacy_groups:
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, (str, int, float, bool)):
                    continue
                requirements.append({
                    "id": f"L{idx}",
                    "kind": kind,
                    "target": str(value)[:240],
                    "expected": str(value)[:240],
                    "unit": "",
                    "hard": False,
                    "explicit": False,
                    "source_quote": "",
                    "source_supported": False,
                    "verification_hint": "legacy_advisory",
                })
                idx += 1

    uncertainties = data.get("uncertainties")
    if not isinstance(uncertainties, list):
        uncertainties = []

    return {
        "version": 2,
        "summary": str(data.get("summary") or "")[:800],
        "components": [str(x)[:160] for x in (data.get("components") or []) if isinstance(x, (str, int, float))][:64],
        "requirements": requirements[:96],
        "uncertainties": [str(x)[:300] for x in uncertainties if isinstance(x, (str, int, float))][:48],
        "policy": {
            "source_of_truth": "raw_user_request",
            "unsupported_requirements_are_advisory_only": True,
        },
    }


def contract_for_planner(contract: dict[str, Any]) -> dict[str, Any]:
    """Return only safe planner context; ungrounded rows cannot override the raw prompt."""
    if not isinstance(contract, dict):
        return {}
    supported = [
        copy.deepcopy(r)
        for r in contract.get("requirements", [])
        if isinstance(r, dict) and r.get("source_supported")
    ]
    return {
        "version": contract.get("version", 2),
        "summary": contract.get("summary", ""),
        "components": copy.deepcopy(contract.get("components", [])),
        "requirements": supported,
        "uncertainties": copy.deepcopy(contract.get("uncertainties", [])),
        "policy": copy.deepcopy(contract.get("policy", {})),
    }




def verified_hybrid_mode() -> str:
    """Return the latency/quality policy for the Nemotron+Codex hybrid.

    balanced (default):
      * obvious one-step/selected edits: Codex + kernel only on the success path;
      * ordinary requests: Codex first, then one combined Nemotron extraction+audit;
      * complex/high-risk requests: pre-plan Nemotron ledger + post-execution audit.
    quality: preserves the v4.0.1 behavior (pre-plan ledger + post audit for every
      non-selected request).
    """
    value = str(os.getenv("CADIA_NEMOTRON_HYBRID_MODE", "balanced") or "balanced").strip().lower()
    return value if value in {"balanced", "quality"} else "balanced"


def _prompt_body(user_prompt: str) -> str:
    text = raw_user_request_text(user_prompt)
    # routes/ai.py may append a verbose selection description; the helper removes it so
    # a tiny user edit does not look like a long/complex request.
    return re.sub(r"\s+", " ", text).strip()


def _selection_size(current_state: dict[str, Any]) -> int:
    state = current_state if isinstance(current_state, dict) else {}
    selection = state.get("selection")
    if not isinstance(selection, dict) or not selection:
        return 0
    total = 0
    for value in selection.values():
        if isinstance(value, (list, tuple, set, dict)):
            total += len(value)
        elif value not in (None, "", False):
            total += 1
    return total


def hybrid_request_profile(user_prompt: str, current_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cheap deterministic routing for the hybrid success path.

    This classifier is deliberately conservative: only very obvious single-operation
    requests bypass Nemotron on a successful first pass.  Anything ambiguous becomes
    standard (one combined post-audit call) and genuinely high-risk work keeps the full
    pre-plan + post-audit supervision used by v4.0.1.  Kernel failures always retain
    Nemotron diagnosis regardless of this profile.
    """
    mode = verified_hybrid_mode()
    body = _prompt_body(user_prompt)
    folded = body.casefold()
    state = current_state if isinstance(current_state, dict) else {}
    selection_count = _selection_size(state)
    numeric_count = len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", body))
    connector_count = len(re.findall(r"(?:\band\b|\bthen\b|\bafter\b|\bplus\b|그리고|그다음|다음|후에|후|및|하면서)", folded))

    occurrences = state.get("occurrences") if isinstance(state.get("occurrences"), list) else []
    joints = state.get("joints") if isinstance(state.get("joints"), list) else []
    constraints = state.get("constraints") if isinstance(state.get("constraints"), list) else []
    doc = state.get("document") if isinstance(state.get("document"), dict) else {}
    doc_type = str(doc.get("type") or doc.get("doc_type") or "").casefold()

    high_risk_terms = (
        "assembly", "assemble", "joint", "mate", "occurrence", "kinematic", "mechanism", "degrees of freedom",
        "조립", "어셈블리", "조인트", "메이트", "구속", "자유도", "기구학", "메커니즘",
        "preserve", "keep existing", "without changing", "do not change", "must remain", "retain existing",
        "유지", "보존", "건드리지", "그대로 두", "바꾸지 말", "변경하지 말",
        "to face", "from-to", "from to", "to next", "through all", "up to face",
        "면까지", "다음 면", "관통", "참조 면", "참조면",
        "persistent topology", "topology", "relink", "re-link", "위상", "재연결",
    )
    has_high_term = any(term in folded for term in high_risk_terms)
    assembly_state = bool(occurrences or joints or "assembly" in doc_type or "어셈블리" in doc_type)

    # Multiple explicit dimensions/features or a long compound instruction benefits from
    # the pre-plan requirement ledger.  Thresholds intentionally err toward quality.
    complex_request = bool(
        has_high_term
        or assembly_state
        or numeric_count >= 5
        or len(body) >= 320
        or (connector_count >= 3 and numeric_count >= 3)
        or (len(constraints) >= 2 and ("modify" in folded or "edit" in folded or "수정" in folded or "변경" in folded))
    )

    direct_edit_terms = (
        "move", "offset", "rotate", "delete", "fillet", "chamfer", "resize", "change", "set ", "edit",
        "이동", "오프셋", "회전", "삭제", "필렛", "모따기", "변경", "수정", "직경", "반지름", "두께",
    )
    multi_feature_terms = (
        " hole", "holes", "bore", "rib", "pattern", "mirror", "loft", "sweep", "shell", "draft",
        "구멍", "홀", "보어", "리브", "패턴", "미러", "로프트", "스윕", "쉘",
    )
    selected_simple = bool(
        selection_count > 0
        and len(body) <= 180
        and numeric_count <= 2
        and connector_count <= 1
        and any(term in folded for term in direct_edit_terms)
        and not has_high_term
        and not any(term in folded for term in multi_feature_terms)
    )

    primitive_terms = ("box", "block", "cube", "cylinder", "sphere", "plate", "shaft", "disk", "disc", "박스", "블록", "직육면체", "원통", "구", "판", "축", "디스크")
    primitive_hits = sum(1 for term in primitive_terms if term in folded)
    feature_combo_terms = ("hole", "bore", "fillet", "chamfer", "rib", "pattern", "mirror", "loft", "sweep", "shell", "구멍", "홀", "보어", "필렛", "모따기", "리브", "패턴", "로프트", "스윕", "쉘")
    primitive_simple = bool(
        primitive_hits == 1
        and len(body) <= 140
        and numeric_count <= 4
        and connector_count <= 1
        and not has_high_term
        and not any(term in folded for term in feature_combo_terms)
    )

    if mode == "quality":
        tier = "complex"
        reason = "quality_mode"
        pre_contract = True
        post_audit = True
        combined_post_audit = False
    elif complex_request:
        tier = "complex"
        reason = "high_risk_or_compound_request"
        pre_contract = True
        post_audit = True
        combined_post_audit = False
    elif selected_simple or primitive_simple:
        tier = "simple"
        reason = "obvious_single_operation"
        pre_contract = False
        post_audit = False
        combined_post_audit = False
    else:
        tier = "standard"
        reason = "single_combined_post_audit"
        pre_contract = False
        post_audit = True
        combined_post_audit = True

    return {
        "mode": mode,
        "tier": tier,
        "reason": reason,
        "pre_contract": pre_contract,
        "post_audit": post_audit,
        "combined_post_audit": combined_post_audit,
        "selection_count": selection_count,
        "numeric_count": numeric_count,
        "connector_count": connector_count,
        "prompt_length": len(body),
        "kernel_failure_diagnosis": True,
    }


def _finite_number(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def geometry_evidence(engine: Any) -> dict[str, Any]:
    """Read-only kernel evidence. Never mutates CAD state and never guesses intent."""
    out: dict[str, Any] = {
        "document_type": None,
        "shape_present": False,
        "shape_valid": None,
        "bbox_mm": None,
        "volume_mm3": None,
        "area_mm2": None,
        "feature_count": 0,
        "face_count": None,
        "edge_count": None,
    }
    try:
        with engine.lock:
            doc = engine.doc
            if doc is None:
                return out
            out["document_type"] = getattr(doc, "doc_type", None)
            out["feature_count"] = len(getattr(doc, "features", []) or [])
            shape = getattr(doc, "shape", None)
            out["shape_present"] = shape is not None
            if shape is None:
                return out
            try:
                out["shape_valid"] = bool(shape.isValid()) if hasattr(shape, "isValid") else None
            except Exception:
                out["shape_valid"] = None
            try:
                bb = shape.BoundingBox()
                vals = {
                    "xmin": _finite_number(bb.xmin), "xmax": _finite_number(bb.xmax), "xlen": _finite_number(bb.xlen),
                    "ymin": _finite_number(bb.ymin), "ymax": _finite_number(bb.ymax), "ylen": _finite_number(bb.ylen),
                    "zmin": _finite_number(bb.zmin), "zmax": _finite_number(bb.zmax), "zlen": _finite_number(bb.zlen),
                }
                if all(v is not None for v in vals.values()):
                    out["bbox_mm"] = vals
            except Exception:
                pass
            try:
                out["volume_mm3"] = _finite_number(shape.Volume())
            except Exception:
                pass
            try:
                out["area_mm2"] = _finite_number(shape.Area())
            except Exception:
                pass
            try:
                topo = doc.topology()
                if isinstance(topo, dict):
                    if isinstance(topo.get("faces"), list):
                        out["face_count"] = len(topo["faces"])
                    if isinstance(topo.get("edges"), list):
                        out["edge_count"] = len(topo["edges"])
            except Exception:
                pass
    except Exception:
        return out
    return out


def state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    b_features = before.get("features") if isinstance(before.get("features"), list) else []
    a_features = after.get("features") if isinstance(after.get("features"), list) else []
    b_names = {str(x.get("name")) for x in b_features if isinstance(x, dict)}
    new_features = [copy.deepcopy(x) for x in a_features if isinstance(x, dict) and str(x.get("name")) not in b_names]
    return {
        "revision_before": ((before.get("document") or {}).get("revision") if isinstance(before.get("document"), dict) else None),
        "revision_after": ((after.get("document") or {}).get("revision") if isinstance(after.get("document"), dict) else None),
        "feature_count_before": len(b_features),
        "feature_count_after": len(a_features),
        "new_features": new_features[-40:],
        "occurrence_count_before": len(before.get("occurrences") or []) if isinstance(before.get("occurrences"), list) else 0,
        "occurrence_count_after": len(after.get("occurrences") or []) if isinstance(after.get("occurrences"), list) else 0,
        "constraint_count_before": len(before.get("constraints") or []) if isinstance(before.get("constraints"), list) else 0,
        "constraint_count_after": len(after.get("constraints") or []) if isinstance(after.get("constraints"), list) else 0,
        "joint_count_before": len(before.get("joints") or []) if isinstance(before.get("joints"), list) else 0,
        "joint_count_after": len(after.get("joints") or []) if isinstance(after.get("joints"), list) else 0,
    }




def _compact_verifier_state(state: dict[str, Any]) -> dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    sketches = []
    for row in state.get("sketches") or []:
        if not isinstance(row, dict):
            continue
        sketches.append({
            "name": row.get("name"),
            "plane": row.get("plane"),
            "closed": row.get("closed"),
            "entity_count": len(row.get("entities") or []) if isinstance(row.get("entities"), list) else None,
            "dimensions": copy.deepcopy(row.get("dimensions") or []),
            "constraints": copy.deepcopy(row.get("constraints") or []),
        })
    occurrences = []
    for row in state.get("occurrences") or []:
        if not isinstance(row, dict):
            continue
        occurrences.append({k: copy.deepcopy(row.get(k)) for k in ("name", "grounded", "position_mm", "rotation_deg_xyz", "interfaces") if row.get(k) is not None})
    faces = []
    for row in (state.get("faces") or [])[:96]:
        if not isinstance(row, dict):
            continue
        faces.append({k: copy.deepcopy(row.get(k)) for k in ("id", "semantic", "semantics", "geom", "center", "area", "normal", "direction", "radius_mm", "axis") if row.get(k) is not None})
    edges = []
    for row in (state.get("edges") or [])[:128]:
        if not isinstance(row, dict):
            continue
        edges.append({k: copy.deepcopy(row.get(k)) for k in ("id", "geom", "center", "length", "direction", "radius_mm") if row.get(k) is not None})
    return {
        "document": copy.deepcopy(state.get("document")),
        "parameters": copy.deepcopy(state.get("parameters") or {}),
        "sketches": sketches,
        "features": copy.deepcopy(state.get("features") or []),
        "work_features": copy.deepcopy(state.get("work_features") or {}),
        "occurrences": occurrences,
        "constraints": copy.deepcopy(state.get("constraints") or []),
        "joints": copy.deepcopy(state.get("joints") or []),
        "selection": copy.deepcopy(state.get("selection") or {}),
        "faces": faces,
        "edges": edges,
    }



def _derived_verifier_metrics(after_state: dict[str, Any]) -> dict[str, Any]:
    state = after_state if isinstance(after_state, dict) else {}
    kind_counts: dict[str, int] = {}
    param_lengths: list[dict[str, Any]] = []
    for feature in state.get("features") or []:
        if not isinstance(feature, dict):
            continue
        kind = str(feature.get("kind") or "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        params = feature.get("params") if isinstance(feature.get("params"), dict) else {}
        for key, value in params.items():
            if isinstance(value, (list, tuple, dict)):
                param_lengths.append({
                    "feature_name": feature.get("name"),
                    "feature_kind": kind,
                    "parameter": str(key),
                    "count": len(value),
                })
    faces = state.get("faces") if isinstance(state.get("faces"), list) else []
    cylindrical_radii = []
    for face in faces:
        if not isinstance(face, dict):
            continue
        radius = face.get("radius_mm")
        if radius is not None:
            value = _finite_number(radius)
            if value is not None:
                cylindrical_radii.append(value)
    return {
        "feature_kind_counts": kind_counts,
        "feature_parameter_counts": param_lengths[:160],
        "occurrence_count": len(state.get("occurrences") or []) if isinstance(state.get("occurrences"), list) else 0,
        "constraint_count": len(state.get("constraints") or []) if isinstance(state.get("constraints"), list) else 0,
        "joint_count": len(state.get("joints") or []) if isinstance(state.get("joints"), list) else 0,
        "face_count": len(faces),
        "edge_count": len(state.get("edges") or []) if isinstance(state.get("edges"), list) else 0,
        "cylindrical_radii_mm": cylindrical_radii[:96],
    }

def build_verification_evidence(
    *,
    engine: Any,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    executed_plan: dict[str, Any],
    execution_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "geometry": geometry_evidence(engine),
        "delta": state_delta(before_state, after_state),
        "derived": _derived_verifier_metrics(after_state),
        "after_state": _compact_verifier_state(after_state),
        "executed_plan": copy.deepcopy(executed_plan),
        "execution_results": copy.deepcopy(execution_results or []),
    }


def _json_pointer_get(document: Any, pointer: str) -> tuple[bool, Any]:
    if pointer in {"", "/"}:
        return True, document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False, None
    cur = document
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if token not in cur:
                return False, None
            cur = cur[token]
        elif isinstance(cur, list):
            try:
                idx = int(token)
            except Exception:
                return False, None
            if idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


def _loosely_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            return abs(fa - fb) <= max(1e-8, 1e-6 * max(abs(fa), abs(fb), 1.0))
    except Exception:
        pass
    return _norm_text(a) == _norm_text(b)


def evaluate_auto_repair_gate(
    audit: Any,
    contract: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Conservative deterministic gate between an LLM audit and CAD mutation.

    Nemotron may *recommend* repair, but it gets no mutation authority by itself. A
    repair is allowed only when a hard, verbatim-grounded requirement is contradicted
    by a concrete after-state/kernel evidence value whose JSON pointer resolves and
    matches the value Nemotron claims it observed.
    """
    threshold = float(min_confidence if min_confidence is not None else os.getenv("CADIA_NEMOTRON_VERIFY_MIN_CONFIDENCE", "0.97"))
    if not isinstance(audit, dict) or str(audit.get("verdict") or "").lower() != "repair":
        return {"allow": False, "reason": "audit_not_repair", "grounded_failures": []}
    requirements = {
        str(r.get("id")): r
        for r in (contract.get("requirements") or [])
        if isinstance(r, dict)
    }
    findings = audit.get("findings")
    if not isinstance(findings, list):
        return {"allow": False, "reason": "missing_findings", "grounded_failures": []}
    grounded: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict) or str(finding.get("status") or "").lower() != "fail":
            continue
        rid = str(finding.get("requirement_id") or "")
        req = requirements.get(rid)
        if not req or not req.get("source_supported") or not req.get("hard") or not req.get("explicit"):
            continue
        try:
            confidence = float(finding.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < threshold:
            continue
        pointer = str(finding.get("evidence_path") or "")
        # Automatic mutation must be grounded in the *actual result*, never just the
        # proposed plan or another LLM statement.
        if not (pointer.startswith("/after_state/") or pointer.startswith("/geometry/") or pointer.startswith("/delta/") or pointer.startswith("/derived/")):
            continue
        ok, actual = _json_pointer_get(evidence, pointer)
        if not ok or not _loosely_equal(actual, finding.get("observed")):
            continue
        expected_contract = req.get("expected")
        if expected_contract is not None and finding.get("expected") is not None and not _loosely_equal(expected_contract, finding.get("expected")):
            continue
        if expected_contract is not None and _loosely_equal(actual, expected_contract):
            # Concrete evidence already satisfies the requirement; never repair it.
            continue
        grounded.append({
            "requirement_id": rid,
            "evidence_path": pointer,
            "actual": copy.deepcopy(actual),
            "expected": copy.deepcopy(expected_contract),
            "confidence": confidence,
            "reason": str(finding.get("reason") or "")[:500],
        })
    if not grounded:
        return {"allow": False, "reason": "no_grounded_hard_failure", "grounded_failures": []}
    return {"allow": True, "reason": "grounded_hard_failure", "grounded_failures": grounded}


def verified_hybrid_enabled() -> bool:
    return _truthy(os.getenv("CADIA_NEMOTRON_POST_VERIFY_ENABLED"), True)


def verified_hybrid_auto_repair_enabled() -> bool:
    return _truthy(os.getenv("CADIA_NEMOTRON_POST_VERIFY_AUTO_REPAIR"), True)
