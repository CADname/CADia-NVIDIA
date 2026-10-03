from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx


PATCH_VERSION = "2.2.0"
DEFAULT_COSMOS_MODEL = "nvidia/cosmos3-nano-reasoner"
DEFAULT_NEMOTRON_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1"

REPAIR_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "minItems": 1,
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments_json": {"type": "string"},
                    "bind": {"type": ["string", "null"]},
                },
                "required": ["tool", "arguments_json", "bind"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["calls", "note"],
    "additionalProperties": False,
}

# Review repairs may edit/delete an incorrect feature, but they may not replace the
# document, import a flattened model, or create a second unrelated document.
FORBIDDEN_REPAIR_TOOLS = {
    "inventor_new_part",
    "inventor_new_assembly",
    "inventor_open_document",
    "inventor_close_document",
    "inventor_switch_target",
    "inventor_save_document",
    "inventor_export_step",
    "inventor_export_stl",
    "inventor_export_dxf",
    "cad_new_part",
    "cad_new_assembly",
    "cad_open_document",
    "cad_close_document",
    "cad_switch_target",
    "cad_save_document",
    "cad_export_step",
    "cad_export_stl",
    "cad_export_dxf",
    "cad_delete_document_file",
}

# NVIDIA-guided correction is intentionally a *local geometry edit* lane.  Do not
# allow document/session/admin/export/view/assembly commands even if they exist in
# the wider CAD catalog.  This keeps the review pass bounded to the current part.
SAFE_REPAIR_TOOLS = {
    "inventor_set_parameter", "inventor_create_parameter",
    "inventor_create_sketch", "inventor_project_geometry",
    "inventor_draw_line", "inventor_draw_circle", "inventor_draw_rectangle", "inventor_draw_arc",
    "inventor_add_sketch_dimension", "inventor_add_sketch_constraint", "inventor_close_sketch",
    "inventor_extrude", "inventor_revolve", "inventor_fillet", "inventor_chamfer",
    "inventor_create_work_plane", "inventor_create_work_axis", "inventor_hole",
    "inventor_circular_pattern", "inventor_rectangular_pattern",
    "cad_create_parameter", "cad_create_sketch", "cad_project_geometry",
    "cad_draw_line", "cad_draw_circle", "cad_draw_rectangle", "cad_draw_arc",
    "cad_add_sketch_dimension", "cad_add_sketch_constraint", "cad_close_sketch",
    "cad_extrude", "cad_revolve", "cad_fillet", "cad_chamfer",
    "cad_create_work_plane", "cad_create_work_axis", "cad_hole",
    "cad_circular_pattern", "cad_rectangular_pattern",
    "cad_create_spur_gear", "cad_create_gear", "cad_create_shaft", "cad_create_parallel_key",
    "cad_create_bearing", "cad_create_spring", "cad_create_v_pulley", "cad_create_coupling",
    "cad_create_lead_screw_nut", "cad_create_sheet_metal_base", "cad_add_sheet_metal_flange",
    "cad_extrude_advanced", "cad_loft", "cad_sweep", "cad_create_box", "cad_create_cylinder",
    "cad_create_cone", "cad_create_sphere", "cad_create_torus", "cad_create_slot", "cad_create_coil",
    "cad_create_external_thread", "cad_create_metric_hex_bolt", "cad_create_metric_hex_nut",
    "cad_create_metric_washer", "cad_shell", "cad_mirror_body", "cad_transform_body",
    "cad_move_face", "cad_delete_face", "cad_set_face_diameter",
    "cad_edit_sketch_entity", "cad_delete_sketch_entity",
    "cad_edit_sketch_dimension", "cad_delete_sketch_dimension",
    "cad_edit_sketch_constraint", "cad_delete_sketch_constraint",
    "cad_edit_feature_definition", "cad_edit_feature", "cad_suppress_feature", "cad_delete_feature",
    "cad_edit_parameter", "cad_edit_sketch", "cad_edit_work_feature", "cad_edit_solid",
}


class NvidiaReviewError(RuntimeError):
    pass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(low, min(high, value))


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(low, min(high, value))


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", "", "null", "none"}:
            return False
    return False


def _safe_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _safe_error(exc: BaseException, limit: int = 900) -> str:
    # Never return request headers / secrets. httpx exception strings normally do not
    # contain Authorization, but redact common bearer/key forms defensively.
    text = str(exc)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1<redacted>", text)
    text = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", text)
    return text[:limit]


def _compact(value: Any, max_chars: int = 24000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


@dataclass(frozen=True)
class NvidiaReviewConfig:
    enabled: bool
    auto_repair: bool
    cosmos_base_url: str
    cosmos_api_key: str
    cosmos_model: str
    nebius_base_url: str
    nebius_api_key: str
    nemotron_model: str
    timeout_seconds: float
    min_confidence: float
    visual_regression_tolerance: float
    max_source_images: int
    max_total_media_bytes: int

    @classmethod
    def from_env(cls) -> "NvidiaReviewConfig":
        return cls(
            enabled=_bool_env("CADIA_NVIDIA_REVIEW_ENABLED", False),
            auto_repair=_bool_env("CADIA_NVIDIA_AUTO_REPAIR", True),
            cosmos_base_url=os.getenv("CADIA_NVIDIA_COSMOS_BASE_URL", "").strip(),
            cosmos_api_key=os.getenv("CADIA_NVIDIA_COSMOS_API_KEY", "").strip(),
            cosmos_model=os.getenv("CADIA_NVIDIA_COSMOS_MODEL", DEFAULT_COSMOS_MODEL).strip() or DEFAULT_COSMOS_MODEL,
            nebius_base_url=os.getenv("NEBIUS_TOKEN_FACTORY_BASE_URL", DEFAULT_NEBIUS_BASE_URL).strip() or DEFAULT_NEBIUS_BASE_URL,
            nebius_api_key=os.getenv("NEBIUS_API_KEY", "").strip(),
            nemotron_model=os.getenv("NEBIUS_NEMOTRON_MODEL", DEFAULT_NEMOTRON_MODEL).strip() or DEFAULT_NEMOTRON_MODEL,
            timeout_seconds=_float_env("CADIA_NVIDIA_REVIEW_TIMEOUT_SECONDS", 120.0, 10.0, 600.0),
            min_confidence=_float_env("CADIA_NVIDIA_REVIEW_MIN_CONFIDENCE", 0.62, 0.0, 1.0),
            visual_regression_tolerance=_float_env("CADIA_NVIDIA_VISUAL_REGRESSION_TOLERANCE", 0.04, 0.0, 0.25),
            max_source_images=_int_env("CADIA_NVIDIA_MAX_SOURCE_IMAGES", 4, 1, 8),
            max_total_media_bytes=_int_env("CADIA_NVIDIA_MAX_MEDIA_BYTES", 32 * 1024 * 1024, 2 * 1024 * 1024, 96 * 1024 * 1024),
        )

    @property
    def cosmos_configured(self) -> bool:
        return bool(self.cosmos_base_url)

    @property
    def nemotron_configured(self) -> bool:
        return bool(self.nebius_base_url and self.nebius_api_key)

    @property
    def configured(self) -> bool:
        # Nemotron is the minimum required NVIDIA path. Cosmos is an optional
        # visual-verification upgrade for Drawing/Photo→CAD.
        return self.enabled and self.nemotron_configured

    @property
    def visual_configured(self) -> bool:
        return self.enabled and self.cosmos_configured and self.nemotron_configured


def capabilities() -> dict[str, Any]:
    cfg = NvidiaReviewConfig.from_env()
    return {
        "version": PATCH_VERSION,
        "enabled": cfg.enabled,
        "configured": cfg.configured,
        "mode": ("cosmos_nemotron" if cfg.visual_configured else ("nemotron_only" if cfg.configured else "unconfigured")),
        "auto_repair": cfg.auto_repair,
        "cosmos": {
            "configured": cfg.cosmos_configured,
            "model": cfg.cosmos_model,
            "endpoint_configured": bool(cfg.cosmos_base_url),
        },
        "nemotron": {
            "configured": cfg.nemotron_configured,
            "model": cfg.nemotron_model,
            "provider": "Nebius Token Factory",
            "tier": "Ultra",
        },
    }


def render_review_views_isolated(
    runtime: Any,
    job_dir: Path,
    *,
    pass_index: int,
) -> dict[str, Path]:
    """Render CAD review views in a disposable subprocess with a hard timeout.

    This is intentionally used by the NVIDIA review lane for Drawing→CAD so an
    OpenGL/VTK renderer stall cannot pin the FastAPI worker forever.  It exports a
    temporary STL, reuses Photo→CAD's already-proven isolated renderer worker, then
    deletes the temporary STL.  Rendering is advisory/fail-open: zero views simply
    causes the NVIDIA review to skip without changing the valid base CAD.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("photo_cad_render_worker.py")
    if not worker.is_file():
        return {}

    orientations = ("front", "top", "right", "iso_top_right")
    prefix = f"render_p{pass_index + 1}"
    timeout_per_view = _float_env("CADIA_NVIDIA_RENDER_TIMEOUT_SECONDS", 45.0, 10.0, 180.0)
    stl_path: Path | None = None
    try:
        stl_path = Path(runtime.export("stl"))
        if not stl_path.is_file():
            return {}
        cmd = [
            sys.executable,
            str(worker),
            "--stl", str(stl_path),
            "--output-dir", str(job_dir),
            "--prefix", prefix,
            "--orientations", ",".join(orientations),
            "--width", "720",
            "--height", "540",
        ]
        env = os.environ.copy()
        env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_per_view * len(orientations),
            check=False,
            env=env,
            start_new_session=True,
        )
    except Exception:
        return {}
    finally:
        if stl_path is not None:
            try:
                stl_path.unlink(missing_ok=True)
            except Exception:
                pass

    output: dict[str, Path] = {}
    for orientation in orientations:
        path = job_dir / f"{prefix}_{orientation}.png"
        try:
            if path.is_file() and path.stat().st_size > 100:
                output[orientation] = path
        except Exception:
            continue
    return output


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_chat(*, base_url: str, api_key: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    if not base_url:
        raise NvidiaReviewError("AI endpoint base URL is not configured.")
    url = _chat_endpoint(base_url)
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.post(url, headers=_headers(api_key), json=payload)
    except Exception as exc:
        raise NvidiaReviewError(f"Could not reach AI endpoint: {_safe_error(exc)}") from exc
    if response.status_code >= 400:
        body = response.text[:700]
        raise NvidiaReviewError(f"AI endpoint HTTP {response.status_code}: {body}")
    try:
        data = response.json()
    except Exception as exc:
        raise NvidiaReviewError("AI endpoint returned non-JSON data.") from exc
    if not isinstance(data, dict):
        raise NvidiaReviewError("AI endpoint returned an unexpected response shape.")
    return data


def _message_content(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
    except Exception as exc:
        raise NvidiaReviewError("AI endpoint response has no choices/message.") from exc
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        if chunks:
            return "\n".join(chunks)
    raise NvidiaReviewError("AI endpoint response contains no text content.")


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    # Reasoner endpoints may include a private-style <think> prelude in text. CADia
    # consumes only the final JSON object and never persists/replays that prelude.
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    raise NvidiaReviewError("AI model did not return a parseable JSON object.")


def _image_data_url(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw)


def _select_media(
    sources: dict[str, Path],
    renders: dict[str, Path],
    *,
    max_source_images: int,
    max_total_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    used_sources: list[str] = []
    used_renders: list[str] = []
    skipped: list[str] = []
    total = 0

    def add_group(kind: str, rows: Iterable[tuple[str, Path]], limit: int) -> None:
        nonlocal total
        count = 0
        for label, path in rows:
            if count >= limit:
                break
            try:
                size = path.stat().st_size
            except Exception:
                skipped.append(f"{kind}:{label}:missing")
                continue
            if total + size > max_total_bytes:
                skipped.append(f"{kind}:{label}:media-budget")
                continue
            try:
                url, size = _image_data_url(path)
            except Exception:
                skipped.append(f"{kind}:{label}:read-failed")
                continue
            parts.append({"type": "text", "text": f"{kind.upper()} IMAGE [{label}]"})
            parts.append({"type": "image_url", "image_url": {"url": url}})
            total += size
            count += 1
            (used_sources if kind == "source" else used_renders).append(label)

    add_group("source", sorted(sources.items()), max_source_images)
    add_group("cad_render", sorted(renders.items()), 4)
    if not used_sources or not used_renders:
        raise NvidiaReviewError("NVIDIA visual review requires at least one source image and one CAD render.")
    return parts, {
        "source_images": used_sources,
        "cad_renders": used_renders,
        "media_bytes": total,
        "skipped": skipped,
    }


def _normalize_discrepancies(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(value, list):
        return rows
    for item in value[:20]:
        if isinstance(item, str):
            rows.append({"severity": "medium", "category": "geometry", "description": item[:700], "evidence": "", "target_hint": ""})
            continue
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "medium").lower()
        if severity not in {"high", "medium", "low"}:
            severity = "medium"
        rows.append({
            "severity": severity,
            "category": _safe_text(item.get("category") or "geometry", 100),
            "description": _safe_text(item.get("description"), 700),
            "evidence": _safe_text(item.get("evidence"), 700),
            "target_hint": _safe_text(item.get("target_hint"), 300),
        })
    return rows


def cosmos_compare(
    *,
    source_kind: str,
    source_images: dict[str, Path],
    cad_renders: dict[str, Path],
    spec: dict[str, Any],
    validation: dict[str, Any],
    notes: str,
    dimension_hint: str,
    config: NvidiaReviewConfig,
) -> dict[str, Any]:
    media_parts, media_meta = _select_media(
        source_images,
        cad_renders,
        max_source_images=config.max_source_images,
        max_total_bytes=config.max_total_media_bytes,
    )
    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Compare the source engineering visual(s) with the CAD render views of the SAME part. "
            "This is verification, not redesign. Perspective and lighting differences are expected. "
            "Judge engineering geometry: silhouette/proportions, hole-slot-pocket-boss counts and placement, "
            "ribs/handles, symmetry/repetition, fillets/chamfers, and cross-view contradictions. "
            "Known/measured dimensions are hard constraints. Do not infer a hidden feature as certain unless "
            "supported by another view, symmetry, or the design spec.\n\n"
            f"SOURCE_KIND={source_kind}\n"
            f"USER_NOTES={notes or '(none)'}\n"
            f"KNOWN_DIMENSION={dimension_hint or '(none)'}\n"
            f"DESIGN_SPEC={_compact(spec, 18000)}\n"
            f"DETERMINISTIC_VALIDATION={_compact(validation, 8000)}\n"
        ),
    }]
    user_content.extend(media_parts)
    user_content.append({
        "type": "text",
        "text": (
            "Return ONLY one JSON object with exactly these top-level fields: "
            "action, confidence, visual_agreement, discrepancies, preserve, summary. "
            "action must be accept or repair. confidence and visual_agreement are numbers 0..1. "
            "discrepancies is an array of objects {severity: high|medium|low, category, description, evidence, target_hint}. "
            "preserve is an array of constraints/features that must not change. "
            "If there is no clear engineering mismatch, action=accept and discrepancies=[]. "
            "Do not include markdown and do not expose chain-of-thought."
        ),
    })
    payload = {
        "model": config.cosmos_model,
        "messages": [
            {"role": "system", "content": "You are NVIDIA Cosmos acting as an independent mechanical CAD visual verifier. Return concise engineering conclusions only."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 1800,
        "stream": False,
    }
    started = time.perf_counter()
    response = _post_chat(
        base_url=config.cosmos_base_url,
        api_key=config.cosmos_api_key,
        payload=payload,
        timeout_seconds=config.timeout_seconds,
    )
    decoded = _extract_json_object(_message_content(response))
    action = str(decoded.get("action") or "accept").strip().lower()
    if action not in {"accept", "repair"}:
        action = "repair" if decoded.get("discrepancies") else "accept"
    discrepancies = _normalize_discrepancies(decoded.get("discrepancies"))
    # The response contract says accept => no discrepancies.  Treat contradictory
    # output conservatively as a repair candidate rather than a false verification.
    if action == "accept" and discrepancies:
        action = "repair"
    return {
        "ok": True,
        "model": config.cosmos_model,
        "action": action,
        "confidence": _clamp01(decoded.get("confidence")),
        "visual_agreement": _clamp01(decoded.get("visual_agreement")),
        "discrepancies": discrepancies,
        "preserve": [_safe_text(x, 300) for x in (decoded.get("preserve") or []) if str(x).strip()][:30],
        "summary": _safe_text(decoded.get("summary"), 1200),
        "media": media_meta,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def nemotron_audit(
    *,
    source_kind: str,
    cosmos: dict[str, Any],
    spec: dict[str, Any],
    validation: dict[str, Any],
    current_state: dict[str, Any],
    notes: str,
    dimension_hint: str,
    config: NvidiaReviewConfig,
) -> dict[str, Any]:
    system = (
        "You are NVIDIA Nemotron 3 Ultra, the engineering review planner for an editable mechanical CAD system. "
        "Cosmos has already compared source visuals with CAD renders. Decide whether the reported discrepancy is "
        "actionable without violating explicit dimensions or existing valid geometry. Produce a conservative repair brief, "
        "not executable code and not CAD tool calls. Preserve design intent and prefer the smallest local change."
    )
    user = f"""SOURCE_KIND={source_kind}
USER_NOTES={notes or '(none)'}
KNOWN_DIMENSION={dimension_hint or '(none)'}
COSMOS_VISUAL_REVIEW={_compact(cosmos, 18000)}
DESIGN_SPEC={_compact(spec, 18000)}
CURRENT_VALIDATION={_compact(validation, 9000)}
CURRENT_CAD_STATE={_compact(current_state, 22000)}

Return ONLY one JSON object with fields:
should_repair (boolean), confidence (0..1), objective (string), preserve (array of strings),
repairs (array of objects with target, change, reason), verification (array of strings), warnings (array of strings).
Do not emit source code, shell commands, CAD tool names, or chain-of-thought.
If Cosmos evidence is weak/ambiguous or conflicts with a hard dimension, should_repair=false.
"""
    payload = {
        "model": config.nemotron_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1800,
        "stream": False,
    }
    started = time.perf_counter()
    response = _post_chat(
        base_url=config.nebius_base_url,
        api_key=config.nebius_api_key,
        payload=payload,
        timeout_seconds=config.timeout_seconds,
    )
    decoded = _extract_json_object(_message_content(response))
    repairs: list[dict[str, str]] = []
    for item in (decoded.get("repairs") or [])[:20]:
        if not isinstance(item, dict):
            continue
        repairs.append({
            "target": _safe_text(item.get("target"), 300),
            "change": _safe_text(item.get("change"), 600),
            "reason": _safe_text(item.get("reason"), 600),
        })
    return {
        "ok": True,
        "model": config.nemotron_model,
        "provider": "Nebius Token Factory",
        "should_repair": _strict_bool(decoded.get("should_repair")),
        "confidence": _clamp01(decoded.get("confidence")),
        "objective": _safe_text(decoded.get("objective"), 1000),
        "preserve": [_safe_text(x, 300) for x in (decoded.get("preserve") or []) if str(x).strip()][:30],
        "repairs": repairs,
        "verification": [_safe_text(x, 500) for x in (decoded.get("verification") or []) if str(x).strip()][:30],
        "warnings": [_safe_text(x, 500) for x in (decoded.get("warnings") or []) if str(x).strip()][:20],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def nemotron_spec_audit(
    *,
    source_kind: str,
    spec: dict[str, Any],
    validation: dict[str, Any],
    current_state: dict[str, Any],
    notes: str,
    dimension_hint: str,
    config: NvidiaReviewConfig,
) -> dict[str, Any]:
    """Conservative fallback when Cosmos visual access is unavailable."""
    system = (
        "You are NVIDIA Nemotron 3 Ultra acting as a conservative mechanical CAD engineering reviewer. "
        "No visual model is available in this pass: you CANNOT inspect the source image or drawing pixels. "
        "Audit only explicit evidence present in the design spec, known dimensions, deterministic validation, "
        "user notes, and current CAD state. Do not invent geometry that is not represented in those inputs. "
        "Recommend a repair only for a concrete high-confidence mismatch supported by that evidence, and prefer "
        "the smallest local edit. Return a repair brief, never executable code or CAD tool calls."
    )
    user = f"""SOURCE_KIND={source_kind}
REVIEW_MODE=nemotron_only_no_visual_access
USER_NOTES={notes or '(none)'}
KNOWN_DIMENSION={dimension_hint or '(none)'}
DESIGN_SPEC={_compact(spec, 18000)}
CURRENT_VALIDATION={_compact(validation, 10000)}
CURRENT_CAD_STATE={_compact(current_state, 24000)}

Return ONLY one JSON object with fields:
should_repair (boolean), confidence (0..1), objective (string), preserve (array of strings),
repairs (array of objects with target, change, reason), verification (array of strings), warnings (array of strings).
Do not emit source code, shell commands, CAD tool names, visual claims, or chain-of-thought.
If the available non-visual evidence does not prove a mismatch, should_repair=false.
"""
    payload = {
        "model": config.nemotron_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1800,
        "stream": False,
    }
    started = time.perf_counter()
    response = _post_chat(
        base_url=config.nebius_base_url,
        api_key=config.nebius_api_key,
        payload=payload,
        timeout_seconds=config.timeout_seconds,
    )
    decoded = _extract_json_object(_message_content(response))
    repairs: list[dict[str, str]] = []
    for item in (decoded.get("repairs") or [])[:20]:
        if not isinstance(item, dict):
            continue
        repairs.append({
            "target": _safe_text(item.get("target"), 300),
            "change": _safe_text(item.get("change"), 600),
            "reason": _safe_text(item.get("reason"), 600),
        })
    return {
        "ok": True,
        "mode": "nemotron_only",
        "visual_access": False,
        "model": config.nemotron_model,
        "provider": "Nebius Token Factory",
        "should_repair": _strict_bool(decoded.get("should_repair")),
        "confidence": _clamp01(decoded.get("confidence")),
        "objective": _safe_text(decoded.get("objective"), 1000),
        "preserve": [_safe_text(x, 300) for x in (decoded.get("preserve") or []) if str(x).strip()][:30],
        "repairs": repairs,
        "verification": [_safe_text(x, 500) for x in (decoded.get("verification") or []) if str(x).strip()][:30],
        "warnings": [_safe_text(x, 500) for x in (decoded.get("warnings") or []) if str(x).strip()][:20],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _decode_repair_plan(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise NvidiaReviewError("Codex NVIDIA repair plan is not an object.")
    calls_raw = raw.get("calls")
    if not isinstance(calls_raw, list) or not calls_raw:
        raise NvidiaReviewError("Codex NVIDIA repair plan contains no tool calls.")
    calls: list[dict[str, Any]] = []
    for index, item in enumerate(calls_raw[:24], 1):
        if not isinstance(item, dict):
            raise NvidiaReviewError(f"Repair call {index} is invalid.")
        tool = str(item.get("tool") or "").strip()
        if tool in FORBIDDEN_REPAIR_TOOLS:
            raise NvidiaReviewError(f"NVIDIA repair attempted a forbidden document-level tool: {tool}")
        if tool not in SAFE_REPAIR_TOOLS:
            raise NvidiaReviewError(f"NVIDIA repair attempted a non-local or unsupported repair tool: {tool}")
        try:
            arguments = json.loads(str(item.get("arguments_json") or "{}"))
        except Exception as exc:
            raise NvidiaReviewError(f"Repair call {index} arguments_json is invalid JSON.") from exc
        if not isinstance(arguments, dict):
            raise NvidiaReviewError(f"Repair call {index} arguments must decode to an object.")
        row: dict[str, Any] = {"tool": tool, "arguments": arguments}
        bind = item.get("bind")
        if isinstance(bind, str) and bind.strip():
            row["bind"] = bind.strip()
        calls.append(row)
    return {"calls": calls, "note": _safe_text(raw.get("note"), 1600)}


def codex_repair_plan(
    *,
    agent: Any,
    source_kind: str,
    repair_brief: dict[str, Any],
    cosmos: dict[str, Any],
    spec: dict[str, Any],
    validation: dict[str, Any],
    current_state: dict[str, Any],
    on_event: Callable[[str, Any], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    catalog = getattr(agent, "_catalog", None)
    client = getattr(agent, "_web_client", None)
    if client is None or catalog is None:
        raise NvidiaReviewError("Connected Codex agent does not expose the expected CAD planner interface.")
    evidence_intro = (
        "NVIDIA Cosmos found a visual discrepancy and NVIDIA Nemotron produced a conservative engineering repair brief."
        if cosmos
        else "NVIDIA Nemotron found a high-confidence engineering mismatch from explicit spec/state/validation evidence; no visual Cosmos evidence is available."
    )
    prompt = f"""You are CADia's executor-side repair planner.
{evidence_intro}
Translate ONLY that approved brief into the smallest set of CADia typed EDIT operations on the CURRENT model.
Do not create a new part/assembly, do not open/import another model, and do not replace the whole document.
Do not emit Python, CadQuery, shell, filesystem, network calls, STEP imports, or prose outside schema JSON.
Preserve all explicit/known dimensions and every item in the preserve lists.
If a wrong local feature must be removed, use a feature-level edit/delete only when safe.

SOURCE_KIND={source_kind}
COSMOS_REVIEW={_compact(cosmos, 14000)}
NEMOTRON_REPAIR_BRIEF={_compact(repair_brief, 16000)}
DESIGN_SPEC={_compact(spec, 16000)}
CURRENT_VALIDATION={_compact(validation, 7000)}
CURRENT_STATE={_compact(current_state, 22000)}
TOOL_CATALOG={_compact(catalog, 50000)}

For every call, encode the complete arguments object as JSON text in arguments_json.
Return only schema-conforming JSON.
"""
    raw = client.complete_json(
        prompt,
        output_schema=REPAIR_PLAN_SCHEMA,
        model=getattr(agent, "model", None),
        effort=getattr(agent, "reasoning", None),
        on_event=on_event,
        cancel_check=cancel_check,
        stall_timeout_seconds=90,
        hard_timeout_seconds=300,
    )
    return _decode_repair_plan(raw)


def _revision_of(state: dict[str, Any] | None) -> int | None:
    if not isinstance(state, dict):
        return None
    doc = state.get("document")
    if not isinstance(doc, dict):
        return None
    try:
        return int(doc.get("revision"))
    except Exception:
        return None


def _severity_count(review: dict[str, Any], severity: str) -> int:
    return sum(1 for row in (review.get("discrepancies") or []) if isinstance(row, dict) and row.get("severity") == severity)


def _severity_burden(review: dict[str, Any]) -> int:
    weights = {"high": 3, "medium": 2, "low": 1}
    total = 0
    for row in (review.get("discrepancies") or []):
        if isinstance(row, dict):
            total += weights.get(str(row.get("severity") or "medium").lower(), 2)
    return total


def run_engineering_review(
    *,
    source_kind: str,
    source_images: dict[str, Path],
    spec: dict[str, Any],
    validation_before: dict[str, Any],
    notes: str,
    dimension_hint: str,
    agent: Any,
    current_state_provider: Callable[[], dict[str, Any]],
    render_before: Callable[[], dict[str, Path]],
    render_after: Callable[[], dict[str, Path]],
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    validate_after: Callable[[], dict[str, Any]],
    validation_accept: Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str] | bool],
    validate_plan: Callable[[dict[str, Any]], None],
    execute_plan: Callable[[dict[str, Any], str], tuple[bool, str | None]],
    on_event: Callable[[str, Any], None] | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    cfg = NvidiaReviewConfig.from_env()
    mode = "cosmos_nemotron" if cfg.visual_configured else "nemotron_only"
    result: dict[str, Any] = {
        "version": PATCH_VERSION,
        "status": "disabled",
        "configured": cfg.configured,
        "mode": mode,
        "accepted": False,
        "auto_repair": cfg.auto_repair,
        "cosmos_model": cfg.cosmos_model,
        "nemotron_model": cfg.nemotron_model,
    }
    if not cfg.enabled:
        result["reason"] = "CADIA_NVIDIA_REVIEW_ENABLED is false."
        return result
    if not cfg.nemotron_configured:
        result["status"] = "not_configured"
        result["reason"] = "Missing configuration: NEBIUS_API_KEY"
        return result
    if not cfg.cosmos_configured:
        result["cosmos_optional"] = True
        result["visual_verification"] = False

    def check_cancel() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("NVIDIA Engineering Review was cancelled.")

    def stage(message: str, percent: int) -> None:
        check_cancel()
        if on_stage:
            try:
                on_stage(message, percent)
            except Exception:
                pass

    rollback_snapshot: Any | None = None
    mutated = False
    try:
        cosmos_before: dict[str, Any] = {}
        if cfg.cosmos_configured:
            stage("NVIDIA Cosmos · comparing source visuals with CAD…", 92)
            state_before_render = current_state_provider()
            before_renders = render_before()
            if not before_renders:
                result["status"] = "skipped"
                result["reason"] = "CAD render views were unavailable for Cosmos visual review; base CAD was kept."
                return result
            review_state = current_state_provider()
            revision_before_render = _revision_of(state_before_render)
            review_revision = _revision_of(review_state)
            if revision_before_render is not None and review_revision is not None and revision_before_render != review_revision:
                result["status"] = "stale_review"
                result["reason"] = "CAD changed while validation renders were being captured; base CAD was kept."
                return result
            result["review_revision"] = review_revision
            cosmos_before = cosmos_compare(source_kind=source_kind,source_images=source_images,cad_renders=before_renders,spec=spec,validation=validation_before,notes=notes,dimension_hint=dimension_hint,config=cfg)
            result["cosmos_before"] = cosmos_before
            check_cancel()
            if cosmos_before.get("action") == "accept" and not cosmos_before.get("discrepancies"):
                if float(cosmos_before.get("confidence") or 0.0) >= cfg.min_confidence:
                    result["status"] = "verified"; result["accepted"] = True; result["reason"] = "NVIDIA Cosmos found no actionable engineering discrepancy."
                else:
                    result["status"] = "reviewed_no_repair"; result["reason"] = "Cosmos did not find a clear discrepancy, but confidence was below the verification threshold."
                return result
            if not (cosmos_before.get("action") == "repair" and cosmos_before.get("discrepancies") and float(cosmos_before.get("confidence") or 0.0) >= cfg.min_confidence):
                result["status"] = "reviewed_no_repair"; result["reason"] = "Visual evidence was not strong enough for an automatic repair."; return result
            stage("Nebius Nemotron 3 Ultra · auditing engineering repair need…", 94)
            audit = nemotron_audit(source_kind=source_kind,cosmos=cosmos_before,spec=spec,validation=validation_before,current_state=review_state,notes=notes,dimension_hint=dimension_hint,config=cfg)
        else:
            stage("Nebius Nemotron 3 Ultra · auditing CAD spec/state (Cosmos optional)…", 92)
            review_state = current_state_provider(); review_revision = _revision_of(review_state); result["review_revision"] = review_revision
            audit = nemotron_spec_audit(source_kind=source_kind,spec=spec,validation=validation_before,current_state=review_state,notes=notes,dimension_hint=dimension_hint,config=cfg)

        result["nemotron"] = audit
        check_cancel()
        actionable = bool(audit.get("should_repair") and float(audit.get("confidence") or 0.0) >= cfg.min_confidence and audit.get("repairs"))
        if not actionable:
            result["status"] = "reviewed_no_repair"
            result["reason"] = "No jointly approved high-confidence repair was required." if cfg.cosmos_configured else "Nemotron found no high-confidence mismatch in explicit spec/state/validation evidence; visual Cosmos verification is unavailable."
            return result
        if not cfg.auto_repair:
            result["status"] = "issues_found"; result["reason"] = "NVIDIA found an actionable discrepancy; automatic repair is disabled."; return result

        latest_state = current_state_provider(); latest_revision = _revision_of(latest_state)
        if review_revision is not None and latest_revision is not None and latest_revision != review_revision:
            result["status"] = "stale_review"; result["reason"] = "CAD changed while NVIDIA review was running; stale repair was not applied."; return result

        stage("Codex · translating NVIDIA repair brief into CAD edits…", 95)
        plan = codex_repair_plan(agent=agent,source_kind=source_kind,repair_brief=audit,cosmos=cosmos_before,spec=spec,validation=validation_before,current_state=latest_state,on_event=on_event,cancel_check=cancel_check)
        validate_plan(plan); check_cancel()
        result["repair"] = {"attempted": True,"tool_count": len(plan.get("calls") or []),"note": _safe_text(plan.get("note"), 1000)}
        latest_state = current_state_provider(); latest_revision = _revision_of(latest_state)
        if review_revision is not None and latest_revision is not None and latest_revision != review_revision:
            result["status"] = "stale_review"; result["reason"] = "CAD changed after repair planning; stale repair was not applied."; return result

        rollback_snapshot = snapshot(); stage("CADia · applying one bounded NVIDIA-guided repair…", 96)
        prompt = "Apply the approved NVIDIA Engineering Review repair to the current CAD model only. Preserve explicit dimensions and unrelated existing features. Repair brief: " + _compact(audit, 9000)
        mutated = True
        try:
            ok, error = execute_plan(plan, prompt)
        except Exception:
            restore(rollback_snapshot); mutated = False; raise
        check_cancel()
        if not ok:
            restore(rollback_snapshot); mutated = False; result["status"] = "repair_failed"; result["reason"] = _safe_text(error or "CAD execution failed", 900); return result

        after_validation = validate_after(); accepted = validation_accept(validation_before, after_validation)
        if isinstance(accepted, tuple): objective_ok, objective_reason = bool(accepted[0]), str(accepted[1])
        else: objective_ok, objective_reason = bool(accepted), ""
        if not objective_ok:
            restore(rollback_snapshot); mutated = False; result["status"] = "repair_rejected"; result["reason"] = objective_reason or "Deterministic CAD validation regressed."; result["validation_after"] = after_validation; return result

        if not cfg.cosmos_configured:
            result["status"] = "corrected"; result["accepted"] = True; result["reason"] = "Nemotron-guided repair passed deterministic CAD validation; Cosmos visual re-verification was not configured."; result["validation_after"] = after_validation
            if isinstance(result.get("repair"), dict): result["repair"]["accepted"] = True; result["repair"]["verification"] = "deterministic_only"
            return result

        stage("NVIDIA Cosmos · re-verifying repaired CAD…", 98)
        after_renders = render_after()
        if not after_renders:
            restore(rollback_snapshot); mutated = False; result["status"] = "repair_rejected"; result["reason"] = "Could not render the repaired CAD for independent re-verification."; return result
        cosmos_after = cosmos_compare(source_kind=source_kind,source_images=source_images,cad_renders=after_renders,spec=spec,validation=after_validation,notes=notes,dimension_hint=dimension_hint,config=cfg)
        result["cosmos_after"] = cosmos_after; check_cancel()
        before_score=float(cosmos_before.get("visual_agreement") or 0.0); after_score=float(cosmos_after.get("visual_agreement") or 0.0)
        high_before=_severity_count(cosmos_before,"high"); high_after=_severity_count(cosmos_after,"high"); total_before=len(cosmos_before.get("discrepancies") or []); total_after=len(cosmos_after.get("discrepancies") or []); burden_before=_severity_burden(cosmos_before); burden_after=_severity_burden(cosmos_after); after_confident=float(cosmos_after.get("confidence") or 0.0) >= cfg.min_confidence
        non_regressing=after_score + cfg.visual_regression_tolerance >= before_score and high_after <= high_before
        visibly_improved=bool((cosmos_after.get("action") == "accept" and not cosmos_after.get("discrepancies")) or burden_after < burden_before or total_after < total_before)
        if not (after_confident and non_regressing and visibly_improved):
            restore(rollback_snapshot); mutated=False; result["status"]="repair_rejected"; result["reason"]="Independent NVIDIA visual re-check did not confirm a real improvement; the base CAD was restored."; result["validation_after"]=after_validation; return result
        result["status"]="corrected"; result["accepted"]=True; result["reason"]="NVIDIA-guided repair passed CAD validation and independent visual re-verification."; result["validation_after"]=after_validation; result["visual_delta"]=round(after_score-before_score,5)
        if isinstance(result.get("repair"),dict): result["repair"]["accepted"]=True; result["repair"]["verification"]="cosmos_and_deterministic"
        return result
    except InterruptedError:
        if mutated and rollback_snapshot is not None:
            try: restore(rollback_snapshot)
            except Exception: pass
        raise
    except Exception as exc:
        if mutated and rollback_snapshot is not None:
            try: restore(rollback_snapshot)
            except Exception: pass
        result["status"] = "degraded"; result["reason"] = _safe_error(exc); result["accepted"] = False; return result
