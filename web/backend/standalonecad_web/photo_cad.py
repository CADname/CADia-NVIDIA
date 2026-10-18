from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from standalonecad.recovery import execute_with_failure_recovery

from .cad_runtime import CadRuntime, runtime_manager
from .config import settings
from .heavy_jobs import HeavyJobBusy, heavy_job_manager
from .isolated_execution import run_isolated
from .photo_cad_external import (
    ExternalEngineResult,
    build_reference_mesh_from_agentic_evidence,
    capabilities as external_capabilities,
    run_agentic_cad,
    run_cadcoder,
    run_cadfit,
    run_pixal3d,
    run_cadreasoner,
    run_cadena,
    score_with_agentic_evidence,
)
from .providers import selected_provider_name
from .providers.app_server import app_server_manager
from .state import planning_state
from .visual_cad import (
    REFINEMENT_SCHEMA,
    VISUAL_PLAN_SCHEMA,
    _decode_refinement,
    _decode_visual_plan,
    _render_validation_views,
    _validate_plan,
    _write_json,
)
from .web_agent import DesktopParityCodexAgent
from .nvidia_review import run_engineering_review
from .opencv_cad import (
    analyze_images as opencv_analyze_images,
    compact_analysis as opencv_compact_analysis,
    compact_verification as opencv_compact_verification,
    prompt_summary as opencv_prompt_summary,
    verification_prompt as opencv_verification_prompt,
    verify_source_to_renders as opencv_verify_source_to_renders,
)


ALLOWED_PHOTO_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_PHOTOS = 8
PHOTO_REFINEMENTS = 2
PHOTO_WORKERS = max(1, min(1, settings.max_concurrent_ai_turns))

PHOTO_RENDER_SUBPROCESS_GUARD = True
PHOTO_RENDER_TIMEOUT_SECONDS = max(10, int(os.getenv("CADIA_PHOTO_RENDER_TIMEOUT", "45")))
PHOTO_CASCADE_MIN_NATIVE = max(0.0, min(1.0, float(os.getenv("CADIA_PHOTO_CASCADE_MIN_NATIVE", "0.90"))))
PHOTO_CASCADE_MIN_OBSERVATION = max(0.0, min(1.0, float(os.getenv("CADIA_PHOTO_CASCADE_MIN_OBSERVATION", "0.78"))))
PHOTO_CASCADE_MIN_COMBINED = max(0.0, min(1.0, float(os.getenv("CADIA_PHOTO_CASCADE_MIN_COMBINED", "0.86"))))


def _render_validation_views(
    runtime: CadRuntime,
    job_dir: Path,
    *,
    pass_index: int,
    stl_path: Path | None = None,
) -> dict[str, Path]:
    """Render the four Photo→CAD QA views in one isolated subprocess.

    The API process exports the current solid at most once. The worker loads that
    STL once and renders all orientations in one process, avoiding four Python/VTK
    startups and four STL reads while keeping native OpenGL failures isolated from
    FastAPI/Uvicorn.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    stl = stl_path
    if stl is None:
        try:
            stl = runtime.export("stl")
        except Exception:
            return {}
    stl = Path(stl)
    if not stl.is_file():
        return {}
    worker = Path(__file__).with_name("photo_cad_render_worker.py")
    if not worker.is_file():
        return {}

    orientations = ("front", "top", "right", "iso_top_right")
    prefix = f"render_p{pass_index + 1}"
    cmd = [
        sys.executable,
        str(worker),
        "--stl", str(stl),
        "--output-dir", str(job_dir),
        "--prefix", prefix,
        "--orientations", ",".join(orientations),
        "--width", "720",
        "--height", "540",
    ]
    env = os.environ.copy()
    env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    try:
        completed = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PHOTO_RENDER_TIMEOUT_SECONDS * len(orientations),
            check=False,
            env=env,
            start_new_session=True,
        )
    except Exception:
        return {}
    # Preserve the old per-view behaviour: if one orientation fails but others
    # were written successfully, keep the successful views instead of discarding
    # the whole render set.
    output: dict[str, Path] = {}
    for orientation in orientations:
        path = job_dir / f"{prefix}_{orientation}.png"
        if path.is_file() and path.stat().st_size > 100:
            output[orientation] = path
    return output


def cascade_quality_pass(validation: dict[str, Any], observation: float, combined: float) -> bool:
    """Conservative quality gate for skipping the next expensive geometry engine."""
    return bool(
        validation.get("brep_valid")
        and validation.get("hard_dimensions_passed")
        and float(validation.get("score") or 0.0) >= PHOTO_CASCADE_MIN_NATIVE
        and float(observation) >= PHOTO_CASCADE_MIN_OBSERVATION
        and float(combined) >= PHOTO_CASCADE_MIN_COMBINED
    )

PHOTO_JOB_HARD_TIMEOUT_SECONDS = max(300, int(os.getenv("CADIA_PHOTO_JOB_HARD_TIMEOUT", "7200")))
TERMINAL_STATUSES = {"done", "failed", "cancelled"}


_DIMENSION_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value_mm": {"type": ["number", "null"]},
        "source": {"type": "string", "enum": ["known", "measured", "estimated", "inferred_hidden", "unknown"]},
        "confidence": {"type": "number"},
    },
    "required": ["value_mm", "source", "confidence"],
    "additionalProperties": False,
}

PHOTO_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "part_name": {"type": "string"},
        "object_summary": {"type": "string"},
        "views_detected": {"type": "array", "items": {"type": "string"}},
        "principal_axes_assumption": {"type": "string"},
        "overall_dimensions_mm": {
            "type": "object",
            "properties": {"x": _DIMENSION_VALUE_SCHEMA, "y": _DIMENSION_VALUE_SCHEMA, "z": _DIMENSION_VALUE_SCHEMA},
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "description": {"type": "string"},
                    "count": {"type": ["integer", "null"]},
                    "axis": {"type": ["string", "null"]},
                    "operation_hint": {"type": ["string", "null"]},
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value_mm": {"type": ["number", "null"]},
                                "source": {"type": "string", "enum": ["known", "measured", "estimated", "inferred_hidden", "unknown"]},
                                "confidence": {"type": "number"},
                            },
                            "required": ["name", "value_mm", "source", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                    "location_mm": {
                        "type": "object",
                        "properties": {
                            "x": {"type": ["number", "null"]},
                            "y": {"type": ["number", "null"]},
                            "z": {"type": ["number", "null"]},
                        },
                        "required": ["x", "y", "z"],
                        "additionalProperties": False,
                    },
                    "visible_in_views": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "description", "count", "axis", "operation_hint", "dimensions", "location_mm", "visible_in_views", "confidence"],
                "additionalProperties": False,
            },
        },
        "symmetry": {"type": "array", "items": {"type": "string"}},
        "repetitions": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "hidden_geometry_assumptions": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "part_name", "object_summary", "views_detected", "principal_axes_assumption", "overall_dimensions_mm",
        "features", "symmetry", "repetitions", "assumptions", "hidden_geometry_assumptions", "uncertainties", "confidence",
    ],
    "additionalProperties": False,
}

PHOTO_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["accept", "refine"]},
        "confidence": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "visual_agreement": {"type": "number"},
        "calls": REFINEMENT_SCHEMA["properties"]["calls"],
        "note": {"type": "string"},
    },
    "required": ["action", "confidence", "issues", "visual_agreement", "calls", "note"],
    "additionalProperties": False,
}

PHOTO_ASSESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "visual_agreement": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["confidence", "issues", "visual_agreement", "note"],
    "additionalProperties": False,
}


@dataclass
class PhotoCadJob:
    id: str
    user_id: str
    project_id: str
    input_files: dict[str, str]
    notes: str = ""
    dimension_hint: str = ""
    model: str = "gpt-5.6-sol"
    effort: str = "high"
    vision_mode: str = "standard"
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline_at: float | None = None
    timed_out: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "vision_mode": self.vision_mode,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_at": self.deadline_at,
        }


class PhotoCadCodexAgent(DesktopParityCodexAgent):
    def extract_spec(
        self,
        images: dict[str, Path],
        *,
        notes: str,
        dimension_hint: str,
        on_event: Callable[[str, Any], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        prompt = f"""You are the perception stage of CADia Photo-to-CAD.
The attached images are photographs of the SAME physical product/part from different viewpoints.
Inspect the attached images directly. Do not call shell, Python, filesystem or image tools.

Goal: infer a MANUFACTURABLE, EDITABLE parametric CAD representation, not an artistic mesh.
Separate direct visual evidence from assumptions. Never report an estimated dimension as measured.
Use millimetres. A user-supplied known dimension is a hard metric anchor. Without a metric anchor,
choose plausible editable dimensions but mark them estimated. Hidden/back-side geometry must be marked
inferred_hidden unless it is supported by another view or symmetry evidence.

Prefer engineering features: sketch/extrude, revolve, holes, pockets/cuts, bosses, slots, shells,
patterns, fillets, chamfers, sweeps and lofts. Ignore texture, logos, labels, scratches and lighting.
Infer a stable right-handed principal-axis frame for the object and explain it briefly.
For every feature, identify which uploaded views support it.

User notes: {notes or '(none)'}
Known dimension / scale anchor: {dimension_hint or '(none)'}
Uploaded views: {', '.join(images.keys())}

Return only schema-conforming JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=PHOTO_SPEC_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            local_images=list(images.values()),
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        return normalize_photo_spec(raw)

    def plan_from_photo_spec(
        self,
        spec: dict[str, Any],
        *,
        current_state: dict[str, Any],
        notes: str,
        dimension_hint: str,
        external_guidance: str,
        validator_feedback: str,
        attempt: int,
        on_event: Callable[[str, Any], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        catalog = json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":"))
        guidance = external_guidance.strip()
        if len(guidance) > 50000:
            guidance = guidance[:50000] + "\n...<truncated>"
        prompt = f"""You are CADia's Photo-to-CAD native planner.
The photographs have already been converted into PHOTO_DESIGN_SPEC. Build ONE new editable parametric
part using ONLY TOOL_CATALOG operations. Never output Python/CadQuery/shell/filesystem/network calls.
Start with inventor_new_part. Prefer a compact semantic feature history over primitive soup.

Hard constraints:
- dimensions whose source is known or measured must be preserved exactly;
- estimated/inferred dimensions may be adjusted to produce a coherent manufacturable solid;
- preserve visible hole/slot/pocket/boss counts, symmetry and repetitions;
- do not model texture, labels or lighting;
- when hidden geometry is uncertain, choose the simplest editable engineering assumption;
- use native sketches/extrudes/revolves/holes/patterns/fillets/chamfers/shells where available;
- do not import a flattened STEP as the final answer.

PHOTO_DESIGN_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
USER_NOTES={notes or '(none)'}
METRIC_ANCHOR={dimension_hint or '(none)'}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
VALIDATOR_FEEDBACK={validator_feedback or '(none)'}
EXTERNAL_GEOMETRY_GUIDANCE={guidance or '(none available)'}
ATTEMPT={attempt}
ATTEMPT_STRATEGY={'direct PhotoDesignSpec reconstruction; prefer the simplest faithful native feature tree' if attempt == 1 else 'alternative reconstruction; use validator feedback and external geometry proposals to correct or challenge the first interpretation'}
TOOL_CATALOG={catalog}

External geometry guidance is ADVISORY only. It may contain CadQuery/DSL text from a public reverse-engineering
engine. Do NOT execute or reproduce that code. Translate useful geometry/feature intent into CADia typed tools.
For each call encode the complete argument object as JSON text in arguments_json. Return only schema JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=VISUAL_PLAN_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        return _decode_visual_plan(raw, allow_empty=False)

    def review_photo_match(
        self,
        *,
        spec: dict[str, Any],
        originals: dict[str, Path],
        rendered: dict[str, Path],
        current_state: dict[str, Any],
        pass_index: int,
        external_guidance: str,
        opencv_feedback: dict[str, Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        images = list(originals.values()) + list(rendered.values())
        prompt = f"""You are the greedy visual correction stage of CADia Photo-to-CAD.
The attachments contain original PHOTOGRAPHS first and CAD RENDERS second. Compare the same physical object.
This is NOT a fresh redesign. Keep the existing valid CAD and make only feature-level corrections with TOOL_CATALOG.

PHOTO_DESIGN_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
PASS={pass_index + 1}
EXTERNAL_GUIDANCE_SUMMARY={external_guidance[:12000] if external_guidance else '(none)'}
OPENCV_5_GEOMETRIC_EVIDENCE={opencv_verification_prompt(opencv_feedback) if opencv_feedback else '(not enabled for this pass)'}
TOOL_CATALOG={json.dumps(self._catalog, ensure_ascii=False, separators=(',', ':'))}

Evaluate: gross proportions, silhouette, number/placement of holes and slots, pockets, bosses, handles/ribs,
symmetry, repeated features, fillets/chamfers and obvious hidden-geometry contradictions across views.
Perspective differences are expected: do not force an orthographic silhouette match. Never change a known/measured
dimension merely to look better in perspective. When OPENCV_5_GEOMETRIC_EVIDENCE is present, use it as a deterministic
shape/edge trigger and correction clue, but never treat pixel distance as a metric dimension without a known scale anchor.
If there is no clear engineering correction, action=accept.
If refining, calls must EDIT the existing model; never create a replacement part and never import STEP.
`visual_agreement` is your 0..1 assessment of how well the CAD explains all source photos.
For every call encode arguments as JSON text. Return only schema-conforming JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=PHOTO_REVIEW_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            local_images=images,
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        # PHOTO_REVIEW_SCHEMA is a superset of the existing refinement decoder.
        decoded = _decode_refinement(raw)
        decoded["visual_agreement"] = clamp01(raw.get("visual_agreement"))
        return decoded


    def assess_photo_match(
        self,
        *,
        spec: dict[str, Any],
        originals: dict[str, Path],
        rendered: dict[str, Path],
        pass_label: str,
        on_event: Callable[[str, Any], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        images = list(originals.values()) + list(rendered.values())
        prompt = f"""You are CADia's visual scoring stage for Photo-to-CAD.
Original product PHOTOGRAPHS are attached first and CAD RENDERS second.
Score how well ONE editable CAD solid explains ALL photographs. Do not propose tool calls.
Account for perspective differences. Judge engineering geometry only: proportions, silhouette, holes/slots,
pockets, bosses, ribs/handles, symmetry, repeated features, fillets/chamfers and cross-view consistency.
Do not penalize texture, color, labels, scratches or lighting. Known/measured dimensions are hard constraints.

PHOTO_DESIGN_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
PASS_LABEL={pass_label}

visual_agreement must be 0..1. Return only schema-conforming JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=PHOTO_ASSESS_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            local_images=images,
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        return {
            "confidence": clamp01(raw.get("confidence")),
            "issues": [str(x)[:500] for x in (raw.get("issues") or [])][:30],
            "visual_agreement": clamp01(raw.get("visual_agreement")),
            "note": str(raw.get("note") or "")[:1200],
        }


class PhotoCadJobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, PhotoCadJob] = {}
        self._futures: dict[str, Any] = {}
        self._pool = ThreadPoolExecutor(max_workers=PHOTO_WORKERS, thread_name_prefix="cadia-photo-cad")

    def start(self) -> None:
        with self._lock:
            if getattr(self._pool, "_shutdown", False):
                self._pool = ThreadPoolExecutor(max_workers=PHOTO_WORKERS, thread_name_prefix="cadia-photo-cad")

    def _state_path(self, job: PhotoCadJob) -> Path:
        return runtime_manager.project_root(job.user_id, job.project_id) / "photo_cad" / "jobs" / f"{job.id}.json"

    def _persist(self, job: PhotoCadJob) -> None:
        try:
            _write_json(self._state_path(job), {
                "id": job.id, "user_id": job.user_id, "project_id": job.project_id,
                "vision_mode": job.vision_mode,
                "status": job.status, "progress": job.progress, "message": job.message,
                "error": job.error, "result": job.result,
                "created_at": job.created_at, "updated_at": job.updated_at,
                "deadline_at": job.deadline_at,
            })
        except Exception:
            pass

    def _token(self, job_id: str) -> str:
        return f"photo:{job_id}"

    def create(self, job: PhotoCadJob) -> PhotoCadJob:
        token = self._token(job.id)
        heavy_job_manager.reserve(token, job.user_id, job.project_id, "photo-to-cad", job.cancel_event)
        try:
            runtime_manager.drop(job.user_id, job.project_id)
            job.deadline_at = None
            with self._lock:
                self._jobs[job.id] = job
                self._persist(job)
            future = self._pool.submit(self._run, job.id)
            with self._lock:
                self._futures[job.id] = future
            future.add_done_callback(lambda completed, jid=job.id: self._worker_done(jid, completed))
            return job
        except Exception:
            heavy_job_manager.finish(token)
            raise

    def _worker_done(self, job_id: str, future) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
            job = self._jobs.get(job_id)
        if future.cancelled():
            heavy_job_manager.finish(self._token(job_id))
            if job is not None and job.status not in TERMINAL_STATUSES:
                job.error = "Photo-to-CAD job was cancelled before it started."
                self._update(job, status="cancelled", message="Cancelled")
            return
        try:
            exc = future.exception()
        except BaseException as callback_exc:
            exc = callback_exc
        if exc is None:
            return
        heavy_job_manager.finish(self._token(job_id))
        if job is None or job.status in TERMINAL_STATUSES:
            return
        job.error = f"Photo-to-CAD worker terminated unexpectedly: {exc}"
        self._update(job, status="failed", message="Photo reconstruction worker stopped unexpectedly")

    def _watchdog(self, job_id: str) -> None:
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status in TERMINAL_STATUSES:
                    return
                deadline = job.deadline_at
            if deadline is None:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None or job.status in TERMINAL_STATUSES:
                        return
                    job.timed_out = True
                    job.cancel_event.set()
                    job.error = f"Photo-to-CAD exceeded the hard runtime limit of {PHOTO_JOB_HARD_TIMEOUT_SECONDS} seconds."
                self._update(job, status="failed", message="Photo reconstruction timed out")
                return
            time.sleep(min(1.0, max(0.1, remaining)))

    def _load(self, job_id: str, user_id: str, project_id: str) -> PhotoCadJob | None:
        path = runtime_manager.project_root(user_id, project_id) / "photo_cad" / "jobs" / f"{job_id}.json"
        if not path.is_file():
            return None
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            status = str(row.get("status") or "failed")
            error = row.get("error")
            message = str(row.get("message") or "")
            if status in {"queued", "running", "cancelling"}:
                status = "failed"
                error = "Photo-to-CAD job was interrupted by a server restart."
                message = "Interrupted by server restart"
            return PhotoCadJob(
                id=job_id, user_id=user_id, project_id=project_id, input_files={},
                vision_mode=str(row.get("vision_mode") or "standard"), status=status,
                progress=int(row.get("progress") or 0), message=message,
                error=None if error is None else str(error), result=row.get("result"),
                created_at=float(row.get("created_at") or time.time()), updated_at=float(row.get("updated_at") or time.time()),
                deadline_at=(float(row.get("deadline_at")) if row.get("deadline_at") is not None else None),
            )
        except Exception:
            return None

    def get(self, job_id: str, user_id: str, project_id: str) -> PhotoCadJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._load(job_id, user_id, project_id)
                if job is not None:
                    self._jobs[job.id] = job
            if job is None or job.user_id != user_id or job.project_id != project_id:
                raise KeyError(job_id)
            return job

    def cancel(self, job_id: str, user_id: str, project_id: str) -> PhotoCadJob:
        job = self.get(job_id, user_id, project_id)
        if job.status in TERMINAL_STATUSES:
            return job
        job.cancel_event.set()
        heavy_job_manager.cancel_token(self._token(job.id))
        with self._lock:
            future = self._futures.get(job.id)
        if future is not None and future.cancel():
            self._update(job, status="cancelled", message="Cancelled")
        else:
            self._update(job, status="cancelling", message="Cancelling…")
        return job

    def forget_project(self, user_id: str, project_id: str) -> None:
        with self._lock:
            stale = [jid for jid, job in self._jobs.items() if job.user_id == str(user_id) and job.project_id == str(project_id)]
            for jid in stale:
                self._jobs.pop(jid, None)
                self._futures.pop(jid, None)

    def forget_user(self, user_id: str) -> None:
        with self._lock:
            stale = [jid for jid, job in self._jobs.items() if job.user_id == str(user_id)]
            for jid in stale:
                self._jobs.pop(jid, None)
                self._futures.pop(jid, None)

    def shutdown(self) -> None:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.status not in TERMINAL_STATUSES]
            futures = list(self._futures.values())
        for job in jobs:
            job.cancel_event.set()
            heavy_job_manager.cancel_token(self._token(job.id))
        for future in futures:
            future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _update(self, job: PhotoCadJob, *, status: str | None = None, progress: int | None = None, message: str | None = None) -> None:
        with self._lock:
            # Once a job is terminal, late worker callbacks may not resurrect or
            # overwrite it. This is important when a watchdog/cancel races with AI work.
            if job.status in TERMINAL_STATUSES and status != job.status:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, max(0, min(100, int(progress))))
            if message is not None:
                job.message = message
            job.updated_at = time.time()
            self._persist(job)

    def _check(self, job: PhotoCadJob) -> None:
        if job.timed_out or (job.deadline_at is not None and time.time() >= job.deadline_at):
            job.timed_out = True
            job.cancel_event.set()
            raise TimeoutError("Photo-to-CAD exceeded its hard runtime limit.")
        if job.cancel_event.is_set():
            raise InterruptedError("Photo-to-CAD job was cancelled.")

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            heavy_job_manager.finish(self._token(job_id))
            return
        token = self._token(job.id)
        try:
            self._update(job, status="queued", progress=1, message="Waiting for an isolated CAD worker…")
            if not heavy_job_manager.acquire(token):
                raise InterruptedError("Photo-to-CAD job was cancelled before it started.")
            job.deadline_at = time.time() + PHOTO_JOB_HARD_TIMEOUT_SECONDS
            self._persist(job)
            threading.Thread(target=self._watchdog, args=(job.id,), name=f"cadia-photo-watchdog-{job.id[:8]}", daemon=True).start()
            self._update(job, status="running", progress=3, message="Starting isolated photo reconstruction…")

            def relay(kind: str, value: Any) -> None:
                self._check(job)
                if kind == "progress":
                    if isinstance(value, dict):
                        message = str(value.get("message") or job.message)
                        percent = value.get("percent")
                        progress = int(percent) if isinstance(percent, (int, float)) else None
                    else:
                        message, progress = str(value), None
                    self._update(job, progress=progress, message=message)

            job.result = run_isolated(
                token=token,
                kind="photo",
                payload={"job": {
                    "id": job.id, "user_id": job.user_id, "project_id": job.project_id,
                    "input_files": dict(job.input_files), "notes": job.notes,
                    "dimension_hint": job.dimension_hint, "model": job.model,
                    "effort": job.effort, "vision_mode": job.vision_mode,
                }},
                cancel_event=job.cancel_event,
                hard_timeout_seconds=PHOTO_JOB_HARD_TIMEOUT_SECONDS,
                on_event=relay,
            )
            self._check(job)
            self._update(job, status="done", progress=100, message="Editable CAD reconstructed from photos")
        except TimeoutError as exc:
            job.error = str(exc)
            self._update(job, status="failed", message="Photo reconstruction timed out")
        except InterruptedError as exc:
            job.error = str(exc)
            self._update(job, status="cancelled", message="Cancelled")
        except Exception as exc:
            job.error = str(exc)
            self._update(job, status="failed", message="Photo reconstruction failed")
        finally:
            heavy_job_manager.finish(token)

    def _reconstruct(self, job: PhotoCadJob) -> dict[str, Any]:
        runtime = runtime_manager.get(job.user_id, job.project_id)
        client = app_server_manager.get(job.user_id)
        codex_dir = (client.workspace / "photo-cad" / job.id).resolve()
        codex_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        report_dir = runtime.root / "photo_cad" / job.id
        report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        image_paths: dict[str, Path] = {}
        for label, raw in sorted(job.input_files.items()):
            source = Path(raw).resolve()
            target = codex_dir / f"{label}{source.suffix.lower()}"
            shutil.copy2(source, target)
            os.chmod(target, 0o600)
            image_paths[label] = target

        opencv_source: dict[str, Any] | None = None
        effective_notes = job.notes
        if job.vision_mode == "opencv5":
            self._update(job, progress=5, message="OpenCV 5 · extracting multi-view photo geometry evidence…")
            opencv_source = opencv_analyze_images(
                image_paths,
                output_dir=report_dir / "opencv" / "source",
                source_kind="photo",
            )
            effective_notes = (
                (job.notes + "\n\n" if job.notes else "")
                + "OpenCV 5 deterministic visual evidence follows. Use it for contours, line/circle evidence, view consistency and proportions only. "
                  "Never claim a pixel-derived measurement is millimetres unless the user supplied a metric anchor.\n"
                + opencv_prompt_summary(opencv_source)
            )

        agent = PhotoCadCodexAgent(
            client=client,
            target_id=f"photo-cad:{job.project_id}:{job.id}",
            project_root=runtime.root,
            model=job.model or "gpt-5.6-sol",
            reasoning=job.effort or "high",
        )

        def event(kind: str, value: Any) -> None:
            self._check(job)
            if kind == "progress":
                text = str(value.get("message") or "Analyzing product photos…") if isinstance(value, dict) else str(value)
                self._update(job, progress=max(job.progress, min(91, job.progress + 1)), message=text)

        def snapshot() -> dict[str, Any]:
            with runtime.engine.lock:
                return copy.deepcopy(runtime.engine._workspace_snapshot())

        def restore(snap: dict[str, Any]) -> None:
            with runtime.engine.lock:
                runtime.engine._restore_workspace(copy.deepcopy(snap), bump_revision=True)

        initial_snapshot = snapshot()
        self._check(job)
        self._update(job, progress=7, message="Extracting engineering design intent from product photos…")
        spec = agent.extract_spec(
            image_paths,
            notes=effective_notes,
            dimension_hint=job.dimension_hint,
            on_event=event,
            cancel_check=job.cancel_event.is_set,
        )
        validate_photo_spec(spec)
        _write_json(report_dir / "photo_design_spec.json", spec)

        # Public-runtime cascade. CADia keeps orchestration/native B-Rep ownership while
        # released upstream runtimes supply proposals or geometry. Heavy stages remain
        # conditional so adding public models does not force every request through all of them.
        # With no external runtime installed, retain the original two-native-proposal
        # behaviour so baseline CADia quality is never traded away for speed.
        self._update(job, progress=20, message="Preparing conditional photo-geometry reconstruction…")
        engine_results: list[ExternalEngineResult] = []
        external_guidance: list[str] = []
        reference_mesh: Path | None = None
        agentic_result: ExternalEngineResult | None = None
        engine_root = report_dir / "external"
        capabilities_now = external_capabilities()
        external_mode = os.getenv("CADIA_PHOTO_EXTERNAL_MODE", "auto").strip().lower() != "off"
        primary_image = next(iter(image_paths.values()))

        def engine_available(name: str) -> bool:
            row = capabilities_now.get(name) or {}
            return bool(external_mode and row.get("available") and row.get("configured", True))

        def current_external_text() -> str:
            value = "\n\n".join(external_guidance)
            return value if len(value) <= 70000 else value[:70000] + "\n...<truncated>"

        def record_skipped(name: str, reason: str) -> None:
            # Preserve a truthful engine_status row even when the cascade intentionally
            # avoids launching an expensive runtime.
            if any(item.name == name for item in engine_results):
                return
            available = bool((capabilities_now.get(name) or {}).get("available"))
            engine_results.append(ExternalEngineResult(name=name, available=available, attempted=False, ok=False, message=reason))

        def ensure_reference_mesh() -> Path | None:
            """Build Agentic observation mesh only when an escalation actually needs it.

            This intentionally avoids the Open3D fusion cost on the common fast path
            where the baseline CADia proposals already satisfy the quality gate.
            """
            nonlocal reference_mesh
            if reference_mesh is not None and reference_mesh.is_file():
                return reference_mesh
            if agentic_result is None or not agentic_result.ok or not agentic_result.evidence_dir:
                return None
            self._check(job)
            self._update(job, progress=max(job.progress, 46), message="Preparing shared observation mesh for geometry escalation…")
            reference_mesh = build_reference_mesh_from_agentic_evidence(
                agentic_result,
                output_dir=engine_root / "agentic_reference",
                cancel_check=job.cancel_event.is_set,
                deadline_at=job.deadline_at,
            )
            return reference_mesh

        # Agentic CAD is the only external stage allowed before the first native CAD.
        # Its multi-view masks/depth/cameras are useful both as planner evidence and as
        # an independent numeric observation score for the generated CADia solid.
        if engine_available("agentic_cad"):
            self._check(job)
            self._update(job, progress=24, message="Building multi-view photo geometry (Agentic CAD)…")
            prompt = f"{spec.get('part_name')}: {spec.get('object_summary')}. {job.notes}".strip()
            agentic_result = run_agentic_cad(
                image_paths.values(),
                output_dir=engine_root / "agentic",
                prompt=prompt,
                dimension_hint=job.dimension_hint,
                cancel_check=job.cancel_event.is_set,
                deadline_at=job.deadline_at,
            )
            engine_results.append(agentic_result)
            if agentic_result.ok and agentic_result.guidance:
                external_guidance.append(agentic_result.guidance)
        elif not external_mode:
            record_skipped("agentic_cad", "External mode disabled by CADIA_PHOTO_EXTERNAL_MODE=off.")
        else:
            record_skipped("agentic_cad", "Agentic CAD runtime is not available/configured.")

        # CAD-Coder has an official one-image LLaVA inference loader. Use it only for
        # single-photo jobs, where its proposal can challenge/strengthen Agentic/Codex
        # without inventing a custom image-to-CAD algorithm inside CADia.
        if len(image_paths) == 1 and engine_available("cadcoder"):
            self._check(job)
            self._update(job, progress=max(job.progress, 27), message="Generating an independent CAD-Coder image proposal…")
            cadcoder = run_cadcoder(
                primary_image,
                output_dir=engine_root / "cadcoder",
                cancel_check=job.cancel_event.is_set,
                deadline_at=job.deadline_at,
            )
            engine_results.append(cadcoder)
            if cadcoder.ok and cadcoder.guidance:
                external_guidance.append(cadcoder.guidance)
        elif len(image_paths) != 1:
            record_skipped("cadcoder", "Skipped: CAD-Coder's released inference path is single-image; multi-photo jobs use Agentic hybrid geometry instead.")
        elif not external_mode:
            record_skipped("cadcoder", "External mode disabled by CADIA_PHOTO_EXTERNAL_MODE=off.")
        else:
            record_skipped("cadcoder", "CAD-Coder runtime is not available/configured.")

        attempts: list[dict[str, Any]] = []
        best_snapshot: dict[str, Any] | None = None
        best_validation: dict[str, Any] | None = None
        best_observation_score = -1.0
        best_combined_score = -1.0
        best_sort_key: tuple[int, int, float] | None = None
        best_rendered: dict[str, Path] = {}
        feedback = ""
        cascade_stopped_after: str | None = None
        reasoner_deferred_for_target = False

        def best_quality_pass() -> bool:
            if best_validation is None:
                return False
            return cascade_quality_pass(best_validation, best_observation_score, best_combined_score)

        def run_native_candidate(attempt: int, *, stage: str, guidance: str) -> bool:
            nonlocal best_snapshot, best_validation, best_observation_score, best_combined_score, best_sort_key, best_rendered, feedback
            self._check(job)
            self._update(job, progress=max(job.progress, min(78, 28 + attempt * 10)), message=f"Planning editable native CAD ({stage})…")
            with runtime.operation_lock:
                restore(initial_snapshot)
                plan = agent.plan_from_photo_spec(
                    spec,
                    current_state=planning_state(runtime.engine),
                    notes=effective_notes,
                    dimension_hint=job.dimension_hint,
                    external_guidance=guidance,
                    validator_feedback=feedback,
                    attempt=attempt,
                    on_event=event,
                    cancel_check=job.cancel_event.is_set,
                )
                _validate_plan(plan)
                executed = execute_with_failure_recovery(
                    executor=runtime.executor,
                    agent=agent,
                    user_prompt=photo_request_text(spec, effective_notes, job.dimension_hint),
                    state_provider=lambda: planning_state(runtime.engine),
                    plan=plan,
                    on_event=event,
                    initial_progress_span=(50, 70),
                    max_ai_repairs=2,
                )
                if not executed.ok:
                    attempts.append({"attempt": attempt, "stage": stage, "error": str(executed.error or "execution failed")})
                    feedback = str(executed.error or "execution failed")
                    return False

                runtime.persist()
                validation = validate_photo_runtime(runtime, spec)

                # Export exactly once for this candidate. The same STL is reused by
                # the isolated four-view renderer and Agentic geometry scorer.
                candidate_stl: Path | None = None
                try:
                    candidate_stl = Path(runtime.export("stl"))
                except Exception:
                    candidate_stl = None

                candidate_render_dir = report_dir / f"candidate_{attempt}"
                candidate_render_dir.mkdir(parents=True, exist_ok=True)
                rendered = _render_validation_views(
                    runtime,
                    candidate_render_dir,
                    pass_index=attempt,
                    stl_path=candidate_stl,
                )
                assessment = (
                    agent.assess_photo_match(
                        spec=spec,
                        originals=image_paths,
                        rendered=rendered,
                        pass_label=f"candidate-{attempt}-{stage}",
                        on_event=event,
                        cancel_check=job.cancel_event.is_set,
                    )
                    if rendered
                    else {"confidence": 0.0, "issues": ["headless renderer unavailable"], "visual_agreement": 0.0, "note": ""}
                )

                observation = float(assessment.get("visual_agreement") or 0.0)
                evidence_score = None
                if candidate_stl is not None and validation.get("brep_valid") and agentic_result is not None:
                    evidence_score = score_with_agentic_evidence(
                        candidate_stl,
                        agentic_result,
                        output_dir=report_dir / "geometry_scores" / f"candidate_{attempt}",
                        cancel_check=job.cancel_event.is_set,
                        deadline_at=job.deadline_at,
                    )
                    if evidence_score and evidence_score.get("ok"):
                        observation = clamp01(evidence_score.get("consistency_score"))

                opencv_candidate = None
                if job.vision_mode == "opencv5" and rendered:
                    opencv_candidate = opencv_verify_source_to_renders(
                        image_paths,
                        rendered,
                        output_dir=report_dir / "opencv" / "candidates" / f"candidate_{attempt}",
                        source_kind="photo",
                    )
                    cv_score = float(opencv_candidate.get("score") or 0.0)
                    # OpenCV is a substantive candidate-selection signal. Existing VLM
                    # and depth/mesh evidence remain complementary semantic/3D signals.
                    cv_weight = 0.35 if evidence_score and evidence_score.get("ok") else 0.45
                    observation = (1.0 - cv_weight) * observation + cv_weight * cv_score

                combined = 0.58 * float(validation.get("score") or 0.0) + 0.42 * observation
                passed_gate = cascade_quality_pass(validation, observation, combined)
                sort_key = (
                    1 if validation.get("hard_dimensions_passed") else 0,
                    1 if validation.get("brep_valid") else 0,
                    combined,
                )
                snap = snapshot()
                attempt_row = {
                    "attempt": attempt,
                    "stage": stage,
                    "validation": validation,
                    "visual_assessment": assessment,
                    "observation_score": round(observation, 5),
                    "combined_score": round(combined, 5),
                    "cascade_quality_pass": passed_gate,
                    "cascade_thresholds": {
                        "native": PHOTO_CASCADE_MIN_NATIVE,
                        "observation": PHOTO_CASCADE_MIN_OBSERVATION,
                        "combined": PHOTO_CASCADE_MIN_COMBINED,
                    },
                }
                if evidence_score:
                    attempt_row["geometry_evidence"] = evidence_score
                if opencv_candidate is not None:
                    attempt_row["opencv_verification"] = opencv_compact_verification(opencv_candidate)
                attempts.append(attempt_row)
                if best_sort_key is None or sort_key > best_sort_key:
                    best_sort_key = sort_key
                    best_validation = validation
                    best_snapshot = snap
                    best_observation_score = observation
                    best_combined_score = combined
                    best_rendered = dict(rendered)
                feedback = validation_feedback(validation)
                if assessment.get("issues"):
                    feedback = (feedback + "; " if feedback else "") + "; ".join(assessment.get("issues") or [])
                return passed_gate

        # Baseline quality is preserved: always build the same two CADia-native
        # proposals before deciding whether the expensive mesh/CAD engines are needed.
        # Proposal #1 is semantic-only; proposal #2 incorporates Agentic guidance when
        # available (otherwise it remains CADia's independent alternative proposal).
        run_native_candidate(1, stage="semantic-only", guidance="")
        run_native_candidate(
            2,
            stage=("semantic+public-proposals" if current_external_text() else "native-alternative"),
            guidance=current_external_text(),
        )

        if best_quality_pass():
            cascade_stopped_after = "baseline-native-pair"
            record_skipped("cadfit", "Skipped by cascade: best of the two baseline native candidates met the quality gate.")
            record_skipped("cadreasoner", "Skipped by cascade: best of the two baseline native candidates met the quality gate.")
            record_skipped("pixal3d", "Skipped by cascade: best of the two baseline native candidates met the quality gate.")
            record_skipped("cadena", "Skipped by cascade: best of the two baseline native candidates met the quality gate.")
        else:
            # Escalation 1: CADFit. Build the Agentic observation mesh lazily only
            # now; if the baseline pair passed, this potentially expensive fusion is
            # never performed. If no Agentic mesh can be built, CADFit may use its
            # own public image-to-mesh route.
            if engine_available("cadfit"):
                self._check(job)
                ensure_reference_mesh()
                self._update(job, progress=max(job.progress, 55), message="Baseline quality below gate; escalating to CADFit…")
                cadfit = run_cadfit(
                    image_paths.values(),
                    output_dir=engine_root / "cadfit",
                    reference_mesh=reference_mesh,
                    cancel_check=job.cancel_event.is_set,
                    deadline_at=job.deadline_at,
                )
                engine_results.append(cadfit)
                if cadfit.ok and cadfit.guidance:
                    external_guidance.append(cadfit.guidance)
                if reference_mesh is None and cadfit.reference_mesh:
                    candidate = Path(cadfit.reference_mesh)
                    if candidate.is_file():
                        reference_mesh = candidate
                if cadfit.ok and cadfit.guidance:
                    run_native_candidate(3, stage="cadfit", guidance=current_external_text())
            else:
                record_skipped("cadfit", "CADFit runtime is not available/configured; baseline two-native quality is retained.")

            if best_quality_pass():
                cascade_stopped_after = "cadfit/native"
                record_skipped("cadreasoner", "Skipped by cascade: current native candidate met the quality gate after CADFit.")
                record_skipped("pixal3d", "Skipped by cascade: current native candidate met the quality gate after CADFit.")
                record_skipped("cadena", "Skipped by cascade: current native candidate met the quality gate after CADFit.")
            else:
                # Escalation 2: CADReasoner. Its released test.py accepts a local
                # directory of STL targets directly, so no custom reverse-engineering
                # algorithm is introduced here.
                if engine_available("cadreasoner"):
                    ensure_reference_mesh()
                    if reference_mesh is not None and reference_mesh.is_file():
                        self._check(job)
                        self._update(job, progress=max(job.progress, 64), message="Quality still below gate; escalating to CADReasoner…")
                        reasoner = run_cadreasoner(
                            reference_mesh,
                            output_dir=engine_root / "cadreasoner",
                            cancel_check=job.cancel_event.is_set,
                            deadline_at=job.deadline_at,
                        )
                        engine_results.append(reasoner)
                        if reasoner.ok and reasoner.guidance:
                            external_guidance.append(reasoner.guidance)
                            run_native_candidate(4, stage="cadreasoner", guidance=current_external_text())
                    else:
                        # A single-photo Pixal3D fallback may still create a target mesh
                        # later in the cascade. Defer the skip decision until then.
                        reasoner_deferred_for_target = True
                else:
                    record_skipped("cadreasoner", "CADReasoner runtime is not available/configured.")

                if best_quality_pass():
                    cascade_stopped_after = "cadreasoner/native"
                    record_skipped("pixal3d", "Skipped by cascade: current native candidate met the quality gate after CADReasoner.")
                    record_skipped("cadena", "Skipped by cascade: current native candidate met the quality gate after CADReasoner.")
                else:
                    # Escalation 3: Pixal3D is an alternative single-image 3D prior.
                    # It is intentionally late because it is GPU-heavy. Its mesh is
                    # converted only as file-format glue, then the public CADFit core
                    # performs Mesh-to-CAD fitting.
                    if len(image_paths) == 1 and engine_available("pixal3d") and engine_available("cadfit"):
                        self._check(job)
                        self._update(job, progress=max(job.progress, 70), message="Trying Pixal3D alternate geometry prior…")
                        pixal = run_pixal3d(
                            primary_image,
                            output_dir=engine_root / "pixal3d",
                            cancel_check=job.cancel_event.is_set,
                            deadline_at=job.deadline_at,
                        )
                        engine_results.append(pixal)
                        if pixal.ok and pixal.reference_mesh:
                            pixal_mesh = Path(pixal.reference_mesh)
                            if reference_mesh is None or not reference_mesh.is_file():
                                reference_mesh = pixal_mesh
                            pixal_fit = run_cadfit(
                                image_paths.values(),
                                output_dir=engine_root / "cadfit_pixal3d",
                                reference_mesh=pixal_mesh,
                                cancel_check=job.cancel_event.is_set,
                                deadline_at=job.deadline_at,
                            )
                            pixal_fit.name = "cadfit_pixal3d"
                            engine_results.append(pixal_fit)
                            if pixal_fit.ok and pixal_fit.guidance:
                                external_guidance.append(pixal.guidance)
                                external_guidance.append(pixal_fit.guidance)
                                run_native_candidate(5, stage="pixal3d+cadfit", guidance=current_external_text())
                            if (
                                not best_quality_pass()
                                and reasoner_deferred_for_target
                                and engine_available("cadreasoner")
                                and reference_mesh is not None
                                and reference_mesh.is_file()
                            ):
                                self._check(job)
                                self._update(job, progress=max(job.progress, 73), message="Using the new Pixal3D target with CADReasoner…")
                                reasoner = run_cadreasoner(
                                    reference_mesh,
                                    output_dir=engine_root / "cadreasoner_after_pixal",
                                    cancel_check=job.cancel_event.is_set,
                                    deadline_at=job.deadline_at,
                                )
                                engine_results.append(reasoner)
                                if reasoner.ok and reasoner.guidance:
                                    external_guidance.append(reasoner.guidance)
                                    run_native_candidate(6, stage="cadreasoner-after-pixal", guidance=current_external_text())
                    elif len(image_paths) != 1:
                        record_skipped("pixal3d", "Skipped: current Pixal3D fallback uses its official single-image inference path.")
                    elif not engine_available("pixal3d"):
                        record_skipped("pixal3d", "Pixal3D runtime is not available/configured.")
                    elif not engine_available("cadfit"):
                        record_skipped("pixal3d", "Pixal3D mesh fallback requires the public CADFit runtime for Mesh-to-CAD conversion.")

                    if best_quality_pass():
                        cascade_stopped_after = "pixal3d-cadfit/native"
                        record_skipped("cadena", "Skipped by cascade: current native candidate met the quality gate after Pixal3D/CADFit.")
                    else:
                        # Final escalation: CADENA. It requires a target mesh, so use
                        # the shared Agentic/CADFit target when available.
                        if engine_available("cadena"):
                            ensure_reference_mesh()
                            if reference_mesh is not None and reference_mesh.is_file():
                                self._check(job)
                                self._update(job, progress=max(job.progress, 76), message="Quality still below gate; escalating to CADENA…")
                                cadena = run_cadena(
                                    reference_mesh,
                                    output_dir=engine_root / "cadena",
                                    cancel_check=job.cancel_event.is_set,
                                    deadline_at=job.deadline_at,
                                )
                                engine_results.append(cadena)
                                if cadena.ok and cadena.guidance:
                                    external_guidance.append(cadena.guidance)
                                    run_native_candidate(7, stage="cadena", guidance=current_external_text())
                                    cascade_stopped_after = "cadena/native"
                                else:
                                    cascade_stopped_after = "best-before-cadena"
                            else:
                                record_skipped("cadena", "CADENA skipped because no shared reference mesh was available.")
                                cascade_stopped_after = "best-available-native"
                        else:
                            record_skipped("cadena", "CADENA runtime is not available/configured.")
                            cascade_stopped_after = "best-available-native"

        external_text = current_external_text()
        _write_json(
            report_dir / "engine_status.json",
            {
                "mode": "conditional-cascade",
                "quality_gate": {
                    "native": PHOTO_CASCADE_MIN_NATIVE,
                    "observation": PHOTO_CASCADE_MIN_OBSERVATION,
                    "combined": PHOTO_CASCADE_MIN_COMBINED,
                },
                "stopped_after": cascade_stopped_after,
                "capabilities": capabilities_now,
                "runs": [x.public() for x in engine_results],
            },
        )

        if best_snapshot is None or best_validation is None:
            restore(initial_snapshot)
            raise RuntimeError("Photo → CAD could not produce a valid native CAD candidate.")

        restore(best_snapshot)
        current_validation = best_validation
        current_observation = best_observation_score
        refinements: list[dict[str, Any]] = []
        # Reuse the already-rendered best candidate as refinement pass #1 input.
        # After an accepted edit, reuse that pass's after-render for the next review.
        current_rendered: dict[str, Path] = dict(best_rendered)

        # Greedy feature-level correction. A proposed edit is retained only when the
        # B-Rep/hard metric anchors remain valid AND photo/geometry consistency does
        # not regress. Agentic evidence gives a deterministic numeric check when active;
        # otherwise a second independent VLM assessment is used.
        for pass_index in range(PHOTO_REFINEMENTS):
            self._check(job)
            self._update(job, progress=max(job.progress, 74 + pass_index * 8), message=f"Comparing CAD against source photos ({pass_index + 1}/{PHOTO_REFINEMENTS})…")
            refinement_before_dir = report_dir / "refinement_before"
            refinement_before_dir.mkdir(parents=True, exist_ok=True)
            rendered = dict(current_rendered) if len(current_rendered) == 4 else {}
            if not rendered:
                rendered = _render_validation_views(runtime, refinement_before_dir, pass_index=pass_index)
            if not rendered:
                refinements.append({"pass": pass_index + 1, "action": "skipped", "reason": "headless renderer unavailable"})
                break
            opencv_before = None
            if job.vision_mode == "opencv5":
                opencv_before = opencv_verify_source_to_renders(
                    image_paths,
                    rendered,
                    output_dir=report_dir / "opencv" / "refinements" / f"pass_{pass_index + 1}_before",
                    source_kind="photo",
                )
                if (
                    opencv_before.get("action") == "pass"
                    and float(opencv_before.get("score") or 0.0) >= float(opencv_before.get("threshold") or 0.0) + 0.08
                    and float(opencv_before.get("confidence") or 0.0) >= 0.45
                ):
                    refinements.append({
                        "pass": pass_index + 1,
                        "action": "opencv_pass",
                        "reason": "OpenCV 5 high-confidence geometry gate passed; no repair action triggered.",
                        "opencv_before": opencv_compact_verification(opencv_before),
                    })
                    break
            review = agent.review_photo_match(
                spec=spec,
                originals=image_paths,
                rendered=rendered,
                current_state=planning_state(runtime.engine),
                pass_index=pass_index,
                external_guidance=external_text,
                opencv_feedback=opencv_before,
                on_event=event,
                cancel_check=job.cancel_event.is_set,
            )
            row = {k: v for k, v in review.items() if k != "calls"}
            row["pass"] = pass_index + 1
            row["observation_before"] = round(current_observation, 5)
            if opencv_before is not None:
                row["opencv_before"] = opencv_compact_verification(opencv_before)
            refinements.append(row)
            if review.get("action") != "refine":
                break

            before = snapshot()
            plan = {"calls": review.get("calls") or [], "note": review.get("note") or "photo visual refinement"}
            try:
                _validate_plan(plan)
            except Exception as exc:
                row["accepted"] = False
                row["error"] = f"invalid refinement plan: {exc}"
                break

            with runtime.operation_lock:
                executed = execute_with_failure_recovery(
                    executor=runtime.executor,
                    agent=agent,
                    user_prompt=photo_request_text(spec, effective_notes, job.dimension_hint) + " Correct only these discrepancies: " + "; ".join(review.get("issues") or []),
                    state_provider=lambda: planning_state(runtime.engine),
                    plan=plan,
                    on_event=event,
                    initial_progress_span=(82, 92),
                    max_ai_repairs=1,
                )
                if not executed.ok:
                    restore(before)
                    row["accepted"] = False
                    row["error"] = str(executed.error or "refinement execution failed")
                    break

                after = validate_photo_runtime(runtime, spec)
                # Export the edited solid once, then share it between rendering and
                # geometry scoring instead of exporting the same state twice.
                after_stl: Path | None = None
                try:
                    after_stl = Path(runtime.export("stl"))
                except Exception:
                    after_stl = None
                refinement_after_dir = report_dir / "refinement_after"
                refinement_after_dir.mkdir(parents=True, exist_ok=True)
                after_rendered = _render_validation_views(
                    runtime,
                    refinement_after_dir,
                    pass_index=pass_index,
                    stl_path=after_stl,
                )
                after_assessment = (
                    agent.assess_photo_match(
                        spec=spec,
                        originals=image_paths,
                        rendered=after_rendered,
                        pass_label=f"refinement-{pass_index + 1}-after",
                        on_event=event,
                        cancel_check=job.cancel_event.is_set,
                    )
                    if after_rendered
                    else {"confidence": 0.0, "issues": ["headless renderer unavailable"], "visual_agreement": 0.0, "note": ""}
                )
                after_observation = float(after_assessment.get("visual_agreement") or 0.0)
                evidence_score = None
                if after.get("brep_valid") and agentic_result is not None and after_stl is not None:
                    evidence_score = score_with_agentic_evidence(
                        after_stl,
                        agentic_result,
                        output_dir=report_dir / "geometry_scores" / f"refinement_{pass_index + 1}",
                        cancel_check=job.cancel_event.is_set,
                        deadline_at=job.deadline_at,
                    )
                    if evidence_score and evidence_score.get("ok"):
                        after_observation = clamp01(evidence_score.get("consistency_score"))

                opencv_after = None
                if job.vision_mode == "opencv5" and after_rendered:
                    opencv_after = opencv_verify_source_to_renders(
                        image_paths,
                        after_rendered,
                        output_dir=report_dir / "opencv" / "refinements" / f"pass_{pass_index + 1}_after",
                        source_kind="photo",
                    )
                    cv_after_score = float(opencv_after.get("score") or 0.0)
                    cv_weight = 0.35 if evidence_score and evidence_score.get("ok") else 0.45
                    after_observation = (1.0 - cv_weight) * after_observation + cv_weight * cv_after_score

                # Deterministic geometry scores should not regress. VLM-only scores are
                # noisier, so permit a tiny 0.03 tolerance while still rejecting clear harm.
                tolerance = 0.005 if evidence_score and evidence_score.get("ok") else 0.03
                geometry_ok = after_observation + tolerance >= current_observation
                opencv_ok = True
                if opencv_before is not None and opencv_after is not None:
                    opencv_ok = float(opencv_after.get("score") or 0.0) + 0.01 >= float(opencv_before.get("score") or 0.0)
                if not acceptable_refinement(current_validation, after) or not geometry_ok or not opencv_ok:
                    restore(before)
                    row["accepted"] = False
                    row["observation_after"] = round(after_observation, 5)
                    row["error"] = "greedy rollback: B-Rep, hard metric anchor, photo/geometry consistency, or OpenCV agreement regressed"
                    break

                current_validation = after
                current_observation = after_observation
                current_rendered = dict(after_rendered)
                runtime.persist()
                row["accepted"] = True
                row["validation"] = after
                row["visual_assessment_after"] = after_assessment
                row["observation_after"] = round(after_observation, 5)
                if evidence_score:
                    row["geometry_evidence_after"] = evidence_score
                if opencv_after is not None:
                    row["opencv_after"] = opencv_compact_verification(opencv_after)

        opencv_vision = None
        if job.vision_mode == "opencv5" and opencv_source is not None:
            final_rendered = dict(current_rendered) if current_rendered else _render_validation_views(
                runtime, report_dir / "opencv" / "final_renders", pass_index=99
            )
            final_cv = opencv_verify_source_to_renders(
                image_paths,
                final_rendered,
                output_dir=report_dir / "opencv" / "final_compare",
                source_kind="photo",
            )
            opencv_vision = {
                "mode": "opencv5",
                "source": opencv_compact_analysis(opencv_source),
                "final": opencv_compact_verification(final_cv),
                "score": final_cv.get("score", 0.0),
                "action": final_cv.get("action", "ambiguous"),
                "artifacts": [
                    *[{**row, "path": f"source/{row.get('path', '')}"} for row in (opencv_source.get("artifacts") or [])],
                    *[{**row, "path": f"final_compare/{row.get('path', '')}"} for row in (final_cv.get("artifacts") or [])],
                ],
            }
            _write_json(report_dir / "opencv" / "closed_loop.json", opencv_vision)

        # CADIA_NVIDIA_REVIEW_PHOTO_V210
        # Post-generation only: the proven Photo→CAD path produces a valid editable
        # B-Rep first. NVIDIA then independently compares source photos against CAD
        # renders. Only a Cosmos discrepancy that Nemotron independently approves can
        # reach Codex as a bounded edit; any failure/regression restores the valid base CAD.
        def _nvidia_photo_execute(plan: dict[str, Any], prompt: str) -> tuple[bool, str | None]:
            # Exactly ONE atomic executor attempt for the NVIDIA correction lane.
            # PlanExecutor.execute owns an internal snapshot/rollback; the outer
            # Engineering Review snapshot remains a second fail-safe.
            with runtime.operation_lock:
                executed = runtime.executor.execute(plan, on_event=event, progress_span=(96, 97))
            return bool(executed.ok), None if executed.ok else str(executed.error or "NVIDIA-guided repair failed")

        def _nvidia_photo_accept(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, str]:
            ok = acceptable_refinement(before, after)
            return ok, "" if ok else "Photo CAD hard metrics or B-Rep validity regressed."

        if job.vision_mode == "opencv5":
            nvidia_review = {
                "status": "preserved_not_run",
                "accepted": False,
                "reason": "OpenCV 5 Vision mode selected. Existing NVIDIA/Cosmos/Nemotron review remains installed but is bypassed for this run.",
            }
        else:
            nvidia_review = run_engineering_review(
                source_kind="product_photo",
                source_images=image_paths,
                spec=spec,
                validation_before=current_validation,
                notes=job.notes,
                dimension_hint=job.dimension_hint,
                agent=agent,
                current_state_provider=lambda: planning_state(runtime.engine),
                render_before=lambda: (
                    dict(current_rendered)
                    if len(current_rendered) == 4
                    else _render_validation_views(runtime, report_dir / "nvidia_review" / "before", pass_index=0)
                ),
                render_after=lambda: _render_validation_views(
                    runtime, report_dir / "nvidia_review" / "after", pass_index=1
                ),
                snapshot=snapshot,
                restore=restore,
                validate_after=lambda: validate_photo_runtime(runtime, spec),
                validation_accept=_nvidia_photo_accept,
                validate_plan=_validate_plan,
                execute_plan=_nvidia_photo_execute,
                on_event=event,
                on_stage=lambda message, percent: self._update(
                    job, progress=max(job.progress, percent), message=message
                ),
                cancel_check=job.cancel_event.is_set,
            )
            _write_json(report_dir / "nvidia_review" / "review.json", nvidia_review)
            if nvidia_review.get("accepted") and isinstance(nvidia_review.get("validation_after"), dict):
                current_validation = nvidia_review["validation_after"]

        self._check(job)
        if not current_validation.get("brep_valid"):
            restore(initial_snapshot)
            raise RuntimeError("Photo reconstruction did not produce a valid editable B-Rep.")
        runtime.persist()
        state = runtime.state()
        report = {
            "job_id": job.id,
            "pipeline": "CADia Photo-to-CAD Full v1.5 public-runtime conditional cascade",
            "provider": "codex",
            "model": job.model,
            "vision_mode": job.vision_mode,
            "spec": spec,
            "validation": current_validation,
            "observation_score": round(current_observation, 5),
            "attempts": attempts,
            "refinements": refinements,
            "opencv_vision": opencv_vision,
            "nvidia_review": nvidia_review,
            "engines": [x.public() for x in engine_results],
            "cascade_stopped_after": cascade_stopped_after,
            "external_capabilities": external_capabilities(),
            "state_revision": (state.get("document") or {}).get("revision"),
        }
        _write_json(report_dir / "report.json", report)
        report["report"] = f"photo_cad/{job.id}/report.json"
        return {"state": state, **report}


def normalize_photo_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Photo extraction did not return an object.")
    dims = raw.get("overall_dimensions_mm") if isinstance(raw.get("overall_dimensions_mm"), dict) else {}

    def dim(value: Any) -> dict[str, Any]:
        value = value if isinstance(value, dict) else {}
        source = str(value.get("source") or "unknown")
        if source not in {"known", "measured", "estimated", "inferred_hidden", "unknown"}:
            source = "unknown"
        return {"value_mm": positive_or_none(value.get("value_mm")), "source": source, "confidence": clamp01(value.get("confidence"))}

    features = []
    for row in (raw.get("features") or [])[:100]:
        if not isinstance(row, dict):
            continue
        dimensions = []
        for item in (row.get("dimensions") or [])[:24]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "unknown")
            if source not in {"known", "measured", "estimated", "inferred_hidden", "unknown"}:
                source = "unknown"
            dimensions.append({
                "name": str(item.get("name") or "dimension")[:80],
                "value_mm": positive_or_none(item.get("value_mm")),
                "source": source,
                "confidence": clamp01(item.get("confidence")),
            })
        loc = row.get("location_mm") if isinstance(row.get("location_mm"), dict) else {}
        features.append({
            "kind": str(row.get("kind") or "feature")[:80],
            "description": str(row.get("description") or "")[:800],
            "count": int(row["count"]) if isinstance(row.get("count"), int) else None,
            "axis": None if row.get("axis") is None else str(row.get("axis"))[:40],
            "operation_hint": None if row.get("operation_hint") is None else str(row.get("operation_hint"))[:120],
            "dimensions": dimensions,
            "location_mm": {"x": finite_or_none(loc.get("x")), "y": finite_or_none(loc.get("y")), "z": finite_or_none(loc.get("z"))},
            "visible_in_views": [str(x)[:60] for x in (row.get("visible_in_views") or [])][:16],
            "confidence": clamp01(row.get("confidence")),
        })
    return {
        "part_name": str(raw.get("part_name") or "Photo Part")[:120],
        "object_summary": str(raw.get("object_summary") or "")[:1200],
        "views_detected": [str(x)[:80] for x in (raw.get("views_detected") or [])][:16],
        "principal_axes_assumption": str(raw.get("principal_axes_assumption") or "")[:500],
        "overall_dimensions_mm": {"x": dim(dims.get("x")), "y": dim(dims.get("y")), "z": dim(dims.get("z"))},
        "features": features,
        "symmetry": [str(x)[:400] for x in (raw.get("symmetry") or [])][:20],
        "repetitions": [str(x)[:400] for x in (raw.get("repetitions") or [])][:20],
        "assumptions": [str(x)[:500] for x in (raw.get("assumptions") or [])][:30],
        "hidden_geometry_assumptions": [str(x)[:500] for x in (raw.get("hidden_geometry_assumptions") or [])][:30],
        "uncertainties": [str(x)[:500] for x in (raw.get("uncertainties") or [])][:30],
        "confidence": clamp01(raw.get("confidence")),
    }


def validate_photo_spec(spec: dict[str, Any]) -> None:
    if clamp01(spec.get("confidence")) <= 0:
        raise RuntimeError("Product-photo analysis returned zero confidence.")
    if not spec.get("views_detected"):
        raise RuntimeError("Product-photo analysis could not identify any usable view.")
    if not spec.get("features") and not any((spec.get("overall_dimensions_mm") or {}).get(axis, {}).get("value_mm") for axis in ("x", "y", "z")):
        raise RuntimeError("Product-photo analysis could not infer manufacturable geometry.")


def photo_request_text(spec: dict[str, Any], notes: str, dimension_hint: str) -> str:
    return (
        "Reconstruct an editable CADia-native parametric model of the photographed physical object. "
        "Use typed CAD tools and preserve known/measured dimensions. "
        f"PHOTO_DESIGN_SPEC={json.dumps(spec, ensure_ascii=False)}. "
        f"Notes={notes or '(none)'}. Metric anchor={dimension_hint or '(none)'}"
    )


def _resolve_numeric(value: Any, numeric_params: dict[str, float]) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return finite_or_none(value)
    if isinstance(value, str):
        key = value.strip()
        if key in numeric_params:
            return finite_or_none(numeric_params[key])
        try:
            return finite_or_none(float(key.rstrip(" mmdeg")))
        except Exception:
            return None
    return None


def _flatten_feature_scalars(value: Any, numeric_params: dict[str, float], path: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    scalar = _resolve_numeric(value, numeric_params)
    if scalar is not None:
        out.append((path.lower(), scalar))
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            out.extend(_flatten_feature_scalars(child, numeric_params, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            out.extend(_flatten_feature_scalars(child, numeric_params, f"{path}[{index}]"))
    return out


def _feature_kind_matches(spec_kind: str, actual_kind: str, operation: str, name: str) -> bool:
    spec = str(spec_kind or "").lower().replace("_", " ").strip()
    actual = " ".join([str(actual_kind or ""), str(operation or ""), str(name or "")]).lower().replace("_", " ")
    if spec and spec in actual:
        return True
    groups = {
        "hole": ("hole", "counterbore", "countersink"),
        "fillet": ("fillet", "round"),
        "chamfer": ("chamfer", "bevel"),
        "slot": ("slot",),
        "revolve": ("revolve",),
        "shell": ("shell",),
        "pattern": ("pattern",),
        "extrude": ("extrude", "box", "prism"),
        "boss": ("boss", "extrude", "cylinder"),
        "pocket": ("pocket", "cut"),
        "cut": ("cut", "pocket"),
    }
    for key, aliases in groups.items():
        if key in spec or any(alias in spec for alias in aliases):
            return any(alias in actual for alias in aliases)
    return False


def _dimension_mode(name: str) -> str | None:
    raw = str(name or "").lower().replace("ø", " diameter ").replace("φ", " diameter ")
    if "diameter" in raw or " dia" in raw or raw.startswith("dia"):
        return "diameter"
    if "radius" in raw or "fillet r" in raw:
        return "radius"
    for key in ("thickness", "depth", "width", "height", "length", "pitch", "distance", "offset", "clearance"):
        if key in raw:
            return key
    return None


def _semantic_dimension_candidates(
    mode: str | None,
    feature_params: Any,
    numeric_params: dict[str, float],
) -> list[tuple[str, float]]:
    scalars = _flatten_feature_scalars(feature_params, numeric_params)
    if mode is None:
        return []
    terms = {
        "diameter": ("diameter", "dia", "radius"),
        "radius": ("radius", "diameter"),
        "thickness": ("thickness", "wall", "distance", "extent"),
        "depth": ("depth", "distance", "extent", "length"),
        "width": ("width",),
        "height": ("height",),
        "length": ("length", "distance", "extent"),
        "pitch": ("pitch",),
        "distance": ("distance", "offset", "position"),
        "offset": ("offset", "distance", "position"),
        "clearance": ("clearance", "gap"),
    }.get(mode, (mode,))
    out: list[tuple[str, float]] = []
    for path, value in scalars:
        if not any(term in path for term in terms):
            continue
        adjusted = value
        if mode == "diameter" and "radius" in path and "diameter" not in path:
            adjusted = value * 2.0
        elif mode == "radius" and "diameter" in path and "radius" not in path:
            adjusted = value / 2.0
        if math.isfinite(adjusted) and adjusted > 0:
            out.append((path, adjusted))
    return out


def validate_photo_runtime(runtime: CadRuntime, spec: dict[str, Any]) -> dict[str, Any]:
    with runtime.engine.lock:
        doc = runtime.engine.doc
        shape = doc.shape if doc is not None else None
        if shape is None:
            return {
                "solid_present": False,
                "brep_valid": False,
                "score": 0.0,
                "hard_dimensions": [],
                "hard_feature_dimensions": [],
                "hard_dimensions_passed": False,
            }
        try:
            brep_valid = bool(shape.isValid())
        except Exception:
            brep_valid = True
        try:
            box = shape.BoundingBox()
            actual = {"x": float(box.xlen), "y": float(box.ylen), "z": float(box.zlen)}
        except Exception:
            actual = {"x": None, "y": None, "z": None}
        try:
            numeric_params = {str(k): float(v) for k, v in doc.numeric_params().items()}
        except Exception:
            numeric_params = {}
        actual_features = [
            {
                "name": str(getattr(feature, "name", "")),
                "kind": str(getattr(feature, "kind", "")),
                "operation": str(getattr(feature, "operation", "")),
                "params": copy.deepcopy(getattr(feature, "params", {}) or {}),
            }
            for feature in getattr(doc, "features", [])
        ]

    hard = []
    estimated = []
    for axis in ("x", "y", "z"):
        row = (spec.get("overall_dimensions_mm") or {}).get(axis) or {}
        expected = positive_or_none(row.get("value_mm"))
        measured = positive_or_none(actual.get(axis))
        if expected is None or measured is None:
            continue
        rel = abs(measured - expected) / max(expected, 1e-9)
        entry = {
            "kind": "overall",
            "axis": axis,
            "expected_mm": round(expected, 5),
            "actual_mm": round(measured, 5),
            "source": row.get("source"),
            "relative_error": round(rel, 6),
        }
        if row.get("source") in {"known", "measured"}:
            entry["pass"] = rel <= 0.025
            hard.append(entry)
        else:
            entry["pass"] = rel <= 0.20
            estimated.append(entry)

    hard_feature: list[dict[str, Any]] = []
    for feature in spec.get("features") or []:
        if not isinstance(feature, dict):
            continue
        feature_kind = str(feature.get("kind") or "feature")
        matches = [
            row for row in actual_features
            if _feature_kind_matches(feature_kind, row["kind"], row["operation"], row["name"])
        ]
        for dim in feature.get("dimensions") or []:
            if not isinstance(dim, dict) or dim.get("source") not in {"known", "measured"}:
                continue
            expected = positive_or_none(dim.get("value_mm"))
            if expected is None:
                continue
            mode = _dimension_mode(str(dim.get("name") or ""))
            candidates: list[tuple[str, float, str]] = []
            for match in matches:
                for path, value in _semantic_dimension_candidates(mode, match.get("params"), numeric_params):
                    candidates.append((path, value, match["name"]))
            if candidates:
                path, measured, feature_name = min(
                    candidates,
                    key=lambda row: abs(float(row[1]) - expected) / max(expected, 1e-9),
                )
                rel = abs(measured - expected) / max(expected, 1e-9)
                hard_feature.append({
                    "kind": "feature",
                    "feature_kind": feature_kind,
                    "dimension": str(dim.get("name") or "dimension")[:100],
                    "expected_mm": round(expected, 5),
                    "actual_mm": round(measured, 5),
                    "matched_feature": feature_name,
                    "matched_parameter": path,
                    "source": dim.get("source"),
                    "relative_error": round(rel, 6),
                    "verified": True,
                    "pass": rel <= 0.025,
                })
            else:
                hard_feature.append({
                    "kind": "feature",
                    "feature_kind": feature_kind,
                    "dimension": str(dim.get("name") or "dimension")[:100],
                    "expected_mm": round(expected, 5),
                    "actual_mm": None,
                    "source": dim.get("source"),
                    "relative_error": None,
                    "verified": False,
                    "pass": False,
                })

    all_hard = hard + hard_feature
    hard_pass = all(bool(x.get("pass")) for x in all_hard)
    hard_score = 1.0 if not all_hard else sum(
        1.0 if x.get("pass") else max(0.0, 1.0 - float(x.get("relative_error") or 1.0))
        for x in all_hard
    ) / len(all_hard)
    estimated_score = 1.0 if not estimated else sum(
        max(0.0, 1.0 - min(1.0, float(x["relative_error"]))) for x in estimated
    ) / len(estimated)
    feature_conf = 0.0
    features = spec.get("features") or []
    if features:
        feature_conf = sum(clamp01(x.get("confidence")) for x in features if isinstance(x, dict)) / max(1, len(features))
    score = (0.48 if brep_valid else 0.0) + 0.37 * hard_score + 0.10 * estimated_score + 0.05 * feature_conf
    return {
        "solid_present": True,
        "brep_valid": brep_valid,
        "bounding_box_mm": actual,
        "hard_dimensions": hard,
        "hard_feature_dimensions": hard_feature,
        "estimated_dimensions": estimated,
        "hard_dimensions_passed": hard_pass,
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def acceptable_refinement(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if not after.get("solid_present") or not after.get("brep_valid"):
        return False
    if before.get("hard_dimensions_passed") and not after.get("hard_dimensions_passed"):
        return False
    before_hard = hard_metric_score(before)
    after_hard = hard_metric_score(after)
    if after_hard + 1e-9 < before_hard:
        return False
    return True


def hard_metric_score(validation: dict[str, Any]) -> float:
    rows = list(validation.get("hard_dimensions") or []) + list(validation.get("hard_feature_dimensions") or [])
    if not rows:
        return 1.0
    return sum(
        1.0 if row.get("pass") else max(0.0, 1.0 - float(row.get("relative_error") or 1.0))
        for row in rows
    ) / len(rows)


def validation_feedback(validation: dict[str, Any]) -> str:
    issues = []
    if not validation.get("brep_valid"):
        issues.append("B-Rep is invalid")
    for row in validation.get("hard_dimensions") or []:
        if not row.get("pass"):
            issues.append(f"{row['axis']} hard dimension expected {row['expected_mm']} mm but got {row['actual_mm']} mm")
    for row in validation.get("hard_feature_dimensions") or []:
        if not row.get("pass"):
            if row.get("verified"):
                issues.append(
                    f"{row.get('feature_kind')} {row.get('dimension')} expected {row.get('expected_mm')} mm "
                    f"but got {row.get('actual_mm')} mm"
                )
            else:
                issues.append(
                    f"Could not verify hard feature dimension {row.get('feature_kind')} {row.get('dimension')}="
                    f"{row.get('expected_mm')} mm in the native feature parameters"
                )
    return "; ".join(issues)


def positive_or_none(value: Any) -> float | None:
    try:
        n = float(value)
    except Exception:
        return None
    return n if math.isfinite(n) and n > 0 else None


def finite_or_none(value: Any) -> float | None:
    try:
        n = float(value)
    except Exception:
        return None
    return n if math.isfinite(n) else None


def clamp01(value: Any) -> float:
    try:
        n = float(value)
    except Exception:
        n = 0.0
    if not math.isfinite(n):
        n = 0.0
    return max(0.0, min(1.0, n))


photo_cad_jobs = None if os.getenv("CADIA_ISOLATED_WORKER") == "1" else PhotoCadJobManager()
