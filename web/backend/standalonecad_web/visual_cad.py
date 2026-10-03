from __future__ import annotations

import copy
import json
import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from standalonecad.core.render import render_shape_png
from standalonecad.recovery import execute_with_failure_recovery

from .cad_runtime import CadRuntime, runtime_manager
from .drawing2cad_native import (
    compile_partspec_to_native_plan,
    report_score as native_report_score,
    validate_runtime_against_partspec,
    validate_runtime_payload,
)
from .drawing2cad_upstream import extract_partspec
from .config import settings
from .providers.app_server import app_server_manager
from .state import planning_state
from .web_agent import DesktopParityCodexAgent, desktop_catalog
from .nvidia_review import run_engineering_review, render_review_views_isolated
from .opencv_cad import (
    analyze_images as opencv_analyze_images,
    compact_analysis as opencv_compact_analysis,
    compact_verification as opencv_compact_verification,
    verification_prompt as opencv_verification_prompt,
    verify_source_to_renders as opencv_verify_source_to_renders,
)
from .cadimage_protocol import (
    MAX_AUDIT_REPAIR as CADIMAGE_MAX_AUDIT_REPAIR,
    MAX_VISUAL_LOOPS as CADIMAGE_MAX_VISUAL_LOOPS,
    UPSTREAM_COMMIT as CADIMAGE_UPSTREAM_COMMIT,
    UPSTREAM_PROTOCOL as CADIMAGE_UPSTREAM_PROTOCOL,
    UPSTREAM_REPO as CADIMAGE_UPSTREAM_REPO,
    audit_crosscheck as cadimage_audit_crosscheck,
    candidate_public as cadimage_candidate_public,
    circle_pair_error as cadimage_circle_pair_error,
    needs_feature_audit as cadimage_needs_feature_audit,
    needs_targeted_refinement as cadimage_needs_targeted_refinement,
    pick_best_candidate as cadimage_pick_best_candidate,
    prepare_semantic_inputs as cadimage_prepare_semantic_inputs,
    visual_score as cadimage_visual_score,
)


# Phase 1 intentionally uses the user's existing ChatGPT/Codex connection.
# No GPU is required on the CADia server. Drawing images are attached to the
# Codex app-server turn as native localImage inputs; no CLI view_image helper is
# used. The model returns typed CADia tool plans, never executable Python.

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_REFINEMENTS = 0
VISUAL_CAD_WORKERS = max(1, min(2, settings.max_concurrent_ai_turns))
DRAWING_VALIDATION_THRESHOLD = 0.90
# Product/OpenCV Drawing-to-CAD does not run the NVIDIA/Cosmos/Nemotron reviewer by
# default.  The implementation remains installed for the separate NVIDIA branch.
DRAWING_NVIDIA_REVIEW_ENABLED = False  # product/OpenCV branch: preserved in source, never on critical path
# OpenCV is deliberately off the successful modeling critical path.  It may be used
# once as preprocessing only after the normal CADia routes miss the objective 90%
# Drawing2CAD threshold, and it never supplies pixel-derived dimensions.
OPENCV_RESCUE_ENABLED = False  # v4.1.2: no extra PartSpec/CAD regeneration is triggered by OpenCV
OPENCV_RESCUE_MIN_DELTA = max(0.0, min(0.25, float(os.getenv("CADIA_OPENCV_RESCUE_MIN_DELTA", "0.01"))))
OPENCV_INSPECTION_WORKERS = 1
OPENCV_REPAIR_FEEDBACK_MAX_CHARS = 6000
OPENCV_REPAIR_EFFORT = "medium"
VISUAL_JOB_HARD_TIMEOUT_SECONDS = max(300, int(os.getenv("CADIA_VISUAL_JOB_HARD_TIMEOUT", "7200")))
VISUAL_TERMINAL_STATUSES = {"done", "failed", "cancelled"}


DRAWING_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "part_name": {"type": "string"},
        "units": {"type": "string"},
        "views_detected": {"type": "array", "items": {"type": "string"}},
        "overall_dimensions_mm": {
            "type": "object",
            "properties": {
                "x": {"type": ["number", "null"]},
                "y": {"type": ["number", "null"]},
                "z": {"type": ["number", "null"]},
            },
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
                            },
                            "required": ["name", "value_mm"],
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
                },
                "required": ["kind", "description", "count", "axis", "operation_hint", "dimensions", "location_mm"],
                "additionalProperties": False,
            },
        },
        "symmetry": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "part_name",
        "units",
        "views_detected",
        "overall_dimensions_mm",
        "features",
        "symmetry",
        "assumptions",
        "uncertainties",
        "confidence",
    ],
    "additionalProperties": False,
}


VISUAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "minItems": 1,
            "maxItems": 80,
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


REFINEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["accept", "refine"]},
        "confidence": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "calls": {
            "type": "array",
            "maxItems": 40,
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
    "required": ["action", "confidence", "issues", "calls", "note"],
    "additionalProperties": False,
}


FEATURE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["accept", "refine"]},
        "confidence": {"type": "number"},
        "features": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "expected_count": {"type": ["integer", "null"]},
                    "model_count": {"type": ["integer", "null"]},
                    "status": {"type": "string", "enum": ["match", "missing", "wrong", "uncertain"]},
                    "evidence": {"type": "string"},
                },
                "required": ["kind", "expected_count", "model_count", "status", "evidence"],
                "additionalProperties": False,
            },
        },
        "issues": {"type": "array", "items": {"type": "string"}},
        "calls": {
            "type": "array",
            "maxItems": 40,
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
    "required": ["action", "confidence", "features", "issues", "calls", "note"],
    "additionalProperties": False,
}

@dataclass
class VisualCadJob:
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


class VisualCadJobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, VisualCadJob] = {}
        self._pool = ThreadPoolExecutor(max_workers=VISUAL_CAD_WORKERS, thread_name_prefix="cadia-visual-cad")
        self._inspection_pool = ThreadPoolExecutor(max_workers=OPENCV_INSPECTION_WORKERS, thread_name_prefix="cadia-opencv-inspect")

    def _job_state_path(self, job_id: str, user_id: str, project_id: str) -> Path:
        return runtime_manager.project_root(user_id, project_id) / "visual_cad" / "jobs" / f"{job_id}.json"

    def _persist_job(self, job: VisualCadJob) -> None:
        try:
            path = self._job_state_path(job.id, job.user_id, job.project_id)
            _write_json(path, {
                "id": job.id,
                "user_id": job.user_id,
                "project_id": job.project_id,
                "vision_mode": job.vision_mode,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "error": job.error,
                "result": job.result,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "deadline_at": job.deadline_at,
            })
        except Exception:
            # Job persistence must never make a modeling operation fail.
            pass

    def _load_persisted(self, job_id: str, user_id: str, project_id: str) -> VisualCadJob | None:
        path = self._job_state_path(job_id, user_id, project_id)
        if not path.is_file():
            return None
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            status = str(row.get("status") or "failed")
            error = row.get("error")
            message = str(row.get("message") or "")
            if status in {"queued", "running", "cancelling"}:
                status = "failed"
                error = "Visual CAD job was interrupted by a server restart."
                message = "Interrupted by server restart"
            return VisualCadJob(
                id=job_id, user_id=user_id, project_id=project_id, input_files={},
                vision_mode=str(row.get("vision_mode") or "standard"),
                status=status, progress=int(row.get("progress") or 0), message=message,
                error=None if error is None else str(error), result=row.get("result"),
                created_at=float(row.get("created_at") or time.time()),
                updated_at=float(row.get("updated_at") or time.time()),
                deadline_at=(float(row.get("deadline_at")) if row.get("deadline_at") is not None else None),
            )
        except Exception:
            return None

    def create(self, job: VisualCadJob) -> VisualCadJob:
        with self._lock:
            if job.deadline_at is None:
                job.deadline_at = time.time() + VISUAL_JOB_HARD_TIMEOUT_SECONDS
            self._jobs[job.id] = job
            self._purge_locked()
            self._persist_job(job)
        future = self._pool.submit(self._run, job.id)
        future.add_done_callback(lambda completed, jid=job.id: self._worker_done(jid, completed))
        threading.Thread(target=self._watchdog, args=(job.id,), name=f"cadia-visual-watchdog-{job.id[:8]}", daemon=True).start()
        return job

    def _worker_done(self, job_id: str, future) -> None:
        try:
            exc = future.exception()
        except BaseException as callback_exc:
            exc = callback_exc
        if exc is None:
            return
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status in VISUAL_TERMINAL_STATUSES:
            return
        job.error = f"Drawing-to-CAD worker terminated unexpectedly: {exc}"
        self._update(job, status="failed", message="Drawing reconstruction worker stopped unexpectedly")

    def _watchdog(self, job_id: str) -> None:
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status in VISUAL_TERMINAL_STATUSES:
                    return
                deadline = job.deadline_at
            if deadline is None:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None or job.status in VISUAL_TERMINAL_STATUSES:
                        return
                    job.timed_out = True
                    job.cancel_event.set()
                    job.error = f"Drawing-to-CAD exceeded the hard runtime limit of {VISUAL_JOB_HARD_TIMEOUT_SECONDS} seconds."
                self._update(job, status="failed", message="Drawing reconstruction timed out")
                return
            time.sleep(min(1.0, max(0.1, remaining)))

    def get(self, job_id: str, user_id: str, project_id: str) -> VisualCadJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._load_persisted(job_id, user_id, project_id)
                if job is not None:
                    self._jobs[job.id] = job
            if job is None or job.user_id != user_id or job.project_id != project_id:
                raise KeyError(job_id)
            return job

    def cancel(self, job_id: str, user_id: str, project_id: str) -> VisualCadJob:
        job = self.get(job_id, user_id, project_id)
        if job.status in VISUAL_TERMINAL_STATUSES:
            return job
        job.cancel_event.set()
        self._update(job, status="cancelling", message="Cancelling…")
        return job

    def _purge_locked(self) -> None:
        cutoff = time.time() - 6 * 3600
        stale = [key for key, value in self._jobs.items() if value.updated_at < cutoff and value.status in {"done", "failed", "cancelled"}]
        for key in stale:
            self._jobs.pop(key, None)

    def _update(self, job: VisualCadJob, *, status: str | None = None, progress: int | None = None, message: str | None = None) -> None:
        with self._lock:
            # Once terminal, a late worker/watchdog callback must never resurrect the job.
            if job.status in VISUAL_TERMINAL_STATUSES and status != job.status:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, max(0, min(100, int(progress))))
            if message is not None:
                job.message = message
            job.updated_at = time.time()
            self._persist_job(job)

    def _check_cancel(self, job: VisualCadJob) -> None:
        if job.timed_out or (job.deadline_at is not None and time.time() >= job.deadline_at):
            job.timed_out = True
            job.cancel_event.set()
            raise TimeoutError("Drawing-to-CAD exceeded its hard runtime limit.")
        if job.cancel_event.is_set():
            raise InterruptedError("Visual CAD job was cancelled.")

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        try:
            # Drawing→CAD always uses CADia's dedicated Codex app-server client inside
            # _reconstruct(). The globally selected chat provider must not block it.
            self._update(job, status="running", progress=3, message="Preparing drawing reconstruction…")
            result = self._reconstruct(job)
            queue_postcheck = False
            if job.vision_mode == "opencv5":
                inspection = result.get("opencv_vision") if isinstance(result, dict) else None
                queue_postcheck = not (isinstance(inspection, dict) and inspection.get("status") == "done")
                if queue_postcheck:
                    result["opencv_vision"] = {
                        "mode": "opencv5",
                        "status": "queued",
                        "read_only": True,
                        "action": "inspection_pending",
                        "rescue": result.get("opencv_rescue"),
                    }
            job.result = result
            self._check_cancel(job)
            self._update(job, status="done", progress=100, message="Editable CAD reconstructed")
            if queue_postcheck:
                self._inspection_pool.submit(self._run_opencv_postcheck, job.id)
        except TimeoutError as exc:
            job.error = str(exc)
            self._update(job, status="failed", message="Drawing reconstruction timed out")
        except InterruptedError as exc:
            job.error = str(exc)
            self._update(job, status="cancelled", message="Cancelled")
        except Exception as exc:
            job.error = str(exc)
            self._update(job, status="failed", message="Drawing reconstruction failed")

    def _set_postcheck_result(self, job: VisualCadJob, payload: dict[str, Any]) -> None:
        with self._lock:
            if job.result is None:
                return
            job.result["opencv_vision"] = payload
            job.updated_at = time.time()
            self._persist_job(job)

    def _run_opencv_postcheck(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.vision_mode != "opencv5" or not isinstance(job.result, dict):
            return
        rescue = job.result.get("opencv_rescue")
        self._set_postcheck_result(job, {
            "mode": "opencv5", "status": "running", "read_only": True,
            "action": "inspecting", "rescue": rescue,
        })
        try:
            runtime = runtime_manager.get(job.user_id, job.project_id)
            report_dir = runtime.root / "visual_cad" / job.id / "opencv" / "postcheck"
            expected_revision = job.result.get("state_revision")
            current_state = runtime.state()
            current_revision = (current_state.get("document") or {}).get("revision")
            if expected_revision is not None and current_revision != expected_revision:
                self._set_postcheck_result(job, {
                    "mode": "opencv5", "status": "skipped_stale", "read_only": True,
                    "action": "skipped",
                    "reason": "CAD changed after reconstruction; stale OpenCV inspection was not run.",
                    "rescue": rescue,
                })
                return
            source_paths = {label: Path(path).resolve() for label, path in job.input_files.items() if Path(path).is_file()}
            source = opencv_analyze_images(
                source_paths, output_dir=report_dir / "source", source_kind="drawing"
            )
            rendered = _render_validation_views(runtime, report_dir / "renders", pass_index=40)
            verification = opencv_verify_source_to_renders(
                source_paths, rendered, output_dir=report_dir / "compare", source_kind="drawing"
            )
            # Never turn a pixel/CV discrepancy into an automatic CAD edit.  The raw
            # verifier action is retained for diagnostics only.
            final = opencv_compact_verification(verification)
            raw_action = str(final.get("action") or "ambiguous")
            final["raw_action"] = raw_action
            final["action"] = "pass" if raw_action == "pass" else "review"
            artifacts: list[dict[str, Any]] = []
            for row in source.get("artifacts") or []:
                item = dict(row)
                item["path"] = f"postcheck/source/{item.get('path', '')}".strip("/")
                artifacts.append(item)
            for row in verification.get("artifacts") or []:
                item = dict(row)
                item["path"] = f"postcheck/compare/{item.get('path', '')}".strip("/")
                artifacts.append(item)
            payload = {
                "mode": "opencv5",
                "status": "done",
                "read_only": True,
                "source": opencv_compact_analysis(source),
                "final": final,
                "score": final.get("score", 0.0),
                "action": final.get("action", "review"),
                "raw_action": raw_action,
                "issues": final.get("issues") or [],
                "artifacts": artifacts,
                "performance": verification.get("performance") or {},
                "rescue": rescue,
            }
            _write_json(report_dir / "postcheck.json", payload)
            self._set_postcheck_result(job, payload)
        except Exception as exc:
            # Post-check is intentionally fail-open. It must never change a completed
            # modeling job into failed or roll back a valid model.
            self._set_postcheck_result(job, {
                "mode": "opencv5", "status": "failed", "read_only": True,
                "action": "unavailable", "reason": str(exc), "rescue": rescue,
            })

    def _reconstruct(self, job: VisualCadJob) -> dict[str, Any]:
        """Drawing -> CAD with Hotfix-5 native modeling preserved.

        Primary route:
          Drawing2CAD PartSpec -> deterministic CADia native plan when safe ->
          CADia typed-tool execution/recovery -> Drawing2CAD validator ->
          bounded topology-aware planner replans. OpenCV is excluded from the
          successful critical path and is read-only after completion.

        If PartSpec extraction or the boosted route cannot produce a usable solid, the
        original Hotfix-5 DrawingSpec -> CADia planner path is retained as the final
        no-regression fallback.  No generated Python is executed and no STEP is imported.
        """
        runtime = runtime_manager.get(job.user_id, job.project_id)
        client = app_server_manager.get(job.user_id)
        codex_job_dir = (client.workspace / "visual-cad" / job.id).resolve()
        codex_job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        report_dir = runtime.root / "visual_cad" / job.id
        report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        image_paths: dict[str, Path] = {}
        for label, raw in job.input_files.items():
            source = Path(raw).resolve()
            suffix = source.suffix.lower()
            target = codex_job_dir / f"input_{label}{suffix}"
            shutil.copy2(source, target)
            os.chmod(target, 0o600)
            image_paths[label] = target

        # OpenCV mode adds bounded low-level geometry evidence to the SAME first
        # PartSpec extraction call. It does not add an AI turn or CAD run. Originals
        # remain source of truth; low-level primitives/regions are advisory only.
        semantic_images = list(image_paths.values())
        extraction_notes = job.notes
        opencv_input_assist: dict[str, Any] = {
            "mode": "standard", "used_for_modeling": False,
            "additional_ai_calls": 0, "additional_cad_runs": 0,
        }
        if job.vision_mode == "opencv5":
            try:
                assist = cadimage_prepare_semantic_inputs(
                    image_paths, output_dir=codex_job_dir / "opencv-input-assist", max_aux_images=2
                )
                aux = [Path(x).resolve() for x in (assist.get("assist_images") or []) if Path(x).is_file()]
                semantic_images.extend(aux)
                opencv_input_assist = {k: v for k, v in assist.items() if k != "assist_images"}
                opencv_input_assist["assist_count"] = len(aux)
                if aux:
                    extraction_notes = ((job.notes or "").strip() + "\n\n" +
                        "OpenCV assist images are low-level geometry evidence appended after the original drawing. "
                        "The original drawing and written dimensions are authoritative. Lines/circles/contours and "
                        "unlabeled regions are candidates only; do not infer feature semantics or millimetres from pixels.").strip()
            except Exception as exc:
                semantic_images = list(image_paths.values())
                opencv_input_assist = {
                    "mode": "single_pass_geometry_assist", "used_for_modeling": False,
                    "additional_ai_calls": 0, "additional_cad_runs": 0,
                    "status": "unavailable", "reason": str(exc),
                }

        effective_notes = job.notes
        opencv_rescue: dict[str, Any] | None = None
        opencv_rescue_images: dict[str, Path] | None = None
        opencv_rescue_source: dict[str, Any] | None = None

        agent = VisualDrawingCodexAgent(
            client=client,
            target_id=f"visual-cad:{job.project_id}:{job.id}",
            project_root=runtime.root,
            model=job.model or "gpt-5.6-sol",
            reasoning=job.effort or "high",
        )

        def event(kind: str, value: Any) -> None:
            self._check_cancel(job)
            if kind != "progress":
                return
            text = str(value.get("message") or "Analyzing…") if isinstance(value, dict) else str(value)
            # Keep UI progress monotonic even when nested Codex/executor stages emit
            # their own local percentages.
            if job.progress < 28:
                pct = max(job.progress, 12)
            elif job.progress < 58:
                pct = max(job.progress, 40)
            else:
                pct = max(job.progress, 70)
            self._update(job, progress=pct, message=text)

        def snapshot() -> dict[str, Any]:
            with runtime.engine.lock:
                return copy.deepcopy(runtime.engine._workspace_snapshot())

        def restore(snap: dict[str, Any]) -> None:
            with runtime.engine.lock:
                runtime.engine._restore_workspace(copy.deepcopy(snap), bump_revision=True)

        def run_plan(
            plan: dict[str, Any],
            *,
            user_prompt: str,
            label: str,
            initial_snapshot: dict[str, Any],
            progress_span: tuple[int, int],
            max_ai_repairs: int = 2,
            validation_spec: Any | None = None,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
            # Serialize every document mutation for this project exactly like the original
            # Hotfix-5 path.  Visual-CAD jobs and normal browser edits must never race.
            with runtime.operation_lock:
                restore(initial_snapshot)
                _validate_plan(plan)
                executed = execute_with_failure_recovery(
                    executor=runtime.executor,
                    agent=agent,
                    user_prompt=user_prompt,
                    state_provider=lambda: planning_state(runtime.engine),
                    plan=plan,
                    on_event=event,
                    initial_progress_span=progress_span,
                    max_ai_repairs=max_ai_repairs,
                )
                if not executed.ok:
                    return None, None, str(executed.error or f"{label} execution failed")
                runtime.persist()
                try:
                    payload = validate_runtime_payload(runtime, validation_spec or partspec)
                except Exception as exc:
                    return None, None, f"{label} validator error: {exc}"
                return payload, snapshot(), None

        def get_opencv_rescue_images() -> tuple[dict[str, Path], dict[str, Any]]:
            nonlocal opencv_rescue_images, opencv_rescue_source
            if opencv_rescue_images is not None and opencv_rescue_source is not None:
                return opencv_rescue_images, opencv_rescue_source
            self._check_cancel(job)
            rescue_dir = report_dir / "opencv" / "rescue_source"
            source = opencv_analyze_images(
                image_paths, output_dir=rescue_dir, source_kind="drawing"
            )
            prepared: dict[str, Path] = {}
            for label, original in image_paths.items():
                row = (source.get("views") or {}).get(label) or {}
                cleaned_name = ((row.get("artifacts") or {}).get("cleaned") if isinstance(row, dict) else None)
                cleaned = rescue_dir / str(cleaned_name or "")
                if cleaned_name and cleaned.is_file():
                    target = codex_job_dir / f"opencv_rescue_{label}.png"
                    shutil.copy2(cleaned, target)
                    os.chmod(target, 0o600)
                    prepared[label] = target
                else:
                    prepared[label] = original
            opencv_rescue_images = prepared
            opencv_rescue_source = source
            return prepared, source

        def rescue_notes() -> str:
            prefix = (job.notes + "\n\n" if job.notes else "")
            return prefix + (
                "OpenCV RESCUE PREPROCESSING ONLY: the attached drawing images may have been "
                "deskewed/perspective-corrected for readability. Treat the drawing annotations "
                "and explicit written dimensions as the only dimensional source of truth. Do not "
                "infer millimetres, feature count, topology, or design intent from pixel lengths, "
                "Hough primitives, crops, or OpenCV measurements."
            )

        self._check_cancel(job)
        initial_snapshot = snapshot()
        attempts: list[dict[str, Any]] = []
        partspec = None
        partspec_error: str | None = None
        best_snapshot: dict[str, Any] | None = None
        best_payload: dict[str, Any] | None = None
        best_label = ""

        # ------------------------------------------------------------------ PartSpec
        try:
            self._update(job, progress=8, message="Extracting structured PartSpec from drawing…")
            partspec = extract_partspec(
                client,
                semantic_images,
                model=job.model or "gpt-5.6-sol",
                effort=job.effort or "high",
                notes=extraction_notes,
                dimension_hint=job.dimension_hint,
                on_event=event,
                cancel_check=job.cancel_event.is_set,
            )
            errors = partspec.sanity_check()
            if errors:
                raise ValueError("; ".join(errors))
            _write_json(report_dir / "partspec.json", partspec.model_dump(mode="json"))
        except InterruptedError:
            raise
        except Exception as exc:
            partspec_error = str(exc)
            partspec = None

        # If semantic extraction itself failed, OpenCV mode gets exactly one rescue
        # attempt using only readability-preprocessed images. No CV measurements are
        # inserted into the prompt. Standard mode remains byte-for-byte on its old path.
        if partspec is None and job.vision_mode == "opencv5" and OPENCV_RESCUE_ENABLED:
            try:
                self._update(job, progress=12, message="OpenCV 5 · one-shot readability rescue after extraction failure…")
                rescue_map, rescue_source = get_opencv_rescue_images()
                rescued_spec = extract_partspec(
                    client,
                    list(rescue_map.values()),
                    model=job.model or "gpt-5.6-sol",
                    effort=job.effort or "high",
                    notes=rescue_notes(),
                    dimension_hint=job.dimension_hint,
                    on_event=event,
                    cancel_check=job.cancel_event.is_set,
                )
                errors = rescued_spec.sanity_check()
                if errors:
                    raise ValueError("; ".join(errors))
                partspec = rescued_spec
                opencv_rescue = {
                    "attempted": True, "trigger": "partspec_extraction_failed",
                    "partspec_recovered": True, "accepted_for_modeling": None,
                    "source": opencv_compact_analysis(rescue_source),
                }
                _write_json(report_dir / "partspec_opencv_rescue.json", partspec.model_dump(mode="json"))
            except InterruptedError:
                raise
            except Exception as rescue_exc:
                opencv_rescue = {
                    "attempted": True, "trigger": "partspec_extraction_failed",
                    "accepted_for_modeling": False, "reason": str(rescue_exc),
                }

        # --------------------------------------------------------- boosted native route
        if partspec is not None:
            request_text = agent.partspec_request_text(partspec, image_paths, effective_notes, job.dimension_hint)
            compile_result = compile_partspec_to_native_plan(partspec)
            feedback = ""

            if compile_result.plan is not None:
                self._check_cancel(job)
                self._update(job, progress=28, message="Compiling PartSpec directly to CADia native features…")
                payload, snap, error = run_plan(
                    compile_result.plan,
                    user_prompt=request_text,
                    label="deterministic native compile",
                    initial_snapshot=initial_snapshot,
                    progress_span=(32, 58),
                )
                row = {"route": "deterministic_native", "reason": compile_result.reason, "error": error}
                if payload is not None:
                    row["validation"] = payload
                    if best_payload is None or float(payload["score"]) > float(best_payload["score"]):
                        best_payload, best_snapshot, best_label = payload, snap, "deterministic_native"
                    feedback = str(payload.get("feedback") or "")
                attempts.append(row)

            # If deterministic compilation is unavailable or fails Layer-1 validation,
            # ask CADia's existing typed-tool planner to model the exact PartSpec.  Each
            # replan starts from the original workspace so failed geometry never accumulates.
            threshold = DRAWING_VALIDATION_THRESHOLD
            if best_payload is None or float(best_payload.get("score", 0.0)) < threshold:
                for replan_index in range(1, 3):
                    self._check_cancel(job)
                    self._update(
                        job,
                        progress=46 + (replan_index - 1) * 8,
                        message=f"Planning native CAD from PartSpec ({replan_index}/2)…",
                    )
                    restore(initial_snapshot)
                    current_state = planning_state(runtime.engine)
                    plan = agent.plan_from_partspec(
                        partspec,
                        image_paths,
                        current_state,
                        validator_feedback=feedback,
                        on_event=event,
                        cancel_check=job.cancel_event.is_set,
                    )
                    payload, snap, error = run_plan(
                        plan,
                        user_prompt=request_text + (" Validator feedback: " + feedback if feedback else ""),
                        label=f"PartSpec native planner {replan_index}",
                        initial_snapshot=initial_snapshot,
                        progress_span=(52, 72),
                    )
                    row = {"route": f"partspec_planner_{replan_index}", "error": error}
                    if payload is not None:
                        row["validation"] = payload
                        if best_payload is None or float(payload["score"]) > float(best_payload["score"]):
                            best_payload, best_snapshot, best_label = payload, snap, f"partspec_planner_{replan_index}"
                        feedback = str(payload.get("feedback") or "")
                        if float(payload.get("score", 0.0)) >= threshold:
                            attempts.append(row)
                            break
                    attempts.append(row)

            # ----------------------------------------------------- Hotfix-5 no-regression rescue
            # Only pay for the legacy extraction/planner when the PartSpec-native routes
            # could not meet the upstream 90% validation threshold.
            if best_payload is None or float(best_payload.get("score", 0.0)) < threshold:
                try:
                    restore(initial_snapshot)
                    self._update(job, progress=70, message="Trying original Hotfix-5 native planner fallback…")
                    legacy_spec = agent.extract_spec(
                        image_paths,
                        notes=effective_notes,
                        dimension_hint=job.dimension_hint,
                        on_event=event,
                    )
                    legacy_spec = _normalize_spec(legacy_spec)
                    _validate_extraction(legacy_spec)
                    with runtime.operation_lock:
                        legacy_plan = agent.plan_from_drawing(
                            legacy_spec,
                            image_paths,
                            planning_state(runtime.engine),
                            on_event=event,
                        )
                        _validate_plan(legacy_plan)
                        legacy_exec = execute_with_failure_recovery(
                            executor=runtime.executor,
                            agent=agent,
                            user_prompt=agent.visual_request_text(
                                legacy_spec, image_paths, effective_notes, job.dimension_hint
                            ),
                            state_provider=lambda: planning_state(runtime.engine),
                            plan=legacy_plan,
                            on_event=event,
                            initial_progress_span=(72, 84),
                            max_ai_repairs=2,
                        )
                    if legacy_exec.ok:
                        legacy_payload = validate_runtime_payload(runtime, partspec)
                        legacy_snap = snapshot()
                        attempts.append({"route": "hotfix5_fallback", "validation": legacy_payload})
                        if best_payload is None or float(legacy_payload["score"]) > float(best_payload["score"]):
                            best_payload, best_snapshot, best_label = legacy_payload, legacy_snap, "hotfix5_fallback"
                    else:
                        attempts.append({"route": "hotfix5_fallback", "error": str(legacy_exec.error or "execution failed")})
                except InterruptedError:
                    raise
                except Exception as exc:
                    attempts.append({"route": "hotfix5_fallback", "error": str(exc)})

            # ----------------------------------------------------- OpenCV hard-failure rescue
            # Do NOT treat a sub-0.90 objective score as an OpenCV failure trigger.
            # The normal CADia routes already spent their bounded retries selecting the
            # best candidate.  Starting another vision/planner cascade for every merely
            # low-scoring-but-valid candidate creates serial high-reasoning Codex turns
            # and can dominate wall-clock latency.  OpenCV rescue is reserved for the
            # hard case where the normal routes produced no candidate at all.
            if (
                job.vision_mode == "opencv5"
                and OPENCV_RESCUE_ENABLED
                and best_payload is None
            ):
                baseline_score = 0.0
                rescue_row: dict[str, Any] = {
                    "attempted": True, "trigger": "no_valid_candidate",
                    "baseline_score": baseline_score, "accepted_for_modeling": False,
                }
                try:
                    self._update(job, progress=max(job.progress, 82), message="OpenCV 5 · one-shot rescue after native reconstruction failure…")
                    rescue_map, rescue_source = get_opencv_rescue_images()
                    rescue_spec = extract_partspec(
                        client,
                        list(rescue_map.values()),
                        model=job.model or "gpt-5.6-sol",
                        effort=job.effort or "high",
                        notes=rescue_notes(),
                        dimension_hint=job.dimension_hint,
                        on_event=event,
                        cancel_check=job.cancel_event.is_set,
                    )
                    errors = rescue_spec.sanity_check()
                    if errors:
                        raise ValueError("; ".join(errors))
                    rescue_row["source"] = opencv_compact_analysis(rescue_source)
                    rescue_row["partspec"] = rescue_spec.model_dump(mode="json")
                    rescue_request = agent.partspec_request_text(rescue_spec, rescue_map, rescue_notes(), job.dimension_hint)
                    rescue_compile = compile_partspec_to_native_plan(rescue_spec)
                    rescue_best_payload = None
                    rescue_best_snapshot = None
                    rescue_best_label = ""
                    rescue_feedback = ""
                    if rescue_compile.plan is not None:
                        payload, snap, error = run_plan(
                            rescue_compile.plan, user_prompt=rescue_request,
                            label="OpenCV rescue deterministic compile",
                            initial_snapshot=initial_snapshot, progress_span=(84, 91),
                            max_ai_repairs=1, validation_spec=rescue_spec,
                        )
                        rescue_row["deterministic_error"] = error
                        if payload is not None:
                            rescue_best_payload, rescue_best_snapshot, rescue_best_label = payload, snap, "opencv_rescue_deterministic"
                            rescue_feedback = str(payload.get("feedback") or "")
                    if rescue_best_payload is None or float(rescue_best_payload.get("score", 0.0)) < threshold:
                        restore(initial_snapshot)
                        plan = agent.plan_from_partspec(
                            rescue_spec, rescue_map, planning_state(runtime.engine),
                            validator_feedback=rescue_feedback, on_event=event,
                            cancel_check=job.cancel_event.is_set,
                        )
                        payload, snap, error = run_plan(
                            plan, user_prompt=rescue_request + (" Validator feedback: " + rescue_feedback if rescue_feedback else ""),
                            label="OpenCV rescue native planner", initial_snapshot=initial_snapshot,
                            progress_span=(88, 94), max_ai_repairs=1, validation_spec=rescue_spec,
                        )
                        rescue_row["planner_error"] = error
                        if payload is not None and (rescue_best_payload is None or float(payload.get("score", 0.0)) > float(rescue_best_payload.get("score", 0.0))):
                            rescue_best_payload, rescue_best_snapshot, rescue_best_label = payload, snap, "opencv_rescue_planner"
                    rescue_score = float(rescue_best_payload.get("score", 0.0)) if rescue_best_payload is not None else 0.0
                    rescue_row["rescue_score"] = rescue_score
                    rescue_basic = None
                    if rescue_best_snapshot is not None:
                        restore(rescue_best_snapshot)
                        bbox = rescue_spec.bbox()
                        rescue_basic = _validate_result(runtime, {
                            "overall_dimensions_mm": {"x": float(bbox[0]), "y": float(bbox[1]), "z": float(bbox[2])}
                        })
                    rescue_row["brep_validation"] = rescue_basic
                    required_score = max(threshold, baseline_score + OPENCV_RESCUE_MIN_DELTA)
                    if (
                        rescue_best_snapshot is not None
                        and rescue_best_payload is not None
                        and rescue_score + 1e-9 >= required_score
                        and rescue_basic is not None
                        and rescue_basic.get("solid_present")
                        and rescue_basic.get("brep_valid")
                    ):
                        best_payload, best_snapshot, best_label = rescue_best_payload, rescue_best_snapshot, rescue_best_label
                        partspec = rescue_spec
                        rescue_row["accepted_for_modeling"] = True
                        rescue_row["required_score"] = required_score
                    else:
                        rescue_row["reason"] = (
                            f"Rescue candidate did not clear objective gate: score={rescue_score:.3f}, required={required_score:.3f}."
                        )
                except InterruptedError:
                    raise
                except Exception as rescue_exc:
                    rescue_row["reason"] = str(rescue_exc)
                opencv_rescue = rescue_row

            if best_snapshot is None or best_payload is None:
                restore(initial_snapshot)
                raise RuntimeError("Drawing reconstruction failed in all native modeling routes.")

            restore(best_snapshot)
            current_validation = best_payload
            if opencv_rescue and opencv_rescue.get("trigger") == "partspec_extraction_failed" and opencv_rescue.get("accepted_for_modeling") is None:
                recovered_score = float(best_payload.get("score", 0.0))
                opencv_rescue["rescue_score"] = recovered_score
                opencv_rescue["accepted_for_modeling"] = recovered_score + 1e-9 >= threshold
                if not opencv_rescue["accepted_for_modeling"]:
                    opencv_rescue["reason"] = f"Recovered PartSpec did not clear objective threshold ({recovered_score:.3f} < {threshold:.3f})."

            # ----------------------------------------------------- cad-image-to-3d adaptive protocol
            # Upstream ideas retained: one-shot candidate, render-back comparison,
            # conservative self-selection, feature completeness audit. Production
            # budget is adaptive: clean one-shot => zero extra VLM review turns.
            refinements: list[dict[str, Any]] = []
            cadimage_protocol: dict[str, Any] | None = None
            opencv_vision = None
            if job.vision_mode == "opencv5":
                protocol_dir = report_dir / "cadimage_protocol"
                source_paths = {label: Path(raw).resolve() for label, raw in job.input_files.items() if Path(raw).is_file()}
                candidates: list[dict[str, Any]] = []

                def capture_candidate(stage: str, *, action: str, issues: list[str] | None = None) -> dict[str, Any]:
                    stage_dir = protocol_dir / stage
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    rendered = _render_validation_views(runtime, stage_dir, pass_index=len(candidates))
                    verification: dict[str, Any] = {}
                    if rendered and source_paths:
                        try:
                            verification = opencv_verify_source_to_renders(
                                source_paths, rendered, output_dir=stage_dir / "compare", source_kind="drawing"
                            )
                        except Exception as exc:
                            verification = {"score": 0.0, "confidence": 0.0, "action": "ambiguous", "issues": [str(exc)], "pairs": []}
                    row = {
                        "stage": stage, "action": action, "issues": list(issues or []),
                        "snapshot": snapshot(), "validation": current_validation,
                        "rendered": rendered, "verification": verification,
                        "visual_score": cadimage_visual_score(verification),
                        "circle_error": cadimage_circle_pair_error(verification), "accepted": True,
                    }
                    candidates.append(row)
                    return row

                self._update(job, progress=max(job.progress, 82), message="OpenCV · fast render-back verification…")
                current_candidate = capture_candidate("exec0", action="one_shot")
                exec0_verification = current_candidate.get("verification") or {}
                native_score = float((current_validation or {}).get("score") or 0.0)
                needs_visual_refine = cadimage_needs_targeted_refinement(native_score, exec0_verification)

                if needs_visual_refine and CADIMAGE_MAX_VISUAL_LOOPS > 0 and current_candidate.get("rendered"):
                    self._check_cancel(job)
                    self._update(job, progress=max(job.progress, 88), message="Checking one targeted CAD refinement…")
                    review = agent.refine_from_views(
                        spec=partspec.model_dump(mode="json"), originals=image_paths,
                        rendered=current_candidate.get("rendered") or {},
                        current_state=planning_state(runtime.engine), pass_index=0,
                        opencv_feedback=exec0_verification,
                        on_event=event, cancel_check=job.cancel_event.is_set,
                    )
                    review_row = {k: v for k, v in review.items() if k != "calls"}
                    review_row.update({"pass": 1, "stage": "loop1"})
                    refinements.append(review_row)
                    if review.get("action") == "refine":
                        candidate_before_refine = snapshot()
                        validation_before = current_validation
                        refine_plan = {"calls": review.get("calls") or [], "note": review.get("note") or "targeted visual refinement"}
                        try:
                            _validate_plan(refine_plan)
                            with runtime.operation_lock:
                                refined = execute_with_failure_recovery(
                                    executor=runtime.executor, agent=agent,
                                    user_prompt=request_text + " Targeted visual corrections: " + "; ".join(review.get("issues") or []),
                                    state_provider=lambda: planning_state(runtime.engine), plan=refine_plan,
                                    on_event=event, initial_progress_span=(89, 93), max_ai_repairs=0,
                                )
                            if not refined.ok:
                                raise RuntimeError(str(refined.error or "targeted refinement execution failed"))
                            after_payload = validate_runtime_payload(runtime, partspec)
                            if float(after_payload.get("score", 0.0)) + 1e-9 < float(validation_before.get("score", 0.0)):
                                raise RuntimeError("targeted refinement regressed objective validation")
                            current_validation = after_payload
                            refinements[-1]["accepted"] = True
                            refinements[-1]["validation"] = after_payload
                            current_candidate = capture_candidate("loop1", action="refine", issues=review.get("issues") or [])
                        except Exception as exc:
                            restore(candidate_before_refine)
                            current_validation = validation_before
                            refinements[-1]["accepted"] = False
                            refinements[-1]["error"] = str(exc)
                    else:
                        refinements[-1]["accepted"] = True
                else:
                    refinements.append({
                        "pass": 0, "stage": "fast_path", "action": "skip_visual_refinement",
                        "reason": "one-shot native/OpenCV evidence did not require an AI edit",
                    })

                selected = cadimage_pick_best_candidate(candidates)
                if selected is not None and selected.get("snapshot") is not None:
                    restore(selected["snapshot"])
                    current_validation = selected.get("validation") or current_validation

                selected_rendered = (selected or {}).get("rendered") or {}
                selected_verification = (selected or {}).get("verification") or {}
                crosscheck = cadimage_audit_crosscheck(source_paths, selected_rendered) if selected_rendered else {}
                needs_audit = cadimage_needs_feature_audit(selected_verification)
                audit: dict[str, Any] = {
                    "action": "skipped_fast_path", "features": [], "issues": [],
                    "note": "deterministic evidence did not require a VLM feature audit",
                }
                if selected_rendered and needs_audit and CADIMAGE_MAX_AUDIT_REPAIR > 0:
                    self._check_cancel(job)
                    self._update(job, progress=max(job.progress, 95), message="Auditing unresolved drawing features…")
                    audit = agent.audit_features_from_views(
                        spec=partspec.model_dump(mode="json"), originals=image_paths, rendered=selected_rendered,
                        current_state=planning_state(runtime.engine), crosscheck=crosscheck,
                        verification=opencv_compact_verification(selected_verification) if selected_verification else {},
                        on_event=event, cancel_check=job.cancel_event.is_set,
                    )
                    if audit.get("action") == "refine":
                        before_audit = snapshot(); before_validation = current_validation
                        audit_plan = {"calls": audit.get("calls") or [], "note": audit.get("note") or "feature audit"}
                        try:
                            _validate_plan(audit_plan)
                            with runtime.operation_lock:
                                audit_exec = execute_with_failure_recovery(
                                    executor=runtime.executor, agent=agent,
                                    user_prompt=request_text + " Targeted feature-audit corrections: " + "; ".join(audit.get("issues") or []),
                                    state_provider=lambda: planning_state(runtime.engine), plan=audit_plan,
                                    on_event=event, initial_progress_span=(96, 98), max_ai_repairs=0,
                                )
                            if not audit_exec.ok:
                                raise RuntimeError(str(audit_exec.error or "audit execution failed"))
                            after_validation = validate_runtime_payload(runtime, partspec)
                            if float(after_validation.get("score", 0.0)) + 1e-9 < float(before_validation.get("score", 0.0)):
                                raise RuntimeError("feature audit regressed objective validation")
                            audit_dir = protocol_dir / "loop3"; audit_dir.mkdir(parents=True, exist_ok=True)
                            after_rendered = _render_validation_views(runtime, audit_dir, pass_index=2)
                            after_verification = opencv_verify_source_to_renders(
                                source_paths, after_rendered, output_dir=audit_dir / "compare", source_kind="drawing"
                            ) if after_rendered else {}
                            before_circle = cadimage_circle_pair_error(selected_verification)
                            after_circle = cadimage_circle_pair_error(after_verification)
                            before_visual = cadimage_visual_score(selected_verification)
                            after_visual = cadimage_visual_score(after_verification)
                            if not ((after_circle < before_circle) or (after_visual >= before_visual + 0.005)):
                                raise RuntimeError("feature-audit edit did not improve deterministic evidence")
                            current_validation = after_validation
                            audit.update({
                                "accepted": True, "before_visual_score": before_visual, "after_visual_score": after_visual,
                                "before_circle_error": before_circle, "after_circle_error": after_circle,
                            })
                            selected_rendered = after_rendered; selected_verification = after_verification
                        except Exception as exc:
                            restore(before_audit); current_validation = before_validation
                            audit["accepted"] = False; audit["error"] = str(exc)

                cadimage_protocol = {
                    "upstream_repo": CADIMAGE_UPSTREAM_REPO, "upstream_commit": CADIMAGE_UPSTREAM_COMMIT,
                    "protocol": CADIMAGE_UPSTREAM_PROTOCOL,
                    "cadia_adaptation": "single-pass OpenCV geometry assist + render-back + conservative self-selection + conditional targeted audit",
                    "fast_path": not needs_visual_refine and not needs_audit,
                    "visual_ai_budget": CADIMAGE_MAX_VISUAL_LOOPS, "audit_ai_budget": CADIMAGE_MAX_AUDIT_REPAIR,
                    "candidates": [cadimage_candidate_public(row) for row in candidates],
                    "selected_stage": ("loop3" if audit.get("accepted") else (selected or {}).get("stage")),
                    "feature_crosscheck": crosscheck,
                    "feature_audit": {k: v for k, v in audit.items() if k != "calls"},
                    "final_visual": opencv_compact_verification(selected_verification) if selected_verification else {},
                }
                _write_json(protocol_dir / "protocol.json", cadimage_protocol)
                opencv_vision = {
                    "mode": "opencv5", "status": "done", "read_only": False,
                    "action": ((cadimage_protocol.get("final_visual") or {}).get("action") or "review"),
                    "score": ((cadimage_protocol.get("final_visual") or {}).get("score") or 0.0),
                    "final": cadimage_protocol.get("final_visual") or {},
                    "issues": ((cadimage_protocol.get("final_visual") or {}).get("issues") or []),
                    "protocol": "cad-image-to-3d adaptive port",
                }

            # CADIA_NVIDIA_REVIEW_DRAWING_PRIMARY_V210
            # Independent post-generation engineering review. Drawing2CAD remains the
            # generator/source of geometric constraints; NVIDIA is allowed only to make
            # a bounded edit that preserves or improves the objective PartSpec score.
            def _nvidia_primary_validate() -> dict[str, Any]:
                payload = validate_runtime_payload(runtime, partspec)
                bbox = {
                    "overall_dimensions_mm": {
                        "x": float(partspec.bbox()[0]),
                        "y": float(partspec.bbox()[1]),
                        "z": float(partspec.bbox()[2]),
                    }
                }
                basic = _validate_result(runtime, bbox)
                payload["_cadia_brep_valid"] = bool(basic.get("solid_present") and basic.get("brep_valid"))
                return payload

            def _nvidia_primary_accept(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, str]:
                if not after.get("_cadia_brep_valid"):
                    return False, "Drawing repair did not preserve a valid CADia B-Rep."
                if float(after.get("score", 0.0)) + 1e-9 < float(before.get("score", 0.0)):
                    return False, "Drawing2CAD objective validation score regressed."
                return True, ""

            def _nvidia_primary_execute(plan: dict[str, Any], prompt: str) -> tuple[bool, str | None]:
                # Exactly ONE atomic executor attempt for the NVIDIA correction lane.
                # PlanExecutor.execute owns an internal snapshot/rollback; the outer
                # Engineering Review snapshot remains a second fail-safe.
                with runtime.operation_lock:
                    executed = runtime.executor.execute(plan, on_event=event, progress_span=(96, 97))
                return bool(executed.ok), None if executed.ok else str(executed.error or "NVIDIA-guided repair failed")

            if not DRAWING_NVIDIA_REVIEW_ENABLED or job.vision_mode == "opencv5":
                nvidia_review = {
                    "status": "disabled_for_product_pipeline",
                    "accepted": False,
                    "reason": "NVIDIA/Cosmos/Nemotron review is preserved in source but disabled on the product/OpenCV Drawing-to-CAD critical path.",
                }
            else:
                nvidia_review = run_engineering_review(
                    source_kind="engineering_drawing",
                    source_images=image_paths,
                    spec=partspec.model_dump(mode="json"),
                    validation_before=current_validation,
                    notes=job.notes,
                    dimension_hint=job.dimension_hint,
                    agent=agent,
                    current_state_provider=lambda: planning_state(runtime.engine),
                    render_before=lambda: render_review_views_isolated(
                        runtime, report_dir / "nvidia_review" / "before", pass_index=0
                    ),
                    render_after=lambda: render_review_views_isolated(
                        runtime, report_dir / "nvidia_review" / "after", pass_index=1
                    ),
                    snapshot=snapshot,
                    restore=restore,
                    validate_after=_nvidia_primary_validate,
                    validation_accept=_nvidia_primary_accept,
                    validate_plan=_validate_plan,
                    execute_plan=_nvidia_primary_execute,
                    on_event=event,
                    on_stage=lambda message, percent: self._update(
                        job, progress=max(job.progress, percent), message=message
                    ),
                    cancel_check=job.cancel_event.is_set,
                )
                _write_json(report_dir / "nvidia_review" / "review.json", nvidia_review)
                if nvidia_review.get("accepted") and isinstance(nvidia_review.get("validation_after"), dict):
                    current_validation = {
                        key: value for key, value in nvidia_review["validation_after"].items()
                        if not str(key).startswith("_cadia_")
                    }

            self._check_cancel(job)
            runtime.persist()
            basic_validation = _validate_result(
                runtime,
                {
                    "overall_dimensions_mm": {
                        "x": float(partspec.bbox()[0]),
                        "y": float(partspec.bbox()[1]),
                        "z": float(partspec.bbox()[2]),
                    }
                },
            )
            if not basic_validation.get("solid_present") or not basic_validation.get("brep_valid"):
                restore(initial_snapshot)
                raise RuntimeError("Drawing reconstruction did not produce a valid editable B-Rep.")
            state = runtime.state()
            report = {
                "job_id": job.id,
                "pipeline": "CADia Hotfix-5 + Drawing2CAD Native Boost",
                "provider": "codex",
                "model": job.model,
                "vision_mode": job.vision_mode,
                "route": best_label,
                "partspec": partspec.model_dump(mode="json"),
                "validation": current_validation,
                "brep_validation": basic_validation,
                "attempts": attempts,
                "refinements": refinements,
                "opencv_input_assist": opencv_input_assist if job.vision_mode == "opencv5" else None,
                "cadimage_protocol": cadimage_protocol,
                "opencv_vision": opencv_vision,
                "opencv_rescue": opencv_rescue,
                "nvidia_review": nvidia_review,
                "state_revision": (state.get("document") or {}).get("revision"),
            }
            report_path = report_dir / "report.json"
            _write_json(report_path, report)
            report["report"] = f"visual_cad/{job.id}/report.json"
            return {"state": state, **report}

        # ------------------------------------------------------------- exact Hotfix-5 fallback
        # PartSpec extraction itself failed.  Preserve Hotfix-5's original production
        # behavior instead of turning a new extractor failure into a regression.
        restore(initial_snapshot)
        self._check_cancel(job)
        self._update(job, progress=10, message="PartSpec extraction unavailable; using Hotfix-5 drawing analysis…")
        legacy_spec = agent.extract_spec(image_paths, notes=effective_notes, dimension_hint=job.dimension_hint, on_event=event)
        legacy_spec = _normalize_spec(legacy_spec)
        _validate_extraction(legacy_spec)
        _write_json(report_dir / "drawing_spec.json", legacy_spec)
        with runtime.operation_lock:
            plan = agent.plan_from_drawing(legacy_spec, image_paths, planning_state(runtime.engine), on_event=event)
            _validate_plan(plan)
            executed = execute_with_failure_recovery(
                executor=runtime.executor,
                agent=agent,
                user_prompt=agent.visual_request_text(legacy_spec, image_paths, effective_notes, job.dimension_hint),
                state_provider=lambda: planning_state(runtime.engine),
                plan=plan,
                on_event=event,
                initial_progress_span=(45, 76),
                max_ai_repairs=2,
            )
            if not executed.ok:
                raise RuntimeError("Hotfix-5 drawing reconstruction failed: " + str(executed.error or "unknown error"))
            runtime.persist()
            validation = _validate_result(runtime, legacy_spec)
            if not validation.get("solid_present") or not validation.get("brep_valid"):
                raise RuntimeError("Drawing reconstruction did not produce a valid editable B-Rep.")
            state = runtime.state()
        opencv_vision = None

        # CADIA_NVIDIA_REVIEW_DRAWING_FALLBACK_V210
        def _nvidia_fallback_accept(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, str]:
            if not after.get("solid_present") or not after.get("brep_valid"):
                return False, "Fallback drawing repair did not preserve a valid B-Rep."
            if float(after.get("score", 0.0)) + 1e-9 < float(before.get("score", 0.0)):
                return False, "Fallback drawing validation score regressed."
            return True, ""

        def _nvidia_fallback_execute(plan: dict[str, Any], prompt: str) -> tuple[bool, str | None]:
            # Exactly ONE atomic executor attempt for the NVIDIA correction lane.
            # PlanExecutor.execute owns an internal snapshot/rollback; the outer
            # Engineering Review snapshot remains a second fail-safe.
            with runtime.operation_lock:
                executed = runtime.executor.execute(plan, on_event=event, progress_span=(96, 97))
            return bool(executed.ok), None if executed.ok else str(executed.error or "NVIDIA-guided repair failed")

        if not DRAWING_NVIDIA_REVIEW_ENABLED or job.vision_mode == "opencv5":
            nvidia_review = {
                "status": "disabled_for_product_pipeline",
                "accepted": False,
                "reason": "NVIDIA/Cosmos/Nemotron review is preserved in source but disabled on the product/OpenCV Drawing-to-CAD critical path.",
            }
        else:
            nvidia_review = run_engineering_review(
                source_kind="engineering_drawing",
                source_images=image_paths,
                spec=legacy_spec,
                validation_before=validation,
                notes=job.notes,
                dimension_hint=job.dimension_hint,
                agent=agent,
                current_state_provider=lambda: planning_state(runtime.engine),
                render_before=lambda: render_review_views_isolated(
                    runtime, report_dir / "nvidia_review" / "before", pass_index=0
                ),
                render_after=lambda: render_review_views_isolated(
                    runtime, report_dir / "nvidia_review" / "after", pass_index=1
                ),
                snapshot=snapshot,
                restore=restore,
                validate_after=lambda: _validate_result(runtime, legacy_spec),
                validation_accept=_nvidia_fallback_accept,
                validate_plan=_validate_plan,
                execute_plan=_nvidia_fallback_execute,
                on_event=event,
                on_stage=lambda message, percent: self._update(
                    job, progress=max(job.progress, percent), message=message
                ),
                cancel_check=job.cancel_event.is_set,
            )
            _write_json(report_dir / "nvidia_review" / "review.json", nvidia_review)
            if nvidia_review.get("accepted") and isinstance(nvidia_review.get("validation_after"), dict):
                validation = nvidia_review["validation_after"]
        runtime.persist()
        state = runtime.state()

        report = {
            "job_id": job.id,
            "pipeline": "CADia Hotfix-5 fallback",
            "provider": "codex",
            "model": job.model,
            "vision_mode": job.vision_mode,
            "partspec_error": partspec_error,
            "spec": legacy_spec,
            "validation": validation,
            "refinements": [],
            "opencv_input_assist": opencv_input_assist if job.vision_mode == "opencv5" else None,
            "cadimage_protocol": None,
            "opencv_vision": opencv_vision,
            "opencv_rescue": opencv_rescue,
            "nvidia_review": nvidia_review,
            "state_revision": (state.get("document") or {}).get("revision"),
        }
        report_path = report_dir / "report.json"
        _write_json(report_path, report)
        report["report"] = f"visual_cad/{job.id}/report.json"
        return {"state": state, **report}


class VisualDrawingCodexAgent(DesktopParityCodexAgent):
    # Keep the repair catalog deliberately smaller than the 118-tool construction
    # catalog.  Refinement is allowed to edit/add local features, but it must not
    # start a replacement document, export files, manipulate assemblies, or invoke
    # unrelated read/admin operations.  The reduced schema materially shrinks the
    # repair prompt while preserving the feature-edit surface used by CADia.
    _REPAIR_TOOL_NAMES = {
        "inventor_set_parameter", "inventor_create_parameter",
        "inventor_create_sketch", "inventor_project_geometry",
        "inventor_draw_line", "inventor_draw_circle", "inventor_draw_rectangle", "inventor_draw_arc",
        "inventor_add_sketch_dimension", "inventor_add_sketch_constraint", "inventor_close_sketch",
        "inventor_extrude", "inventor_revolve", "inventor_fillet", "inventor_chamfer", "inventor_hole",
        "inventor_circular_pattern", "inventor_rectangular_pattern",
        "cad_get_topology", "cad_extrude_advanced", "cad_loft", "cad_sweep", "cad_create_slot",
        "cad_shell", "cad_mirror_body", "cad_transform_body", "cad_move_face", "cad_delete_face",
        "cad_set_face_diameter", "cad_edit_sketch_entity", "cad_delete_sketch_entity",
        "cad_edit_sketch_dimension", "cad_delete_sketch_dimension", "cad_edit_sketch_constraint",
        "cad_delete_sketch_constraint", "cad_edit_feature_definition", "cad_edit_feature",
        "cad_suppress_feature", "cad_delete_feature", "cad_edit_parameter", "cad_edit_sketch",
        "cad_edit_work_feature", "cad_edit_solid",
    }

    def _image_summary(self, images: dict[str, Path]) -> str:
        return ", ".join(images.keys())

    def _repair_catalog(self) -> list[dict[str, Any]]:
        return [row for row in self._catalog if str(row.get("name") or "") in self._REPAIR_TOOL_NAMES]

    def _stage_images_for_codex(self, images: dict[str, Path], *, purpose: str) -> list[Path]:
        """Return image paths guaranteed to live under the isolated Codex workspace.

        Original uploads are already staged there.  CAD renders are produced under the
        project report directory, so standard visual refinement must copy those renders
        before app-server localImage attachment validation.
        """
        workspace = Path(self._web_client.workspace).resolve()
        safe_target = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in self.target_id)[-96:] or "visual-cad"
        safe_purpose = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in purpose)[-64:] or "repair"
        stage_dir = workspace / "visual-cad-staged" / safe_target / safe_purpose
        stage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        staged: list[Path] = []
        for index, (label, raw_path) in enumerate(images.items()):
            src = Path(raw_path).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Visual repair image does not exist: {src}")
            try:
                src.relative_to(workspace)
                staged.append(src)
                continue
            except ValueError:
                pass
            suffix = src.suffix.lower() if src.suffix else ".png"
            safe_label = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(label))[:48] or f"image_{index}"
            dst = stage_dir / f"{index:02d}_{safe_label}{suffix}"
            shutil.copy2(src, dst)
            os.chmod(dst, 0o600)
            staged.append(dst.resolve())
        return staged

    def extract_spec(
        self,
        images: dict[str, Path],
        *,
        notes: str,
        dimension_hint: str,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""You are the structured extraction stage of an engineering Drawing-to-CAD system.
The attached local images are the actual drawing inputs ({self._image_summary(images)}). Inspect the attached images directly; do not call view_image, shell tools, Python, or filesystem tools.
If one sheet contains multiple orthographic views, inspect every view on that sheet.

Interpret the images as mechanical engineering drawings, not artistic references.
Coordinate convention for the CAD model:
- front-view horizontal = +X
- front-view vertical = +Z
- top-view vertical/depth = +Y
- right-view horizontal/depth = +Y and vertical = +Z
Use millimetres. Read written dimensions literally when legible. Never invent a precise labelled dimension from pixel size.
If a required dimension is absent, use null and record the ambiguity.
Capture holes, slots, pockets, bosses, counterbores/countersinks, fillets/chamfers, repeated patterns, axes/centrelines and symmetry when visible.
Every feature object must use the provided schema. Put human-readable detail in description and name/value pairs in dimensions.

User notes: {notes or '(none)'}
Known dimension/reference hint: {dimension_hint or '(none)'}

Return only schema-conforming JSON.
"""
        return self._web_client.complete_json(
            prompt,
            output_schema=DRAWING_SPEC_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            local_images=list(images.values()),
        )

    def visual_request_text(self, spec: dict[str, Any], images: dict[str, Path], notes: str, dimension_hint: str) -> str:
        return (
            "Reconstruct a new editable parametric part from the supplied engineering drawing. "
            "Preserve feature intent and all explicit dimensions. "
            f"Drawing spec: {json.dumps(spec, ensure_ascii=False)}. "
            f"User notes: {notes or '(none)'}. Dimension hint: {dimension_hint or '(none)'}. "
            f"Images: {', '.join(str(p) for p in images.values())}."
        )

    def plan_from_drawing(
        self,
        spec: dict[str, Any],
        images: dict[str, Path],
        current_state: dict[str, Any],
        *,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> dict[str, Any]:
        catalog = json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":"))
        prompt = f"""You are CADia's Drawing-to-CAD planning engine.
The attached drawing has already been inspected and normalized into DRAWING_SPEC. Do not use image/file/shell tools in this stage.

A structured extraction was produced first:
DRAWING_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}

CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
TOOL_CATALOG={catalog}

Create a NEW editable part from the drawing using only TOOL_CATALOG operations. Never emit Python, shell, CadQuery source, eval/exec, file operations, network calls, or prose.
The new model must be feature-based and editable. Prefer a compact engineering history: sketches + extrude/revolve, then holes/cuts, patterns, fillets/chamfers. Prefer native/Inventor-style operations and deterministic CADia tools over long chains of primitive booleans.
Use explicit drawing dimensions exactly. Do not estimate a labelled dimension from pixels. For omitted dimensions, choose a minimal symmetric/standard assumption and keep it easy to edit.
If CURRENT_STATE already has a document or geometry, start with inventor_new_part so the existing model is not overwritten.
Map orthographic views using X/Y/Z convention stated in the extraction stage.
For each call, encode the complete tool arguments object as a JSON string in arguments_json.
Use bind=null unless a later call needs the result.
Return only schema-conforming JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=VISUAL_PLAN_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
        )
        return _decode_visual_plan(raw, allow_empty=False)

    def partspec_request_text(self, spec, images: dict[str, Path], notes: str, dimension_hint: str) -> str:
        return (
            "Reconstruct a new editable CADia-native part from the Drawing2CAD PartSpec. "
            "Use only typed CAD tools; do not emit or execute Python and do not flatten through STEP. "
            f"PARTSPEC={spec.model_dump_json()}. "
            f"User notes: {notes or '(none)'}. Dimension hint: {dimension_hint or '(none)'}. "
            f"Images: {', '.join(str(p) for p in images.values())}."
        )

    def plan_from_partspec(
        self,
        spec,
        images: dict[str, Path],
        current_state: dict[str, Any],
        *,
        validator_feedback: str = "",
        on_event: Callable[[str, Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        catalog = json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":"))
        feedback = validator_feedback.strip() or "(none)"
        prompt = f"""You are CADia's native Drawing-to-CAD planner.
The drawing was already converted into the exact public Drawing2CAD PartSpec below.
PARTSPEC={spec.model_dump_json()}
VALIDATOR_FEEDBACK={feedback}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
TOOL_CATALOG={catalog}

Build a NEW editable CADia part using only TOOL_CATALOG operations. The PartSpec is the
primary geometric contract. Preserve its explicit dimensions, hole types/count/locations,
cuts, body placement and modeling intent. Start with inventor_new_part. Prefer native CADia
features (sketch/extrude/revolve/hole/pattern and deterministic cad_* primitives) so feature
history remains editable. Never emit Python, CadQuery source, eval/exec, filesystem calls,
network calls, or STEP-import workarounds. If VALIDATOR_FEEDBACK lists a mismatch, rebuild
the plan to correct that mismatch rather than accumulating compensating geometry.
Use the attached original drawing only to resolve topology that PartSpec does not encode
explicitly (for example which edges receive a listed fillet/chamfer); never override a
legible explicit PartSpec dimension from perspective appearance.
For each call encode the complete tool arguments object as JSON text in arguments_json.
Use bind=null unless a later call needs the result. Return only schema-conforming JSON.
"""
        raw = self._web_client.complete_json(
            prompt,
            output_schema=VISUAL_PLAN_SCHEMA,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            local_images=list(images.values()),
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        return _decode_visual_plan(raw, allow_empty=False)

    def refine_from_views(
        self,
        *,
        spec: dict[str, Any],
        originals: dict[str, Path],
        rendered: dict[str, Path],
        current_state: dict[str, Any],
        pass_index: int,
        opencv_feedback: dict[str, Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        all_images = {**{f"original_{k}": v for k, v in originals.items()}, **{f"cad_{k}": v for k, v in rendered.items()}}
        repair_catalog = self._repair_catalog()

        if opencv_feedback is not None:
            # Fast OpenCV path: the source and CAD renders have already been measured
            # deterministically.  Do not pay for a second multimodal inspection turn,
            # and do not attach project-report renders that live outside the Codex
            # workspace.  The model only maps objective discrepancy evidence back to
            # the existing editable feature history.
            evidence = opencv_verification_prompt(
                opencv_feedback, max_chars=OPENCV_REPAIR_FEEDBACK_MAX_CHARS
            )
            prompt = f"""You are the bounded feature-repair planner of an engineering Drawing-to-CAD pipeline.
No images are attached in this pass. OpenCV 5 has already compared the original drawing with the CAD renders and supplied deterministic discrepancy evidence below. Do not request or inspect image/filesystem tools.

This is repair of the CURRENT editable model, not a redesign.
DRAWING_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
OPENCV_5_GEOMETRIC_EVIDENCE={evidence}
REPAIR_TOOL_CATALOG={json.dumps(repair_catalog, ensure_ascii=False, separators=(',', ':'))}

Use explicit dimensions in DRAWING_SPEC as the metric source of truth. OpenCV pixel/normalized deltas identify WHICH existing feature is wrong; never convert those deltas directly to millimetres unless the drawing spec establishes that value. Correct only discrepancies supported by the evidence/spec. Preserve correct geometry and feature history.
If the evidence cannot be mapped safely to an engineering edit, action must be "accept" and calls must be [].
If a safe correction is clear, action must be "refine" and every call must use only REPAIR_TOOL_CATALOG. Never create a replacement document.
Pass number: {pass_index + 1}
Encode every complete arguments object as JSON text in arguments_json; use bind=null unless required.
Return only schema-conforming JSON.
"""
            local_images = None
            repair_effort = OPENCV_REPAIR_EFFORT
        else:
            # Standard (non-OpenCV) visual refinement still needs the images.  CAD
            # renders are generated under the project report root, so stage them into
            # the isolated app-server workspace before attaching them as localImage.
            prompt = f"""You are the visual verification/refinement stage of an engineering Drawing-to-CAD pipeline.
The attached local images contain both original drawing views and CAD render views ({self._image_summary(all_images)}). Inspect the attachments directly; do not call view_image, shell, Python, or filesystem tools.

Compare the reconstructed CAD renders to the original drawing views. This is verification, not a fresh redesign.
DRAWING_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
TOOL_CATALOG={json.dumps(repair_catalog, ensure_ascii=False, separators=(',', ':'))}

Check silhouette, relative proportions, hole/slot count and placement, symmetry, visible pockets/bosses, and explicit dimensions from the spec. Do not change a correct explicit dimension merely to improve perspective similarity. Do not add cosmetic details unsupported by the drawing.
If the model is already consistent enough, action must be "accept" and calls must be [].
If there is a clear engineering mismatch that can be corrected safely, action must be "refine" and calls may contain only TOOL_CATALOG operations that EDIT the current model. Do not create a new replacement part during visual refinement.
Pass number: {pass_index + 1}
For every refinement call, encode the complete tool arguments object as JSON text in arguments_json and use bind=null unless needed.
If action is accept, calls must be an empty array.
Return only schema-conforming JSON.
"""
            local_images = self._stage_images_for_codex(all_images, purpose=f"refine-{pass_index + 1}")
            repair_effort = self.reasoning

        raw = self._web_client.complete_json(
            prompt,
            output_schema=REFINEMENT_SCHEMA,
            model=self.model,
            effort=repair_effort,
            on_event=on_event,
            local_images=local_images,
            cancel_check=cancel_check,
            stall_timeout_seconds=180,
            hard_timeout_seconds=900,
        )
        return _decode_refinement(raw)


    def audit_features_from_views(
        self,
        *,
        spec: dict[str, Any],
        originals: dict[str, Path],
        rendered: dict[str, Path],
        current_state: dict[str, Any],
        crosscheck: dict[str, Any],
        verification: dict[str, Any],
        on_event: Callable[[str, Any], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        all_images = {**{f"original_{k}": v for k, v in originals.items()}, **{f"cad_{k}": v for k, v in rendered.items()}}
        repair_catalog = self._repair_catalog()
        prompt = f"""You are the final feature-audit stage of CADia's engineering Drawing-to-CAD pipeline.
Inspect the attached original drawing and CAD projections directly. This is a targeted completeness audit, not a redesign.

DRAWING_PARTSPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}
CURRENT_STATE={json.dumps(current_state, ensure_ascii=False, separators=(',', ':'))}
DETERMINISTIC_CIRCULAR_MARK_CROSSCHECK={json.dumps(crosscheck, ensure_ascii=False, separators=(',', ':'))}
DETERMINISTIC_RENDER_COMPARISON={json.dumps(verification, ensure_ascii=False, separators=(',', ':'))}
REPAIR_TOOL_CATALOG={json.dumps(repair_catalog, ensure_ascii=False, separators=(',', ':'))}

Inventory physical features visible in the drawing: through/blind holes, counterbores/countersinks, slots, notches, bosses, pockets, steps, teeth, repeated patterns, fillets and chamfers. Compare counts/patterns/placement with the current CAD.
OpenCV ring/dot counts are only a cross-check: rings may also be bosses, counterbores, center marks, text or annotations. Never infer semantics or millimetres from pixels alone. Written dimensions and PartSpec are authoritative.
Ignore title blocks, text, leaders, dimension/centre lines, section arrows and surrounding context.
If physical features are complete, action=accept and calls=[].
If a clearly supported feature is missing/wrong, action=refine and return only REPAIR_TOOL_CATALOG operations that edit the CURRENT model. Do not create a replacement document and do not alter correct explicit dimensions.
Return only schema-conforming JSON.
"""
        staged = self._stage_images_for_codex(all_images, purpose="feature-audit")
        raw = self._web_client.complete_json(
            prompt, output_schema=FEATURE_AUDIT_SCHEMA, model=self.model, effort=OPENCV_REPAIR_EFFORT,
            on_event=on_event, local_images=staged, cancel_check=cancel_check,
            stall_timeout_seconds=180, hard_timeout_seconds=900,
        )
        if not isinstance(raw, dict):
            raise ValueError("Feature audit response is not an object.")
        action = str(raw.get("action") or "accept").lower()
        if action not in {"accept", "refine"}: action = "accept"
        calls = _decode_call_rows(raw.get("calls") or [], allow_empty=True)
        if action == "accept": calls = []
        if action == "refine" and not calls: action = "accept"
        return {
            "action": action, "confidence": _finite_float(raw.get("confidence"), 0.0),
            "features": [x for x in (raw.get("features") or []) if isinstance(x, dict)][:80],
            "issues": [str(x) for x in (raw.get("issues") or [])][:30],
            "calls": calls, "note": str(raw.get("note") or ""),
        }


visual_cad_jobs = VisualCadJobManager()


def _normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Drawing extraction did not return an object.")
    dims = raw.get("overall_dimensions_mm") if isinstance(raw.get("overall_dimensions_mm"), dict) else {}
    normalized = {
        "part_name": str(raw.get("part_name") or "Drawing Part")[:120],
        "units": "mm",
        "views_detected": [str(x)[:40] for x in (raw.get("views_detected") or [])][:12],
        "overall_dimensions_mm": {
            "x": _positive_or_none(dims.get("x")),
            "y": _positive_or_none(dims.get("y")),
            "z": _positive_or_none(dims.get("z")),
        },
        "features": [x for x in (raw.get("features") or []) if isinstance(x, dict)][:80],
        "symmetry": [str(x)[:300] for x in (raw.get("symmetry") or [])][:20],
        "assumptions": [str(x)[:500] for x in (raw.get("assumptions") or [])][:30],
        "uncertainties": [str(x)[:500] for x in (raw.get("uncertainties") or [])][:30],
        "confidence": max(0.0, min(1.0, _finite_float(raw.get("confidence"), 0.0))),
    }
    return normalized


def _decode_call_rows(rows: Any, *, allow_empty: bool) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("CAD plan calls are not an array.")
    if not rows and not allow_empty:
        raise ValueError("CAD plan contains no tool calls.")
    decoded: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"CAD plan call {index} is invalid.")
        raw_args = row.get("arguments_json")
        if not isinstance(raw_args, str):
            raise ValueError(f"CAD plan call {index} arguments_json is missing.")
        try:
            arguments = json.loads(raw_args)
        except Exception as exc:
            raise ValueError(f"CAD plan call {index} arguments_json is not valid JSON.") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"CAD plan call {index} arguments must decode to an object.")
        call: dict[str, Any] = {"tool": str(row.get("tool") or ""), "arguments": arguments}
        bind = row.get("bind")
        if isinstance(bind, str) and bind.strip():
            call["bind"] = bind.strip()
        decoded.append(call)
    return decoded


def _decode_visual_plan(raw: dict[str, Any], *, allow_empty: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("CAD plan is not an object.")
    return {
        "calls": _decode_call_rows(raw.get("calls"), allow_empty=allow_empty),
        "note": str(raw.get("note") or ""),
    }


def _decode_refinement(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Visual refinement response is not an object.")
    action = str(raw.get("action") or "accept").lower()
    if action not in {"accept", "refine"}:
        raise ValueError("Visual refinement action is invalid.")
    calls = _decode_call_rows(raw.get("calls"), allow_empty=(action == "accept"))
    if action == "accept" and calls:
        calls = []
    if action == "refine" and not calls:
        raise ValueError("Visual refinement requested refine but returned no tool calls.")
    return {
        "action": action,
        "confidence": _finite_float(raw.get("confidence"), 0.0),
        "issues": [str(x) for x in (raw.get("issues") or [])][:20],
        "calls": calls,
        "note": str(raw.get("note") or ""),
    }


def _validate_extraction(spec: dict[str, Any]) -> None:
    dims = spec.get("overall_dimensions_mm") or {}
    has_dimension = any(_positive_or_none(dims.get(axis)) is not None for axis in ("x", "y", "z"))
    has_features = bool(spec.get("features"))
    has_views = bool(spec.get("views_detected"))
    confidence = _finite_float(spec.get("confidence"), 0.0)
    if confidence <= 0.0 or not has_views or (not has_dimension and not has_features):
        raise RuntimeError(
            "Drawing image analysis failed: no reliable geometry/dimensions were extracted from the attached image. "
            "The CAD model was not generated."
        )


def _positive_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def _validate_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("CAD plan is not an object.")
    calls = plan.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("CAD plan contains no tool calls.")
    if len(calls) > 80:
        raise ValueError("CAD plan is too large.")
    allowed = {row["name"] for row in desktop_catalog()}
    for index, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            raise ValueError(f"CAD plan call {index} is invalid.")
        tool = str(call.get("tool") or "")
        if tool not in allowed:
            raise ValueError(f"CAD plan attempted a non-CAD/unknown tool: {tool}")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError(f"CAD plan call {index} arguments must be an object.")


def _render_validation_views(runtime: CadRuntime, job_dir: Path, *, pass_index: int) -> dict[str, Path]:
    """Render validation views outside the long-lived API process.

    CADia already ships a disposable VTK renderer for engineering review.  Reuse
    it here so repeated Drawing-to-CAD render-back passes cannot accumulate VTK/
    OpenGL native allocations in the FastAPI worker.  The helper exports one STL,
    renders all four views in a child process, finalizes VTK there, and is fail-open.
    """
    return render_review_views_isolated(runtime, job_dir, pass_index=pass_index)


def _validate_result(runtime: CadRuntime, spec: dict[str, Any]) -> dict[str, Any]:
    with runtime.engine.lock:
        doc = runtime.engine.doc
        shape = doc.shape if doc is not None else None
        if shape is None:
            return {"solid_present": False, "brep_valid": False, "dimension_checks": [], "score": 0.0}
        try:
            brep_valid = bool(shape.isValid())
        except Exception:
            brep_valid = True
        try:
            box = shape.BoundingBox()
            actual = {"x": float(box.xlen), "y": float(box.ylen), "z": float(box.zlen)}
        except Exception:
            actual = {"x": None, "y": None, "z": None}

    checks: list[dict[str, Any]] = []
    expected = spec.get("overall_dimensions_mm") or {}
    for axis in ("x", "y", "z"):
        exp = _positive_or_none(expected.get(axis))
        act = _positive_or_none(actual.get(axis))
        if exp is None or act is None:
            continue
        rel = abs(act - exp) / max(exp, 1e-9)
        checks.append({
            "axis": axis,
            "expected_mm": round(exp, 5),
            "actual_mm": round(act, 5),
            "relative_error": round(rel, 6),
            "pass": rel <= 0.025,
        })
    dimension_score = 1.0 if not checks else sum(1.0 if row["pass"] else max(0.0, 1.0 - row["relative_error"]) for row in checks) / len(checks)
    score = (0.55 if brep_valid else 0.0) + 0.45 * dimension_score
    return {
        "solid_present": True,
        "brep_valid": brep_valid,
        "bounding_box_mm": actual,
        "dimension_checks": checks,
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
