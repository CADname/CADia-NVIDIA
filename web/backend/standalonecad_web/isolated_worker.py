from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
from typing import Any

_PREFIX = "CADIA_WORKER\t"


def emit(packet: dict[str, Any]) -> None:
    sys.stdout.write(_PREFIX + json.dumps(packet, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def event(kind: str, value: Any) -> None:
    emit({"type": "event", "kind": kind, "value": value})


def _prefer_worker_for_oom() -> None:
    # Best effort: if the host enters global OOM, sacrifice this disposable worker
    # before the long-lived API process. Unprivileged processes may raise the score.
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write("800")
    except Exception:
        pass


def _visual(payload: dict[str, Any], hard_timeout: float) -> dict[str, Any]:
    from .visual_cad import VisualCadJob, VisualCadJobManager

    class Manager(VisualCadJobManager):
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._jobs = {}

        def _update(self, job, *, status=None, progress=None, message=None):
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, max(0, min(100, int(progress))))
            if message is not None:
                job.message = message
            job.updated_at = time.time()
            event("progress", {"message": job.message, "percent": job.progress})

    row = payload["job"]
    job = VisualCadJob(
        id=str(row["id"]), user_id=str(row["user_id"]), project_id=str(row["project_id"]),
        input_files={str(k): str(v) for k, v in dict(row.get("input_files") or {}).items()},
        notes=str(row.get("notes") or ""), dimension_hint=str(row.get("dimension_hint") or ""),
        model=str(row.get("model") or "gpt-5.6-sol"), effort=str(row.get("effort") or "high"),
        vision_mode=str(row.get("vision_mode") or "standard"), status="running",
        deadline_at=time.time() + hard_timeout,
    )
    manager = Manager()
    return manager._reconstruct(job)


def _photo(payload: dict[str, Any], hard_timeout: float) -> dict[str, Any]:
    from .photo_cad import PhotoCadJob, PhotoCadJobManager

    class Manager(PhotoCadJobManager):
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._jobs = {}

        def _update(self, job, *, status=None, progress=None, message=None):
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, max(0, min(100, int(progress))))
            if message is not None:
                job.message = message
            job.updated_at = time.time()
            event("progress", {"message": job.message, "percent": job.progress})

    row = payload["job"]
    job = PhotoCadJob(
        id=str(row["id"]), user_id=str(row["user_id"]), project_id=str(row["project_id"]),
        input_files={str(k): str(v) for k, v in dict(row.get("input_files") or {}).items()},
        notes=str(row.get("notes") or ""), dimension_hint=str(row.get("dimension_hint") or ""),
        model=str(row.get("model") or "gpt-5.6-sol"), effort=str(row.get("effort") or "high"),
        vision_mode=str(row.get("vision_mode") or "standard"), status="running",
        deadline_at=time.time() + hard_timeout,
    )
    manager = Manager()
    return manager._reconstruct(job)


def _prompt(payload: dict[str, Any], _hard_timeout: float) -> dict[str, Any]:
    from .routes.ai import _execute_prompt_body
    from .schemas import PromptRequest

    request = PromptRequest.model_validate(payload["request"])
    return _execute_prompt_body(
        str(payload["user_id"]), str(payload["project_id"]), request, on_event=event,
    )



def _worker_user_id(request: dict[str, Any]) -> str:
    payload = dict(request.get("payload") or {})
    if request.get("kind") == "prompt":
        return str(payload.get("user_id") or "")
    job = dict(payload.get("job") or {})
    return str(job.get("user_id") or "")


def _prepare_codex_sandbox(request: dict[str, Any]):
    user_id = _worker_user_id(request)
    token = str(request.get("token") or f"worker-{os.getpid()}")
    if not user_id:
        return None
    from .config import settings
    safe = "".join(ch for ch in token if ch.isalnum() or ch in "-_.")[:120] or f"worker-{os.getpid()}"
    worker_root = settings.data_root / "worker-sandboxes" / safe
    # Share only the user's authenticated CODEX_HOME, just like two terminal
    # sessions using the same account. Give every modeling worker an independent
    # workspace so threads/files from separate CAD jobs never collide.
    shared_home = settings.data_root / "codex-users" / user_id / "codex-home"
    worker_workspace = worker_root / "agent-workspace"
    shared_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    worker_workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["CADIA_CODEX_HOME_OVERRIDE"] = str(shared_home)
    os.environ["CADIA_CODEX_WORKSPACE_OVERRIDE"] = str(worker_workspace)
    return worker_root

def main() -> int:
    _prefer_worker_for_oom()
    raw = sys.stdin.readline()
    if not raw:
        return 2
    request = json.loads(raw)
    worker_root = _prepare_codex_sandbox(request)
    kind = str(request.get("kind") or "")
    payload = dict(request.get("payload") or {})
    hard_timeout = float(request.get("hard_timeout_seconds") or 7200)
    try:
        if kind == "prompt":
            result = _prompt(payload, hard_timeout)
        elif kind == "visual":
            result = _visual(payload, hard_timeout)
        elif kind == "photo":
            result = _photo(payload, hard_timeout)
        else:
            raise ValueError(f"Unknown isolated CAD job kind: {kind}")
        emit({"type": "result", "value": result})
        return 0
    except BaseException as exc:
        emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        try:
            from .cad_runtime import runtime_manager
            runtime_manager.close_all()
        except Exception:
            pass
        try:
            from .providers.app_server import app_server_manager
            app_server_manager.close_all()
        except Exception:
            pass
        if worker_root is not None:
            try:
                shutil.rmtree(worker_root, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
