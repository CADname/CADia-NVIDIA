from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .heavy_jobs import heavy_job_manager

_PREFIX = "CADIA_WORKER\t"


def _json_default(value: Any):
    try:
        return vars(value)
    except Exception:
        return str(value)


def _rss_tree_kb(root_pid: int) -> int:
    """Best-effort RSS sum for one worker process and its descendants."""
    try:
        rows: dict[int, tuple[int, int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                status = (entry / "status").read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            ppid = 0
            rss = 0
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
            rows[pid] = (ppid, rss)
        wanted = {int(root_pid)}
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _rss) in rows.items():
                if pid not in wanted and ppid in wanted:
                    wanted.add(pid)
                    changed = True
        return sum(rows.get(pid, (0, 0))[1] for pid in wanted)
    except Exception:
        return 0


def run_isolated(
    *,
    token: str,
    kind: str,
    payload: dict[str, Any],
    cancel_event: threading.Event,
    hard_timeout_seconds: float,
    on_event: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CADIA_ISOLATED_WORKER"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "standalonecad_web.isolated_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    heavy_job_manager.attach_process(token, process)
    messages: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=80)

    def stdout_reader() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                messages.put(("stdout", line.rstrip("\n")))
        finally:
            messages.put(("stdout_eof", None))

    def stderr_reader() -> None:
        try:
            assert process.stderr is not None
            for line in process.stderr:
                stderr_tail.append(line.rstrip("\n"))
        finally:
            messages.put(("stderr_eof", None))

    threading.Thread(target=stdout_reader, name=f"cadia-worker-out-{process.pid}", daemon=True).start()
    threading.Thread(target=stderr_reader, name=f"cadia-worker-err-{process.pid}", daemon=True).start()

    request = {"token": token, "kind": kind, "payload": payload, "hard_timeout_seconds": float(hard_timeout_seconds)}
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request, ensure_ascii=False, default=_json_default) + "\n")
        process.stdin.flush()
        process.stdin.close()
    except Exception:
        heavy_job_manager.cancel_token(token)
        heavy_job_manager.detach_process(token, process)
        raise

    started = time.monotonic()
    result: dict[str, Any] | None = None
    worker_error: str | None = None
    peak_tree_rss_kb = 0
    last_sample = 0.0

    try:
        while True:
            if cancel_event.is_set():
                heavy_job_manager.cancel_token(token)
                raise InterruptedError("The modeling task was canceled.")
            elapsed = time.monotonic() - started
            if elapsed >= hard_timeout_seconds:
                heavy_job_manager.cancel_token(token)
                raise TimeoutError(f"The isolated CAD worker exceeded {int(hard_timeout_seconds)} seconds.")
            now = time.monotonic()
            if now - last_sample >= 1.0 and process.poll() is None:
                last_sample = now
                peak_tree_rss_kb = max(peak_tree_rss_kb, _rss_tree_kb(process.pid))
            try:
                channel, line = messages.get(timeout=0.2)
            except queue.Empty:
                if process.poll() is not None:
                    # Drain any final lines that were already queued.
                    try:
                        while True:
                            channel, line = messages.get_nowait()
                            if channel == "stdout" and line:
                                result, worker_error = _handle_line(line, result, worker_error, on_event)
                    except queue.Empty:
                        break
                continue
            if channel == "stdout" and line:
                result, worker_error = _handle_line(line, result, worker_error, on_event)
            if process.poll() is not None and messages.empty():
                break

        return_code = process.wait(timeout=2)
        peak_tree_rss_kb = max(peak_tree_rss_kb, _rss_tree_kb(process.pid))
        if worker_error:
            raise RuntimeError(worker_error)
        if return_code != 0:
            suffix = "\n".join(stderr_tail)
            if return_code in {-9, 137}:
                hint = "The isolated CAD worker was killed (possible memory pressure/OOM)."
            else:
                hint = f"The isolated CAD worker exited with code {return_code}."
            raise RuntimeError((hint + ("\n" + suffix if suffix else "")).strip())
        if result is None:
            suffix = "\n".join(stderr_tail)
            raise RuntimeError(("The isolated CAD worker returned no result." + ("\n" + suffix if suffix else "")).strip())
        metrics = result.setdefault("_worker_metrics", {}) if isinstance(result, dict) else {}
        if isinstance(metrics, dict):
            metrics.setdefault("peak_process_tree_rss_mb", round(peak_tree_rss_kb / 1024.0, 1))
            metrics.setdefault("elapsed_seconds", round(time.monotonic() - started, 2))
            metrics.setdefault("worker_pid", process.pid)
        return result
    finally:
        if process.poll() is None:
            heavy_job_manager.cancel_token(token)
        heavy_job_manager.detach_process(token, process)


def _handle_line(
    line: str,
    result: dict[str, Any] | None,
    worker_error: str | None,
    on_event: Callable[[str, Any], None] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not line.startswith(_PREFIX):
        return result, worker_error
    try:
        packet = json.loads(line[len(_PREFIX):])
    except Exception:
        return result, worker_error
    packet_type = packet.get("type")
    if packet_type == "event":
        if on_event:
            on_event(str(packet.get("kind") or "progress"), packet.get("value"))
    elif packet_type == "result":
        value = packet.get("value")
        result = value if isinstance(value, dict) else {"value": value}
    elif packet_type == "error":
        worker_error = str(packet.get("error") or "Isolated CAD worker failed.")
    return result, worker_error
