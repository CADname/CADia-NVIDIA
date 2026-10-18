from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import settings


class HeavyJobBusy(RuntimeError):
    pass


@dataclass
class _Reservation:
    token: str
    owner_id: str
    project_id: str
    kind: str
    cancel_event: threading.Event
    acquired: bool = False
    process: subprocess.Popen[str] | None = None
    created_at: float = 0.0
    started_at: float | None = None


class HeavyJobManager:
    """Process-slot coordinator for expensive CAD work.

    The coordinator lives only in the API process.  Expensive CAD work itself is
    executed in disposable child processes; this object only owns reservations,
    bounded concurrency, cancellation, and process-group teardown.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(settings.heavy_cad_workers)
        self._items: dict[str, _Reservation] = {}
        self._stopping = False

    def start(self) -> None:
        with self._lock:
            self._stopping = False

    def reserve(
        self,
        token: str,
        owner_id: str,
        project_id: str,
        kind: str,
        cancel_event: threading.Event,
    ) -> None:
        owner_id = str(owner_id)
        project_id = str(project_id)
        with self._lock:
            if self._stopping:
                raise RuntimeError("CAD worker pool is shutting down.")
            if token in self._items:
                raise RuntimeError(f"Duplicate CAD job token: {token}")
            for item in self._items.values():
                if item.owner_id == owner_id and item.project_id == project_id:
                    raise HeavyJobBusy("A modeling task is already queued or running in this project.")
            self._items[token] = _Reservation(
                token=token,
                owner_id=owner_id,
                project_id=project_id,
                kind=str(kind),
                cancel_event=cancel_event,
                created_at=time.time(),
            )

    def acquire(self, token: str) -> bool:
        """Wait for a worker slot. Returns False if cancelled before start."""
        deadline = time.monotonic() + settings.heavy_queue_timeout_seconds
        while True:
            with self._lock:
                item = self._items.get(token)
                if item is None:
                    return False
                if self._stopping or item.cancel_event.is_set():
                    return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The CAD job waited too long for an available worker slot.")
            if not self._slots.acquire(timeout=min(0.5, remaining)):
                continue
            with self._lock:
                item = self._items.get(token)
                if item is None or self._stopping or item.cancel_event.is_set():
                    self._slots.release()
                    return False
                item.acquired = True
                item.started_at = time.time()
                return True

    def attach_process(self, token: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            item = self._items.get(token)
            if item is None:
                raise RuntimeError("CAD job reservation disappeared before worker start.")
            item.process = process
            cancelled = self._stopping or item.cancel_event.is_set()
        if cancelled:
            self._terminate_process(process)

    def detach_process(self, token: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            item = self._items.get(token)
            if item is not None and item.process is process:
                item.process = None

    def finish(self, token: str) -> None:
        release = False
        with self._lock:
            item = self._items.pop(token, None)
            if item is not None and item.acquired:
                release = True
        if release:
            try:
                self._slots.release()
            except ValueError:
                pass

    def cancel_token(self, token: str) -> None:
        process = None
        with self._lock:
            item = self._items.get(token)
            if item is not None:
                item.cancel_event.set()
                process = item.process
        if process is not None:
            self._terminate_process(process)

    def cancel_project(self, owner_id: str, project_id: str, *, wait_timeout: float = 20.0) -> bool:
        owner_id, project_id = str(owner_id), str(project_id)
        with self._lock:
            tokens = [
                token for token, item in self._items.items()
                if item.owner_id == owner_id and item.project_id == project_id
            ]
        for token in tokens:
            self.cancel_token(token)
        return self._wait_absent(
            lambda item: item.owner_id == owner_id and item.project_id == project_id,
            wait_timeout,
        )

    def cancel_user(self, owner_id: str, *, wait_timeout: float = 25.0) -> bool:
        owner_id = str(owner_id)
        with self._lock:
            tokens = [token for token, item in self._items.items() if item.owner_id == owner_id]
        for token in tokens:
            self.cancel_token(token)
        return self._wait_absent(lambda item: item.owner_id == owner_id, wait_timeout)

    def project_reserved(self, owner_id: str, project_id: str) -> bool:
        owner_id, project_id = str(owner_id), str(project_id)
        with self._lock:
            return any(
                item.owner_id == owner_id and item.project_id == project_id
                for item in self._items.values()
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "token": item.token,
                    "owner_id": item.owner_id,
                    "project_id": item.project_id,
                    "kind": item.kind,
                    "state": "running" if item.acquired else "queued",
                    "pid": item.process.pid if item.process is not None and item.process.poll() is None else None,
                    "created_at": item.created_at,
                    "started_at": item.started_at,
                }
                for item in self._items.values()
            ]

    def shutdown(self, *, wait_timeout: float = 20.0) -> None:
        with self._lock:
            self._stopping = True
            tokens = list(self._items)
        for token in tokens:
            self.cancel_token(token)
        self._wait_absent(lambda _item: True, wait_timeout)

    def _wait_absent(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                if not any(predicate(item) for item in self._items.values()):
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except Exception:
                pass
        deadline = time.monotonic() + settings.heavy_cancel_grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.kill()
                except Exception:
                    pass


heavy_job_manager = HeavyJobManager()
