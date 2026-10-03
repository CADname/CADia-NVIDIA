from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

from .engineering_verification import normalize_engineering_contract

def _strip_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    # Keep the same tolerant JSON-object extraction semantics used by CADia's planner,
    # but keep this sidecar independent from the CAD kernel so host-side verification
    # does not require CadQuery/OCP to be installed.
    s = _strip_fence(text)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("Nemotron did not return JSON")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(s[start:i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("Nemotron JSON must be an object")
                return obj
    raise ValueError("Nemotron returned incomplete JSON")


DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1"
DEFAULT_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _compact(value: Any, limit: int = 14000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class NemotronError(RuntimeError):
    pass


class NemotronOrchestrator:
    """Nebius Token Factory / NVIDIA Nemotron engineering-reasoning sidecar.

    It never executes CAD tools and never mutates the document. It extracts a source-
    grounded requirement ledger, audits successful CAD results against concrete runtime
    evidence, and diagnoses concrete kernel failures. If it is unavailable, the caller
    continues with the existing Codex path unchanged.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("NEBIUS_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("NEBIUS_TOKEN_FACTORY_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = (model or os.getenv("NEBIUS_NEMOTRON_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
        self.timeout_seconds = int(timeout_seconds or os.getenv("CADIA_NEMOTRON_TIMEOUT_SECONDS", "90"))
        self.enabled = _truthy(os.getenv("CADIA_NEMOTRON_ENABLED"), True) and bool(self.api_key)
        self._transport = transport
        self._cancelled = threading.Event()
        self._lock = threading.RLock()

    @property
    def configured(self) -> bool:
        return self.enabled

    def cancel(self) -> None:
        self._cancelled.set()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        if not self.enabled:
            raise NemotronError("Nebius Nemotron is not configured.")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                detail = (parsed.get("error") or {}).get("message") or body
            except Exception:
                detail = body
            raise NemotronError(f"Nebius Token Factory error ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise NemotronError(f"Nebius Token Factory connection error: {exc.reason}") from exc

    @staticmethod
    def _response_text(response: dict[str, Any]) -> str:
        try:
            value = response["choices"][0]["message"]["content"]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "".join(parts)
        except Exception:
            pass
        direct = response.get("output_text")
        return direct if isinstance(direct, str) else ""

    def _complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        on_event=None,
        progress_message: str,
        progress_percent: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        with self._lock:
            self._cancelled.clear()
            if on_event:
                on_event("progress", {"message": progress_message, "percent": progress_percent})
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                response = self._post(payload)
            except NemotronError as first:
                # Some OpenAI-compatible endpoints/models reject response_format even
                # though they can reliably follow a JSON-only instruction. Retry once
                # without that optional field before falling back to the untouched Codex path.
                msg = str(first).lower()
                if "response_format" not in msg and "json_object" not in msg and "400" not in msg:
                    raise
                payload.pop("response_format", None)
                response = self._post(payload)
            if self._cancelled.is_set():
                raise NemotronError("Nemotron orchestration was canceled by the user.")
            text = self._response_text(response).strip()
            if not text:
                raise NemotronError("Nemotron returned no usable text response.")
            try:
                return _extract_json_object(text)
            except ValueError as first_parse_error:
                # One bounded transport-level recovery for truncated/malformed JSON.
                # This is provider resilience only; it never changes CAD state.
                if on_event:
                    on_event("progress", {"message": "Nemotron JSON was incomplete · retrying once…", "percent": progress_percent})
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                retry_payload["max_tokens"] = max(int(max_tokens), 1600)
                retry_messages = list(payload["messages"])
                retry_messages[0] = {
                    "role": "system",
                    "content": system + "\nReturn one COMPLETE JSON object only. Do not include markdown or stop mid-object.",
                }
                retry_payload["messages"] = retry_messages
                retry_response = self._post(retry_payload)
                retry_text = self._response_text(retry_response).strip()
                if not retry_text:
                    raise NemotronError("Nemotron returned no usable JSON on retry.") from first_parse_error
                try:
                    return _extract_json_object(retry_text)
                except ValueError as retry_error:
                    raise NemotronError("Nemotron returned incomplete JSON after one retry.") from retry_error

    def engineering_contract(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        system = (
            "You are CADia's engineering requirements extraction agent. "
            "Do not write CAD code and do not choose CAD API/tool names. The raw user request is the source of truth. "
            "Extract explicit requirements into a verification-ready ledger without adding any hard requirement the user did not state. "
            "Every requirement MUST include a short source_quote copied verbatim from USER_REQUEST. If you cannot quote it, do not mark it hard. "
            "Use stable ids R1, R2, ... and preserve explicit dimensions, counts, components, relations, edit targets, motion requirements, and preservation constraints. "
            "Do not infer missing hard dimensions. Return JSON only with keys: summary, components, requirements, uncertainties. "
            "requirements must be an array of objects with keys: id, kind, target, expected, unit, hard, explicit, source_quote, verification_hint. "
            "expected should be a JSON number/bool/string or a compact object/list when necessary; unit is empty when not applicable. "
            "verification_hint describes what concrete CAD evidence could verify the requirement, not a tool call."
        )
        user = (
            "USER_REQUEST:\n" + user_prompt.strip() +
            "\n\nCURRENT_CAD_STATE (read-only context):\n" + _compact(current_state, 10000)
        )
        raw = self._complete_json(
            system=system,
            user=user,
            max_tokens=1800,
            on_event=on_event,
            progress_message="Nemotron · extracting source-grounded engineering requirements…",
            progress_percent=3,
        )
        return normalize_engineering_contract(raw, user_prompt)

    def result_audit_with_contract(
        self,
        *,
        user_prompt: str,
        evidence: dict[str, Any],
        on_event=None,
    ) -> dict[str, Any]:
        """One-call success-path supervisor for ordinary requests.

        It extracts only verbatim-grounded requirements and audits them against the
        already-executed CAD evidence in the same Nemotron round trip.  This preserves
        post-execution supervision while removing the pre-plan Nemotron latency for
        ordinary requests.
        """
        system = (
            "You are CADia's combined engineering requirement extraction and post-execution verification agent. "
            "The CAD operation already executed successfully. RAW_USER_REQUEST is the source of truth. "
            "First extract ONLY explicit requirements that can be anchored by a short source_quote copied verbatim from RAW_USER_REQUEST. "
            "Then audit those requirements against RUNTIME_EVIDENCE. Do not invent missing requirements, geometry, measurements, or evidence. "
            "Absence of evidence is uncertain, never fail. Set audit.verdict=repair ONLY when a hard explicit requirement is concretely contradicted by RUNTIME_EVIDENCE. "
            "For every fail finding, evidence_path MUST be an exact RFC6901-style JSON pointer into /after_state, /geometry, /delta, or /derived, and observed MUST exactly match the value there. "
            "Do not use /executed_plan as proof of final geometry. Return one JSON object only with keys contract and audit. "
            "contract has keys summary, components, requirements, uncertainties. contract.requirements items have keys id, kind, target, expected, unit, hard, explicit, source_quote, verification_hint. "
            "audit has keys verdict, summary, findings, repair_guidance, preserve. audit.verdict is pass, repair, or uncertain. "
            "audit.findings items have keys requirement_id, status, expected, observed, evidence_path, confidence, reason. Confidence is 0..1. Keep everything concise."
        )
        user = (
            "RAW_USER_REQUEST:\n" + user_prompt.strip()
            + "\n\nRUNTIME_EVIDENCE:\n" + _compact(evidence, 28000)
        )
        raw = self._complete_json(
            system=system,
            user=user,
            max_tokens=1800,
            on_event=on_event,
            progress_message="Nemotron · verifying executed CAD in one combined pass…",
            progress_percent=91,
        )
        contract = normalize_engineering_contract(raw.get("contract") if isinstance(raw, dict) else {}, user_prompt)
        audit = raw.get("audit") if isinstance(raw, dict) and isinstance(raw.get("audit"), dict) else {}
        return {"contract": contract, "audit": audit}

    def result_audit(
        self,
        *,
        user_prompt: str,
        contract: dict[str, Any],
        evidence: dict[str, Any],
        on_event=None,
        recheck: bool = False,
    ) -> dict[str, Any]:
        """Audit a *successful* CAD execution against grounded requirements.

        The audit has no mutation authority. A separate deterministic gate decides whether
        any finding is concrete enough to permit a bounded Codex repair.
        """
        system = (
            "You are CADia's post-execution engineering verification agent. The CAD operation already executed successfully. "
            "Audit only the source-grounded requirements in ENGINEERING_CONTRACT against RUNTIME_EVIDENCE. "
            "RAW USER REQUEST is always the source of truth. Do not invent requirements, geometry, measurements, or evidence. "
            "A successful CAD kernel call is not proof that design intent is satisfied. Conversely, absence of evidence is NOT a failure: mark it uncertain. "
            "Set verdict=repair ONLY when a hard explicit requirement is concretely contradicted by RUNTIME_EVIDENCE. "
            "For every fail finding, evidence_path MUST be an exact RFC6901-style JSON pointer into /after_state, /geometry, /delta, or /derived, and observed MUST exactly match the value at that path. "
            "Do not use /executed_plan as proof of final geometry. Use status pass/fail/uncertain per requirement. "
            "Return JSON only with keys: verdict, summary, findings, repair_guidance, preserve. "
            "verdict is pass, repair, or uncertain. findings is an array with keys requirement_id, status, expected, observed, evidence_path, confidence, reason. "
            "repair_guidance and preserve are concise string arrays. Confidence is 0..1."
        )
        user = (
            "RAW_USER_REQUEST:\n" + user_prompt.strip() +
            "\n\nENGINEERING_CONTRACT:\n" + _compact(contract, 12000) +
            "\n\nRUNTIME_EVIDENCE:\n" + _compact(evidence, 30000) +
            ("\n\nThis is a bounded re-check after one repair. Do not request another repair unless concrete evidence still contradicts a hard requirement." if recheck else "")
        )
        return self._complete_json(
            system=system,
            user=user,
            max_tokens=1800,
            on_event=on_event,
            progress_message=("Nemotron · re-checking repaired CAD against requirements…" if recheck else "Nemotron · verifying executed CAD against requirements…"),
            progress_percent=(98 if recheck else 91),
        )

    def recovery_guidance(
        self,
        *,
        user_prompt: str,
        current_state: dict[str, Any],
        failed_plan: Any,
        error: str,
        contract: dict[str, Any] | None,
        recovery_context: dict[str, Any] | None = None,
        on_event=None,
    ) -> dict[str, Any]:
        system = (
            "You are CADia's engineering failure-diagnosis agent. A deterministic CAD kernel rejected an operation. "
            "Do not emit executable CAD tool calls or Python. Diagnose the concrete failure and give a bounded repair "
            "strategy to a separate Codex execution agent. Preserve the original user dimensions/counts and already-valid "
            "geometry. Do not recommend repeating an operation with identical arguments after the kernel proved it fails. "
            "Return JSON only with keys: failure_class, likely_cause, must_preserve, repair_strategy, avoid, verification_after_repair. "
            "All list values must be concise strings."
        )
        user = (
            "USER_REQUEST:\n" + user_prompt.strip() +
            "\n\nENGINEERING_CONTRACT:\n" + _compact(contract or {}, 5000) +
            "\n\nCURRENT_CAD_STATE:\n" + _compact(current_state, 8000) +
            "\n\nFAILED_PLAN_OR_CALL:\n" + _compact(failed_plan, 8000) +
            "\n\nKERNEL_ERROR:\n" + str(error)[:4000] +
            "\n\nRECOVERY_CONTEXT:\n" + _compact(recovery_context or {}, 6000)
        )
        return self._complete_json(
            system=system,
            user=user,
            max_tokens=900,
            on_event=on_event,
            progress_message="Nemotron · diagnosing CAD kernel failure…",
            progress_percent=73,
        )
