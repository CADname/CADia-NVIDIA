from __future__ import annotations

# CADIA_RESILIENT_LLM_ROUTING_V2

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from standalonecad.codex_agent import CodexAgent, PLANNER_PREFIX, REPAIR_PREFIX, _validate_planner_plan, _planner_cancelled, _retry_planner_prompt, _legacy_direct_first_enabled
from standalonecad.mcp.tools import make_tools
from standalonecad.planning import direct_plan, _validate_value

from .providers.app_server import AppServerClient
from .providers.base import AIProvider


logger = logging.getLogger(__name__)


# The app-server path is used only for authentication/account/model discovery.
# Actual CAD planning in parity mode is delegated to the *original* CodexAgent
# through `codex exec`, using the same per-user CODEX_HOME established by the
# app-server login.  This deliberately preserves the desktop/MCP planner path.


class DesktopParityCodexAgent(CodexAgent):
    """Run the exact desktop CodexAgent against a web user's isolated login.

    Inherited unchanged from standalonecad.codex_agent.CodexAgent:
      - tool catalog construction
      - resilient LLM-first planning with deterministic final fallback
      - PLANNER_PREFIX / REPAIR_PREFIX
      - JSON prompt serialization
      - codex exec command line
      - JSON extraction
      - plan()/repair() control flow
      - cancel() semantics

    The only web-specific adaptation is the executable wrapper below, which sets
    CODEX_HOME/HOME to the current web user's isolated authentication directory
    before exec'ing the same Codex CLI binary.  No CAD/planning algorithm is
    reimplemented here.
    """

    def __init__(
        self,
        client: AppServerClient,
        target_id: str,
        project_root: Path,
        model: str = "gpt-5.6-sol",
        reasoning: str = "medium",
    ) -> None:
        self._web_client = client
        self._wrapper_path = client.codex_home / "bin" / "codex-desktop-parity"
        super().__init__(target_id=target_id, project_root=project_root, model=model, reasoning=reasoning)
        # Keep generated CAD artifacts inside the current project. Authentication
        # remains per-user through AppServerClient/CODEX_HOME, but a user-global CAD
        # workspace can create persisted assembly paths outside project storage.
        self.workspace = (Path(project_root) / "agent-workspace").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _ensure_user_codex_wrapper(client: AppServerClient) -> Path:
        executable = Path(client._resolve_executable()).resolve()
        user_home = (client.codex_home.parent / "home").resolve()
        user_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        bin_dir = client.codex_home / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        wrapper = bin_dir / "codex-desktop-parity"
        script = "\n".join(
            [
                "#!/bin/sh",
                f"export CODEX_HOME={shlex.quote(str(client.codex_home))}",
                f"export HOME={shlex.quote(str(user_home))}",
                f"exec {shlex.quote(str(executable))} \"$@\"",
                "",
            ]
        )
        if not wrapper.exists() or wrapper.read_text(encoding="utf-8", errors="ignore") != script:
            wrapper.write_text(script, encoding="utf-8")
            os.chmod(wrapper, 0o700)
        return wrapper

    @property
    def codex_exe(self) -> Path | None:
        # Resolve/install the wrapper only when Codex is actually needed.  This
        # preserves the desktop direct_plan fast path, which never requires Codex.
        try:
            self._wrapper_path = self._ensure_user_codex_wrapper(self._web_client)
        except FileNotFoundError:
            return None
        return self._wrapper_path if self._wrapper_path.is_file() else None


# API fallback is intentionally separate.  It cannot be byte-for-byte equivalent
# to the desktop ChatGPT/Codex CLI execution transport, but retains the same
# prompt/tool-catalog/retry/fallback contract for deployments that explicitly choose
# operator-paid API mode.
# CADIA_CONVERSATIONAL_PLAN_SCHEMA_V1
PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response_type": {"type": "string", "enum": ["action", "clarify", "answer"]},
        "calls": {
            "type": "array",
            "minItems": 0,
            "maxItems": 80,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                    "bind": {"type": "string"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
        "message": {"type": "string"},
        "options": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "note": {"type": "string"},
    },
    "required": ["response_type", "calls", "message", "options", "note"],
    "additionalProperties": False,
}



def desktop_catalog() -> list[dict[str, Any]]:
    """Exact copy of CodexAgent._make_catalog(), kept explicit for audit tests."""
    base = make_tools("inventor")
    inventor_extended = make_tools("inventor", assembly_extensions=True)
    base_inventory = {x["name"] for x in base}
    assembly_extensions = [x for x in inventor_extended if x["name"] not in base_inventory]
    native_all = make_tools("cad", extensions=True)
    base_names = {x["name"].replace("inventor_", "cad_", 1) for x in base}
    native_extensions = [x for x in native_all if x["name"] not in base_names]
    rows = base + assembly_extensions + native_extensions
    return [
        {
            "name": r["name"],
            "description": r.get("description", ""),
            "inputSchema": r.get("inputSchema", {}),
        }
        for r in rows
    ]


def _validate_catalog_argument_contract(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate every LLM-produced call against the exact public CAD tool schema.

    The transport-level plan validator intentionally checks only the outer plan shape.
    API-hosted models need one additional pre-execution gate because state snapshots may
    contain metadata fields (status/suppression/health/etc.) that are *not* valid inputs
    to a CAD tool.  Catching those here lets the same planner repair its JSON before the
    failure-only kernel recovery budget is consumed.  This is tool-schema driven and
    contains no feature-, part-, or model-specific exceptions.
    """
    checked = _validate_planner_plan(plan, catalog)
    schemas = {
        str(row.get("name")): row.get("inputSchema") or {}
        for row in catalog
        if row.get("name")
    }
    for idx, call in enumerate(checked.get("calls") or []):
        tool = str(call.get("tool") or "")
        args = call.get("arguments") or {}
        schema = schemas.get(tool)
        if schema is None:
            # _validate_planner_plan already rejects unknown tool names.
            raise ValueError(f"Planner call {idx} uses an unknown tool: {tool!r}")
        try:
            _validate_value(schema, args, f"calls[{idx}].{tool}.arguments")
        except ValueError as exc:
            logger.warning(
                "[NEMOTRON_DIAG] schema_violation call_index=%d tool=%s arguments=%s error=%s",
                idx,
                tool,
                json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                str(exc),
            )
            raise ValueError(
                f"Planner call {idx} violates the exact TOOL_CATALOG inputSchema for {tool}: {exc}"
            ) from exc
    return checked


def _schema_retry_prompt(base_prompt: str, error: Exception, *, repair: bool = False) -> str:
    msg = str(error).replace("\x00", " ").strip()
    if len(msg) > 1800:
        msg = msg[-1800:]
    stage = "repair plan" if repair else "planner response"
    return (
        base_prompt
        + "\n\nTOOL_SCHEMA_RETRY_FEEDBACK:\n"
        + f"The previous {stage} was rejected before CAD execution: {msg}\n"
        + "Return a COMPLETE replacement JSON plan. For every call, use only argument keys, "
        + "types, enums, and nesting explicitly present in that tool's TOOL_CATALOG inputSchema. "
        + "Do not copy fields from CURRENT_STATE into arguments unless that exact inputSchema "
        + "contains the field. Preserve all requirements of ORIGINAL_REQUEST/USER_REQUEST.\n"
    )


class APICompatibleAgent:
    """Same planning contract over the optional operator-paid API transport."""

    def __init__(self, provider: AIProvider, model: str | None = None, reasoning: str | None = None):
        self.provider = provider
        self.model = model
        self.reasoning = reasoning
        self._catalog = desktop_catalog()

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        # API transport mirrors desktop routing policy: full request first, one planner
        # retry with structured feedback, deterministic direct_plan only after both fail.
        if _legacy_direct_first_enabled():
            local = direct_plan(user_prompt, current_state)
            if local is not None:
                if on_event:
                    on_event("progress", {"message": "Standard geometry plan completed", "percent": 20})
                return local

        base_prompt = PLANNER_PREFIX.format(
            tool_catalog=json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":")),
            current_state=json.dumps(current_state, ensure_ascii=False, separators=(",", ":")),
            user_request=user_prompt.strip(),
        )
        last_exc: Exception | None = None
        prompt = base_prompt
        for attempt in (1, 2):
            try:
                planned = self.provider.complete_json(
                    prompt,
                    output_schema=PLAN_OUTPUT_SCHEMA,
                    model=self.model,
                    effort=self.reasoning,
                    on_event=on_event,
                )
                return _validate_catalog_argument_contract(planned, self._catalog)
            except Exception as exc:
                if _planner_cancelled(exc):
                    raise
                last_exc = exc
                if attempt == 1:
                    if on_event:
                        on_event("progress", {"message": "Planner response invalid; retrying with error feedback", "percent": 14})
                    prompt = _schema_retry_prompt(base_prompt, exc, repair=False)

        local = direct_plan(user_prompt, current_state)
        if local is not None:
            if on_event:
                on_event("progress", {"message": "AI planner unavailable; using deterministic compatibility fallback", "percent": 20})
            return local
        assert last_exc is not None
        raise last_exc

    def repair(
        self,
        user_prompt: str,
        current_state: dict[str, Any],
        failed_plan: dict[str, Any],
        error: str,
        on_event=None,
        recovery_context: dict[str, Any] | None = None,
        progress_span: tuple[int, int] = (72, 78),
    ) -> dict[str, Any]:
        prompt = REPAIR_PREFIX.format(
            tool_catalog=json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":")),
            current_state=json.dumps(current_state, ensure_ascii=False, separators=(",", ":")),
            user_request=user_prompt.strip(),
            failed_plan=json.dumps(failed_plan, ensure_ascii=False, separators=(",", ":")),
            error=str(error),
            recovery_context=json.dumps(recovery_context or {}, ensure_ascii=False, separators=(",", ":")),
        )
        base_prompt = prompt
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                repaired = self.provider.complete_json(
                    prompt,
                    output_schema=PLAN_OUTPUT_SCHEMA,
                    model=self.model,
                    effort=self.reasoning,
                    on_event=on_event,
                )
                return _validate_catalog_argument_contract(repaired, self._catalog)
            except Exception as exc:
                if _planner_cancelled(exc):
                    raise
                last_exc = exc
                if attempt == 1:
                    if on_event:
                        on_event(
                            "progress",
                            {
                                "message": "Repair plan violates CAD tool schema; correcting before execution",
                                "percent": max(progress_span[0], min(progress_span[1], progress_span[0] + 1)),
                            },
                        )
                    prompt = _schema_retry_prompt(base_prompt, exc, repair=True)
        assert last_exc is not None
        raise last_exc

    def cancel(self) -> None:
        self.provider.cancel()


# Backward-compatible symbol for older tests/imports.  In app-server mode the
# route uses DesktopParityCodexAgent directly.
WebCodexAgent = APICompatibleAgent
