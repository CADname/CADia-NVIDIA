from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from standalonecad.planning import _validate_value

from .providers.nebius_nemotron import NebiusNemotronProvider
from .web_agent import desktop_catalog


logger = logging.getLogger(__name__)

_FINISH_TOOL = {
    "name": "cadia_finish",
    "description": (
        "Finish the CAD task only after every explicit requirement in the user's request "
        "has been implemented and the current CAD state is usable."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
}


class NemotronNativeToolAgent:
    """Direct Nemotron CAD agent using native OpenAI-compatible tool calling.

    Unlike the legacy JSON-plan adapter, this agent never asks Nemotron to serialize a
    long list of CAD calls into one free-form JSON object.  Each turn exposes CADia's
    exact public tool schemas as model tools, executes the selected tool against the live
    document, returns the structured tool result to Nemotron, and lets the next turn
    decide the next operation.  This is the same agent/tool separation used by normal
    tool-calling systems: the model chooses; CADia validates and executes deterministically.
    """

    def __init__(
        self,
        provider: NebiusNemotronProvider,
        *,
        executor,
        state_provider: Callable[[], dict[str, Any]],
        project_root: Path,
        model: str | None = None,
        max_steps: int = 80,
    ) -> None:
        self.provider = provider
        self.executor = executor
        self.state_provider = state_provider
        self.project_root = Path(project_root).resolve()
        self.document_dir = (self.project_root / "documents").resolve()
        self.model = model or provider.default_model
        self.max_steps = max(8, min(int(max_steps), 120))
        self.catalog = desktop_catalog()
        self.schemas = {str(row["name"]): row.get("inputSchema") or {} for row in self.catalog}
        self.tools = [self._as_openai_tool(row) for row in self.catalog]
        self.tools.append(self._as_openai_tool(_FINISH_TOOL))
        self._cancelled = False

    @staticmethod
    def _as_openai_tool(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": str(row["name"]),
                "description": str(row.get("description") or ""),
                "parameters": row.get("inputSchema") or {"type": "object", "properties": {}},
            },
        }

    def cancel(self) -> None:
        self._cancelled = True
        self.provider.cancel()

    def _system_prompt(self) -> str:
        return f"""You are CADia's direct NVIDIA Nemotron CAD agent.
Operate CADia only by calling the supplied tools. Do not output a long JSON plan and do not write CAD code.
Work iteratively: normally call ONE CAD tool, read its structured result, then decide the next tool.
The tool schema is authoritative. Never invent argument names, enum values, references, occurrence names, sketch names, or file paths.
Use exact values returned by prior tool results when a later operation depends on them.
Prefer deterministic cad_* primitives/generators when they directly express the requested geometry; otherwise use inventor_* tools.
For a normal first solid in an empty Part, an inventor_* feature with operation=join is valid because CADia treats join on an empty Part as creation of the first body. Never send an enum value that the selected tool schema does not allow.
new_part/new_assembly switch the active document. For multi-document assemblies, save each unsaved editable document under PROJECT_DOCUMENT_DIR using a .scad.json filename, and open the intended assembly again before placing components or adding assembly relationships.
Do not guess Part:1-style occurrence names. Use the occurrence_name actually returned by place_occurrence.
If a CAD tool returns an error, use the error and the current state to choose a corrected tool call; do not repeat the same invalid call unchanged.
Do not export files unless the user explicitly requests export.
Preserve existing geometry on edit requests unless replacement/deletion is explicit.
Call cadia_finish only after the complete user request is satisfied. Never call cadia_finish merely because one sub-step succeeded.
PROJECT_DOCUMENT_DIR: {self.document_dir}
"""

    @staticmethod
    def _tool_result_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str),
        }

    def _validate_call(self, name: str, arguments: Any) -> dict[str, Any]:
        if name == "cadia_finish":
            schema = _FINISH_TOOL["inputSchema"]
        else:
            schema = self.schemas.get(name)
            if schema is None:
                raise ValueError(f"Unknown CAD tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError(f"{name}.arguments must be an object")
        _validate_value(schema, arguments, f"{name}.arguments")
        return arguments

    @staticmethod
    def _compact_state(state: dict[str, Any], limit: int = 18000) -> str:
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def run(self, user_prompt: str, *, on_event=None) -> str:
        self._cancelled = False
        before = int(self.executor.engine.revision)
        outer_snapshot = self.executor._snapshot()
        executed_rows: list[dict[str, Any]] = []
        tool_failures = 0
        invalid_model_calls = 0

        initial_state = self.state_provider()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "USER_REQUEST:\n" + user_prompt.strip()
                    + "\n\nCURRENT_STATE:\n" + self._compact_state(initial_state)
                    + "\n\nBegin the task by calling the single best next CAD tool."
                ),
            },
        ]

        try:
            for step in range(1, self.max_steps + 1):
                if self._cancelled:
                    raise RuntimeError("The Nemotron CAD task was canceled by the user.")
                if on_event:
                    pct = min(92, 8 + int((step - 1) * 82 / max(1, self.max_steps)))
                    on_event("progress", {"message": f"NVIDIA Nemotron · CAD tool step {step}", "percent": pct})

                response = self.provider.complete_tool_turn(
                    messages,
                    tools=self.tools,
                    model=self.model,
                    on_event=on_event,
                )
                assistant = response.get("message") or {}
                tool_calls = assistant.get("tool_calls") or []
                if not isinstance(tool_calls, list) or not tool_calls:
                    invalid_model_calls += 1
                    if invalid_model_calls > 2:
                        raise RuntimeError("Nemotron stopped without selecting a CAD tool.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "No valid tool call was returned. Continue the unfinished CAD task using exactly one supplied tool. "
                            "Call cadia_finish only when every explicit requirement is complete."
                        ),
                    })
                    continue

                # Preserve the assistant tool-call message exactly in the tool-calling history.
                messages.append({
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": tool_calls,
                })

                finish_summary: str | None = None
                for raw_call in tool_calls:
                    call_id = str(raw_call.get("id") or f"cadia_call_{step}")
                    fn = raw_call.get("function") or {}
                    name = str(fn.get("name") or "")
                    raw_args = fn.get("arguments")
                    try:
                        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception as exc:
                        invalid_model_calls += 1
                        err = {"ok": False, "error": f"Tool arguments were not valid JSON: {exc}"}
                        messages.append(self._tool_result_message(call_id, name or "unknown", err))
                        continue

                    try:
                        arguments = self._validate_call(name, arguments)
                    except Exception as exc:
                        invalid_model_calls += 1
                        err = {
                            "ok": False,
                            "error": str(exc),
                            "instruction": "Choose a supplied tool and obey that tool's exact argument schema on the next turn.",
                        }
                        logger.warning(
                            "[NEMOTRON_TOOL] invalid_call step=%d tool=%s arguments=%s error=%s",
                            step,
                            name,
                            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str),
                            str(exc),
                        )
                        messages.append(self._tool_result_message(call_id, name or "unknown", err))
                        continue

                    if name == "cadia_finish":
                        finish_summary = str(arguments.get("summary") or "CAD modeling complete.").strip()
                        messages.append(self._tool_result_message(call_id, name, {"ok": True, "accepted": True}))
                        continue

                    step_snapshot = self.executor._snapshot()
                    try:
                        runtime_args = self.executor._resolve_runtime_refs(arguments, [], {})
                        result = self.executor._dispatch(name, runtime_args)
                        executed_rows.append({"tool": name, "result": result})
                        tool_failures = 0
                        tool_payload = {
                            "ok": True,
                            "result": result,
                            "current_state_json": self._compact_state(self.state_provider()),
                        }
                        logger.info(
                            "[NEMOTRON_TOOL] ok step=%d tool=%s",
                            step,
                            name,
                        )
                    except Exception as exc:
                        try:
                            self.executor._rollback(step_snapshot)
                        except Exception:
                            pass
                        tool_failures += 1
                        tool_payload = {
                            "ok": False,
                            "error": str(exc),
                            "current_state_json": self._compact_state(self.state_provider()),
                            "instruction": "Repair this failure with a different valid next tool call; do not repeat the same failing call unchanged.",
                        }
                        logger.warning(
                            "[NEMOTRON_TOOL] tool_error step=%d tool=%s arguments=%s error=%s",
                            step,
                            name,
                            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str),
                            str(exc),
                        )
                    messages.append(self._tool_result_message(call_id, name, tool_payload))

                    if tool_failures >= 5:
                        raise RuntimeError("Nemotron exhausted the CAD tool-repair budget after repeated kernel/tool failures.")

                if finish_summary is not None:
                    after = int(self.executor.engine.revision)
                    if not executed_rows or after == before:
                        # The task claimed completion without making a model change. Keep the
                        # conversation alive instead of accepting a false success.
                        messages.append({
                            "role": "user",
                            "content": (
                                "cadia_finish was rejected because the CAD document has not changed. "
                                "Continue with the required CAD tool calls."
                            ),
                        })
                        continue
                    if on_event:
                        on_event("progress", {"message": "Validating Nemotron CAD result…", "percent": 96})
                    return self.executor._summary(executed_rows, before, after) + (f"\n{finish_summary}" if finish_summary else "")

            raise RuntimeError(f"Nemotron reached the {self.max_steps}-step CAD tool limit before completing the request.")
        except Exception:
            try:
                self.executor._rollback(outer_snapshot)
            except Exception:
                pass
            raise
