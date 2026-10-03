from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Callable

from ..nemotron_orchestrator import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    NemotronError,
    NemotronOrchestrator,
    _extract_json_object,
)


logger = logging.getLogger(__name__)


# Nebius Token Factory's constrained-decoding grammar currently rejects the
# JSON-Schema keyword ``uniqueItems`` (HTTP 400: "Grammar error: Unimplemented
# keys: [\"uniqueItems\"]").  CADia keeps its *authoritative* schema unchanged
# and relaxes only the transport copy sent to the remote grammar compiler.
# The full schema is still validated locally before any CAD operation executes.
#
# Keep this deliberately narrow.  Current v10 CADia schema uses no other
# xgrammar-known unsupported features (contains/minContains/maxContains,
# multipleOf, patternProperties, propertyNames).  If one appears later we fail
# locally instead of silently weakening the contract.
_REMOTE_RELAXABLE_SCHEMA_KEYS = frozenset({"uniqueItems"})
_REMOTE_HARD_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "contains", "minContains", "maxContains", "multipleOf",
    "patternProperties", "propertyNames",
})


def _schema_keyword_paths(value: Any, wanted: set[str] | frozenset[str], path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}"
            if key in wanted:
                found.append(here)
            found.extend(_schema_keyword_paths(child, wanted, here))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(_schema_keyword_paths(child, wanted, f"{path}[{i}]"))
    return found


def _nebius_transport_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a grammar-compatible copy without weakening CADia's local contract.

    Only ``uniqueItems`` is removed because Token Factory/xgrammar cannot compile
    it.  ``validate_program`` (or the legacy tool-schema validator on older direct
    paths) still receives the original schema and enforces uniqueness locally.
    """
    hard = _schema_keyword_paths(schema, _REMOTE_HARD_UNSUPPORTED_SCHEMA_KEYS)
    if hard:
        raise NemotronError(
            "CADia schema uses constrained-decoding keywords unsupported by the "
            "Nebius grammar backend: " + ", ".join(hard[:8])
        )

    removed = 0

    def clone(node: Any) -> Any:
        nonlocal removed
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, child in node.items():
                if key in _REMOTE_RELAXABLE_SCHEMA_KEYS:
                    removed += 1
                    continue
                out[key] = clone(child)
            return out
        if isinstance(node, list):
            return [clone(x) for x in node]
        return copy.deepcopy(node)

    out = clone(schema)
    if removed:
        logger.info(
            "[NEMOTRON_SCHEMA] relaxed %d remote grammar-only keyword(s); "
            "full CADia schema remains locally authoritative",
            removed,
        )
    return out


def _nebius_transport_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize only function parameter schemas in a copied OpenAI tools list."""
    out = copy.deepcopy(tools)
    for row in out:
        if not isinstance(row, dict) or row.get("type") != "function":
            continue
        fn = row.get("function")
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            fn["parameters"] = _nebius_transport_schema(params)
    return out


class NebiusNemotronProvider(NemotronOrchestrator):
    """Direct CAD planner transport for NVIDIA Nemotron via Nebius Token Factory.

    This provider is intentionally separate from ``NemotronAugmentedCodexAgent``.
    When selected, Nemotron itself receives CADia's normal planner/repair prompts and
    returns the executable typed-tool plan consumed by the existing deterministic
    CAD executor.  No Codex process or ChatGPT authentication is involved.

    The server-owned ``NEBIUS_API_KEY`` is never exposed to the browser or stored in
    a user's provider row.  The provider row stores only the selected model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: int | None = None,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or os.getenv("NEBIUS_TOKEN_FACTORY_BASE_URL", DEFAULT_BASE_URL),
            model=default_model or os.getenv("NEBIUS_NEMOTRON_MODEL", DEFAULT_MODEL),
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        self.default_model = self.model

    def list_models(self) -> list[dict[str, Any]]:
        if not self.configured:
            raise NemotronError("Nebius Nemotron is not configured on this server.")
        # Keep this provider NVIDIA-specific instead of exposing every Token Factory
        # model through the Nemotron card. Additional Nemotron variants can be added
        # later without changing the common provider/executor architecture.
        return [
            {
                "id": self.default_model,
                "model": self.default_model,
                "displayName": self.default_model,
                "isDefault": True,
            }
        ]

    def complete_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
        model: str | None = None,
        effort: str | None = None,
        on_event=None,
    ) -> dict[str, Any]:
        if not self.configured:
            raise NemotronError("Nebius Nemotron is not configured on this server.")

        with self._lock:
            self._cancelled.clear()
            selected_model = (model or self.default_model or DEFAULT_MODEL).strip()
            if on_event:
                on_event(
                    "progress",
                    {
                        "message": "NVIDIA Nemotron · generating executable CAD plan…",
                        "percent": 8,
                    },
                )

            schema_text = json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
            system = (
                "You are the direct CAD planning agent for CADia. "
                "The user message contains CADia's authoritative tool catalog, current CAD state, "
                "and modeling or repair request. Produce only one JSON object matching the required "
                "schema exactly. Use only tool names and argument shapes present in the supplied catalog. "
                "Treat CURRENT_STATE fields as read-only observations: never copy state/status metadata into a tool's arguments "
                "unless that exact argument key is explicitly present in the tool's inputSchema. "
                "Do not emit markdown, prose outside the JSON object, Python, shell commands, or invented tools. "
                "Preserve explicit dimensions, counts, selections, constraints, and already-valid geometry.\n\n"
                "REQUIRED_OUTPUT_SCHEMA:\n" + schema_text
            )
            # Nemotron 3 Ultra reasons by default. For constrained JSON planning, keep
            # thinking disabled: NVIDIA documents that reasoning-enabled constrained
            # output can produce malformed/truncated JSON on some OpenAI-compatible
            # runtimes. force_nonempty_content is recommended for coding/agent use.
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_DIRECT_MAX_TOKENS", "16384")),
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                },
                "response_format": {"type": "json_object"},
            }

            try:
                response = self._post(payload)
            except NemotronError as first:
                # Token Factory models can differ in optional response_format support.
                # Retry once without that optional field; CADia still validates the
                # returned plan against its own schema/tool catalog before execution.
                message = str(first).lower()
                if "response_format" not in message and "json_object" not in message and "400" not in message:
                    raise
                payload.pop("response_format", None)
                response = self._post(payload)

            if self._cancelled.is_set():
                raise NemotronError("The Nemotron CAD planning request was canceled by the user.")
            text = self._response_text(response).strip()
            try:
                finish_reason = str(response.get("choices", [{}])[0].get("finish_reason") or "")
            except Exception:
                finish_reason = ""
            logger.warning(
                "[NEMOTRON_DIAG] stage=%s model=%s finish_reason=%s chars=%d raw=%s",
                "schema_retry" if "TOOL_SCHEMA_RETRY_FEEDBACK:" in prompt else "planner",
                selected_model,
                finish_reason,
                len(text),
                text,
            )
            if not text:
                error = response.get("error") or {}
                raise NemotronError(str(error.get("message") or "Nemotron did not return CAD plan JSON."))
            if on_event:
                on_event("progress", {"message": "Validating Nemotron CAD plan…", "percent": 20})
            try:
                return _extract_json_object(text)
            except ValueError as first_parse_error:
                # One bounded retry for malformed/truncated structured output. This is
                # provider-level resilience, not a CAD-feature special case: the same
                # path applies to every modeling/edit/repair request.
                if on_event:
                    on_event(
                        "progress",
                        {
                            "message": "Nemotron output was incomplete · retrying JSON plan…",
                            "percent": 18,
                        },
                    )
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                retry_payload["max_tokens"] = max(int(payload.get("max_tokens") or 0), 16384)
                retry_messages = list(payload["messages"])
                retry_messages[0] = {
                    "role": "system",
                    "content": system
                    + "\nIMPORTANT: Return the COMPLETE JSON object. Do not stop mid-object. "
                    + "Do not include reasoning or markdown in content.",
                }
                retry_payload["messages"] = retry_messages
                retry_response = self._post(retry_payload)
                retry_text = self._response_text(retry_response).strip()
                try:
                    retry_finish_reason = str(retry_response.get("choices", [{}])[0].get("finish_reason") or "")
                except Exception:
                    retry_finish_reason = ""
                logger.warning(
                    "[NEMOTRON_DIAG] stage=%s model=%s finish_reason=%s chars=%d raw=%s",
                    "schema_retry_json_recovery" if "TOOL_SCHEMA_RETRY_FEEDBACK:" in prompt else "planner_json_recovery",
                    selected_model,
                    retry_finish_reason,
                    len(retry_text),
                    retry_text,
                )
                if not retry_text:
                    raise NemotronError("Nemotron returned no usable CAD plan JSON on retry.") from first_parse_error
                try:
                    return _extract_json_object(retry_text)
                except ValueError as retry_parse_error:
                    finish_reason = ""
                    try:
                        finish_reason = str(retry_response.get("choices", [{}])[0].get("finish_reason") or "")
                    except Exception:
                        pass
                    suffix = f" (finish_reason={finish_reason})" if finish_reason else ""
                    raise NemotronError(
                        "Nemotron returned incomplete CAD plan JSON after one retry" + suffix + "."
                    ) from retry_parse_error


    @staticmethod
    def _strict_plan_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a plan schema mechanically from CADia's live public tool catalog.

        No feature names, part names, or one-off geometry rules are encoded here. Each
        branch is generated from the exact inputSchema of one currently available tool.
        """
        branches: list[dict[str, Any]] = []
        for row in catalog:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            branches.append({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": [name]},
                    "arguments": row.get("inputSchema") or {"type": "object", "properties": {}},
                    "bind": {"type": "string"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            })
        return {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 80,
                    "items": {"oneOf": branches},
                },
                "note": {"type": "string"},
            },
            "required": ["calls", "note"],
            "additionalProperties": False,
        }

    @staticmethod
    def _tool_call_arguments(message: dict[str, Any], expected_name: str) -> dict[str, Any] | None:
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            return None
        for raw in calls:
            fn = raw.get("function") or {} if isinstance(raw, dict) else {}
            if str(fn.get("name") or "") != expected_name:
                continue
            args = fn.get("arguments")
            if isinstance(args, dict):
                return args
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _effort_kwargs(effort: str | None) -> dict[str, Any]:
        level = (effort or "medium").strip().lower()
        # NVIDIA recommends reasoning for agentic/tool-use requests. Medium effort is
        # the documented starting point; high/max leave the full reasoning budget on.
        kwargs: dict[str, Any] = {
            "enable_thinking": True,
            "force_nonempty_content": True,
        }
        if level in {"low", "medium", "balanced", "auto"}:
            kwargs["medium_effort"] = True
        return kwargs

    def complete_plan(
        self,
        prompt: str,
        *,
        catalog: list[dict[str, Any]],
        model: str | None = None,
        effort: str | None = None,
        on_event=None,
        progress_span: tuple[int, int] = (5, 20),
    ) -> dict[str, Any]:
        """Return one complete CADia IR plan through constrained structured output.

        Public Nemotron CAD projects that are reliable do not ask the model to emit
        arbitrary backend calls: they constrain a typed CAD program/intent object.
        CADia follows that pattern here. The schema is still generated from the live
        request-specific catalog, so no prompt/part-specific operation list is baked in.

        Primary transport is strict ``json_schema`` with thinking disabled because the
        Nebius/NVIDIA OpenAI-compatible stack can have reasoning+guided-decoding parser
        conflicts. A forced function call is retained only as a compatibility fallback.
        """
        if not self.configured:
            raise NemotronError("Nebius Nemotron is not configured on this server.")

        with self._lock:
            self._cancelled.clear()
            selected_model = (model or self.default_model or DEFAULT_MODEL).strip()
            strict_schema = self._strict_plan_schema(catalog)
            transport_schema = _nebius_transport_schema(strict_schema)
            system = (
                "You are CADia's structured CAD planner. The user message already contains "
                "the authoritative planning rules, CURRENT_STATE, USER_REQUEST, and a closed "
                "typed TOOL_CATALOG. Return one complete plan that matches the supplied JSON "
                "schema exactly. Do not emit prose, markdown, hidden backend tools, or arguments "
                "outside the visible schemas. Preserve every explicit dimension, count, relation, "
                "selection, and already-valid feature."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            if on_event:
                on_event(
                    "progress",
                    {
                        "message": "NVIDIA Nemotron · constrained CAD IR planning…",
                        "percent": int(progress_span[0]),
                    },
                )

            primary: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": float(os.getenv("CADIA_NEMOTRON_IR_TEMPERATURE", "0.15")),
                "top_p": float(os.getenv("CADIA_NEMOTRON_IR_TOP_P", "0.95")),
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_PLAN_MAX_TOKENS", "32768")),
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cadia_ir_plan",
                        "strict": True,
                        "schema": transport_schema,
                    },
                },
            }

            first_error: Exception | None = None
            try:
                response = self._post(primary)
                if self._cancelled.is_set():
                    raise NemotronError("The Nemotron CAD planning request was canceled by the user.")
                text = self._response_text(response).strip()
                if not text:
                    raise NemotronError("Nemotron constrained CAD IR returned no plan.")
                plan = _extract_json_object(text)
                logger.info(
                    "[NEMOTRON_IR] transport=json_schema model=%s calls=%d",
                    selected_model,
                    len(plan.get("calls") or []),
                )
                return plan
            except Exception as exc:
                first_error = exc

            # Compatibility fallback for serving stacks that reject nested json_schema.
            # The same strict schema is attached to one function, so the model still
            # cannot switch to CADia's hidden/raw backend vocabulary.
            fn_name = "submit_cadia_ir_plan"
            tool_payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "description": "Submit the complete validated CADia IR plan.",
                        "parameters": transport_schema,
                    },
                }],
                "tool_choice": {"type": "function", "function": {"name": fn_name}},
                "parallel_tool_calls": False,
                "temperature": float(os.getenv("CADIA_NEMOTRON_IR_TEMPERATURE", "0.15")),
                "top_p": float(os.getenv("CADIA_NEMOTRON_IR_TOP_P", "0.95")),
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_PLAN_MAX_TOKENS", "32768")),
                "chat_template_kwargs": self._effort_kwargs(effort),
            }
            try:
                try:
                    response = self._post(tool_payload)
                except NemotronError as exc:
                    msg = str(exc).lower()
                    if "parallel_tool_calls" not in msg and "unknown field" not in msg and "extra_forbidden" not in msg:
                        raise
                    tool_payload.pop("parallel_tool_calls", None)
                    response = self._post(tool_payload)
                if self._cancelled.is_set():
                    raise NemotronError("The Nemotron CAD planning request was canceled by the user.")
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                plan = self._tool_call_arguments(message, fn_name)
                if plan is None:
                    text = self._response_text(response).strip()
                    if text:
                        plan = _extract_json_object(text)
                if plan is None:
                    raise NemotronError("Nemotron tool fallback returned no CAD IR plan.")
                logger.warning(
                    "[NEMOTRON_IR] transport=tool_fallback model=%s calls=%d primary_error=%s",
                    selected_model,
                    len(plan.get("calls") or []),
                    str(first_error),
                )
                return plan
            except Exception as second_error:
                raise NemotronError(
                    "Nemotron could not produce a constrained CAD IR plan. "
                    f"json_schema_error={first_error}; tool_error={second_error}"
                ) from second_error

    def complete_cad_program(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        model: str | None = None,
        effort: str | None = None,
        on_event=None,
        repair: bool = False,
    ) -> dict[str, Any]:
        """Generate one schema-constrained high-level CAD program.

        This is the PromptCAD/GPT4FreeCAD-style transport used by Nemotron Direct.
        The schema is the CAD program contract itself; no CADia backend tool catalog,
        runtime binding syntax, document path, or raw Inventor/CAD command is exposed.
        """
        if not self.configured:
            raise NemotronError("Nebius Nemotron is not configured on this server.")

        with self._lock:
            self._cancelled.clear()
            selected_model = (model or self.default_model or DEFAULT_MODEL).strip()
            transport_schema = _nebius_transport_schema(schema)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are CADia's constrained mechanical CAD program planner. "
                        "Return exactly one JSON object matching the supplied JSON Schema. "
                        "The deterministic CAD interpreter owns all backend tools, files, documents, "
                        "runtime identifiers, and execution bookkeeping. Do not output prose or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            if on_event:
                on_event(
                    "progress",
                    {
                        "message": "NVIDIA Nemotron · generating structured CAD program…" if not repair else "NVIDIA Nemotron · repairing structured CAD program…",
                        "percent": 8 if not repair else 82,
                    },
                )

            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": float(os.getenv("CADIA_NEMOTRON_PROGRAM_TEMPERATURE", "0.1")),
                "top_p": float(os.getenv("CADIA_NEMOTRON_PROGRAM_TOP_P", "0.95")),
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_PROGRAM_MAX_TOKENS", "16384")),
                # As in public Nemotron CAD integrations, structured generation is
                # constrained at decoding time.  Disable hidden reasoning on this
                # transport to avoid reasoning/json parser conflicts on compatible APIs.
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cadia_structured_cad_program",
                        "strict": True,
                        "schema": transport_schema,
                    },
                },
            }

            primary_error: Exception | None = None
            try:
                response = self._post(payload)
                if self._cancelled.is_set():
                    raise NemotronError("The Nemotron CAD program request was canceled by the user.")
                text = self._response_text(response).strip()
                if not text:
                    raise NemotronError("Nemotron returned no structured CAD program.")
                program = _extract_json_object(text)
                logger.info(
                    "[NEMOTRON_PROGRAM] transport=json_schema model=%s components=%d repair=%s",
                    selected_model,
                    len(program.get("components") or []),
                    bool(repair),
                )
                return program
            except Exception as exc:
                primary_error = exc

            # Compatibility fallback: force one function whose arguments are the
            # exact same program schema.  This keeps the single source of truth even
            # when a serving revision does not accept response_format.json_schema.
            fn_name = "submit_cadia_structured_program"
            fallback: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "description": "Submit one validated CADia high-level CAD program.",
                        "parameters": transport_schema,
                    },
                }],
                "tool_choice": {"type": "function", "function": {"name": fn_name}},
                "parallel_tool_calls": False,
                "temperature": float(os.getenv("CADIA_NEMOTRON_PROGRAM_TEMPERATURE", "0.1")),
                "top_p": float(os.getenv("CADIA_NEMOTRON_PROGRAM_TOP_P", "0.95")),
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_PROGRAM_MAX_TOKENS", "16384")),
                "chat_template_kwargs": self._effort_kwargs(effort),
            }
            try:
                try:
                    response = self._post(fallback)
                except NemotronError as exc:
                    msg = str(exc).lower()
                    if "parallel_tool_calls" not in msg and "unknown field" not in msg and "extra_forbidden" not in msg:
                        raise
                    fallback.pop("parallel_tool_calls", None)
                    response = self._post(fallback)
                if self._cancelled.is_set():
                    raise NemotronError("The Nemotron CAD program request was canceled by the user.")
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                program = self._tool_call_arguments(message, fn_name)
                if program is None:
                    text = self._response_text(response).strip()
                    if text:
                        program = _extract_json_object(text)
                if program is None:
                    raise NemotronError("Nemotron function fallback returned no structured CAD program.")
                logger.warning(
                    "[NEMOTRON_PROGRAM] transport=tool_fallback model=%s components=%d repair=%s primary_error=%s",
                    selected_model,
                    len(program.get("components") or []),
                    bool(repair),
                    str(primary_error),
                )
                return program
            except Exception as second_error:
                raise NemotronError(
                    "Nemotron could not produce the constrained high-level CAD program. "
                    f"json_schema_error={primary_error}; tool_error={second_error}"
                ) from second_error

    def complete_tool_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        model: str | None = None,
        on_event=None,
    ) -> dict[str, Any]:
        """Run one native function-calling turn for the direct Nemotron CAD agent.

        The endpoint receives CADia's exact tool schemas through the OpenAI-compatible
        ``tools`` field instead of asking the model to reproduce a long free-form JSON
        plan. Thinking is disabled for constrained tool calls, matching NVIDIA's
        documented guidance for reliable tool-use responses.
        """
        if not self.configured:
            raise NemotronError("Nebius Nemotron is not configured on this server.")

        with self._lock:
            self._cancelled.clear()
            selected_model = (model or self.default_model or DEFAULT_MODEL).strip()
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "tools": _nebius_transport_tools(tools),
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "temperature": 0.1,
                "max_tokens": int(os.getenv("CADIA_NEMOTRON_TOOL_MAX_TOKENS", "2048")),
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                },
            }
            try:
                response = self._post(payload)
            except NemotronError as first:
                # Some OpenAI-compatible serving revisions omit parallel_tool_calls.
                # Retry only by removing that optional transport hint; tools/tool_choice
                # remain required so direct mode never falls back to free-form plans.
                msg = str(first).lower()
                if "parallel_tool_calls" not in msg and "unknown field" not in msg and "extra_forbidden" not in msg:
                    raise
                payload.pop("parallel_tool_calls", None)
                response = self._post(payload)

            if self._cancelled.is_set():
                raise NemotronError("The Nemotron CAD tool request was canceled by the user.")
            try:
                choice = response["choices"][0]
                message = choice.get("message") or {}
            except Exception as exc:
                raise NemotronError("Nebius Nemotron returned an invalid chat-completion response.") from exc
            tool_calls = message.get("tool_calls") or []
            logger.info(
                "[NEMOTRON_TOOL] model=%s finish_reason=%s tool_calls=%d",
                selected_model,
                str(choice.get("finish_reason") or ""),
                len(tool_calls) if isinstance(tool_calls, list) else 0,
            )
            return {"message": message, "finish_reason": choice.get("finish_reason"), "raw": response}

