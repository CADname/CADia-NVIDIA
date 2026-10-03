from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any, Iterator

from standalonecad.codex_agent import CodexAgent, _validate_planner_plan
from standalonecad.planning import _validate_value

from .nemotron_ir import build_ir_catalog, compile_document_lifecycle, normalize_ir_plan, validate_ir_plan
from .providers.nebius_nemotron import NebiusNemotronProvider


NEMOTRON_GROUNDING_RULES = """
NEMOTRON_CAD_IR_RULES:
- Work only through the TOOL_CATALOG visible in this turn. It is a closed, typed CAD vocabulary selected from CADia's live capabilities for the current request.
- Some model-facing operations are cadia_ir_* operations. They are intentional canonical operations, not backend names. Use them exactly as defined; CADia translates them deterministically to its existing backend after validation.
- Do not call hidden/raw alternatives that are absent from TOOL_CATALOG even if a previous failed plan or kernel error mentions them.
- Treat each tool inputSchema as a hard type contract. Omitted fields are not implied; additional fields are forbidden when additionalProperties=false.
- Prefer a short dependency-correct plan. Use bindings for values returned by earlier calls instead of guessing generated document/occurrence/entity/topology values.
- Runtime binding syntax is a safe JSON-object path: `$bind.<name>.field`, `$bind.<name>.items[0].field`, or legacy `$result[N].field`. Only use fields documented by that tool's outputSchema/referenceHint in TOOL_CATALOG. Never invent a result field name.
- Sketch entity ids and B-Rep topology ids are different namespaces. inventor_add_sketch_dimension.entityId must come from inventor_draw_line/circle/arc `.entity_id` (or rectangle `.entity_ids[index]`), never from cad_get_topology face/edge ids. Face/edge ids are only for topology/GeometryIntent inputs.
- For a newly created Part/Assembly that must be reopened or placed later, call inventor_save_document with path:null and bind that save call; later use `$bind.<save>.path`. Do not invent server filesystem paths.
- Document focus is compiled deterministically by CADia for multi-document work: create/build the requested documents and relationships in dependency order; do not add speculative tab/focus workarounds or duplicate reopen loops.
- Treat persistent face/edge ids exposed by CURRENT_STATE as first-class geometry references. For GeometryIntent kind=face, selector may be an exact {"id":"F_..."}; prefer an exact id whenever CURRENT_STATE or a kernel ambiguity error provides one.
- CURRENT_STATE.occurrences[].topology is local component topology. Its ids are valid only for that occurrence and must not be mixed between occurrences.
- If a kernel error reports candidate topology ids, choose only from those real candidates using ORIGINAL_REQUEST/CURRENT_STATE. Never invent a topology id or an arbitrary near_mm just to force a match.
- On repair/continuation, preserve already-valid work and correct only the failed semantic step plus genuinely dependent remainder. Do not change requested dimensions/counts to make the kernel accept a plan.
""".strip()


class NemotronParityAgent(CodexAgent):
    """Nemotron planner with a public-CAD-agent-style typed IR boundary.

    The execution engine remains CADia's existing Codex/PlanExecutor/recovery stack.
    Nemotron sees a smaller, closed, request-relevant vocabulary and a few canonical
    IR operations where the raw backend exposes overlapping or condition-dependent
    contracts.  The validated IR is translated deterministically back to CADia's normal
    tools before execution.  There are no part/prompt/dimension-specific branches.
    """

    def __init__(
        self,
        *,
        provider: NebiusNemotronProvider,
        target_id: str,
        project_root: Path,
        model: str,
        reasoning: str = "medium",
    ) -> None:
        self.provider = provider
        super().__init__(
            target_id=target_id,
            project_root=project_root,
            model=model,
            reasoning=reasoning,
        )
        self._full_catalog = copy.deepcopy(self._catalog)
        self._active_ir_mappings = {}
        self._document_dir = (Path(project_root).resolve() / "documents")

    def cancel(self) -> None:
        self.provider.cancel()

    def _validate_exact_tool_arguments(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Validate against the currently visible catalog before any IR translation."""
        validate_ir_plan(plan, self._catalog)
        return plan

    def _validate_backend_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Second gate: translated calls must match CADia's unchanged live backend."""
        checked = _validate_planner_plan(plan, self._full_catalog)
        schemas = {
            str(row.get("name")): row.get("inputSchema") or {}
            for row in self._full_catalog
            if row.get("name")
        }
        for idx, call in enumerate(checked.get("calls") or []):
            name = str(call.get("tool") or "")
            _validate_value(
                schemas[name],
                call.get("arguments") or {},
                f"calls[{idx}].{name}.arguments",
            )
        return checked

    @staticmethod
    def _query_text(
        user_prompt: str,
        current_state: dict[str, Any],
        *,
        error: str | None = None,
        plans: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = [str(user_prompt or "")]
        doc_type = current_state.get("document_type") or current_state.get("doc_type")
        if doc_type:
            parts.append(str(doc_type))
        selection = current_state.get("selection") or {}
        if isinstance(selection, dict):
            parts.extend(str(x) for x in (selection.get("type"), selection.get("kind")) if x)
        if error:
            parts.append(str(error))
        for plan in plans or []:
            for call in (plan or {}).get("calls") or []:
                if isinstance(call, dict) and call.get("tool"):
                    parts.append(str(call["tool"]))
        return " ".join(parts)

    @contextmanager
    def _ir_context(self, query: str) -> Iterator[None]:
        previous_catalog = self._catalog
        previous_mappings = self._active_ir_mappings
        catalog, mappings = build_ir_catalog(self._full_catalog, query)
        self._catalog = catalog
        self._active_ir_mappings = mappings
        try:
            yield
        finally:
            self._catalog = previous_catalog
            self._active_ir_mappings = previous_mappings

    def _translate_current_plan(
        self,
        plan: dict[str, Any],
        mappings: dict[str, Any],
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        translated = normalize_ir_plan(plan, mappings)
        translated = compile_document_lifecycle(
            translated,
            document_dir=self._document_dir,
            current_state=current_state,
        )
        return self._validate_backend_plan(translated)

    def _run_planner(self, prompt: str, on_event=None, progress_span: tuple[int, int] = (5, 20)) -> dict[str, Any]:
        prompt = NEMOTRON_GROUNDING_RULES + "\n\n" + prompt
        if on_event:
            on_event(
                "progress",
                {
                    "message": "NVIDIA Nemotron · planning through typed CAD IR…",
                    "percent": int(progress_span[0]),
                },
            )
        plan = self.provider.complete_plan(
            prompt,
            catalog=self._catalog,
            model=self.model,
            effort=self.reasoning,
            on_event=on_event,
            progress_span=progress_span,
        )
        return self._validate_exact_tool_arguments(plan)

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        query = self._query_text(user_prompt, current_state)
        with self._ir_context(query):
            mappings = copy.deepcopy(self._active_ir_mappings)
            ir_plan = super().plan(user_prompt, current_state, on_event=on_event)
        return self._translate_current_plan(ir_plan, mappings, current_state)

    def continue_after_failure(
        self,
        user_prompt: str,
        current_state: dict[str, Any],
        completed_prefix: list[dict[str, Any]],
        failed_call: dict[str, Any],
        remaining_calls: list[dict[str, Any]],
        error: str,
        on_event=None,
        progress_span: tuple[int, int] = (72, 80),
    ) -> dict[str, Any]:
        context_plan = {"calls": [*(completed_prefix or []), failed_call, *(remaining_calls or [])]}
        query = self._query_text(user_prompt, current_state, error=error, plans=[context_plan])
        with self._ir_context(query):
            mappings = copy.deepcopy(self._active_ir_mappings)
            ir_plan = super().continue_after_failure(
                user_prompt,
                current_state,
                completed_prefix,
                failed_call,
                remaining_calls,
                error,
                on_event=on_event,
                progress_span=progress_span,
            )
        return self._translate_current_plan(ir_plan, mappings, current_state)

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
        query = self._query_text(user_prompt, current_state, error=error, plans=[failed_plan])
        with self._ir_context(query):
            mappings = copy.deepcopy(self._active_ir_mappings)
            ir_plan = super().repair(
                user_prompt,
                current_state,
                failed_plan,
                error,
                on_event=on_event,
                recovery_context=recovery_context,
                progress_span=progress_span,
            )
        return self._translate_current_plan(ir_plan, mappings, current_state)
