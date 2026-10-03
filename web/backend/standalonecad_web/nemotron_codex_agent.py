from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .nemotron_orchestrator import NemotronOrchestrator
from .engineering_verification import (
    build_verification_evidence,
    contract_for_planner,
    evaluate_auto_repair_gate,
    hybrid_request_profile,
    raw_user_request_text,
    verified_hybrid_auto_repair_enabled,
    verified_hybrid_enabled,
)
from .providers.app_server import AppServerClient
from .web_agent import DesktopParityCodexAgent
from standalonecad.planning import selected_geometry_plan
from standalonecad.codex_agent import _validate_planner_plan


def _compact_json(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "…"

POST_VERIFY_FORBIDDEN_DOCUMENT_TOOLS = {
    "inventor_new_part", "inventor_new_assembly", "inventor_open_document", "inventor_close_document", "inventor_delete_document_file",
    "cad_new_part", "cad_new_assembly", "cad_open_document", "cad_close_document", "cad_delete_document_file",
}


def _validate_post_repair_safety(plan: dict[str, Any]) -> dict[str, Any]:
    for call in plan.get("calls") or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "")
        if tool in POST_VERIFY_FORBIDDEN_DOCUMENT_TOOLS:
            raise ValueError(f"Post-verification repair may not use document lifecycle tool: {tool}")
    return plan


POST_VERIFICATION_REPAIR_PREFIX = """You are applying one bounded, incremental repair to a CAD model that already executed successfully but failed a grounded engineering requirement check.
Return JSON only with the usual CAD plan schema: {{"calls":[{{"tool":"...","arguments":{{...}},"bind":"optional"}}],"note":"..."}}.
Use only TOOL_CATALOG. CURRENT_STATE is the live, already-built model and MUST be preserved except for the smallest changes required to fix GROUNDED_FAILURES.
Do not recreate, reset, delete, or replace geometry that already satisfies the request. Do not change dimensions/counts that are not listed as failed.
RAW_USER_REQUEST is the source of truth. ENGINEERING_CONTRACT is a source-grounded checklist, not permission to invent requirements.
Every GROUNDED_FAILURE has already passed a deterministic evidence gate: the evidence path resolved in the actual post-execution CAD state and contradicted a hard explicit requirement.
Prefer editing the responsible feature/parameter or appending the missing operation over rebuilding the part. Preserve persistent topology and existing selections whenever possible.
If a safe incremental correction cannot be expressed with TOOL_CATALOG, return the smallest no-op-safe plan only if one exists; never guess geometry.

TOOL_CATALOG:
{tool_catalog}

CURRENT_STATE:
{current_state}

RAW_USER_REQUEST:
{user_request}

ENGINEERING_CONTRACT:
{engineering_contract}

VERIFICATION_AUDIT:
{audit}

GROUNDED_FAILURES:
{grounded_failures}
"""



class NemotronAugmentedCodexAgent(DesktopParityCodexAgent):
    """Codex remains the CAD planner; Nemotron acts as a bounded engineering supervisor.

    Nemotron extracts source-grounded requirements, diagnoses concrete kernel failures,
    and audits successful results against read-only runtime evidence. It never executes
    CAD tools itself. Automatic post-success repair requires a separate deterministic
    evidence gate; all Nemotron failures are fail-open to the proven Codex path.
    """

    def __init__(
        self,
        client: AppServerClient,
        target_id: str,
        project_root: Path,
        model: str = "gpt-5.6-sol",
        reasoning: str = "medium",
        nemotron: NemotronOrchestrator | None = None,
    ) -> None:
        super().__init__(client=client, target_id=target_id, project_root=project_root, model=model, reasoning=reasoning)
        self.nemotron = nemotron or NemotronOrchestrator()
        self._engineering_contract: dict[str, Any] = {}
        self._recovery_cache: dict[str, dict[str, Any]] = {}
        self._last_verification: dict[str, Any] = {}
        self._hybrid_profile: dict[str, Any] = {}

    @property
    def nemotron_enabled(self) -> bool:
        return bool(self.nemotron.configured)

    def should_audit_success(self) -> bool:
        if not self.nemotron_enabled or not verified_hybrid_enabled():
            return False
        if not self._hybrid_profile:
            return True
        return bool(self._hybrid_profile.get("post_audit", True))

    def _contract_prompt(self, user_prompt: str) -> str:
        if not self._engineering_contract:
            return user_prompt
        safe_contract = contract_for_planner(self._engineering_contract)
        if not safe_contract.get("requirements") and not safe_contract.get("uncertainties"):
            return user_prompt
        return (
            user_prompt.rstrip()
            + "\n\nNEMOTRON_REQUIREMENT_LEDGER (advisory checklist; RAW USER REQUEST ABOVE IS THE SOURCE OF TRUTH):\n"
            + _compact_json(safe_contract, 12000)
            + "\nUse only source-supported rows as a completeness checklist. If any ledger row conflicts with the raw request, ignore the ledger row. "
              "Do not let this checklist narrow CADia's tool catalog or replace your own CAD reasoning."
        )

    def _get_recovery_guidance(
        self,
        *,
        user_prompt: str,
        current_state: dict[str, Any],
        failed_plan: Any,
        error: str,
        recovery_context: dict[str, Any] | None,
        on_event=None,
    ) -> dict[str, Any]:
        if not self.nemotron_enabled:
            return {}
        key = _compact_json({"failed": failed_plan, "error": error}, 18000)
        cached = self._recovery_cache.get(key)
        if cached is not None:
            return cached
        try:
            guidance = self.nemotron.recovery_guidance(
                user_prompt=raw_user_request_text(user_prompt),
                current_state=current_state,
                failed_plan=failed_plan,
                error=error,
                contract=self._engineering_contract,
                recovery_context=recovery_context,
                on_event=on_event,
            )
            if isinstance(guidance, dict) and guidance:
                self._recovery_cache[key] = guidance
                return guidance
        except Exception as exc:
            if on_event:
                on_event("progress", {"message": f"Nemotron unavailable for recovery; preserving Codex recovery ({type(exc).__name__})", "percent": 74})
        return {}

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None) -> dict[str, Any]:
        self._engineering_contract = {}
        self._recovery_cache.clear()
        self._last_verification = {}
        self._hybrid_profile = hybrid_request_profile(user_prompt, current_state)
        # Exact viewport geometry is stronger evidence than another language-model
        # interpretation. Preserve the user's ordered multi-face selection and route it
        # directly to the general Loft feature. Kernel-failure diagnosis remains available.
        selected = selected_geometry_plan(user_prompt, current_state)
        if selected is not None:
            self._hybrid_profile = dict(self._hybrid_profile, tier="simple", pre_contract=False, post_audit=False, combined_post_audit=False, reason="deterministic_selected_geometry_plan")
            if on_event:
                on_event("progress", {"message": "Selected profiles captured · creating Loft feature…", "percent": 5})
            return selected

        use_pre_contract = bool(self._hybrid_profile.get("pre_contract"))
        if self.nemotron_enabled and use_pre_contract:
            try:
                contract = self.nemotron.engineering_contract(raw_user_request_text(user_prompt), current_state, on_event=on_event)
                if isinstance(contract, dict) and contract:
                    self._engineering_contract = contract
                    if on_event:
                        on_event("progress", {"message": "Nemotron contract ready · Codex generating executable CAD plan…", "percent": 5})
            except Exception as exc:
                if on_event:
                    on_event("progress", {"message": f"Nemotron unavailable; continuing with proven Codex planner ({type(exc).__name__})", "percent": 5})
        elif self.nemotron_enabled and on_event:
            tier = str(self._hybrid_profile.get("tier") or "standard")
            if tier == "simple":
                on_event("progress", {"message": "Hybrid fast path · Codex planning directly; Nemotron remains available on failure", "percent": 3})
            else:
                on_event("progress", {"message": "Hybrid balanced path · Codex planning directly; one Nemotron verification pass follows success", "percent": 3})

        return super().plan(self._contract_prompt(user_prompt), current_state, on_event=on_event)

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
        context = {
            "completed_prefix": completed_prefix,
            "remaining_calls": remaining_calls,
            "mode": "kernel_closed_loop_continuation",
        }
        guidance = self._get_recovery_guidance(
            user_prompt=user_prompt,
            current_state=current_state,
            failed_plan=failed_call,
            error=error,
            recovery_context=context,
            on_event=on_event,
        )
        augmented_error = error
        if guidance:
            augmented_error += "\n\nNEMOTRON_ENGINEERING_RECOVERY_GUIDANCE:\n" + _compact_json(guidance, 7000)
        return super().continue_after_failure(
            user_prompt=user_prompt,
            current_state=current_state,
            completed_prefix=completed_prefix,
            failed_call=failed_call,
            remaining_calls=remaining_calls,
            error=augmented_error,
            on_event=on_event,
            progress_span=progress_span,
        )

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
        guidance = self._get_recovery_guidance(
            user_prompt=user_prompt,
            current_state=current_state,
            failed_plan=failed_plan,
            error=error,
            recovery_context=recovery_context,
            on_event=on_event,
        )
        merged = dict(recovery_context or {})
        if self._engineering_contract:
            merged["nemotron_engineering_contract"] = self._engineering_contract
        if guidance:
            merged["nemotron_recovery_guidance"] = guidance
        return super().repair(
            user_prompt=user_prompt,
            current_state=current_state,
            failed_plan=failed_plan,
            error=error,
            on_event=on_event,
            recovery_context=merged,
            progress_span=progress_span,
        )

    def audit_success(
        self,
        *,
        user_prompt: str,
        engine: Any,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        executed_plan: dict[str, Any],
        execution_results: list[dict[str, Any]] | None,
        on_event=None,
        recheck: bool = False,
    ) -> dict[str, Any]:
        if not self.nemotron_enabled or not verified_hybrid_enabled():
            return {}
        # Balanced mode deliberately spends no Nemotron success-path latency on only
        # the safest obvious one-step requests. Any kernel failure still invokes the
        # normal Nemotron diagnosis path. A re-check is never skipped after a repair.
        if not recheck and self._hybrid_profile and not self._hybrid_profile.get("post_audit", True):
            return {}
        try:
            evidence = build_verification_evidence(
                engine=engine,
                before_state=before_state,
                after_state=after_state,
                executed_plan=executed_plan,
                execution_results=execution_results,
            )

            # Ordinary requests use one combined extraction+audit call after execution,
            # replacing the two sequential Nemotron calls in v4.0.1. Complex requests
            # keep the pre-plan ledger and therefore use the normal audit method.
            if not self._engineering_contract and not recheck:
                bundle = self.nemotron.result_audit_with_contract(
                    user_prompt=raw_user_request_text(user_prompt),
                    evidence=evidence,
                    on_event=on_event,
                )
                if isinstance(bundle, dict) and isinstance(bundle.get("contract"), dict):
                    self._engineering_contract = bundle["contract"]
                audit = bundle.get("audit") if isinstance(bundle, dict) and isinstance(bundle.get("audit"), dict) else {}
            else:
                if not self._engineering_contract:
                    return {}
                audit = self.nemotron.result_audit(
                    user_prompt=raw_user_request_text(user_prompt),
                    contract=self._engineering_contract,
                    evidence=evidence,
                    on_event=on_event,
                    recheck=recheck,
                )

            supported = [r for r in self._engineering_contract.get("requirements", []) if isinstance(r, dict) and r.get("source_supported")]
            if not supported:
                return {}
            gate = evaluate_auto_repair_gate(audit, self._engineering_contract, evidence)
            result = {
                "audit": audit if isinstance(audit, dict) else {},
                "gate": gate,
                "auto_repair_enabled": bool(verified_hybrid_auto_repair_enabled()),
                "source_grounded_requirement_count": len(supported),
                "hybrid_profile": {k: self._hybrid_profile.get(k) for k in ("mode", "tier", "reason") if self._hybrid_profile.get(k) is not None},
            }
            self._last_verification = result
            if on_event:
                verdict = str((audit or {}).get("verdict") or "uncertain")
                if gate.get("allow") and verified_hybrid_auto_repair_enabled() and not recheck:
                    on_event("progress", {"message": "Engineering verification found a grounded mismatch · preparing one local repair…", "percent": 93})
                elif verdict == "pass":
                    on_event("progress", {"message": "Engineering requirement verification passed", "percent": (98 if recheck else 97)})
                elif verdict == "uncertain":
                    on_event("progress", {"message": "Engineering verification found no grounded failure; keeping the successful CAD result", "percent": (98 if recheck else 97)})
            return result
        except Exception as exc:
            self._last_verification = {"audit": {}, "gate": {"allow": False, "reason": f"verification_unavailable:{type(exc).__name__}"}}
            if on_event:
                on_event("progress", {"message": f"Nemotron verification unavailable; preserving successful Codex result ({type(exc).__name__})", "percent": (98 if recheck else 97)})
            return self._last_verification

    def repair_after_verification(
        self,
        *,
        user_prompt: str,
        current_state: dict[str, Any],
        verification: dict[str, Any],
        on_event=None,
        progress_span: tuple[int, int] = (93, 95),
    ) -> dict[str, Any]:
        gate = verification.get("gate") if isinstance(verification, dict) else {}
        if not isinstance(gate, dict) or not gate.get("allow") or not verified_hybrid_auto_repair_enabled():
            return {}
        prompt = POST_VERIFICATION_REPAIR_PREFIX.format(
            tool_catalog=json.dumps(self._catalog, ensure_ascii=False, separators=(",", ":")),
            current_state=json.dumps(current_state, ensure_ascii=False, separators=(",", ":"), default=str),
            user_request=raw_user_request_text(user_prompt),
            engineering_contract=_compact_json(contract_for_planner(self._engineering_contract), 12000),
            audit=_compact_json(verification.get("audit") or {}, 10000),
            grounded_failures=_compact_json(gate.get("grounded_failures") or [], 8000),
        )
        if on_event:
            on_event("progress", {"message": "Codex · applying one evidence-grounded local repair…", "percent": progress_span[0]})
        planned = self._run_planner(prompt, on_event=on_event, progress_span=progress_span)
        return _validate_post_repair_safety(_validate_planner_plan(planned, self._catalog))

    @property
    def last_verification(self) -> dict[str, Any]:
        return self._last_verification

    def cancel(self) -> None:
        try:
            self.nemotron.cancel()
        finally:
            super().cancel()
