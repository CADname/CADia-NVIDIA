from __future__ import annotations

import json
from typing import Any
from .web_agent import APICompatibleAgent
from .nemotron_orchestrator import NemotronOrchestrator
from .engineering_verification import (
    build_verification_evidence, contract_for_planner, evaluate_auto_repair_gate,
    hybrid_request_profile, raw_user_request_text, verified_hybrid_enabled,
)


class NemotronAugmentedAPIAgent(APICompatibleAgent):
    """OpenAI/API Codex-compatible planner + bounded Nemotron verification.

    Does NOT claim to invoke a Codex CLI for API credentials. Uses the same
    Codex CAD planning schema / validated CAD executor but API model transport.
    Missing Nemotron credentials preserve the primary API planner.
    """
    def __init__(self, provider, model=None, reasoning=None, nemotron=None):
        super().__init__(provider, model=model, reasoning=reasoning)
        self.nemotron = nemotron or NemotronOrchestrator()
        self._contract = {}
        self._hybrid_profile = {}

    def plan(self, user_prompt: str, current_state: dict[str, Any], on_event=None):
        self._contract = {}
        self._hybrid_profile = hybrid_request_profile(user_prompt, current_state)
        if self.nemotron.configured and self._hybrid_profile.get('pre_contract'):
            try:
                ledger = self.nemotron.engineering_contract(raw_user_request_text(user_prompt), current_state, on_event=on_event)
                if isinstance(ledger, dict):
                    self._contract = ledger
            except Exception:
                # Advisory service failure must not corrupt or block valid CAD.
                pass
        if self._contract:
            verified = contract_for_planner(self._contract)
            if verified.get('requirements') or verified.get('uncertainties'):
                user_prompt += ("\n\nNEMOTRON_REQUIREMENT_LEDGER (advisory, raw user request wins):\n"
                                + json.dumps(verified, ensure_ascii=False)[:12000])
        return super().plan(user_prompt, current_state, on_event=on_event)

    def repair(self, user_prompt, current_state, failed_plan, error, on_event=None,
               recovery_context=None, progress_span=(72, 78)):
        guidance = {}
        if self.nemotron.configured:
            try:
                guidance = self.nemotron.recovery_guidance(
                    user_prompt=raw_user_request_text(user_prompt), current_state=current_state,
                    failed_plan=failed_plan, error=error, contract=self._contract,
                    recovery_context=recovery_context, on_event=on_event) or {}
            except Exception:
                pass
        merged = dict(recovery_context or {})
        if isinstance(guidance, dict) and guidance:
            merged['nemotron_recovery_guidance'] = guidance
        return super().repair(user_prompt, current_state, failed_plan, error,
                              on_event=on_event, recovery_context=merged,
                              progress_span=progress_span)

    def should_audit_success(self) -> bool:
        return (bool(self.nemotron.configured) and verified_hybrid_enabled()
                and bool(self._hybrid_profile.get('post_audit', True)))

    def audit_success(self, *, user_prompt, engine, before_state, after_state,
                      executed_plan, execution_results, on_event=None, recheck=False):
        if not self.should_audit_success():
            return {}
        try:
            evidence = build_verification_evidence(
                engine=engine, before_state=before_state, after_state=after_state,
                executed_plan=executed_plan, execution_results=execution_results)
            if not self._contract:
                bundle = self.nemotron.result_audit_with_contract(
                    user_prompt=raw_user_request_text(user_prompt), evidence=evidence,
                    on_event=on_event)
                if isinstance(bundle, dict):
                    self._contract = bundle.get('contract') or {}
                    audit = bundle.get('audit') or {}
                else:
                    return {}
            else:
                audit = self.nemotron.result_audit(
                    user_prompt=raw_user_request_text(user_prompt), contract=self._contract,
                    evidence=evidence, on_event=on_event, recheck=recheck)
            gate = evaluate_auto_repair_gate(audit, self._contract, evidence)
            # API variant reports advisory result only. Never auto-change a
            # successful CAD model based on an untrusted model verdict.
            return {'audit': audit, 'gate': dict(gate, allow=False),
                    'auto_repair_enabled': False,
                    'source_grounded_requirement_count': sum(
                        1 for r in self._contract.get('requirements', [])
                        if isinstance(r, dict) and r.get('source_supported'))}
        except Exception:
            return {}  # fail open to the proven API planner

    def cancel(self):
        try: self.nemotron.cancel()
        finally: return super().cancel()
