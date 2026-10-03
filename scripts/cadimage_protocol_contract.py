from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "web/backend/standalonecad_web/visual_cad.py"
PROTO = ROOT / "web/backend/standalonecad_web/cadimage_protocol.py"


def need(text: str, token: str, where: str) -> None:
    if token not in text:
        raise AssertionError(f"missing {token!r} in {where}")


def forbid(text: str, token: str, where: str) -> None:
    if token in text:
        raise AssertionError(f"forbidden {token!r} in {where}")


def main() -> None:
    v = VISUAL.read_text()
    p = PROTO.read_text()
    ast.parse(v); ast.parse(p)

    # Exact public-protocol lineage + bounded production adaptation.
    need(p, 'UPSTREAM_REPO = "skykim/cad-image-to-3d"', 'cadimage_protocol.py')
    need(p, 'MAX_VISUAL_LOOPS = 1', 'cadimage_protocol.py')
    need(p, 'MAX_AUDIT_REPAIR = 1', 'cadimage_protocol.py')
    need(p, 'def prepare_semantic_inputs(', 'cadimage_protocol.py')
    need(p, 'def circular_marks(', 'cadimage_protocol.py')
    need(p, 'def pick_best_candidate(', 'cadimage_protocol.py')
    need(p, 'def needs_targeted_refinement(', 'cadimage_protocol.py')
    need(p, 'def needs_feature_audit(', 'cadimage_protocol.py')

    # Fix the actual server error: selected normal-chat provider cannot gate Drawing->CAD.
    forbid(v, 'selected_provider_name', 'visual_cad.py')
    forbid(v, 'currently requires the ChatGPT / Codex provider', 'visual_cad.py')
    need(v, 'client = app_server_manager.get(job.user_id)', 'visual_cad.py')

    # Preserve current server reliability machinery that the rejected v4.1 payload would have overwritten.
    need(v, 'VISUAL_JOB_HARD_TIMEOUT_SECONDS', 'visual_cad.py')
    need(v, 'def _watchdog(', 'visual_cad.py')
    need(v, 'future.add_done_callback', 'visual_cad.py')
    need(v, 'VISUAL_TERMINAL_STATUSES', 'visual_cad.py')

    # OpenCV contributes to the first semantic pass without an extra AI/CAD turn.
    need(v, 'cadimage_prepare_semantic_inputs(', 'visual_cad.py')
    need(v, 'semantic_images', 'visual_cad.py')
    need(v, 'notes=extraction_notes', 'visual_cad.py')
    need(v, 'OPENCV_RESCUE_ENABLED = False', 'visual_cad.py')

    # Fast adaptive port: no fixed loop1+loop2; one targeted edit max + one audit edit max.
    need(v, 'capture_candidate("exec0"', 'visual_cad.py')
    need(v, 'cadimage_needs_targeted_refinement(', 'visual_cad.py')
    need(v, 'opencv_feedback=exec0_verification', 'visual_cad.py')
    need(v, 'skip_visual_refinement', 'visual_cad.py')
    need(v, 'cadimage_needs_feature_audit(', 'visual_cad.py')
    need(v, 'audit_features_from_views(', 'visual_cad.py')
    need(v, 'cadimage_pick_best_candidate(candidates)', 'visual_cad.py')
    need(v, 'max_ai_repairs=0', 'visual_cad.py')
    forbid(v, 'for pass_index in range(CADIMAGE_MAX_VISUAL_LOOPS)', 'visual_cad.py')

    # Feature audit images must be staged into the isolated Codex workspace.
    need(v, 'self._stage_images_for_codex(all_images, purpose="feature-audit")', 'visual_cad.py')

    # Do not perform the old second OpenCV postcheck after protocol verification already finished.
    need(v, 'queue_postcheck = not (isinstance(inspection, dict) and inspection.get("status") == "done")', 'visual_cad.py')
    need(v, 'if queue_postcheck:', 'visual_cad.py')

    # Keep existing CADia modeling/recovery surface unchanged.
    need(v, 'for replan_index in range(1, 3):', 'visual_cad.py')
    need(v, 'MAX_REFINEMENTS = 0', 'visual_cad.py')
    need(v, 'DRAWING_NVIDIA_REVIEW_ENABLED = False', 'visual_cad.py')
    need(v, 'execute_with_failure_recovery(', 'visual_cad.py')

    print('CADia v4.1.3 current-base adaptive contract: PASS')


if __name__ == '__main__':
    main()
