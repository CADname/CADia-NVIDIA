#!/usr/bin/env python3
"""Static regression guards for CADia OpenCV Assist FastPath v1.3.0.

No network/model calls. Fails if a future edit accidentally puts OpenCV or NVIDIA
back on the successful Drawing-to-CAD critical path.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "web/backend/standalonecad_web/visual_cad.py"
APP_SERVER = ROOT / "web/backend/standalonecad_web/providers/app_server.py"


def _func(tree: ast.AST, cls: str, name: str) -> ast.FunctionDef:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"missing {cls}.{name}")


def main() -> int:
    text = VISUAL.read_text(encoding="utf-8")
    tree = ast.parse(text)
    reconstruct = _func(tree, "VisualCadJobManager", "_reconstruct")
    run = _func(tree, "VisualCadJobManager", "_run")

    # OpenCV analysis is allowed only inside the nested failure-only rescue helper.
    direct_calls: list[tuple[int, str]] = []
    nested_names: list[str] = []
    for node in reconstruct.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_names.append(node.name)
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                fn = child.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == "opencv_analyze_images":
                    direct_calls.append((child.lineno, name))
    if direct_calls:
        raise AssertionError(f"OpenCV leaked onto successful _reconstruct path: {direct_calls}")
    if "get_opencv_rescue_images" not in nested_names:
        raise AssertionError("failure-only OpenCV rescue helper missing")

    if "run_opencv_closed_loop" in text:
        raise AssertionError("legacy OpenCV auto-repair closed loop still present")
    if "opencv_prompt_summary" in text:
        raise AssertionError("OpenCV pixel evidence is still injected into initial semantic prompt")
    if "OPENCV_MAX_REPAIRS" in text:
        raise AssertionError("legacy automatic OpenCV repair count still exists")
    if "DRAWING_NVIDIA_REVIEW_ENABLED = False" not in text:
        raise AssertionError("NVIDIA reviewer is not hard-disabled on product/OpenCV Drawing path")

    # Modeling must become terminal before the async post-check is submitted.
    run_src = ast.get_source_segment(text, run) or ""
    done_pos = run_src.find('status="done"')
    submit_pos = run_src.find("_inspection_pool.submit")
    if done_pos < 0 or submit_pos < 0 or done_pos >= submit_pos:
        raise AssertionError("OpenCV inspection is not scheduled strictly after modeling reaches done")

    # Rescue must be objectively gated and failure-only.
    required = [
        'job.vision_mode == "opencv5"',
        'float(best_payload.get("score", 0.0)) < threshold',
        'required_score = max(threshold, baseline_score + OPENCV_RESCUE_MIN_DELTA)',
        'rescue_basic.get("brep_valid")',
    ]
    for token in required:
        if token not in text:
            raise AssertionError(f"missing rescue gate: {token}")

    app_server = APP_SERVER.read_text(encoding="utf-8")
    for token in ("features.plugins=false", "features.apps=false", '"app-server"'):
        if token not in app_server:
            raise AssertionError(f"Codex planner isolation missing: {token}")

    print("CADia OpenCV Assist FastPath contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
