from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .vendor.drawing2cad.cad_generator import _strip_fences, run_code, spec_to_code
from .vendor.drawing2cad.partspec import PartSpec
from .vendor.drawing2cad.prompts import EXTRACTION_PROMPT, GEN_FROM_DRAWING_SYSTEM
from .vendor.drawing2cad.validator import ValidationReport, validate


UPSTREAM_REPO = "D4V1ND/tech_europe"
UPSTREAM_COMMIT = "9d7026d6441153032b4bc1d93050bfd1d825296a"
DEFAULT_THRESHOLD = 0.90
DEFAULT_TOL_MM = 0.5
MAX_ATTEMPTS = 4  # Same default as drawing2cad.llm_codegen.repair_loop.


@dataclass
class ReconstructionCandidate:
    code: str
    result: Any
    report: ValidationReport
    score: float
    attempt: int
    output_dir: Path
    engine: str


def report_score(report: ValidationReport | None) -> float:
    if not report or not report.checks:
        return 0.0
    return sum(1 for c in report.checks if c.passed) / len(report.checks)


def validation_payload(report: ValidationReport | None) -> dict[str, Any]:
    if report is None:
        return {"score": 0.0, "passed": False, "checks": [], "advisory": []}
    return {
        "score": report_score(report),
        "passed": report.ok(),
        "checks": [row.model_dump() for row in report.checks],
        "advisory": [row.model_dump() for row in report.advisory],
        "feedback": report.feedback(),
    }


def extract_partspec(
    client,
    images: list[Path],
    *,
    model: str,
    effort: str,
    notes: str = "",
    dimension_hint: str = "",
    on_event: Callable[[str, Any], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PartSpec:
    prompt = EXTRACTION_PROMPT
    if notes or dimension_hint:
        prompt += (
            "\n\nUSER-SUPPLIED CONTEXT (use only as additional drawing context):\n"
            f"Notes: {notes or '(none)'}\n"
            f"Known dimension / scale reference: {dimension_hint or '(none)'}\n"
        )
    # Codex app-server structured output is constrained with a tiny transport schema.
    # The inner JSON is validated by the exact upstream Pydantic PartSpec model.  This
    # avoids changing the upstream discriminated-union schema just to satisfy transport.
    import json
    prompt += (
        "\n\nReturn the COMPLETE PartSpec as a JSON string in the single field `partspec_json`. "
        "The JSON inside that string MUST validate against this exact PartSpec JSON Schema:\n"
        + json.dumps(PartSpec.model_json_schema(), ensure_ascii=False)
    )
    raw = client.complete_json(
        prompt,
        output_schema={
            "type": "object",
            "properties": {"partspec_json": {"type": "string"}},
            "required": ["partspec_json"],
            "additionalProperties": False,
        },
        model=model,
        effort=effort,
        on_event=on_event,
        local_images=images,
        cancel_check=cancel_check,
        stall_timeout_seconds=180,
        hard_timeout_seconds=900,
    )
    return PartSpec.model_validate_json(str(raw.get("partspec_json") or ""))


def _repair_code(
    client,
    *,
    spec: PartSpec,
    code: str,
    feedback: str,
    model: str,
    effort: str,
    images: list[Path] | None,
    on_event: Callable[[str, Any], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> str:
    # This is the same information flow as upstream drawing2cad.llm_codegen._repair_code;
    # complete_json is only a transport adapter because CADia uses Codex app-server rather
    # than LangChain's model.invoke().
    prompt = (
        GEN_FROM_DRAWING_SYSTEM
        + "\n\nPartSpec (ground truth — your output must match these dimensions):\n"
        + spec.model_dump_json(indent=2)
        + "\n\nCurrent CadQuery code (fix it, keep what worked):\n"
        + code
        + "\n\nProblems to fix:\n"
        + feedback
        + "\n\nReturn JSON with one field named code containing the full corrected CadQuery script."
    )
    raw = client.complete_json(
        prompt,
        output_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        model=model,
        effort=effort,
        on_event=on_event,
        local_images=images or [],
        cancel_check=cancel_check,
        stall_timeout_seconds=180,
        hard_timeout_seconds=900,
    )
    return _strip_fences(str(raw.get("code") or ""))


def reconstruct(
    client,
    *,
    spec: PartSpec,
    images: list[Path],
    out_dir: Path,
    model: str,
    effort: str,
    threshold: float = DEFAULT_THRESHOLD,
    tol: float = DEFAULT_TOL_MM,
    max_attempts: int = MAX_ATTEMPTS,
    on_event: Callable[[str, Any], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[ReconstructionCandidate, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = spec.sanity_check()
    if errors:
        raise ValueError("PartSpec sanity check failed: " + "; ".join(errors))

    code = spec_to_code(spec)
    feedback = ""
    best: ReconstructionCandidate | None = None
    notes: list[str] = []

    for attempt in range(1, max_attempts + 1):
        if cancel_check and cancel_check():
            raise InterruptedError("Drawing reconstruction was cancelled.")

        if attempt > 1:
            if on_event:
                on_event("progress", {"message": f"Repairing CadQuery from validation feedback ({attempt}/{max_attempts})…"})
            code = _repair_code(
                client,
                spec=spec,
                code=code,
                feedback=feedback,
                model=model,
                effort=effort,
                # Upstream attaches the drawing only on the first repair call.
                images=images if attempt == 2 else None,
                on_event=on_event,
                cancel_check=cancel_check,
            )

        attempt_dir = out_dir / f"attempt-{attempt}"
        try:
            result = run_code(code, attempt_dir, spec=spec)
        except Exception:
            feedback = "The code raised an exception:\n" + traceback.format_exc(limit=6)
            notes.append(f"attempt {attempt}: execution error")
            continue

        try:
            if hasattr(result, "val") and hasattr(result.val(), "isValid") and not bool(result.val().isValid()):
                raise RuntimeError("CadQuery produced an invalid B-Rep")
            report = validate(spec, result, tol=tol)
        except Exception:
            feedback = "The build executed but validation raised an exception:\n" + traceback.format_exc(limit=6)
            notes.append(f"attempt {attempt}: validator error")
            continue

        score = report_score(report)
        candidate = ReconstructionCandidate(
            code=code,
            result=result,
            report=report,
            score=score,
            attempt=attempt,
            output_dir=attempt_dir,
            engine="deterministic" if attempt == 1 else "llm_repair",
        )
        if best is None or candidate.score >= best.score:
            best = candidate

        if score >= threshold:
            notes.append(f"attempt {attempt}: validated at {score:.0%}")
            break

        feedback = (
            "The build ran but failed these validator checks — fix the geometry:\n"
            + report.feedback()
        )
        notes.append(f"attempt {attempt}: validation {score:.0%}")

    if best is None:
        raise RuntimeError("Drawing2CAD could not produce any executable CAD candidate after repair attempts.")

    return best, notes
