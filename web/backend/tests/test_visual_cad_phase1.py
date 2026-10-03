from __future__ import annotations

from pathlib import Path

from standalonecad_web.cad_runtime import runtime_manager
from standalonecad_web.drawing2cad_upstream import extract_partspec, reconstruct, validation_payload
from standalonecad_web.vendor.drawing2cad.cad_generator import spec_to_code
from standalonecad_web.vendor.drawing2cad.partspec import PartSpec
from standalonecad_web.vendor.drawing2cad.validator import validate
from standalonecad_web.visual_cad import VisualCadJob, VisualCadJobManager


def _plate_spec() -> PartSpec:
    return PartSpec.model_validate({
        "units": "mm",
        "geometry": {
            "kind": "extruded",
            "profile_kind": "rectangle",
            "width": 120,
            "height": 70,
            "thickness": 20,
        },
        "holes": [
            {"id": "corner_a", "x": 10, "y": 10, "diameter": 8, "hole_type": "through"},
            {"id": "corner_b", "x": 110, "y": 60, "diameter": 8, "hole_type": "through"},
            {
                "id": "center_cbore", "x": 60, "y": 35, "diameter": 12,
                "hole_type": "counterbore", "counterbore_diameter": 20,
                "counterbore_depth": 5,
            },
        ],
        "cuts": [],
        "fillets": [],
        "chamfers": [],
        "dimensions": [
            {"name": "overall_width", "nominal": 120, "tol_plus": 0.1, "tol_minus": 0.1},
            {"name": "overall_height", "nominal": 70, "tol_plus": 0.1, "tol_minus": 0.1},
            {"name": "overall_depth", "nominal": 20, "tol_plus": 0.1, "tol_minus": 0.1},
        ],
        "notes": "test plate",
    })


class _NoRepairClient:
    def complete_json(self, *args, **kwargs):
        raise AssertionError("deterministic validated build should not call the model repair path")


class _ExtractionClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.kwargs = None

    def complete_json(self, prompt, **kwargs):
        self.kwargs = kwargs
        return {"partspec_json": self.payload}


def test_vendored_generator_cuts_holes_without_cadia_hole_tool(tmp_path):
    spec = _plate_spec()
    code = spec_to_code(spec)
    assert "result.cut(" in code
    assert ".hole(" not in code
    candidate, notes = reconstruct(
        _NoRepairClient(), spec=spec, images=[], out_dir=tmp_path,
        model="test", effort="high",
    )
    assert candidate.result.val().isValid()
    assert candidate.score == 1.0
    assert candidate.attempt == 1
    assert any("validated" in note for note in notes)
    payload = validation_payload(candidate.report)
    assert payload["passed"] is True
    assert all(row["passed"] for row in payload["checks"])


def test_vendored_validator_confirms_hole_voids(tmp_path):
    spec = _plate_spec()
    candidate, _ = reconstruct(
        _NoRepairClient(), spec=spec, images=[], out_dir=tmp_path,
        model="test", effort="high",
    )
    report = validate(spec, candidate.result)
    holes = [row for row in report.checks if row.name.endswith("_present")]
    assert len(holes) == 3
    assert all(row.passed for row in holes)


def test_extraction_transport_validates_exact_upstream_partspec(tmp_path):
    spec = _plate_spec()
    client = _ExtractionClient(spec.model_dump_json())
    image = tmp_path / "drawing.png"
    image.write_bytes(b"not-read-by-fake-client")
    parsed = extract_partspec(
        client, [image], model="test", effort="high", notes="", dimension_hint=""
    )
    assert parsed == spec
    assert client.kwargs is not None
    schema = client.kwargs["output_schema"]
    assert list(schema["properties"]) == ["partspec_json"]


def test_visual_job_progress_is_monotonic_and_restart_state_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_manager, "project_root", lambda user_id, project_id: tmp_path / user_id / project_id)
    manager = VisualCadJobManager()
    try:
        job = VisualCadJob(id="job-1", user_id="user-1", project_id="project-1", input_files={})
        manager._update(job, status="running", progress=72, message="CAD built")
        manager._update(job, progress=42, message="validation")
        assert job.progress == 72
        loaded = manager._load_persisted("job-1", "user-1", "project-1")
        assert loaded is not None
        assert loaded.status == "failed"
        assert "server restart" in (loaded.error or "").lower()
    finally:
        manager._pool.shutdown(wait=False, cancel_futures=True)


def test_visual_pipeline_no_longer_replays_generation_through_cadia_inventor_hole():
    source = Path(__file__).parents[1] / "standalonecad_web" / "visual_cad.py"
    text = source.read_text(encoding="utf-8")
    assert "inventor_hole" not in text
    assert "compile_to_cadia_plan" not in text
    assert "reconstruct_upstream" in text
    assert "runtime.import_document(step_path)" in text
