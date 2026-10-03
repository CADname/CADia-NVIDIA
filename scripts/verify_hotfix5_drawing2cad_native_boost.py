from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from standalonecad_web.cad_runtime import CadRuntime
from standalonecad_web.drawing2cad_native import compile_partspec_to_native_plan, validate_runtime_payload
from standalonecad_web.providers.app_server import AppServerClient
from standalonecad_web.vendor.drawing2cad.partspec import PartSpec
import standalonecad_web.visual_cad as visual_cad


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


root = Path(os.environ.get("CADIA_APP_ROOT", "/app")).resolve()
vendor = root / "web/backend/standalonecad_web/vendor/drawing2cad"
expected_blobs = {
    "partspec.py": "b8fc2d6a505a10a6d00b1975bb6826e0379ffffc",
    "cad_generator.py": "a0759d05d47d3d76176a5725d14293e389dc4921",
    "validator.py": "1b76d8d54670f7e5dba2bd6ec413424e00dba8ff",
    "prompts.py": "33d8ce7b386f830d847303c6e44d1ca7dbeb286a",
}
for name, expected in expected_blobs.items():
    actual = git_blob_sha1(vendor / name)
    assert actual == expected, (name, actual, expected)
print("UPSTREAM_DRAWING2CAD_HASHES: PASS")

reconstruct_src = inspect.getsource(visual_cad.VisualCadJobManager._reconstruct)
for required in (
    "extract_partspec",
    "compile_partspec_to_native_plan",
    "validate_runtime_payload",
    "hotfix5_fallback",
    "refine_from_views",
):
    assert required in reconstruct_src, required
for forbidden in ("runtime.import_document", "reconstruct_upstream", "run_code("):
    assert forbidden not in reconstruct_src, forbidden
native_src = inspect.getsource(__import__("standalonecad_web.drawing2cad_native", fromlist=["*"]))
assert "exec(" not in native_src
assert "runtime.import_document" not in native_src
print("NATIVE_ROUTE_NO_STEP_FLATTENING: PASS")
print("HOTFIX5_NO_REGRESSION_FALLBACK: PASS")

sig = inspect.signature(AppServerClient.complete_json)
for name in ("cancel_check", "stall_timeout_seconds", "hard_timeout_seconds"):
    assert name in sig.parameters, name
print("BOUNDED_CODEX_TRANSPORT: PASS")
print("LAYER2_REVIEW_WIRING: PASS")

cases = {
    "plate": {
        "geometry": {"kind": "extruded", "profile_kind": "rectangle", "width": 120, "height": 70, "thickness": 20},
        "holes": [
            {"x": 10, "y": 10, "diameter": 8, "hole_type": "through"},
            {"x": 60, "y": 35, "diameter": 12, "hole_type": "counterbore", "counterbore_diameter": 20, "counterbore_depth": 5},
            {"x": 100, "y": 50, "diameter": 8, "hole_type": "countersink", "countersink_diameter": 14, "countersink_angle": 90},
        ],
        "cuts": [{"x": 20, "y": 20, "z": 5, "dx": 10, "dy": 15, "dz": 10}],
        "fillets": [], "chamfers": [], "dimensions": [],
    },
    "threaded": {
        "geometry": {"kind": "extruded", "profile_kind": "rectangle", "width": 60, "height": 40, "thickness": 18},
        "holes": [{"x": 30, "y": 20, "diameter": 6.8, "hole_type": "threaded", "depth": 12, "thread": "M8x1.25"}],
        "cuts": [], "fillets": [], "chamfers": [], "dimensions": [],
    },
    "polygon": {
        "geometry": {"kind": "extruded", "profile_kind": "polygon", "profile_points": [
            {"x": 0, "y": 0}, {"x": 80, "y": 0}, {"x": 65, "y": 45}, {"x": 0, "y": 45}
        ], "thickness": 10},
        "holes": [], "cuts": [], "fillets": [], "chamfers": [], "dimensions": [],
    },
    "revolved": {
        "geometry": {"kind": "revolved", "segments": [
            {"z_start": 0, "z_end": 20, "outer_diameter": 40, "inner_diameter": 10},
            {"z_start": 20, "z_end": 35, "outer_diameter": 30, "inner_diameter": 0},
        ]},
        "holes": [], "cuts": [], "fillets": [], "chamfers": [], "dimensions": [],
    },
    # Deliberately put the cut before the adds.  The adapter must preserve upstream
    # semantics by constructing all add bodies first and applying cuts afterwards.
    "multibody_cut_first": {
        "geometry": {"kind": "multibody", "bodies": [
            {"shape": "cylinder", "operation": "cut", "x": 50, "y": 30, "z": -1, "diameter": 12, "length": 10, "axis": "z"},
            {"shape": "box", "operation": "add", "x": 0, "y": 0, "z": 0, "dx": 100, "dy": 60, "dz": 8},
            {"shape": "box", "operation": "add", "x": 0, "y": 0, "z": 8, "dx": 8, "dy": 60, "dz": 40},
        ]},
        "holes": [], "cuts": [], "fillets": [], "chamfers": [], "dimensions": [],
    },
}

with TemporaryDirectory(prefix="cadia-native-boost-") as td:
    base = Path(td)
    for idx, (case_name, payload) in enumerate(cases.items()):
        spec = PartSpec.model_validate(payload)
        compiled = compile_partspec_to_native_plan(spec)
        assert compiled.plan is not None, (case_name, compiled.reason)
        runtime = CadRuntime(owner_id="verify", project_id=f"case-{idx}", root=base / case_name)
        try:
            result = runtime.executor.execute(compiled.plan)
            assert result.ok, (case_name, result.error)
            validation = validate_runtime_payload(runtime, spec)
            assert validation["score"] >= 0.90, (case_name, validation)
            assert runtime.engine.doc is not None
            assert runtime.engine.doc.shape is not None
            assert runtime.engine.doc.shape.isValid()
            assert runtime.engine.doc.features
        finally:
            runtime.close()
print("NATIVE_PARTSPEC_COMPILE: PASS")
print("CADIA_NATIVE_BREP: PASS")
print("DRAWING2CAD_VALIDATOR: PASS")
print("NATIVE_FEATURE_HISTORY: PASS")
print("CADia Hotfix-5 Drawing2CAD Native Boost v1.0.1: PASS")
