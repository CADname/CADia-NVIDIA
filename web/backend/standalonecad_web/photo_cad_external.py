from __future__ import annotations

import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class ExternalEngineResult:
    name: str
    available: bool
    attempted: bool = False
    ok: bool = False
    message: str = ""
    guidance: str = ""
    reference_mesh: str | None = None
    output_dir: str | None = None
    evidence_dir: str | None = None
    baseline_mesh: str | None = None
    score: float | None = None

    def public(self) -> dict:
        # Do not leak server-local paths or generated programs to the browser.
        return {
            "name": self.name,
            "available": self.available,
            "attempted": self.attempted,
            "ok": self.ok,
            "message": self.message,
            "score": self.score,
        }


def _truthy(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _text(path: Path, limit: int = 26000) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(value) > limit:
        return value[:limit] + "\n...<truncated>"
    return value


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def _run(
    args: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    timeout: int,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> tuple[int, str]:
    """Run an optional external engine with bounded, cancellable lifetime.

    stdout is streamed to a temporary file so a verbose child cannot deadlock on a
    full PIPE buffer. The child is launched in its own process group so cancel or
    timeout also terminates subprocesses spawned by vLLM/COLMAP/etc.
    """
    started = time.time()
    local_deadline = started + max(1, int(timeout))
    effective_deadline = min(local_deadline, deadline_at) if deadline_at else local_deadline
    log_path: str | None = None
    proc: subprocess.Popen | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="cadia-photo-engine-", suffix=".log", delete=False) as log:
            log_path = log.name
            proc = subprocess.Popen(
                args,
                cwd=None if cwd is None else str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=False,
                start_new_session=(os.name == "posix"),
            )
            while proc.poll() is None:
                if cancel_check is not None and cancel_check():
                    _terminate_process(proc)
                    return 130, "Cancelled by CADia Photo-to-CAD job."
                if time.time() >= effective_deadline:
                    _terminate_process(proc)
                    return 124, f"Timed out after {max(1, int(time.time() - started))}s."
                time.sleep(0.25)
            code = int(proc.returncode or 0)
        output = ""
        if log_path:
            try:
                data = Path(log_path).read_bytes()[-16000:]
                output = data.decode("utf-8", errors="replace")
            except Exception:
                output = ""
        return code, output
    except Exception as exc:
        if proc is not None:
            _terminate_process(proc)
        return 125, f"Could not start engine: {exc}"
    finally:
        if log_path:
            try:
                Path(log_path).unlink(missing_ok=True)
            except Exception:
                pass


def _extract_numeric_score(payload) -> float | None:
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        for key in (
            "final_iou", "best_iou", "iou", "score", "mean_silhouette_iou",
            "worst_silhouette_iou",
        ):
            if key in payload:
                value = _extract_numeric_score(payload[key])
                if value is not None:
                    return value
        values = [_extract_numeric_score(v) for v in payload.values()]
        values = [v for v in values if v is not None]
        return max(values) if values else None
    if isinstance(payload, list):
        values = [_extract_numeric_score(v) for v in payload]
        values = [v for v in values if v is not None]
        return max(values) if values else None
    return None


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _agentic_python(executable: str | None) -> str:
    override = os.getenv("CADIA_AGENTIC_CAD_PYTHON", "").strip()
    if override:
        return override
    if executable:
        try:
            first = Path(executable).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if first.startswith("#!") and "python" in first.lower():
                parts = shlex.split(first[2:].strip())
                if parts:
                    # pip console scripts normally use an absolute venv Python.
                    # Also support portable '#!/usr/bin/env python3' launchers.
                    if Path(parts[0]).name == "env" and len(parts) >= 2:
                        resolved = shutil.which(parts[1])
                        if resolved:
                            return resolved
                    elif Path(parts[0]).is_file():
                        return parts[0]
        except Exception:
            pass
    return sys.executable


def capabilities() -> dict[str, dict]:
    agentic = os.getenv("CADIA_AGENTIC_CAD_BIN", "agentic-cad")
    cadcoder_root = Path(os.getenv("CADIA_CADCODER_ROOT", "")).expanduser() if os.getenv("CADIA_CADCODER_ROOT") else None
    cadfit_root = Path(os.getenv("CADIA_CADFIT_ROOT", "")).expanduser() if os.getenv("CADIA_CADFIT_ROOT") else None
    pixal_root = Path(os.getenv("CADIA_PIXAL3D_ROOT", "")).expanduser() if os.getenv("CADIA_PIXAL3D_ROOT") else None
    reasoner_root = Path(os.getenv("CADIA_CADREASONER_ROOT", "")).expanduser() if os.getenv("CADIA_CADREASONER_ROOT") else None
    cadena_root = Path(os.getenv("CADIA_CADENA_ROOT", "")).expanduser() if os.getenv("CADIA_CADENA_ROOT") else None
    cadena_model = Path(os.getenv("CADIA_CADENA_MODEL_PATH", "")).expanduser() if os.getenv("CADIA_CADENA_MODEL_PATH") else None
    return {
        "agentic_cad": {
            "available": bool(shutil.which(agentic)),
            "configured": _truthy("CADIA_PHOTO_AGENTIC_ENABLED", True),
            "detail": shutil.which(agentic) or agentic,
        },
        "cadcoder": {
            "available": bool(cadcoder_root and (cadcoder_root / "llava" / "eval" / "model_vqa_loader.py").is_file()),
            "configured": _truthy("CADIA_PHOTO_CADCODER_ENABLED", True),
            "detail": str(cadcoder_root or "CADIA_CADCODER_ROOT is not set"),
        },
        "cadfit": {
            "available": bool(cadfit_root and (cadfit_root / "run_pipeline.py").is_file()),
            "configured": _truthy("CADIA_PHOTO_CADFIT_ENABLED", True),
            "detail": str(cadfit_root or "CADIA_CADFIT_ROOT is not set"),
        },
        "pixal3d": {
            "available": bool(pixal_root and (pixal_root / "inference.py").is_file()),
            "configured": _truthy("CADIA_PHOTO_PIXAL3D_ENABLED", True),
            "detail": str(pixal_root or "CADIA_PIXAL3D_ROOT is not set"),
        },
        "cadreasoner": {
            "available": bool(reasoner_root and (reasoner_root / "test.py").is_file()),
            "configured": _truthy("CADIA_PHOTO_CADREASONER_ENABLED", True),
            "detail": str(reasoner_root or "CADIA_CADREASONER_ROOT is not set"),
        },
        "cadena": {
            "available": bool(
                cadena_root
                and (cadena_root / "inference" / "run.sh").is_file()
                and (cadena_root / "dataset" / "create_hf_dataset.py").is_file()
                and cadena_model
                and cadena_model.exists()
            ),
            "configured": _truthy("CADIA_PHOTO_CADENA_ENABLED", True),
            "detail": str(cadena_root or "CADIA_CADENA_ROOT is not set"),
        },
    }


def run_agentic_cad(
    images: Iterable[Path],
    *,
    output_dir: Path,
    prompt: str,
    dimension_hint: str,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> ExternalEngineResult:
    bin_name = os.getenv("CADIA_AGENTIC_CAD_BIN", "agentic-cad")
    executable = shutil.which(bin_name)
    result = ExternalEngineResult(name="agentic_cad", available=bool(executable))
    if not executable or not _truthy("CADIA_PHOTO_AGENTIC_ENABLED", True):
        result.message = "Agentic CAD CLI not available/configured; native CADia photo path remains active."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    photo_dir = output_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(images, 1):
        shutil.copy2(source, photo_dir / f"view_{index:02d}{source.suffix.lower()}")

    run_dir = output_dir / "run"
    args = [
        executable,
        "reconstruct",
        str(photo_dir),
        "--prompt",
        prompt or "Reconstruct this physical object as editable parametric CAD.",
        "--output",
        str(run_dir),
        "--reconstruction",
        os.getenv("CADIA_AGENTIC_CAD_RECONSTRUCTION", "auto"),
        "--max-repairs",
        os.getenv("CADIA_AGENTIC_CAD_MAX_REPAIRS", "2"),
        "--timeout",
        os.getenv("CADIA_AGENTIC_CAD_REQUEST_TIMEOUT", "900"),
        "--max-output-tokens",
        os.getenv("CADIA_AGENTIC_CAD_MAX_OUTPUT_TOKENS", "32768"),
    ]
    if dimension_hint.strip():
        args += ["--dimension", dimension_hint.strip()[:800]]

    code, output = _run(
        args,
        cwd=output_dir,
        env=os.environ.copy(),
        timeout=int(os.getenv("CADIA_AGENTIC_CAD_PROCESS_TIMEOUT", "2400")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(run_dir)
    if code != 0:
        result.message = f"Agentic CAD exited with code {code}: {output[-2200:]}"
        return result

    model_py = run_dir / "model.py"
    parameters = run_dir / "parameters.json"
    model_stl = run_dir / "model.stl"
    report = run_dir / "report.json"
    quality = run_dir / "quality.json"
    geometry_review = run_dir / "geometry-review.json"
    evidence = run_dir / "evidence"

    blocks = []
    if model_py.is_file():
        blocks.append("AGENTIC_CAD_PARAMETRIC_PROPOSAL:\n" + _text(model_py))
    if parameters.is_file():
        blocks.append("AGENTIC_CAD_PARAMETERS:\n" + _text(parameters, 10000))
    if report.is_file():
        blocks.append("AGENTIC_CAD_REPORT:\n" + _text(report, 10000))
    if geometry_review.is_file():
        blocks.append("AGENTIC_CAD_GEOMETRY_REVIEW:\n" + _text(geometry_review, 10000))

    result.guidance = "\n\n".join(blocks)
    # model.stl is Agentic CAD's own CAD candidate. Keep it as a baseline proposal,
    # not as the observation target used by CADFit/CADENA. A separate helper below
    # reconstructs an observation-derived mesh from masks/depth/cameras when Open3D
    # is available in the Agentic CAD runtime.
    result.baseline_mesh = str(model_stl) if model_stl.is_file() else None
    result.reference_mesh = None
    if (evidence / "geometry.npz").is_file():
        result.evidence_dir = str(evidence)

    for score_file in (geometry_review, quality, report):
        payload = _load_json(score_file) if score_file.is_file() else {}
        score = _extract_numeric_score(payload)
        if score is not None:
            result.score = max(0.0, min(1.0, float(score)))
            break

    result.ok = bool(result.guidance or result.reference_mesh or result.evidence_dir)
    result.message = (
        "Agentic CAD public reconstruction proposal generated."
        if result.ok
        else "Agentic CAD completed but produced no usable artifact."
    )
    return result


def build_reference_mesh_from_agentic_evidence(
    agentic_result: ExternalEngineResult | None,
    *,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> Path | None:
    """Fuse Agentic CAD's shared multi-view DA3 evidence into ONE target mesh.

    The mesh is reconstructed only from `geometry.npz` (masks, relative depth,
    intrinsics and extrinsics), never from Agentic CAD's generated model.stl.
    Scale remains relative; CADFit/CADENA are shape reconstruction stages and the
    user/CADia metric anchors remain authoritative for millimetres.

    Open3D is intentionally executed in Agentic CAD's Python environment. If its
    optional `dense` dependency is unavailable, this function simply returns None
    and CADFit falls back to its own public image-to-mesh route.
    """
    if agentic_result is None or not agentic_result.evidence_dir:
        return None
    evidence = Path(agentic_result.evidence_dir)
    geometry = evidence / "geometry.npz"
    if not geometry.is_file():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "observation_reference.stl"
    script = output_dir / "fuse_depth_to_mesh.py"
    executable = shutil.which(os.getenv("CADIA_AGENTIC_CAD_BIN", "agentic-cad"))
    python_bin = _agentic_python(executable)
    script.write_text(
        "import numpy as np\n"
        "from pathlib import Path\n"
        "import open3d as o3d\n"
        f"geometry=Path({str(geometry)!r})\n"
        f"target=Path({str(target)!r})\n"
        "with np.load(geometry, allow_pickle=False) as data:\n"
        "    extr=np.asarray(data['extrinsics'], dtype=float)\n"
        "    intr=np.asarray(data['intrinsics'], dtype=float)\n"
        "    depth=np.asarray(data['depth'], dtype=float)\n"
        "    masks=np.asarray(data['masks']).astype(bool)\n"
        "clouds=[]\n"
        "for i in range(len(depth)):\n"
        "    valid=masks[i] & np.isfinite(depth[i]) & (depth[i] > 0)\n"
        "    ys,xs=np.where(valid)\n"
        "    if len(xs) < 32:\n"
        "        continue\n"
        "    if len(xs) > 60000:\n"
        "        pick=np.linspace(0, len(xs)-1, 60000).astype(int)\n"
        "        xs,ys=xs[pick],ys[pick]\n"
        "    pix=np.column_stack([xs,ys,np.ones(len(xs))])\n"
        "    rays=pix @ np.linalg.inv(intr[i]).T\n"
        "    cam=rays * depth[i,ys,xs,None]\n"
        "    e=extr[i]\n"
        "    world=(cam - e[:3,3]) @ e[:3,:3]\n"
        "    clouds.append(world)\n"
        "if not clouds:\n"
        "    raise RuntimeError('no finite masked depth points')\n"
        "pts=np.concatenate(clouds, axis=0)\n"
        "center=np.median(pts, axis=0)\n"
        "rad=np.linalg.norm(pts-center, axis=1)\n"
        "pts=pts[rad <= np.percentile(rad, 98.5)]\n"
        "pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))\n"
        "bbox=pcd.get_axis_aligned_bounding_box()\n"
        "diag=max(float(np.linalg.norm(bbox.get_extent())), 1e-6)\n"
        "voxel=max(diag/320.0, 1e-6)\n"
        "pcd=pcd.voxel_down_sample(voxel)\n"
        "pcd,_=pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.5)\n"
        "pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*6.0, max_nn=60))\n"
        "try:\n"
        "    pcd.orient_normals_consistent_tangent_plane(20)\n"
        "except Exception:\n"
        "    pass\n"
        "mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, linear_fit=False)\n"
        "density=np.asarray(density)\n"
        "if len(density):\n"
        "    mesh.remove_vertices_by_mask(density < np.quantile(density, 0.025))\n"
        "crop=pcd.get_axis_aligned_bounding_box()\n"
        "mn=np.asarray(crop.min_bound); mx=np.asarray(crop.max_bound); pad=(mx-mn)*0.035\n"
        "crop=o3d.geometry.AxisAlignedBoundingBox(mn-pad, mx+pad)\n"
        "mesh=mesh.crop(crop)\n"
        "if len(mesh.triangles):\n"
        "    clusters,counts,_=mesh.cluster_connected_triangles()\n"
        "    clusters=np.asarray(clusters); counts=np.asarray(counts)\n"
        "    if len(counts) > 1:\n"
        "        keep=int(np.argmax(counts))\n"
        "        mesh.remove_triangles_by_mask(clusters != keep)\n"
        "mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()\n"
        "if len(mesh.triangles) < 4:\n"
        "    raise RuntimeError('depth fusion produced an empty mesh')\n"
        "mesh.compute_vertex_normals()\n"
        "if not o3d.io.write_triangle_mesh(str(target), mesh, write_ascii=False):\n"
        "    raise RuntimeError('could not write observation reference mesh')\n",
        encoding="utf-8",
    )
    code, output = _run(
        [python_bin, str(script)],
        cwd=output_dir,
        env=os.environ.copy(),
        timeout=int(os.getenv("CADIA_AGENTIC_CAD_MESH_TIMEOUT", "1200")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    if code != 0 or not target.is_file() or target.stat().st_size < 256:
        (output_dir / "mesh_build.log").write_text(output, encoding="utf-8", errors="replace")
        return None
    return target



def run_cadcoder(
    image: Path,
    *,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> ExternalEngineResult:
    """Run CAD-Coder through its released LLaVA evaluation entry point.

    CADia only creates the one-row question JSONL expected by the public loader;
    model inference itself stays inside CAD-Coder. Generated Python is advisory
    planner guidance and is never executed inside the CADia project.
    """
    root_raw = os.getenv("CADIA_CADCODER_ROOT", "").strip()
    root = Path(root_raw).expanduser() if root_raw else None
    loader = root / "llava" / "eval" / "model_vqa_loader.py" if root else None
    result = ExternalEngineResult(name="cadcoder", available=bool(loader and loader.is_file()))
    if not result.available or not _truthy("CADIA_PHOTO_CADCODER_ENABLED", True):
        result.message = "CAD-Coder is not installed/configured."
        return result
    if not image.is_file():
        result.message = "CAD-Coder input photo is missing."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    local_image = image_dir / ("input" + image.suffix.lower())
    shutil.copy2(image, local_image)
    questions = output_dir / "question.jsonl"
    answers = output_dir / "answers.jsonl"
    questions.write_text(
        json.dumps({
            "question_id": 1,
            "image": local_image.name,
            "text": "Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words.",
            "category": "default",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    python_bin = os.getenv("CADIA_CADCODER_PYTHON", sys.executable)
    model = os.getenv("CADIA_CADCODER_MODEL", "CADCODER/CAD-Coder")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    args = [
        python_bin, "-m", "llava.eval.model_vqa_loader",
        "--model-path", model,
        "--question-file", str(questions),
        "--image-folder", str(image_dir),
        "--answers-file", str(answers),
        "--num-chunks", "1",
        "--chunk-idx", "0",
        "--temperature", "0",
        "--max_new_tokens", os.getenv("CADIA_CADCODER_MAX_NEW_TOKENS", "3450"),
        "--conv-mode", os.getenv("CADIA_CADCODER_CONV_MODE", "vicuna_v1"),
    ]
    code, output = _run(
        args,
        cwd=root,
        env=env,
        timeout=int(os.getenv("CADIA_CADCODER_TIMEOUT", "1800")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(output_dir)
    if code != 0:
        result.message = f"CAD-Coder exited with code {code}: {output[-2200:]}"
        return result
    try:
        rows = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines() if line.strip()]
        proposal = str(rows[0].get("text") or "") if rows else ""
    except Exception as exc:
        result.message = f"CAD-Coder completed but its answer could not be parsed: {exc}"
        return result
    if not proposal.strip():
        result.message = "CAD-Coder completed but returned no CAD program."
        return result
    if len(proposal) > 36000:
        proposal = proposal[:36000] + "\n...<truncated>"
    result.guidance = "CADCODER_IMAGE_PROPOSAL:\n" + proposal
    result.ok = True
    result.message = "CAD-Coder public image-to-CadQuery proposal generated."
    return result


def run_pixal3d(
    image: Path,
    *,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> ExternalEngineResult:
    """Run Pixal3D's released single-image inference and expose only its mesh.

    The GLB->STL conversion is format glue via trimesh; no geometry generation is
    reimplemented in CADia.
    """
    root_raw = os.getenv("CADIA_PIXAL3D_ROOT", "").strip()
    root = Path(root_raw).expanduser() if root_raw else None
    script = root / "inference.py" if root else None
    result = ExternalEngineResult(name="pixal3d", available=bool(script and script.is_file()))
    if not result.available or not _truthy("CADIA_PHOTO_PIXAL3D_ENABLED", True):
        result.message = "Pixal3D is not installed/configured."
        return result
    if not image.is_file():
        result.message = "Pixal3D input photo is missing."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    local_image = output_dir / ("input" + image.suffix.lower())
    shutil.copy2(image, local_image)
    glb = output_dir / "pixal3d.glb"
    stl = output_dir / "pixal3d.stl"
    python_bin = os.getenv("CADIA_PIXAL3D_PYTHON", sys.executable)
    args = [python_bin, str(script), "--image", str(local_image), "--output", str(glb)]
    if _truthy("CADIA_PIXAL3D_LOW_VRAM", False):
        args.append("--low_vram")
    resolution = os.getenv("CADIA_PIXAL3D_RESOLUTION", "").strip()
    if resolution:
        args += ["--resolution", resolution]
    model_path = os.getenv("CADIA_PIXAL3D_MODEL_PATH", "").strip()
    if model_path:
        args += ["--model_path", model_path]
    env = os.environ.copy()
    env.setdefault("ATTN_BACKEND", os.getenv("CADIA_PIXAL3D_ATTN_BACKEND", "sdpa"))
    code, output = _run(
        args,
        cwd=root,
        env=env,
        timeout=int(os.getenv("CADIA_PIXAL3D_TIMEOUT", "3600")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(output_dir)
    if code != 0 or not glb.is_file():
        result.message = f"Pixal3D inference failed ({code}): {output[-2200:]}"
        return result

    convert = (
        "import trimesh,sys\n"
        "src,dst=sys.argv[1],sys.argv[2]\n"
        "obj=trimesh.load(src, force='scene')\n"
        "geoms=list(obj.geometry.values()) if hasattr(obj,'geometry') else [obj]\n"
        "geoms=[g for g in geoms if hasattr(g,'faces') and len(g.faces)]\n"
        "assert geoms, 'no triangle mesh in GLB'\n"
        "mesh=trimesh.util.concatenate(geoms) if len(geoms)>1 else geoms[0]\n"
        "mesh.export(dst)\n"
    )
    code, output = _run(
        [python_bin, "-c", convert, str(glb), str(stl)],
        cwd=root,
        env=env,
        timeout=int(os.getenv("CADIA_PIXAL3D_CONVERT_TIMEOUT", "300")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    if code != 0 or not stl.is_file() or stl.stat().st_size < 256:
        result.message = f"Pixal3D generated GLB but STL conversion failed ({code}): {output[-1600:]}"
        return result
    result.reference_mesh = str(stl)
    result.baseline_mesh = str(glb)
    result.guidance = "PIXAL3D_TARGET_MESH: public Pixal3D single-image reconstruction is available as a geometry target for downstream public Mesh-to-CAD engines."
    result.ok = True
    result.message = "Pixal3D public single-image mesh generated."
    return result


def run_cadreasoner(
    reference_mesh: Path | None,
    *,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> ExternalEngineResult:
    """Run CADReasoner's released iterative inference on a local STL directory."""
    root_raw = os.getenv("CADIA_CADREASONER_ROOT", "").strip()
    root = Path(root_raw).expanduser() if root_raw else None
    script = root / "test.py" if root else None
    result = ExternalEngineResult(name="cadreasoner", available=bool(script and script.is_file()))
    if not result.available or not _truthy("CADIA_PHOTO_CADREASONER_ENABLED", True):
        result.message = "CADReasoner is not installed/configured."
        return result
    if reference_mesh is None or not reference_mesh.is_file():
        result.message = "CADReasoner is available but no target mesh exists."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "target_mesh"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "target.stl"
    shutil.copy2(reference_mesh, target)
    predictions = output_dir / "predictions"
    python_bin = os.getenv("CADIA_CADREASONER_PYTHON", sys.executable)
    checkpoint = os.getenv("CADIA_CADREASONER_MODEL", "kulibinai/cadreasoner")
    args = [
        python_bin, str(script),
        "--dataset", str(dataset_dir),
        "--checkpoint", checkpoint,
        "--n_iters", os.getenv("CADIA_CADREASONER_ITERS", "3"),
        "--outdir", str(predictions),
        "--greedy", os.getenv("CADIA_CADREASONER_GREEDY", "true"),
        "--top_n", os.getenv("CADIA_CADREASONER_TOP_N", "1"),
        "--n_samples", os.getenv("CADIA_CADREASONER_SAMPLES", "1"),
    ]
    code, output = _run(
        args,
        cwd=root,
        env=os.environ.copy(),
        timeout=int(os.getenv("CADIA_CADREASONER_TIMEOUT", "3600")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(predictions)
    if code != 0:
        result.message = f"CADReasoner exited with code {code}: {output[-2200:]}"
        return result

    py_files = list(predictions.rglob("*.py"))
    if not py_files:
        result.message = "CADReasoner completed but no CadQuery prediction was found."
        return result
    def rank(path: Path) -> tuple[int, float, str]:
        try:
            iteration = max(int(p) for p in path.parts if p.isdigit())
        except Exception:
            iteration = 0
        try:
            cd = float(path.stem)
        except Exception:
            cd = float("inf")
        return (-iteration, cd, str(path))
    py_files.sort(key=rank)
    best = py_files[0]
    program = _text(best, 36000)
    if not program.strip():
        result.message = "CADReasoner produced an empty program."
        return result
    result.guidance = "CADREASONER_ITERATIVE_PROPOSAL:\n" + program
    try:
        cd = float(best.stem)
        result.guidance += f"\nCADREASONER_CHAMFER_DISTANCE={cd:.10g}"
    except Exception:
        pass
    result.reference_mesh = str(target)
    result.ok = True
    result.message = "CADReasoner public iterative program proposal generated."
    return result

def _collect_cadfit_result(cadfit_out: Path) -> tuple[str, str | None, float | None]:
    candidates: list[tuple[float, Path]] = []
    for py_file in cadfit_out.rglob("best_greedy_parallel_iterative.py"):
        score = -1.0
        score_file = py_file.parent / "final_iou.json"
        if score_file.is_file():
            numeric = _extract_numeric_score(_load_json(score_file))
            if numeric is not None:
                score = float(numeric)
        candidates.append((score, py_file))
    if not candidates:
        for py_file in cadfit_out.rglob("best_greedy_parallel.py"):
            candidates.append((-1.0, py_file))
    if not candidates:
        return "", None, None
    candidates.sort(key=lambda row: row[0], reverse=True)
    score, best_py = candidates[0]
    mesh = best_py.with_suffix(".stl")
    if not mesh.is_file():
        meshes = list(best_py.parent.glob("best_greedy_parallel_iterative.stl")) or list(best_py.parent.glob("best_greedy_parallel.stl"))
        mesh = meshes[0] if meshes else None
    guidance = "CADFIT_PARAMETRIC_PROPOSAL:\n" + _text(best_py)
    if score >= 0:
        guidance += f"\nCADFIT_REPORTED_IOU={score:.6f}"
    return guidance, (str(mesh) if mesh and mesh.is_file() else None), (score if score >= 0 else None)


def run_cadfit(
    images: Iterable[Path],
    *,
    output_dir: Path,
    reference_mesh: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> ExternalEngineResult:
    root_raw = os.getenv("CADIA_CADFIT_ROOT", "").strip()
    root = Path(root_raw).expanduser() if root_raw else None
    result = ExternalEngineResult(
        name="cadfit",
        available=bool(root and (root / "run_pipeline.py").is_file()),
    )
    if not result.available or not _truthy("CADIA_PHOTO_CADFIT_ENABLED", True):
        result.message = "CADFit is not installed/configured; native CADia photo path remains active."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = os.getenv("CADIA_CADFIT_PYTHON", sys.executable)
    max_iterations = int(os.getenv("CADIA_CADFIT_MAX_ITERATIONS", "2"))

    # Preferred path: fit the ONE shared multi-view reference mesh produced by the
    # Agentic-CAD/SAM2/COLMAP/DA3 branch. This avoids generating a different 3D
    # object independently from every photograph.
    if reference_mesh is not None and reference_mesh.is_file():
        target = output_dir / "target.stl"
        shutil.copy2(reference_mesh, target)
        cadfit_out = output_dir / "run"
        cadfit_out.mkdir(parents=True, exist_ok=True)
        runner = output_dir / "run_shared_mesh.py"
        runner.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            "from run_pipeline import process_single_stl\n"
            f"process_single_stl({str(target)!r}, folder_name={str(cadfit_out)!r}, "
            f"max_iterations={max_iterations}, apply_fillet_chamfer=True, keep_history=True)\n",
            encoding="utf-8",
        )
        code, output = _run(
            [python_bin, str(runner)],
            cwd=root,
            env=os.environ.copy(),
            timeout=int(os.getenv("CADIA_CADFIT_TIMEOUT", "3600")),
            cancel_check=cancel_check,
            deadline_at=deadline_at,
        )
        result.output_dir = str(cadfit_out)
        if code != 0:
            result.message = f"CADFit shared-mesh fitting failed ({code}): {output[-2200:]}"
            return result
        guidance, reconstructed, score = _collect_cadfit_result(cadfit_out)
        if not guidance:
            result.message = "CADFit completed shared-mesh fitting but no parametric candidate was found."
            return result
        result.guidance = guidance
        # CADENA should keep comparing against the original observation-derived target,
        # not against CADFit's own reconstructed output.
        result.reference_mesh = str(target)
        result.score = score
        result.ok = True
        result.message = "CADFit proposal fitted to the shared multi-view reference mesh."
        if reconstructed:
            result.guidance += f"\nCADFIT_RECONSTRUCTED_MESH={Path(reconstructed).name}"
        return result

    # Fallback when there is no shared reference mesh: use CADFit's public image→mesh
    # entry point and select the highest-IoU parametric proposal.
    script = root / "image_to_cad" / "run_image_to_cad.py"
    if not script.is_file():
        result.message = "CADFit is present but no shared reference mesh and no image_to_cad runner are available."
        return result
    photo_dir = output_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(images, 1):
        shutil.copy2(source, photo_dir / f"view_{index:02d}{source.suffix.lower()}")
    cadfit_out = output_dir / "run"
    args = [
        python_bin,
        str(script),
        "--images-folder",
        str(photo_dir),
        "--output-folder",
        str(cadfit_out),
        "--max-iterations",
        str(max_iterations),
        "--fillet-chamfer",
        "--keep-history",
    ]
    code, output = _run(
        args,
        cwd=root,
        env=os.environ.copy(),
        timeout=int(os.getenv("CADIA_CADFIT_TIMEOUT", "3600")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(cadfit_out)
    if code != 0:
        result.message = f"CADFit image path exited with code {code}: {output[-2200:]}"
        return result
    guidance, _reconstructed, score = _collect_cadfit_result(cadfit_out)
    if not guidance:
        result.message = "CADFit image path completed but no parametric candidate was found."
        return result
    result.guidance = guidance
    result.score = score
    clean_meshes = list(cadfit_out.rglob("clean.stl"))
    result.reference_mesh = str(clean_meshes[0]) if clean_meshes else None
    result.ok = True
    result.message = "CADFit image-to-CAD proposal generated."
    return result


def run_cadena(reference_mesh: Path | None, *, output_dir: Path, cancel_check: Callable[[], bool] | None = None, deadline_at: float | None = None) -> ExternalEngineResult:
    root_raw = os.getenv("CADIA_CADENA_ROOT", "").strip()
    model_raw = os.getenv("CADIA_CADENA_MODEL_PATH", "").strip()
    root = Path(root_raw).expanduser() if root_raw else None
    model = Path(model_raw).expanduser() if model_raw else None
    run_script = root / "inference" / "run.sh" if root else None
    create_script = root / "dataset" / "create_hf_dataset.py" if root else None
    result = ExternalEngineResult(
        name="cadena",
        available=bool(run_script and run_script.is_file() and create_script and create_script.is_file() and model and model.exists()),
    )
    if not result.available or not _truthy("CADIA_PHOTO_CADENA_ENABLED", True):
        result.message = "CADENA is not installed/configured; CADia's native greedy refinement remains active."
        return result
    if reference_mesh is None or not reference_mesh.is_file():
        result.message = "CADENA is available but there is no shared reference mesh from the geometry branch."
        return result

    result.attempted = True
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_root = output_dir / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    target = mesh_root / "target.stl"
    shutil.copy2(reference_mesh, target)
    dataset_dir = output_dir / "dataset"
    env = os.environ.copy()
    env.update({"MESH_ROOT": str(mesh_root), "OUT_DIR": str(dataset_dir)})
    python_bin = os.getenv("CADIA_CADENA_PYTHON", sys.executable)
    try:
        cadena_bin = str(Path(python_bin).expanduser().resolve().parent)
        env["PATH"] = cadena_bin + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    code, output = _run(
        [python_bin, str(create_script)],
        cwd=root,
        env=env,
        timeout=int(os.getenv("CADIA_CADENA_DATASET_TIMEOUT", "600")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    if code != 0:
        result.message = f"CADENA dataset preparation failed ({code}): {output[-2200:]}"
        return result

    env = os.environ.copy()
    try:
        cadena_bin = str(Path(python_bin).expanduser().resolve().parent)
        env["PATH"] = cadena_bin + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    env.update({
        "MODEL_PATH": str(model),
        "DATASET_PATH": str(dataset_dir),
        "OUT_ROOT": str(output_dir / "work"),
        "MODE": os.getenv("CADIA_CADENA_MODE", "greedy"),
        "MAX_ROLLOUT_STEPS": os.getenv("CADIA_CADENA_MAX_STEPS", "16"),
        "NUM_WORKERS_PER_GPU": os.getenv("CADIA_CADENA_WORKERS", "4"),
        "BATCH_SIZE": "64",
        "GET_METRICS": os.getenv("CADIA_CADENA_METRICS", "gms"),
    })
    code, output = _run(
        ["bash", str(run_script)],
        cwd=root,
        env=env,
        timeout=int(os.getenv("CADIA_CADENA_TIMEOUT", "3600")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    result.output_dir = str(output_dir / "work")
    if code != 0:
        result.message = f"CADENA exited with code {code}: {output[-2200:]}"
        return result

    greedy_files = list((output_dir / "work").rglob("greedy.json"))
    best_program = ""
    best_reward = -1.0
    for path in greedy_files:
        row = _load_json(path)
        prefixes = row.get("prefixes") or []
        rewards = row.get("rewards") or []
        # In CADENA greedy rollout, reward[j] evaluates prefixes[j+1]. prefixes[-1]
        # may be the rejected step that caused termination, so never blindly use it.
        for j, reward in enumerate(rewards):
            if not isinstance(reward, (int, float)):
                continue
            prefix_index = j + 1
            if prefix_index >= len(prefixes):
                continue
            if float(reward) >= best_reward:
                best_reward = float(reward)
                best_program = str(prefixes[prefix_index])
    if not best_program:
        result.message = "CADENA completed but no accepted scored prefix was found."
        return result
    if len(best_program) > 32000:
        best_program = best_program[:32000] + "\n...<truncated>"
    result.guidance = "CADENA_STEPWISE_ACCEPTED_PROPOSAL:\n" + best_program
    if best_reward >= 0:
        result.guidance += f"\nCADENA_BEST_IOU={best_reward:.6f}"
        result.score = max(0.0, min(1.0, best_reward))
    result.reference_mesh = str(reference_mesh)
    result.ok = True
    result.message = "CADENA best accepted stepwise proposal generated."
    return result


def score_with_agentic_evidence(
    candidate_mesh: Path,
    agentic_result: ExternalEngineResult | None,
    *,
    output_dir: Path,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> dict | None:
    """Numerically compare a CADia candidate against the SAME multi-view masks/depth/cameras.

    This reuses Agentic CAD's public GeometryObjective. It does not call an LLM and does
    not alter the candidate. Returned `consistency_score` is observation consistency,
    not certified dimensional accuracy.
    """
    if agentic_result is None or not agentic_result.evidence_dir:
        return None
    evidence = Path(agentic_result.evidence_dir)
    if not (evidence / "geometry.npz").is_file() or not candidate_mesh.is_file():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score.json"
    panels = output_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(os.getenv("CADIA_AGENTIC_CAD_BIN", "agentic-cad"))
    python_bin = _agentic_python(executable)
    runner = output_dir / "score_candidate.py"
    runner.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from da3_cad.hybrid_fit import GeometryObjective\n"
        f"evidence=Path({str(evidence)!r})\n"
        f"mesh=Path({str(candidate_mesh)!r})\n"
        f"panels=Path({str(panels)!r})\n"
        "search=GeometryObjective(evidence)\n"
        "search.load_mesh(mesh)\n"
        "registration=search.register()\n"
        "verify=GeometryObjective(evidence, resolution=192)\n"
        "verify.load_mesh(mesh)\n"
        "result=verify.evaluate(registration['pose'], panels=panels)\n"
        "result['registration_method']=registration.get('registration_method','global')\n"
        "result['consistency_score']=max(0.0,min(1.0,1.0-float(result['loss'])))\n"
        f"Path({str(score_path)!r}).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')\n",
        encoding="utf-8",
    )
    code, output = _run(
        [python_bin, str(runner)],
        cwd=output_dir,
        env=os.environ.copy(),
        timeout=int(os.getenv("CADIA_AGENTIC_CAD_SCORE_TIMEOUT", "900")),
        cancel_check=cancel_check,
        deadline_at=deadline_at,
    )
    if code != 0 or not score_path.is_file():
        return {"ok": False, "error": f"Agentic evidence score failed ({code}): {output[-1200:]}"}
    payload = _load_json(score_path)
    payload["ok"] = True
    return payload
