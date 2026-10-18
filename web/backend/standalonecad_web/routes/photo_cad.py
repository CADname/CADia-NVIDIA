from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import CurrentUser
from ..cad_runtime import runtime_manager
from ..config import settings
from ..heavy_jobs import HeavyJobBusy
from ..database import get_db
from ..photo_cad import ALLOWED_PHOTO_SUFFIXES, MAX_PHOTOS, PhotoCadJob, photo_cad_jobs
from ..photo_cad_external import capabilities
from .projects import owned_project


router = APIRouter(prefix="/projects/{project_id}/photo-cad", tags=["photo-cad"])


def _safe_suffix(filename: str) -> str:
    suffix = Path(Path(filename or "photo.jpg").name).suffix.lower()
    if suffix not in ALLOWED_PHOTO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Photo → CAD supports PNG, JPG/JPEG and WEBP images.")
    return suffix


async def _save_upload(file: UploadFile, target: Path) -> None:
    size = 0
    try:
        with target.open("xb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="A product photo exceeds the allowed upload size.")
                output.write(chunk)
        os.chmod(target, 0o600)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/capabilities")
def get_capabilities(project_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    return {
        "max_photos": MAX_PHOTOS,
        "external": capabilities(),
    }


@router.post("/jobs")
async def create_job(
    project_id: str,
    user: CurrentUser,
    db: Session = Depends(get_db),
    photos: list[UploadFile] = File(...),
    notes: str = Form(""),
    dimension_hint: str = Form(""),
    model: str = Form("gpt-5.6-sol"),
    effort: str = Form("high"),
):
    owned_project(db, user.id, project_id)
    if not photos:
        raise HTTPException(status_code=422, detail="Upload at least one product photo.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(status_code=422, detail=f"Upload at most {MAX_PHOTOS} product photos per reconstruction.")
    project_root = runtime_manager.project_root(user.id, project_id)
    job_id = uuid.uuid4().hex
    input_dir = project_root / "photo_cad" / job_id / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    saved: dict[str, str] = {}
    try:
        for index, upload in enumerate(photos, 1):
            suffix = _safe_suffix(upload.filename or f"photo_{index}.jpg")
            target = input_dir / f"photo_{index:02d}{suffix}"
            await _save_upload(upload, target)
            saved[f"photo_{index:02d}"] = str(target)
    except Exception:
        for path in input_dir.glob("*"):
            path.unlink(missing_ok=True)
        try:
            input_dir.rmdir()
        except Exception:
            pass
        raise

    clean_effort = effort.strip().lower()
    if clean_effort not in {"low", "medium", "high"}:
        clean_effort = "high"
    job = PhotoCadJob(
        id=job_id,
        user_id=user.id,
        project_id=project_id,
        input_files=saved,
        notes=notes.strip()[:4000],
        dimension_hint=dimension_hint.strip()[:1000],
        model=(model.strip() or "gpt-5.6-sol")[:100],
        effort=clean_effort,
        vision_mode="standard",
    )
    try:
        photo_cad_jobs.create(job)
    except HeavyJobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.public()


@router.get("/jobs/{job_id}")
def get_job(project_id: str, job_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    try:
        return photo_cad_jobs.get(job_id, user.id, project_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Photo → CAD job was not found.") from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    try:
        return photo_cad_jobs.cancel(job_id, user.id, project_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Photo → CAD job was not found.") from exc
