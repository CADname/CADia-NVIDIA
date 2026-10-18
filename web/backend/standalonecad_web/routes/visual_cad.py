from __future__ import annotations

import os
import uuid
from pathlib import Path

import fitz

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import CurrentUser
from ..cad_runtime import runtime_manager
from ..config import settings
from ..heavy_jobs import HeavyJobBusy
from ..database import get_db
from ..visual_cad import ALLOWED_IMAGE_SUFFIXES, VisualCadJob, visual_cad_jobs
from .projects import owned_project


router = APIRouter(prefix="/projects/{project_id}/visual-cad", tags=["visual-cad"])

ALLOWED_DRAWING_SUFFIXES = {*ALLOWED_IMAGE_SUFFIXES, ".pdf"}
MAX_PDF_PAGES = 12
PDF_RENDER_DPI = 200
PDF_RENDER_MAX_EDGE = 4096


def _safe_upload_name(filename: str, fallback: str) -> tuple[str, str]:
    original = Path(filename or fallback).name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_DRAWING_SUFFIXES:
        raise HTTPException(status_code=415, detail="PDF, PNG, JPG/JPEG 또는 WEBP 도면 파일만 사용할 수 있습니다.")
    return original, suffix


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
                    raise HTTPException(status_code=413, detail="도면 파일이 허용 업로드 크기를 초과했습니다.")
                output.write(chunk)
        os.chmod(target, 0o600)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def _render_pdf_pages(source: Path, input_dir: Path, label: str) -> dict[str, str]:
    rendered: dict[str, str] = {}
    try:
        with fitz.open(source) as document:
            if document.needs_pass:
                raise HTTPException(status_code=422, detail="암호로 보호된 PDF 도면은 사용할 수 없습니다.")
            if document.page_count < 1:
                raise HTTPException(status_code=422, detail="PDF 도면에 페이지가 없습니다.")
            if document.page_count > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF 도면은 파일당 최대 {MAX_PDF_PAGES}페이지까지 사용할 수 있습니다.",
                )

            for index, page in enumerate(document):
                base_scale = PDF_RENDER_DPI / 72.0
                longest = max(float(page.rect.width), float(page.rect.height), 1.0)
                scale = min(base_scale, PDF_RENDER_MAX_EDGE / longest)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                page_label = label if index == 0 else f"{label}_p{index + 1}"
                target = input_dir / f"{page_label}.png"
                pixmap.save(target)
                os.chmod(target, 0o600)
                rendered[page_label] = str(target)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF 도면을 이미지로 변환하지 못했습니다: {exc}") from exc
    return rendered


@router.post("/drawing/jobs")
async def create_drawing_job(
    project_id: str,
    user: CurrentUser,
    db: Session = Depends(get_db),
    primary: UploadFile = File(...),
    top: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
    reference: UploadFile | None = File(None),
    notes: str = Form(""),
    dimension_hint: str = Form(""),
    model: str = Form("gpt-5.6-sol"),
    effort: str = Form("high"),
):
    owned_project(db, user.id, project_id)
    project_root = runtime_manager.project_root(user.id, project_id)
    job_id = uuid.uuid4().hex
    input_dir = project_root / "visual_cad" / job_id / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)

    uploads = {"primary": primary, "top": top, "right": right, "reference": reference}
    saved: dict[str, str] = {}
    try:
        for label, upload in uploads.items():
            if upload is None:
                continue
            _, suffix = _safe_upload_name(upload.filename or f"{label}.png", f"{label}.png")
            target = input_dir / f"{label}{suffix}"
            await _save_upload(upload, target)
            if suffix == ".pdf":
                saved.update(_render_pdf_pages(target, input_dir, label))
                target.unlink(missing_ok=True)
            else:
                saved[label] = str(target)
    except Exception:
        for path in input_dir.glob("*"):
            path.unlink(missing_ok=True)
        try:
            input_dir.rmdir()
        except Exception:
            pass
        raise

    if not saved.get("primary"):
        raise HTTPException(status_code=422, detail="기본 도면 파일은 필수입니다.")
    clean_effort = effort.strip().lower()
    if clean_effort not in {"low", "medium", "high"}:
        clean_effort = "high"
    job = VisualCadJob(
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
        visual_cad_jobs.create(job)
    except HeavyJobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.public()


@router.get("/jobs/{job_id}")
def get_job(project_id: str, job_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    try:
        return visual_cad_jobs.get(job_id, user.id, project_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drawing → CAD 작업을 찾을 수 없습니다.") from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    try:
        return visual_cad_jobs.cancel(job_id, user.id, project_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Drawing → CAD 작업을 찾을 수 없습니다.") from exc
