from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import CurrentUser
from ..cad_runtime import _safe_name, runtime_manager
from ..config import settings
from ..database import get_db
from ..models import Asset3D
from .projects import owned_project

router = APIRouter(prefix="/projects/{project_id}/assets", tags=["3d-assets"])
ASSET_SUFFIXES = {".glb"}


def _asset_row(asset: Asset3D, project_id: str) -> dict:
    metadata = {}
    try:
        metadata = json.loads(asset.metadata_json or "{}")
    except Exception:
        pass
    return {
        "id": asset.id,
        "project_id": asset.project_id,
        "name": asset.name,
        "kind": asset.kind,
        "format": asset.format,
        "provider": asset.provider,
        "source_kind": asset.source_kind,
        "display_scale": asset.display_scale,
        "metadata": metadata,
        "created_at": asset.created_at,
        "content_url": f"{settings.api_prefix}/projects/{project_id}/assets/{asset.id}/content",
    }


def _project_root(user_id: str, project_id: str) -> Path:
    return runtime_manager.project_root(user_id, project_id)


def _safe_project_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=404, detail="Asset file not found.")
    return path


async def _store_upload(file: UploadFile, target: Path, *, total_limit: int) -> int:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ASSET_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'unknown'}")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    size = 0
    with target.open("xb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > total_limit:
                raise HTTPException(status_code=413, detail="The uploaded file exceeds the allowed size.")
            output.write(chunk)
    os.chmod(target, 0o600)
    return size


@router.get("")
def list_assets(project_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    assets = list(db.scalars(select(Asset3D).where(Asset3D.project_id == project_id).order_by(Asset3D.created_at.asc())).all())
    return [_asset_row(asset, project_id) for asset in assets]


@router.post("/import", status_code=201)
async def import_asset(
    project_id: str,
    user: CurrentUser,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    name: str = Form(default=""),
):
    """Import an existing GLB asset. No hosted generation API is involved."""
    owned_project(db, user.id, project_id)
    original = Path(file.filename or "asset.glb").name
    suffix = Path(original).suffix.lower()
    if suffix not in ASSET_SUFFIXES:
        raise HTTPException(status_code=415, detail="CADia 3D Asset import currently accepts binary .glb files.")
    asset_id = str(uuid.uuid4())
    root = _project_root(user.id, project_id)
    target = root / "assets" / asset_id / "model.glb"
    try:
        await _store_upload(file, target, total_limit=settings.max_upload_bytes)
        with target.open("rb") as check:
            magic = check.read(4)
        if target.stat().st_size < 12 or magic != b"glTF":
            raise HTTPException(status_code=422, detail="The uploaded file is not a valid binary GLB container.")
        asset = Asset3D(
            id=asset_id,
            project_id=project_id,
            name=(name.strip() or Path(original).stem)[:160],
            kind="imported",
            format="glb",
            provider="import",
            relative_path=str(target.relative_to(root)),
            source_kind="import",
            display_scale=100.0,
            metadata_json="{}",
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        return _asset_row(asset, project_id)
    except Exception:
        shutil.rmtree(target.parent, ignore_errors=True)
        raise
    finally:
        await file.close()


@router.get("/{asset_id}/content")
def asset_content(project_id: str, asset_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    asset = db.scalar(select(Asset3D).where(Asset3D.id == asset_id, Asset3D.project_id == project_id))
    if asset is None:
        raise HTTPException(status_code=404, detail="3D asset not found.")
    root = _project_root(user.id, project_id)
    path = _safe_project_file(root, asset.relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="3D asset file is missing.")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{_safe_name(asset.name)}.glb")


@router.delete("/{asset_id}", status_code=204)
def delete_asset(project_id: str, asset_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    asset = db.scalar(select(Asset3D).where(Asset3D.id == asset_id, Asset3D.project_id == project_id))
    if asset is None:
        raise HTTPException(status_code=404, detail="3D asset not found.")
    root = _project_root(user.id, project_id)
    path = _safe_project_file(root, asset.relative_path)
    db.delete(asset)
    db.commit()
    shutil.rmtree(path.parent, ignore_errors=True)
    return None
