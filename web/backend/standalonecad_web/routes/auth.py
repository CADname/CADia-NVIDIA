from __future__ import annotations

import hmac
import os
import shutil
import uuid
import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from ..auth import CurrentUser, clear_session_cookie, set_session_cookie
from ..database import get_db
from ..models import (User, Project, ChatMessage, ConversationState, McpAccessToken,
    AIProviderConnection, Asset3D, GenerationJob, OAuthIdentity, AccountPlan, MonthlyModelingUsage)
from ..cad_runtime import runtime_manager
from ..providers.app_server import app_server_manager
from ..heavy_jobs import heavy_job_manager
from ..config import settings
from .studio_orders import StudioOrder

log = logging.getLogger(__name__)
from ..schemas import LoginRequest, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _judge_username() -> str:
    return os.getenv("CADIA_HACKATHON_USERNAME", "123456").strip() or "123456"


def _judge_password() -> str:
    return os.getenv("CADIA_HACKATHON_PASSWORD", "123456") or "123456"


def _judge_email() -> str:
    return os.getenv("CADIA_HACKATHON_USER_EMAIL", "judge@cadia.local").strip().lower() or "judge@cadia.local"


@router.post("/login", response_model=UserOut)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    username = str(payload.email or "").strip()
    password = str(payload.password or "")
    if not hmac.compare_digest(username, _judge_username()) or not hmac.compare_digest(password, _judge_password()):
        raise HTTPException(status_code=401, detail="Invalid judge ID or password.")

    # Each judge sign-in is a new isolated throwaway workspace. The shared
    # login ID/password remain unchanged, but there is no shared user record.
    user = User(email=f"guest-{uuid.uuid4().hex}@cadia.local",
                display_name="Hackathon Judge", password_hash="guest$disabled")
    db.add(user)
    db.commit()
    db.refresh(user)
    set_session_cookie(response, user.id)
    return user


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return user


def _is_ephemeral_judge(user: User) -> bool:
    return (user.password_hash == "guest$disabled"
            and user.email.startswith("guest-") and user.email.endswith("@cadia.local"))


@router.post("/logout", status_code=204)
def logout(response: Response, user: CurrentUser, db: Session = Depends(get_db)):
    # Never delete owner/admin or legacy shared judge records. Only sessions
    # created by the temporary judge login can be erased automatically.
    if _is_ephemeral_judge(user):
        uid = str(user.id)
        if not heavy_job_manager.cancel_user(uid, wait_timeout=25.0):
            raise HTTPException(status_code=409, detail="CAD tasks are still stopping. Retry sign-out shortly.")
        project_ids = list(db.scalars(select(Project.id).where(Project.owner_id == uid)))
        studio_ids = list(db.scalars(select(StudioOrder.id).where(StudioOrder.customer_id == uid)))
        # Stop in-memory workers BEFORE removing their on-disk files.
        for pid in project_ids:
            runtime_manager.drop(uid, pid)
        app_server_manager.drop(uid, logout=True)
        # Explicit order works even when SQLite foreign keys aren't enabled.
        if project_ids:
            for model in (GenerationJob, Asset3D, ChatMessage, ConversationState, McpAccessToken):
                db.execute(delete(model).where(model.project_id.in_(project_ids)))
        db.execute(delete(McpAccessToken).where(McpAccessToken.owner_id == uid))
        db.execute(delete(StudioOrder).where(StudioOrder.customer_id == uid))
        db.execute(delete(AIProviderConnection).where(AIProviderConnection.user_id == uid))
        db.execute(delete(OAuthIdentity).where(OAuthIdentity.user_id == uid))
        db.execute(delete(MonthlyModelingUsage).where(MonthlyModelingUsage.user_id == uid))
        db.execute(delete(AccountPlan).where(AccountPlan.user_id == uid))
        db.execute(delete(Project).where(Project.owner_id == uid))
        db.execute(delete(User).where(User.id == uid))
        db.commit()
        # Shared API daily quotas are stored separately from the main database.
        quota_store = settings.data_root / "shared_openai_calls.sqlite3"
        if quota_store.is_file():
            try:
                with sqlite3.connect(str(quota_store), timeout=10) as quota:
                    quota.execute("DELETE FROM daily_calls WHERE user=?", (uid,))
            except sqlite3.Error:
                log.exception("Could not remove guest API quota ledger")
        for directory in (settings.data_root / "projects" / uid,
                          settings.data_root / "codex-users" / uid,
                          settings.data_root / "copilot-users" / uid,
                          *(settings.data_root / "studio_orders" / oid for oid in studio_ids)):
            # All IDs are DB-generated UUIDs, so no user-controlled paths.
            try:
                if directory.is_dir():
                    shutil.rmtree(directory)
            except OSError:
                log.exception("Guest scratch directory cleanup failed: %s", directory)
        # Clear only the guest session's in-memory auth cookie, not server keys.
    clear_session_cookie(response)
    response.status_code = 204
    return None
