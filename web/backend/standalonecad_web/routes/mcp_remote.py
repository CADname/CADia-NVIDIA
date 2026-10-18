from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..cad_runtime import runtime_manager
from ..database import get_db
from ..heavy_jobs import heavy_job_manager
from ..models import McpAccessToken
from ..remote_mcp import RemoteProjectMcpGateway
from .projects import owned_project

router = APIRouter(tags=["mcp"])


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _challenge(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={
            "WWW-Authenticate": 'Bearer realm="CADia Remote MCP"',
            "Cache-Control": "no-store",
        },
    )


def _token_row(authorization: str | None, db: Session) -> McpAccessToken:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _challenge("CADia MCP Bearer token is required.")
    raw = authorization[7:].strip()
    if not raw.startswith("scad_mcp_"):
        raise _challenge("Invalid CADia MCP token.")
    row = db.scalar(select(McpAccessToken).where(McpAccessToken.token_hash == _hash_token(raw)))
    now = datetime.now(timezone.utc)
    if row is None or row.revoked_at is not None:
        raise _challenge("Invalid CADia MCP token.")
    if row.expires_at is not None:
        expiry = row.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= now:
            raise _challenge("CADia MCP token has expired.")
    row.last_used_at = now
    db.commit()
    return row


def _headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Vary": "Authorization",
        "X-Content-Type-Options": "nosniff",
    }


@router.post("/mcp")
def remote_mcp_post(
    request_payload: dict[str, Any],
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Stateless MCP Streamable-HTTP JSON response endpoint.

    The bearer token identifies exactly one CADia project. All CAD calls are
    delegated to CADia's existing project runtime and tool executor.
    """
    row = _token_row(authorization, db)
    owned_project(db, row.owner_id, row.project_id)
    if heavy_job_manager.project_reserved(row.owner_id, row.project_id) and RemoteProjectMcpGateway.request_mutates(request_payload):
        raise HTTPException(
            status_code=423,
            detail="This project is currently being modeled by an isolated CAD worker. Wait for the job to finish or cancel it first.",
        )
    runtime = runtime_manager.get(row.owner_id, row.project_id)
    response = RemoteProjectMcpGateway(runtime).dispatch(request_payload)
    if response is None:
        # MCP notifications have no JSON-RPC response. Streamable HTTP permits 202.
        return Response(status_code=202, headers=_headers())
    return JSONResponse(response, headers=_headers())


@router.get("/mcp")
def remote_mcp_get(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    # CADia v1.1 is stateless and does not open a server-initiated SSE stream.
    _token_row(authorization, db)
    return Response(status_code=405, headers={**_headers(), "Allow": "POST, DELETE"})


@router.delete("/mcp", status_code=204)
def remote_mcp_delete(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    # No server session exists in stateless mode; validate credentials then acknowledge.
    _token_row(authorization, db)
    return Response(status_code=204, headers=_headers())
