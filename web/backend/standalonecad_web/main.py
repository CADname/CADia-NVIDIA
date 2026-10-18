from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .auth import websocket_user_id
from .cad_runtime import runtime_manager
from .config import settings
from .database import SessionLocal, init_database
from .models import User
from .nvidia_review import capabilities as nvidia_review_capabilities
from .opencv_cad import capabilities as opencv_cad_capabilities
from .providers.app_server import app_server_manager
from .heavy_jobs import heavy_job_manager
from .routes import ai, assets, auth, cad, manufacturing, mcp, projects, visual_cad, photo_cad, mcp_remote, studio_orders, manufacturing_market
from .routes.projects import owned_project
from .schemas import PromptRequest


async def idle_reaper() -> None:
    while True:
        await asyncio.sleep(60)
        await asyncio.to_thread(runtime_manager.evict_idle)
        await asyncio.to_thread(app_server_manager.evict_idle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    heavy_job_manager.start()
    visual_cad.visual_cad_jobs.start()
    photo_cad.photo_cad_jobs.start()
    task = asyncio.create_task(idle_reaper())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # Cancel/drain job managers before closing any persisted CAD/AppServer state.
    await asyncio.to_thread(visual_cad.visual_cad_jobs.shutdown)
    await asyncio.to_thread(photo_cad.photo_cad_jobs.shutdown)
    await asyncio.to_thread(heavy_job_manager.shutdown)
    await asyncio.to_thread(runtime_manager.close_all)
    await asyncio.to_thread(app_server_manager.close_all)


app = FastAPI(title="CADia Web", version="11.0.3", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(self)")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; "
        "img-src 'self' data: blob: https://tile.openstreetmap.org; font-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self' ws: wss:",
    )
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if request.url.path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    return response
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(projects.router, prefix=settings.api_prefix)
app.include_router(cad.router, prefix=settings.api_prefix)
app.include_router(visual_cad.router, prefix=settings.api_prefix)
app.include_router(photo_cad.router, prefix=settings.api_prefix)
app.include_router(assets.router, prefix=settings.api_prefix)
app.include_router(manufacturing.router, prefix=settings.api_prefix)
app.include_router(manufacturing_market.router, prefix=settings.api_prefix)
app.include_router(studio_orders.router, prefix=settings.api_prefix)
app.include_router(ai.router, prefix=settings.api_prefix)
app.include_router(mcp.router, prefix=settings.api_prefix)



@app.get(f"{settings.api_prefix}/health")
def health():
    # CADIA_NVIDIA_ENGINEERING_REVIEW_V210
    _cadia_health_payload = ({"ok": True, "service": "cadia-web", "version": "11.0.3", "cad_kernel": "CadQuery/OCP", "manufacturing_api": True, "visual_3d_api": False, "drawing_to_cad": True, "photo_to_cad": True, "nemotron_orchestration": True, "nemotron_configured": bool(os.getenv("NEBIUS_API_KEY", "").strip()) and os.getenv("CADIA_NEMOTRON_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no", "disabled"}, "nemotron_model": os.getenv("NEBIUS_NEMOTRON_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")})
    nvidia = nvidia_review_capabilities()
    opencv = opencv_cad_capabilities()
    if isinstance(_cadia_health_payload, dict):
        _cadia_health_payload.update({
            "opencv_cad": True,
            "opencv_available": bool(opencv.get("available")),
            "opencv5": bool(opencv.get("opencv5")),
            "opencv_version": opencv.get("version"),
            "opencv_ximgproc_thinning": bool(opencv.get("ximgproc_thinning")),
            "isolated_cad_workers": settings.heavy_cad_workers,
            "active_heavy_jobs": len(heavy_job_manager.snapshot()),
            "nvidia_engineering_review": True,
            "nvidia_review_configured": bool(nvidia.get("configured")),
            "nvidia_cosmos_configured": bool((nvidia.get("cosmos") or {}).get("configured")),
            "nebius_nemotron_configured": bool((nvidia.get("nemotron") or {}).get("configured")),
            "nvidia_cosmos_model": (nvidia.get("cosmos") or {}).get("model"),
            "nebius_nemotron_model": (nvidia.get("nemotron") or {}).get("model"),
        })
    return _cadia_health_payload


@app.websocket(f"{settings.api_prefix}/ws/projects/{{project_id}}")
async def project_socket(websocket: WebSocket, project_id: str):
    try:
        origin = websocket.headers.get("origin")
        if origin and origin not in settings.cors_origins:
            raise RuntimeError("WebSocket origin is not allowed.")
        user_id = websocket_user_id(dict(websocket.cookies), websocket.headers.get("authorization"))
        with SessionLocal() as db:
            if db.get(User, user_id) is None:
                raise RuntimeError("Sign-in is required.")
            owned_project(db, user_id, project_id)
    except Exception:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    send_lock = asyncio.Lock()
    active_task: asyncio.Task | None = None

    async def send(payload):
        async with send_lock:
            await websocket.send_json(payload)

    async def run_prompt(payload: PromptRequest):
        nonlocal runtime
        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_event(kind: str, value):
            loop.call_soon_threadsafe(progress_queue.put_nowait, {"type": kind, "payload": value})

        async def pump_progress():
            while True:
                item = await progress_queue.get()
                await send(item)

        pump = asyncio.create_task(pump_progress())
        try:
            await send({"type": "busy", "payload": True})
            result = await asyncio.to_thread(ai.execute_prompt_sync, user_id, project_id, payload, on_event)
            await send({"type": "result", "payload": result})
            runtime = runtime_manager.get(user_id, project_id)
            mesh = await asyncio.to_thread(runtime.mesh, True)
            await send({"type": "mesh", "payload": mesh})
        except Exception as exc:
            await send({"type": "error", "payload": str(exc)})
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            # The isolated worker owns a separate OCCT runtime. Reload the persisted
            # project so subsequent refresh/selection never uses the pre-job object.
            try:
                runtime = await asyncio.to_thread(runtime_manager.get, user_id, project_id)
            except Exception:
                pass
            await send({"type": "busy", "payload": False})

    try:
        runtime = await asyncio.to_thread(runtime_manager.get, user_id, project_id)
        await send({"type": "state", "payload": runtime.state()})
        await send({"type": "mesh", "payload": await asyncio.to_thread(runtime.mesh, True)})
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "prompt":
                if active_task and not active_task.done():
                    await send({"type": "error", "payload": "A modeling task is already running."})
                    continue
                try:
                    payload = PromptRequest.model_validate(message.get("payload") or {})
                except Exception as exc:
                    await send({"type": "error", "payload": str(exc)})
                    continue
                active_task = asyncio.create_task(run_prompt(payload))
            elif kind == "cancel":
                await asyncio.to_thread(ai.cancel_prompt, user_id, project_id)
            elif kind == "refresh":
                await send({"type": "state", "payload": runtime.state()})
                await send({"type": "mesh", "payload": await asyncio.to_thread(runtime.mesh, True)})
            elif kind == "ping":
                await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            await asyncio.to_thread(ai.cancel_prompt, user_id, project_id)
            try:
                await asyncio.wait_for(active_task, timeout=5)
            except Exception:
                active_task.cancel()


# CADIA_FREE_PLAN_AI_V101_AI_ROUTE_MOUNT_BEGIN
# Server diagnostics on 2026-10-09 confirmed:
#   SOURCE_AI_ROUTES contained 15 routes but APP_AI_ROUTES was empty.
# Preserve the existing AI endpoint functions and mount them explicitly before
# the SPA catch-all so GET endpoints cannot be shadowed by /{path:path}.
from fastapi.routing import APIRoute as _CADiaAPIRouteFreeV101
from .routes import ai as _cadia_ai_free_v101


def _cadia_ai_route_key_free_v101(path: str, methods) -> tuple[str, tuple[str, ...]]:
    return path, tuple(sorted(set(methods or set())))


def _cadia_register_ai_router_free_v101() -> None:
    prefix = settings.api_prefix.rstrip("/")
    existing = {
        _cadia_ai_route_key_free_v101(str(getattr(route, "path", "")), getattr(route, "methods", set()))
        for route in app.routes
        if getattr(route, "path", None)
    }
    for route in _cadia_ai_free_v101.router.routes:
        if not isinstance(route, _CADiaAPIRouteFreeV101):
            continue
        full_path = prefix + route.path
        key = _cadia_ai_route_key_free_v101(full_path, route.methods)
        if key in existing:
            continue
        app.router.add_api_route(
            full_path,
            route.endpoint,
            methods=sorted(route.methods or set()),
            status_code=route.status_code,
            tags=list(route.tags or []),
            dependencies=list(route.dependencies or []),
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=dict(route.responses or {}),
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            response_model=route.response_model,
            response_model_include=route.response_model_include,
            response_model_exclude=route.response_model_exclude,
            response_model_by_alias=route.response_model_by_alias,
            response_model_exclude_unset=route.response_model_exclude_unset,
            response_model_exclude_defaults=route.response_model_exclude_defaults,
            response_model_exclude_none=route.response_model_exclude_none,
            include_in_schema=route.include_in_schema,
            name=route.name,
            openapi_extra=route.openapi_extra,
        )
        existing.add(key)


_cadia_register_ai_router_free_v101()
_cadia_ai_paths_free_v101 = {str(getattr(route, "path", "")) for route in app.routes}
if settings.api_prefix.rstrip("/") + "/ai/provider/select" not in _cadia_ai_paths_free_v101:
    raise RuntimeError("CADia Free AI registration failed: /api/ai/provider/select missing")
# CADIA_FREE_PLAN_AI_V101_AI_ROUTE_MOUNT_END

frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    assets = frontend_dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        requested = (frontend_dist / path).resolve()
        if frontend_dist.resolve() in requested.parents and requested.is_file():
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")

# CADIA_REMOTE_MCP_FORCE_REGISTER_V12_BEGIN
# Live-server diagnostics on 2026-10-09 showed:
#   - routes.mcp.router.routes contained the 5 expected MCP endpoints
#   - standalonecad_web.main.app.routes contained zero MCP endpoints
#   - host/image source hashes matched
#   - deleting __pycache__ and fresh-importing did not change that result
# Therefore this block explicitly mounts the already-defined endpoint functions
# after module import is complete. CAD operations remain in the existing gateway.
from fastapi.routing import APIRoute as _CADiaAPIRouteV12
from .routes import mcp as _cadia_mcp_v12, mcp_remote as _cadia_mcp_remote_v12


def _cadia_route_key_v12(path: str, methods) -> tuple[str, tuple[str, ...]]:
    return path, tuple(sorted(set(methods or set())))


def _cadia_register_router_v12(source_router, prefix: str) -> None:
    prefix = prefix.rstrip("/")
    existing = {
        _cadia_route_key_v12(str(getattr(route, "path", "")), getattr(route, "methods", set()))
        for route in app.routes
        if getattr(route, "path", None)
    }
    for route in source_router.routes:
        if not isinstance(route, _CADiaAPIRouteV12):
            continue
        full_path = prefix + route.path
        key = _cadia_route_key_v12(full_path, route.methods)
        if key in existing:
            continue
        app.router.add_api_route(
            full_path,
            route.endpoint,
            methods=sorted(route.methods or set()),
            status_code=route.status_code,
            tags=list(route.tags or []),
            dependencies=list(route.dependencies or []),
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=dict(route.responses or {}),
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            response_model=route.response_model,
            response_model_include=route.response_model_include,
            response_model_exclude=route.response_model_exclude,
            response_model_by_alias=route.response_model_by_alias,
            response_model_exclude_unset=route.response_model_exclude_unset,
            response_model_exclude_defaults=route.response_model_exclude_defaults,
            response_model_exclude_none=route.response_model_exclude_none,
            include_in_schema=route.include_in_schema,
            name=route.name,
            openapi_extra=route.openapi_extra,
        )
        existing.add(key)


_cadia_register_router_v12(_cadia_mcp_v12.router, settings.api_prefix)
_cadia_register_router_v12(_cadia_mcp_remote_v12.router, settings.api_prefix)

_cadia_mcp_routes_v12: dict[str, set[str]] = {}
for _route_v12 in app.routes:
    _path_v12 = str(getattr(_route_v12, "path", ""))
    if "mcp" not in _path_v12.lower():
        continue
    _cadia_mcp_routes_v12.setdefault(_path_v12, set()).update(getattr(_route_v12, "methods", set()) or set())

if "/api/mcp/projects/{project_id}" not in _cadia_mcp_routes_v12:
    raise RuntimeError("CADia MCP registration failed: legacy project MCP route missing")
if "/api/mcp/bridge.py" not in _cadia_mcp_routes_v12:
    raise RuntimeError("CADia MCP registration failed: stdio bridge route missing")
if "/api/mcp" not in _cadia_mcp_routes_v12 or not {"POST", "GET", "DELETE"}.issubset(_cadia_mcp_routes_v12["/api/mcp"]):
    raise RuntimeError("CADia MCP registration failed: direct /api/mcp POST/GET/DELETE missing")
# CADIA_REMOTE_MCP_FORCE_REGISTER_V12_END
