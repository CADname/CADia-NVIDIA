from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import json
import secrets
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from standalonecad.recovery import execute_with_failure_recovery

from ..ai_credentials import PROVIDER_LABELS, SUPPORTED_PROVIDERS, provider_ai, provider_connection, remove_provider, select_provider, selected_ai
from ..auth import CurrentUser
from ..cad_runtime import runtime_manager
from ..database import db_session, get_db
from ..config import settings
from ..models import AIProviderConnection, ChatMessage
from ..providers import provider_for_user, selected_provider_name
from ..providers.anthropic_api import AnthropicAPIProvider
from ..providers.copilot_sdk import CopilotSDKProvider
from ..providers.app_server import AppServerError, app_server_manager
from ..providers.gemini_api import GeminiAPIProvider
from ..providers.nebius_nemotron import NebiusNemotronProvider
from ..providers.openai_api import OpenAIAPIProvider
from ..schemas import AIProviderConnectRequest, AIProviderSelectRequest, PromptRequest
from ..state import planning_state
from ..web_agent import APICompatibleAgent, DesktopParityCodexAgent
from ..nemotron_codex_agent import NemotronAugmentedCodexAgent
from ..nemotron_program_v10 import NemotronProgramAgent, execute_nemotron_program_with_repair
from .projects import owned_project


router = APIRouter(prefix="/ai", tags=["ai"])
ai_turn_slots = threading.BoundedSemaphore(settings.max_concurrent_ai_turns)
cancel_lock = threading.RLock()
cancel_events: dict[tuple[str, str], threading.Event] = {}
active_agents: dict[tuple[str, str], Any] = {}


GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def _github_ready() -> None:
    if not (settings.github_oauth_client_id and settings.github_oauth_client_secret and settings.github_oauth_redirect_uri):
        raise HTTPException(status_code=503, detail="GitHub Copilot OAuth is not configured on this server yet.")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))


def _issue_github_state(user_id: str) -> str:
    body = _b64(json.dumps({"sub": user_id, "exp": int(time.time()) + 600, "nonce": secrets.token_urlsafe(18)}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(settings.session_secret.encode(), ("github-copilot|" + body).encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _verify_github_state(value: str, user_id: str) -> None:
    try:
        body, sig = value.split(".", 1)
        expected = _b64(hmac.new(settings.session_secret.encode(), ("github-copilot|" + body).encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("signature")
        payload = json.loads(_unb64(body))
        if str(payload.get("sub")) != user_id or int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="GitHub OAuth state validation failed. Please connect again.") from exc


def _github_json(request: UrlRequest, timeout: int = 20) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise HTTPException(status_code=502, detail=f"GitHub authorization server error ({exc.code}): {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="Could not reach the GitHub authorization server.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Unexpected GitHub authorization response format.")
    return payload


def _github_profile(token: str) -> dict[str, Any]:
    return _github_json(UrlRequest(GITHUB_USER_URL, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "CADia"}))


def _codex_connected(user_id: str) -> bool:
    try:
        return bool(_client(user_id).account().get("account"))
    except Exception:
        return False


def _stored_connected(db: Session, user_id: str, provider: str) -> bool:
    row = provider_connection(db, user_id, provider)
    return bool(row and row.encrypted_api_key)


def _nemotron_provider(model: str | None = None) -> NebiusNemotronProvider:
    return NebiusNemotronProvider(default_model=model or None)


def _nemotron_connected() -> bool:
    try:
        return bool(_nemotron_provider().configured)
    except Exception:
        return False


def _client(user_id: str):
    try:
        return app_server_manager.get(user_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not start Codex App Server: {exc}") from exc


def _provider_payload(provider: str, *, connected: bool, model: str | None = None, account: dict | None = None) -> dict[str, Any]:
    return {
        "provider": provider,
        "providerLabel": PROVIDER_LABELS.get(provider, provider),
        "connected": connected,
        "model": model,
        "supportsReasoning": provider in {"codex", "codex_direct", "copilot", "openai"},
        "account": account,
        "requiresOpenaiAuth": provider in {"codex", "codex_direct"} and not connected,
    }


def _api_provider(provider: str, api_key: str, model: str | None = None):
    if provider == "nemotron":
        return _nemotron_provider(model)
    if provider == "openai":
        return OpenAIAPIProvider(api_key=api_key, default_model=model or None)
    if provider == "anthropic":
        return AnthropicAPIProvider(api_key=api_key, default_model=model or None)
    if provider == "gemini":
        return GeminiAPIProvider(api_key=api_key, default_model=model or None)
    raise ValueError(f"Not an API provider: {provider}")


def _model_rows(provider, preferred: str = "") -> list[dict[str, Any]]:
    rows = provider.list_models()
    seen: set[str] = set()
    cleaned = []
    for row in rows:
        mid = str(row.get("model") or row.get("id") or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        cleaned.append({
            "id": mid,
            "model": mid,
            "displayName": row.get("displayName") or mid,
            "isDefault": bool(preferred and mid == preferred) or bool(row.get("isDefault")),
            "supportedReasoningEfforts": ([{"reasoningEffort": "low"}, {"reasoningEffort": "medium"}, {"reasoningEffort": "high"}] if isinstance(provider, OpenAIAPIProvider) else []),
            "defaultReasoningEffort": "medium" if isinstance(provider, OpenAIAPIProvider) else "",
        })
    if preferred and preferred not in seen:
        cleaned.insert(0, {
            "id": preferred,
            "model": preferred,
            "displayName": preferred,
            "isDefault": True,
            "supportedReasoningEfforts": ([{"reasoningEffort": "low"}, {"reasoningEffort": "medium"}, {"reasoningEffort": "high"}] if isinstance(provider, OpenAIAPIProvider) else []),
            "defaultReasoningEffort": "medium" if isinstance(provider, OpenAIAPIProvider) else "",
        })
    # Keep the browser selector usable even if a provider account exposes a very large catalog.
    return cleaned[:200]


@router.get("/diagnostics")
def diagnostics(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    if selected.provider == "copilot":
        provider = CopilotSDKProvider(selected.api_key, user_id=user.id, default_model=selected.model or "auto")
        try:
            rows = _model_rows(provider, selected.model)
            return {"provider": "copilot", "providerLabel": PROVIDER_LABELS["copilot"], "models": len(rows), "ok": True}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if selected.provider not in {"codex", "codex_direct"}:
        provider = _api_provider(selected.provider, selected.api_key, selected.model)
        try:
            rows = _model_rows(provider, selected.model)
            return {"provider": selected.provider, "providerLabel": PROVIDER_LABELS[selected.provider], "models": len(rows), "ok": True}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    client = _client(user.id)
    try:
        executable = client._resolve_executable()
        import subprocess
        version = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=10)
        app_help = subprocess.run([executable, "app-server", "--help"], text=True, capture_output=True, timeout=10)
        return {
            "provider": selected.provider,
            "executable": executable,
            "version": (version.stdout or version.stderr).strip(),
            "version_ok": version.returncode == 0,
            "app_server_ok": app_help.returncode == 0,
            "running": client.running,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/providers")
def providers(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    stored = {row.provider: row for row in db.scalars(select(AIProviderConnection).where(AIProviderConnection.user_id == user.id))}
    codex_connected = _codex_connected(user.id)
    result = []
    for provider in SUPPORTED_PROVIDERS:
        if provider in {"codex", "codex_direct"}:
            connected = codex_connected
        elif provider == "nemotron":
            connected = _nemotron_connected()
        else:
            connected = bool(stored.get(provider) and stored[provider].encrypted_api_key)
        configured_model = stored.get(provider).model if stored.get(provider) else None
        if provider == "nemotron" and not configured_model:
            try:
                configured_model = _nemotron_provider().default_model
            except Exception:
                configured_model = None
        result.append({
            "id": provider,
            "label": PROVIDER_LABELS[provider],
            "configured": connected,
            "connected": connected,
            "selected": provider == selected.provider,
            "model": configured_model,
        })
    return {"selected": selected.provider, "autoFallback": settings.ai_auto_fallback, "providers": result}


@router.post("/provider/select")
def choose_provider(payload: AIProviderSelectRequest, user: CurrentUser, db: Session = Depends(get_db)):
    provider_name = payload.provider.strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported AI provider.")
    if provider_name in {"codex", "codex_direct"}:
        if not _codex_connected(user.id):
            raise HTTPException(status_code=401, detail="Connect ChatGPT / Codex first.")
    elif provider_name == "nemotron":
        if not _nemotron_connected():
            raise HTTPException(status_code=503, detail="NVIDIA Nemotron is not configured on this server.")
    elif not _stored_connected(db, user.id, provider_name):
        raise HTTPException(status_code=401, detail=f"Connect {PROVIDER_LABELS[provider_name]} first.")
    current = provider_ai(db, user.id, provider_name)
    selected_model = current.model or (_nemotron_provider().default_model if provider_name == "nemotron" else None)
    select_provider(db, user.id, provider_name, model=selected_model)
    db.commit()
    return _provider_payload(
        provider_name,
        connected=True,
        model=selected_model,
        account={"type": "oauth" if provider_name == "copilot" else ("server" if provider_name == "nemotron" else "stored"), "planType": PROVIDER_LABELS[provider_name]},
    )


@router.get("/copilot/login")
def copilot_login(user: CurrentUser):
    _github_ready()
    state = _issue_github_state(user.id)
    query = urlencode({
        "client_id": settings.github_oauth_client_id,
        "redirect_uri": settings.github_oauth_redirect_uri,
        "state": state,
        "prompt": "select_account",
    })
    return {"authorizationUrl": f"{GITHUB_AUTHORIZE_URL}?{query}"}


@router.get("/copilot/callback")
def copilot_callback(request: Request, user: CurrentUser, db: Session = Depends(get_db)):
    _github_ready()
    error = request.query_params.get("error", "")
    if error:
        return HTMLResponse(f"<!doctype html><meta charset='utf-8'><title>CADia</title><p>GitHub connection failed: {html.escape(error)}</p><script>window.opener&&window.opener.postMessage({{type:'cadia:copilot-oauth',ok:false}},location.origin);window.close();</script>", headers={"Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"})
    code = request.query_params.get("code", "")
    state = request.query_params.get("state", "")
    if not code or not state:
        raise HTTPException(status_code=400, detail="GitHub OAuth code/state is missing.")
    _verify_github_state(state, user.id)
    body = urlencode({
        "client_id": settings.github_oauth_client_id,
        "client_secret": settings.github_oauth_client_secret,
        "code": code,
        "redirect_uri": settings.github_oauth_redirect_uri,
    }).encode("utf-8")
    token_data = _github_json(UrlRequest(GITHUB_TOKEN_URL, data=body, headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "User-Agent": "CADia"}, method="POST"))
    token = str(token_data.get("access_token") or "").strip()
    if not token:
        raise HTTPException(status_code=502, detail=str(token_data.get("error_description") or token_data.get("error") or "Could not receive a GitHub access token."))
    profile = _github_profile(token)
    login = str(profile.get("login") or "GitHub user")
    # Validate that this user token can reach Copilot before persisting it.
    provider = CopilotSDKProvider(token, user_id=user.id, default_model="auto")
    try:
        models = provider.list_models()
    except Exception as exc:
        raise HTTPException(status_code=403, detail=f"Could not verify GitHub Copilot access: {exc}") from exc
    chosen = "auto" if any(str(row.get("id")) == "auto" for row in models) else str(models[0].get("id") or models[0].get("model") or "auto")
    select_provider(db, user.id, "copilot", api_key=token, model=chosen)
    db.commit()
    html = f"""<!doctype html><meta charset='utf-8'><title>CADia</title><style>body{{font-family:system-ui;background:#0f1319;color:#e8eef7;display:grid;place-items:center;height:100vh;margin:0}}div{{text-align:center}}small{{color:#91a0b5}}</style><div><h2>GitHub Copilot connected</h2><small>{html.escape(login)} · This window will close automatically.</small></div><script>window.opener&&window.opener.postMessage({{type:'cadia:copilot-oauth',ok:true}},location.origin);setTimeout(()=>window.close(),350);</script>"""
    return HTMLResponse(html, headers={"Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'", "Referrer-Policy": "no-referrer"})


@router.post("/provider/connect")
def connect_provider(payload: AIProviderConnectRequest, user: CurrentUser, db: Session = Depends(get_db)):
    provider_name = payload.provider.strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported AI provider.")
    if provider_name in {"codex", "codex_direct"}:
        select_provider(db, user.id, provider_name)
        db.commit()
        try:
            account = _client(user.id).account()
            connected = bool(account.get("account"))
            return _provider_payload(provider_name, connected=connected, account=account.get("account"))
        except Exception:
            return _provider_payload(provider_name, connected=False)
    if provider_name == "copilot":
        current = provider_ai(db, user.id, "copilot")
        if not current.api_key:
            raise HTTPException(status_code=401, detail="Connect Copilot with GitHub OAuth first.")
        select_provider(db, user.id, "copilot", model=current.model or "auto")
        db.commit()
        profile = _github_profile(current.api_key)
        return _provider_payload("copilot", connected=True, model=current.model or "auto", account={"type": "oauth", "email": profile.get("login"), "planType": "GitHub Copilot"})
    if provider_name == "nemotron":
        provider = _nemotron_provider(payload.model)
        if not provider.configured:
            raise HTTPException(status_code=503, detail="NVIDIA Nemotron is not configured on this server. Set NEBIUS_API_KEY first.")
        chosen = (payload.model or provider.default_model).strip()
        select_provider(db, user.id, "nemotron", model=chosen)
        db.commit()
        return _provider_payload(
            "nemotron",
            connected=True,
            model=chosen,
            account={"type": "server", "email": None, "planType": "Nebius Token Factory"},
        )

    api_key = (payload.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=422, detail="Enter an API key.")
    candidate = _api_provider(provider_name, api_key, payload.model)
    try:
        models = _model_rows(candidate, payload.model or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"API connection check failed: {exc}") from exc
    if not models and not payload.model:
        raise HTTPException(status_code=422, detail="Could not find available models. Enter a model ID manually.")
    chosen = (payload.model or "").strip() or str((models[0].get("model") if models else "") or "")
    select_provider(db, user.id, provider_name, api_key=api_key, model=chosen)
    db.commit()
    return _provider_payload(
        provider_name,
        connected=True,
        model=chosen,
        account={"type": "api", "email": None, "planType": PROVIDER_LABELS[provider_name]},
    )


@router.post("/provider/disconnect")
def disconnect_provider(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    if selected.provider in {"codex", "codex_direct"}:
        try:
            _client(user.id).logout()
        except Exception:
            pass
        return _provider_payload(selected.provider, connected=False)
    remove_provider(db, user.id, selected.provider)
    db.commit()
    return _provider_payload("codex", connected=False)


@router.get("/account")
def account(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    if selected.provider in {"codex", "codex_direct"}:
        try:
            raw = _client(user.id).account()
            account_value = raw.get("account")
            return _provider_payload(selected.provider, connected=bool(account_value), account=account_value)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if selected.provider == "copilot":
        if not selected.api_key:
            return _provider_payload("copilot", connected=False, model=selected.model)
        try:
            profile = _github_profile(selected.api_key)
            return _provider_payload("copilot", connected=True, model=selected.model or "auto", account={"type": "oauth", "email": profile.get("login"), "planType": "GitHub Copilot"})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if selected.provider == "nemotron":
        provider = _nemotron_provider(selected.model)
        connected = provider.configured
        return _provider_payload(
            "nemotron",
            connected=connected,
            model=selected.model or provider.default_model,
            account={"type": "server", "email": None, "planType": "Nebius Token Factory"} if connected else None,
        )
    if not selected.api_key:
        return _provider_payload(selected.provider, connected=False, model=selected.model)
    return _provider_payload(
        selected.provider,
        connected=True,
        model=selected.model,
        account={"type": "api", "email": None, "planType": PROVIDER_LABELS[selected.provider]},
    )


@router.post("/login/device")
def login_device(user: CurrentUser, db: Session = Depends(get_db), provider: str = "codex"):
    provider = provider.strip().lower()
    if provider not in {"codex", "codex_direct"}:
        raise HTTPException(status_code=422, detail="Device login is available only for ChatGPT / Codex providers.")
    select_provider(db, user.id, provider)
    db.commit()
    try:
        return _client(user.id).start_device_login()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/login/{login_id}/cancel")
def cancel_login(login_id: str, user: CurrentUser):
    try:
        return _client(user.id).cancel_login(login_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/login/{login_id}/status")
def login_status(login_id: str, user: CurrentUser):
    return _client(user.id).login_status(login_id)


@router.post("/logout")
def codex_logout(user: CurrentUser):
    try:
        return _client(user.id).logout()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/models")
def models(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    if selected.provider in {"codex", "codex_direct"}:
        try:
            return {"models": _client(user.id).models()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if selected.provider == "copilot":
        if not selected.api_key:
            raise HTTPException(status_code=401, detail="GitHub Copilot connection is required.")
        try:
            return {"models": _model_rows(CopilotSDKProvider(selected.api_key, user_id=user.id, default_model=selected.model or "auto"), selected.model or "auto")}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if selected.provider == "nemotron":
        provider = _nemotron_provider(selected.model)
        if not provider.configured:
            raise HTTPException(status_code=503, detail="NVIDIA Nemotron is not configured on this server.")
        return {"models": _model_rows(provider, selected.model or provider.default_model)}
    if not selected.api_key:
        raise HTTPException(status_code=401, detail="AI API connection is required.")
    try:
        return {"models": _model_rows(_api_provider(selected.provider, selected.api_key, selected.model), selected.model)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/rate-limits")
def rate_limits(user: CurrentUser, db: Session = Depends(get_db)):
    selected = selected_ai(db, user.id)
    if selected.provider not in {"codex", "codex_direct"}:
        return {"rateLimits": {}}
    try:
        return _client(user.id).rate_limits()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _execute_prompt_body(user_id: str, project_id: str, payload: PromptRequest, on_event=None):
    runtime = runtime_manager.get(user_id, project_id)
    provider_name = selected_provider_name(user_id)

    # Copilot follows the ipt-mcp architecture directly:
    # Copilot SDK -> CADia project MCP -> existing 118 tools -> existing executor/core.
    # It does not use the JSON-plan adapter and does not modify the Codex path below.
    if provider_name == "copilot":
        with db_session() as db:
            cfg = provider_ai(db, user_id, "copilot")
        if not cfg.api_key:
            raise RuntimeError("GitHub Copilot connection is required.")
        agent = CopilotSDKProvider(cfg.api_key, user_id=user_id, default_model=cfg.model or "auto")
        prompt = payload.prompt.strip()
        key = (user_id, project_id)
        with cancel_lock:
            active_agents[key] = agent
        try:
            message = agent.run_mcp_turn(
                project_id,
                prompt,
                model=payload.model or cfg.model or "auto",
                effort=payload.effort,
                on_event=on_event,
            )
            with db_session() as db:
                db.add(ChatMessage(project_id=project_id, role="user", content=prompt))
                db.add(ChatMessage(project_id=project_id, role="assistant", content=message))
            return {"message": message, "state": runtime.state()}
        finally:
            with cancel_lock:
                if active_agents.get(key) is agent:
                    active_agents.pop(key, None)

    if provider_name == "nemotron":
        # Nemotron Direct uses a small PromptCAD/GPT4FreeCAD-style structured CAD
        # program.  The model never sees CADia raw tool names, document lifecycle,
        # runtime bindings, file paths, or occurrence ids.  A deterministic interpreter
        # maps the validated high-level program onto the existing CADia engine.
        with db_session() as db:
            cfg = provider_ai(db, user_id, "nemotron")
        provider = _nemotron_provider(payload.model or cfg.model)
        if not provider.configured:
            raise RuntimeError("NVIDIA Nemotron is not configured on this server.")
        model = payload.model or cfg.model or provider.default_model
        reasoning = payload.effort or "medium"
        agent = NemotronProgramAgent(provider=provider, model=model, reasoning=reasoning)
    elif provider_name in {"codex", "codex_direct"}:
        # Both entries share the same isolated ChatGPT/Codex authentication.
        # `codex` keeps the existing Nemotron-assisted hybrid path.
        # `codex_direct` bypasses Nemotron completely and restores the exact
        # DesktopParityCodexAgent planner/repair path.
        client = app_server_manager.get(user_id)
        model = payload.model or "gpt-5.6-sol"
        reasoning = payload.effort or "medium"
        agent_cls = NemotronAugmentedCodexAgent if provider_name == "codex" else DesktopParityCodexAgent
        agent = agent_cls(
            client=client,
            target_id=f"web:{project_id}",
            project_root=runtime.root,
            model=model,
            reasoning=reasoning,
        )
    else:
        # Other API providers retain the pre-Copilot API-compatible path.
        provider = provider_for_user(user_id)
        agent = APICompatibleAgent(provider, model=payload.model, reasoning=payload.effort)

    # All planner-style providers converge on the common execution path below.

    prompt = payload.prompt.strip()
    with runtime.engine.lock:
        selection = dict(runtime.engine.selection)
    agent_prompt = prompt
    if selection:
        agent_prompt += f"\n\nSelected CAD context: {selection}. If the request says this/selected/here, use this selection."

    key = (user_id, project_id)
    with cancel_lock:
        active_agents[key] = agent
    verification_response: dict[str, Any] | None = None
    try:
        with runtime.operation_lock:
            if provider_name == "nemotron":
                executed_program = execute_nemotron_program_with_repair(
                    runtime=runtime,
                    agent=agent,
                    user_prompt=agent_prompt,
                    current_state=planning_state(runtime.engine),
                    on_event=on_event,
                    max_repairs=1,
                )
                message = str(executed_program["message"])
            else:
                # CADIA_VERIFIED_HYBRID_ROUTE_V4
                # Keep Codex as the executable planner. For the hybrid `codex` provider only,
                # Nemotron may audit the *successful* result against source-grounded requirements.
                # Any automatic correction is bounded to one local repair and requires a separate
                # deterministic evidence gate; audit/repair failures preserve the successful model.
                before_state = planning_state(runtime.engine)
                plan = agent.plan(agent_prompt, before_state, on_event=on_event)
                executed = execute_with_failure_recovery(
                    executor=runtime.executor,
                    agent=agent,
                    user_prompt=agent_prompt,
                    state_provider=lambda: planning_state(runtime.engine),
                    plan=plan,
                    on_event=on_event,
                    initial_progress_span=(25, 70),
                    max_ai_repairs=2,
                )
                if not executed.ok:
                    raise RuntimeError("Modeling failed: " + str(executed.error or "unknown error"))
                message = executed.message

                if provider_name == "codex" and isinstance(agent, NemotronAugmentedCodexAgent) and agent.should_audit_success():
                    after_state = planning_state(runtime.engine)
                    actual_plan = {"calls": list(executed.calls or []), "note": str(plan.get("note") or "")}
                    verification = agent.audit_success(
                        user_prompt=agent_prompt,
                        engine=runtime.engine,
                        before_state=before_state,
                        after_state=after_state,
                        executed_plan=actual_plan,
                        execution_results=list(executed.results or []),
                        on_event=on_event,
                    )
                    if verification:
                        verification_response = dict(verification)
                    gate = verification.get("gate") if isinstance(verification, dict) else {}
                    if isinstance(gate, dict) and gate.get("allow") and verification.get("auto_repair_enabled"):
                        try:
                            repair_plan = agent.repair_after_verification(
                                user_prompt=agent_prompt,
                                current_state=after_state,
                                verification=verification,
                                on_event=on_event,
                                progress_span=(93, 95),
                            )
                            if isinstance(repair_plan, dict) and repair_plan.get("calls"):
                                repaired = execute_with_failure_recovery(
                                    executor=runtime.executor,
                                    agent=agent,
                                    user_prompt=agent_prompt,
                                    state_provider=lambda: planning_state(runtime.engine),
                                    plan=repair_plan,
                                    on_event=on_event,
                                    initial_progress_span=(94, 97),
                                    max_ai_repairs=1,
                                )
                                if repaired.ok:
                                    message = repaired.message or message
                                    final_state = planning_state(runtime.engine)
                                    combined_plan = {
                                        "calls": list(executed.calls or []) + list(repaired.calls or []),
                                        "note": "initial execution + one evidence-grounded verification repair",
                                    }
                                    recheck = agent.audit_success(
                                        user_prompt=agent_prompt,
                                        engine=runtime.engine,
                                        before_state=before_state,
                                        after_state=final_state,
                                        executed_plan=combined_plan,
                                        execution_results=list(executed.results or []) + list(repaired.results or []),
                                        on_event=on_event,
                                        recheck=True,
                                    )
                                    if isinstance(recheck, dict):
                                        recheck["repair_applied"] = True
                                        recheck["repair_call_count"] = len(repair_plan.get("calls") or [])
                                        verification_response = recheck
                                else:
                                    # The failed repair plan is atomically rolled back by the executor,
                                    # so the already-successful initial result remains intact.
                                    if isinstance(verification_response, dict):
                                        verification_response["repair_applied"] = False
                                        verification_response["repair_error"] = str(repaired.error or "unknown repair error")[:1200]
                                    if on_event:
                                        on_event("progress", {"message": "Verification repair did not execute cleanly · keeping the successful pre-repair model", "percent": 98})
                        except Exception as exc:
                            # Fail-open: a supervisor/repair outage must never destroy a successful Codex result.
                            if isinstance(verification_response, dict):
                                verification_response["repair_applied"] = False
                                verification_response["repair_error"] = f"{type(exc).__name__}: {exc}"[:1200]
                            if on_event:
                                on_event("progress", {"message": f"Verification repair unavailable · keeping successful Codex result ({type(exc).__name__})", "percent": 98})
            if on_event:
                on_event("progress", {"message": "Saving project…", "percent": 99})
            runtime.persist()

        with db_session() as db:
            db.add(ChatMessage(project_id=project_id, role="user", content=prompt))
            db.add(ChatMessage(project_id=project_id, role="assistant", content=message))
        response = {"message": message, "state": runtime.state()}
        if verification_response:
            response["engineeringVerification"] = verification_response
        return response
    finally:
        with cancel_lock:
            if active_agents.get(key) is agent:
                active_agents.pop(key, None)


def execute_prompt_sync(user_id: str, project_id: str, payload: PromptRequest, on_event=None):
    key = (user_id, project_id)
    event = threading.Event()
    with cancel_lock:
        if key in cancel_events:
            raise RuntimeError("A modeling task is already running in this project.")
        cancel_events[key] = event
    acquired = False
    try:
        acquired = ai_turn_slots.acquire(timeout=10)
        if not acquired:
            raise RuntimeError("The server has reached its concurrent modeling limit. Please try again shortly.")

        def guarded_event(kind, value):
            if event.is_set():
                raise RuntimeError("The modeling task was canceled by the user.")
            if on_event:
                on_event(kind, value)

        if event.is_set():
            raise RuntimeError("The modeling task was canceled by the user.")
        return _execute_prompt_body(user_id, project_id, payload, guarded_event)
    finally:
        if acquired:
            ai_turn_slots.release()
        with cancel_lock:
            if cancel_events.get(key) is event:
                cancel_events.pop(key, None)


def cancel_prompt(user_id: str, project_id: str) -> None:
    key = (user_id, project_id)
    with cancel_lock:
        event = cancel_events.get(key)
        agent = active_agents.get(key)
        if event:
            event.set()
    if agent is not None:
        try:
            agent.cancel()
        except Exception:
            pass


@router.post("/projects/{project_id}/prompt")
async def prompt(project_id: str, payload: PromptRequest, user: CurrentUser, db: Session = Depends(get_db)):
    owned_project(db, user.id, project_id)
    try:
        return await asyncio.to_thread(execute_prompt_sync, user.id, project_id, payload)
    except AppServerError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
