"""Worldwide, opt-in local supplier discovery and read-only real manufacturing quotes.

There is intentionally no checkout, order, payment, or account-creation endpoint.
"""
from __future__ import annotations

import base64
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import CurrentUser
from ..database import get_db
from ..manufacturing.slant3d import Slant3DConfig, Slant3DError, call_slant_tool, slant3d_capability
from ..manufacturing.places_search import SearchProblem, find_nearby, search_locations
from .cad import runtime_for

router = APIRouter(prefix='/manufacturing/market', tags=['manufacturing-market'])

# Managed places providers with server-side free usage caps and private API keys.


class NearbySearch(BaseModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    radius_m: int = Field(default=12000, ge=1000, le=25000)
    consent: bool


class RealQuote(BaseModel):
    project_id: str = Field(min_length=3, max_length=64)
    consent_to_external_transfer: bool
    material: str = Field(default='PLA', max_length=60)
    color: str = Field(default='black', max_length=60)
    filament_id: str = Field(default='', max_length=120)
    quantity: int = Field(ge=1, le=1000)


class ShippingQuote(BaseModel):
    consent_to_external_transfer: bool
    quote_id: str = Field(min_length=1, max_length=200)
    shipping_service: str | None = Field(default=None, max_length=200)
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=5, max_length=250)
    line1: str = Field(min_length=1, max_length=200)
    line2: str = Field(default='', max_length=200)
    city: str = Field(min_length=1, max_length=120)
    state: str = Field(default='', max_length=120)
    postal_code: str = Field(min_length=1, max_length=40)
    country: str = Field(min_length=2, max_length=80)


@router.get('/locations')
def locations(q: str, user: CurrentUser):
    try:
        return {'results': search_locations(q)}
    except SearchProblem as exc:
        status=429 if exc.code=='budget' else 400 if exc.code=='location_query' else 503
        raise HTTPException(status,detail=str(exc)) from exc


@router.post('/nearby')
def nearby(payload: NearbySearch, user: CurrentUser):
    if not payload.consent:
        raise HTTPException(403, detail='Please allow location-based lookup first.')
    try:
        return find_nearby(payload.lat, payload.lon, payload.radius_m)
    except SearchProblem as exc:
        # Safe to display: never include key, private coordinates or provider raw errors.
        raise HTTPException(429 if exc.code=='budget' else 503,detail=str(exc)) from exc


@router.get('/slant3d/capabilities')
def slant_capabilities(user: CurrentUser):
    cfg = slant3d_capability()
    return {'provider':cfg['provider'], 'enabled':cfg['enabled'], 'real_quote':cfg['enabled'],
            'payment_enabled':False, 'checkout_enabled':False,
            'warning':'STL is sent to Slant 3D only after explicit opt-in. No payments or orders are created.'}


def _require_quote_enabled() -> None:
    if not Slant3DConfig.from_env().enabled:
        raise HTTPException(503, detail='Slant 3D quote service disabled on this server.')


def _call(tool: str, payload: dict[str, Any]) -> Any:
    try:
        return call_slant_tool(tool, payload)
    except (Slant3DError, httpx.HTTPError) as exc:
        # Never relay private payloads; errors come from the provider or tool schema only.
        raise HTTPException(502, detail=f'Slant 3D quotation unavailable: {str(exc)[:240]}') from exc


@router.get('/slant3d/materials')
def real_materials(user: CurrentUser):
    _require_quote_enabled()
    return {'provider':'Slant 3D', 'result':_call('list_materials', {})}


@router.post('/slant3d/quote')
def quote(payload: RealQuote, user: CurrentUser, db: Session = Depends(get_db)):
    _require_quote_enabled()
    if not payload.consent_to_external_transfer:
        raise HTTPException(403,detail='Consent to sharing the STL model with Slant 3D is required.')
    runtime = runtime_for(payload.project_id, user, db)
    try:
        stl = runtime.export('stl')
        data = stl.read_bytes()
    except Exception as exc:
        raise HTTPException(422,detail='Export a valid model as STL before quoting.') from exc
    if not data or len(data)>Slant3DConfig.from_env().max_stl_bytes:
        raise HTTPException(422,detail='STL is empty or exceeds the Slant 3D model size limit.')
    quote_result = _call('quote_upload_part', {'stl_base64':base64.b64encode(data).decode('ascii'),
                  'filename': 'CADia-model.stl', 'material':payload.material, 'color':payload.color,
                  'filament_id':payload.filament_id or None})
    # Do not silently issue any checkout, paid order, or printing request.
    return {'provider':'Slant 3D', 'stage':'real_quote', 'payment_enabled':False, 'result':quote_result}


@router.post('/slant3d/quantity')
def quote_quantity(quote_id: str, quantity: int, user: CurrentUser):
    _require_quote_enabled()
    if not 1 <= quantity <= 1000 or not 1 <= len(quote_id) <= 200:
        raise HTTPException(422,detail='Invalid quote or quantity.')
    return {'provider':'Slant 3D', 'stage':'quantity_quote', 'payment_enabled':False,
            'result':_call('quote_set_quantity', {'quote_id':quote_id, 'quantity': quantity})}


@router.post('/slant3d/shipping')
def shipping(payload: ShippingQuote, user: CurrentUser):
    _require_quote_enabled()
    if not payload.consent_to_external_transfer:
        raise HTTPException(403,detail='Consent to sharing the shipping address with Slant 3D is required.')
    values = payload.model_dump(exclude={'consent_to_external_transfer'})
    return {'provider':'Slant 3D', 'stage':'shipping_quote', 'payment_enabled':False,
            'result':_call('quote_shipping_options', values)}
