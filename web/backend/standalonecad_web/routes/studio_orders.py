"""CADia Studio: private, manual-quote print requests. No payment or printer auto-start."""
from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, select, func
from sqlalchemy.orm import Mapped, Session, mapped_column

from ..auth import CurrentUser
from ..config import settings
from ..database import Base, get_db
from ..models import utcnow
from .cad import runtime_for

router = APIRouter(tags=["studio-orders"])


class StudioOrder(Base):
    __tablename__ = "cadia_studio_orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    model_name: Mapped[str] = mapped_column(String(160))
    model_revision: Mapped[int] = mapped_column(Integer, default=0)
    snapshot_file: Mapped[str] = mapped_column(String(255))
    process: Mapped[str] = mapped_column(String(32), default='fdm')
    material: Mapped[str] = mapped_column(String(32))
    color: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[int] = mapped_column(Integer)
    finish: Mapped[str] = mapped_column(String(32), default='standard')
    delivery: Mapped[str] = mapped_column(String(32), default='shipping')
    notes: Mapped[str] = mapped_column(Text, default='')
    customer_name: Mapped[str] = mapped_column(String(100))
    customer_email: Mapped[str] = mapped_column(String(320))
    phone: Mapped[str] = mapped_column(String(40), default='')
    shipping_address: Mapped[str] = mapped_column(Text, default='')
    status: Mapped[str] = mapped_column(String(24), default='requested', index=True)
    quote_krw: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_note: Mapped[str] = mapped_column(Text, default='')
    eta_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shipping_carrier: Mapped[str] = mapped_column(String(80), default='')
    tracking_number: Mapped[str] = mapped_column(String(120), default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


Process = Literal['fdm']
Material = Literal['pla', 'petg', 'abs', 'tpu']
Delivery = Literal['shipping', 'pickup']


class CreateRequest(BaseModel):
    material: Material
    color: Literal['black', 'white', 'gray', 'blue', 'red', 'custom'] = 'black'
    quantity: int = Field(ge=1, le=100)
    finish: Literal['standard', 'support-removal', 'surface-finish'] = 'standard'
    delivery: Delivery
    notes: str = Field(default='', max_length=1200)
    customer_name: str = Field(min_length=2, max_length=100)
    customer_email: str = Field(min_length=5, max_length=320)
    consent: Literal[True]

    @field_validator('customer_email')
    @classmethod
    def valid_email(cls, value: str) -> str:
        from ..auth import normalize_email
        return normalize_email(value)

    @field_validator('customer_name', 'notes')
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class QuoteUpdate(BaseModel):
    quote_krw: int = Field(ge=0, le=100_000_000)
    eta_days: int = Field(ge=1, le=365)
    quote_note: str = Field(default='', max_length=1500)


class ConfirmRequest(BaseModel):
    phone: str = Field(min_length=7, max_length=40)
    shipping_address: str = Field(default='', max_length=500)

    @field_validator('phone', 'shipping_address')
    @classmethod
    def trim(cls, value: str) -> str:
        return value.strip()


class StatusUpdate(BaseModel):
    status: Literal['printing', 'quality_check', 'shipped', 'completed', 'rejected']
    shipping_carrier: str = Field(default='', max_length=80)
    tracking_number: str = Field(default='', max_length=120)
    quote_note: str | None = Field(default=None, max_length=1500)


STATES = {
    'requested': {'rejected'},
    'quoted': {'rejected'},
    'confirmed': {'printing', 'rejected'},
    'printing': {'quality_check'},
    'quality_check': {'shipped', 'completed'},
    'shipped': {'completed'},
    'completed': set(), 'rejected': set(), 'cancelled': set(),
}


def _studio_admin(user: CurrentUser) -> bool:
    configured = {x.strip().casefold() for x in os.getenv('CADIA_STUDIO_ADMIN_EMAILS', '').split(',') if x.strip()}
    return bool(configured) and user.email.casefold() in configured


def _require_admin(user: CurrentUser) -> None:
    if not _studio_admin(user):
        raise HTTPException(status_code=403, detail='Studio administrator access required.')


def _order_data(o: StudioOrder, *, admin: bool = False) -> dict:
    data = {
        'id': o.id, 'project_id': o.project_id, 'model_name': o.model_name,
        'model_revision': o.model_revision, 'process': o.process,
        'material': o.material, 'color': o.color, 'quantity': o.quantity,
        'finish': o.finish, 'delivery': o.delivery, 'notes': o.notes,
        'customer_name': o.customer_name, 'customer_email': o.customer_email,
        'status': o.status, 'quote_krw': o.quote_krw, 'quote_note': o.quote_note,
        'eta_days': o.eta_days, 'shipping_carrier': o.shipping_carrier,
        'tracking_number': o.tracking_number, 'created_at': o.created_at,
        'updated_at': o.updated_at,
    }
    if admin:
        data.update(phone=o.phone, shipping_address=o.shipping_address)
    return data


def _mine(db: Session, user: CurrentUser, order_id: str) -> StudioOrder:
    order = db.scalar(select(StudioOrder).where(StudioOrder.id == order_id, StudioOrder.customer_id == user.id))
    if order is None:
        raise HTTPException(status_code=404, detail='Request not found.')
    return order


@router.get('/studio/meta')
def studio_meta(user: CurrentUser):
    return {'admin': _studio_admin(user), 'checkout_enabled': False, 'quote_mode': 'manual'}


@router.get('/studio/orders')
def my_orders(user: CurrentUser, db: Session = Depends(get_db)):
    return [_order_data(o) for o in db.scalars(select(StudioOrder).where(StudioOrder.customer_id == user.id).order_by(StudioOrder.created_at.desc()).limit(100))]


@router.post('/projects/{project_id}/manufacturing/studio/request', status_code=201)
def request_print(project_id: str, payload: CreateRequest, user: CurrentUser, db: Session = Depends(get_db)):
    # A durable copy is created *at submission*. Later CAD edits do not alter the request.
    recent = db.scalar(select(func.count()).select_from(StudioOrder).where(
        StudioOrder.customer_id == user.id,
        StudioOrder.created_at >= datetime.now(timezone.utc) - timedelta(days=1),
    )) or 0
    if recent >= 10:
        raise HTTPException(status_code=429, detail='Daily request limit reached. Please try again tomorrow.')
    if len(payload.customer_name.strip()) < 2:
        raise HTTPException(status_code=422, detail='Enter a valid contact name.')
    runtime = runtime_for(project_id, user, db)
    try:
        with runtime.operation_lock, runtime.engine.lock:
            doc = runtime.engine.doc
            if doc is None or doc.shape is None:
                raise ValueError('Create a solid CAD model before requesting printing.')
            model_name = str(doc.title or 'CAD Model')[:160]
            revision = int(getattr(doc, 'revision', 0) or 0)
        stl = runtime.export('stl')
        if stl.stat().st_size > 40 * 1024 * 1024:
            raise ValueError('The STL file exceeds the 40 MB print-request limit.')
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    order_id = str(uuid.uuid4())
    directory = settings.data_root / 'studio_orders' / order_id
    try:
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        destination = directory / 'model.stl'
        shutil.copyfile(stl, destination)
        destination.chmod(0o600)
        order = StudioOrder(
            id=order_id, customer_id=user.id, project_id=project_id, model_name=model_name,
            model_revision=revision, snapshot_file='model.stl',
            process='fdm', material=payload.material, color=payload.color, quantity=payload.quantity,
            finish=payload.finish, delivery=payload.delivery, notes=payload.notes,
            customer_name=payload.customer_name, customer_email=str(payload.customer_email),
        )
        db.add(order)
        db.commit()
        db.refresh(order)
        return _order_data(order)
    except Exception:
        db.rollback()
        shutil.rmtree(directory, ignore_errors=True)
        raise


@router.post('/studio/orders/{order_id}/confirm')
def confirm_quote(order_id: str, payload: ConfirmRequest, user: CurrentUser, db: Session = Depends(get_db)):
    order = _mine(db, user, order_id)
    if order.status != 'quoted' or order.quote_krw is None:
        raise HTTPException(status_code=409, detail='A valid quote must be issued before confirming.')
    if not payload.phone.strip():
        raise HTTPException(status_code=422, detail='Phone number is required.')
    if order.delivery == 'shipping' and len(payload.shipping_address.strip()) < 8:
        raise HTTPException(status_code=422, detail='Please enter a complete shipping address.')
    order.phone = payload.phone.strip()
    order.shipping_address = payload.shipping_address.strip() if order.delivery == 'shipping' else ''
    order.status = 'confirmed'
    db.commit()
    db.refresh(order)
    return _order_data(order)


@router.post('/studio/orders/{order_id}/cancel')
def cancel_order(order_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    order = _mine(db, user, order_id)
    if order.status not in {'requested', 'quoted'}:
        raise HTTPException(status_code=409, detail='This order can no longer be cancelled here.')
    order.status = 'cancelled'
    db.commit()
    db.refresh(order)
    return _order_data(order)


@router.get('/studio/admin/orders')
def admin_orders(user: CurrentUser, db: Session = Depends(get_db)):
    _require_admin(user)
    return [_order_data(o, admin=True) for o in db.scalars(select(StudioOrder).order_by(StudioOrder.created_at.desc()).limit(250))]


@router.post('/studio/admin/orders/{order_id}/quote')
def admin_quote(order_id: str, payload: QuoteUpdate, user: CurrentUser, db: Session = Depends(get_db)):
    _require_admin(user)
    order = db.get(StudioOrder, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail='Request not found.')
    if order.status not in {'requested', 'quoted'}:
        raise HTTPException(status_code=409, detail='Quote cannot be changed at this stage.')
    order.quote_krw, order.eta_days = payload.quote_krw, payload.eta_days
    order.quote_note, order.status = payload.quote_note.strip(), 'quoted'
    db.commit()
    db.refresh(order)
    return _order_data(order, admin=True)


@router.patch('/studio/admin/orders/{order_id}/status')
def admin_status(order_id: str, payload: StatusUpdate, user: CurrentUser, db: Session = Depends(get_db)):
    _require_admin(user)
    order = db.get(StudioOrder, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail='Request not found.')
    if payload.status not in STATES.get(order.status, set()):
        raise HTTPException(status_code=409, detail=f'Cannot move from {order.status} to {payload.status}.')
    if payload.status == 'shipped' and order.delivery == 'shipping' and not payload.tracking_number.strip():
        raise HTTPException(status_code=422, detail='Tracking number is required for shipped orders.')
    order.status = payload.status
    if payload.quote_note is not None:
        order.quote_note = payload.quote_note.strip()
    if payload.status == 'shipped':
        order.shipping_carrier = payload.shipping_carrier.strip()
        order.tracking_number = payload.tracking_number.strip()
    db.commit()
    db.refresh(order)
    return _order_data(order, admin=True)


@router.get('/studio/admin/orders/{order_id}/model')
def admin_model(order_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    _require_admin(user)
    order = db.get(StudioOrder, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail='Request not found.')
    path = settings.data_root / 'studio_orders' / order.id / 'model.stl'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='Model snapshot not found.')
    return FileResponse(path, media_type='model/stl', filename=f'CADia-{order.id[:8]}.stl', headers={'Cache-Control': 'private, no-store'})
