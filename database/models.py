"""
Ghost Ledger v3 — SQLAlchemy models (new tables only).

These tables live in the SAME SQLite file as the v2 tables (no separation).
v2 tables are defined in ``database/schema.sql`` and are untouched.

Tables
------
merchants        up to ``MERCHANT_MAX_COUNT`` (10) merchants, hashed API keys
recovery_cases   cached current state of each recovery (one row per recovery)
recovery_events  immutable, append-only log of every stage of every recovery
webhook_events   every verified Razorpay webhook, keyed by event id
                 (the idempotency key: a duplicate delivery cannot insert twice)

Append-only and the merchant cap are enforced by SQL triggers, created in
``database/migrations.py``, so they hold even for code that bypasses this
module.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

RECOVERY_STATUSES: tuple[str, ...] = ("pending", "settled", "failed", "escalated")


class Base(DeclarativeBase):
    """Declarative base shared by every v3 model."""


class Merchant(Base):
    """A merchant that uses Ghost Ledger. The API key itself is never stored."""

    __tablename__ = "merchants"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class RecoveryCase(Base):
    """Cached current state of one recovery. Source of truth is ``recovery_events``."""

    __tablename__ = "recovery_cases"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'settled', 'failed', 'escalated')",
            name="ck_recovery_status",
        ),
        Index("ix_cases_merchant_status", "merchant_id", "status"),
        Index("ix_cases_status", "status"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    merchant_id: Mapped[str] = mapped_column(
        String, ForeignKey("merchants.id"), nullable=False
    )
    txn_id: Mapped[str] = mapped_column(String, nullable=False)
    failure_id: Mapped[str | None] = mapped_column(String, nullable=True)
    cause: Mapped[str | None] = mapped_column(String, nullable=True)
    amount_inr: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    settled_at: Mapped[str | None] = mapped_column(String, nullable=True)


class RecoveryEvent(Base):
    """One immutable stage transition of a recovery. Never updated or deleted."""

    __tablename__ = "recovery_events"
    __table_args__ = (Index("ix_events_recovery_ts", "recovery_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recovery_id: Mapped[str] = mapped_column(
        String, ForeignKey("recovery_cases.id"), nullable=False
    )
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_by: Mapped[str] = mapped_column(String, nullable=False)


class WebhookEvent(Base):
    """A verified inbound webhook. ``event_id`` is the idempotency key."""

    __tablename__ = "webhook_events"
    __table_args__ = (Index("ix_webhook_process_status", "process_status"),)

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str | None] = mapped_column(String, nullable=True)
    received_at: Mapped[str] = mapped_column(String, nullable=False)
    signature_valid: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # received | processed | anomaly | ignored | failed
    process_status: Mapped[str] = mapped_column(String, nullable=False, default="received")
    recovery_id: Mapped[str | None] = mapped_column(String, nullable=True)
    processed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)  # raw JSON body
