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

from sqlalchemy import CheckConstraint, ForeignKey, Float, Index, Integer, String, Text, UniqueConstraint
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
    customer_id: Mapped[str | None] = mapped_column(String, nullable=True)
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


APPROVAL_STATUSES: tuple[str, ...] = ("pending", "approved", "rejected")


class Approval(Base):
    """A human decision request for a recovery above the auto-approve ceiling (A3)."""

    __tablename__ = "approvals"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_approval_status"),
        Index("ix_approvals_status_created", "status", "created_at"),
        Index("ix_approvals_merchant_status", "merchant_id", "status"),
    )

    approval_id: Mapped[str] = mapped_column(String, primary_key=True)
    recovery_id: Mapped[str] = mapped_column(
        String, ForeignKey("recovery_cases.id"), nullable=False
    )
    merchant_id: Mapped[str] = mapped_column(
        String, ForeignKey("merchants.id"), nullable=False
    )
    txn_id: Mapped[str] = mapped_column(String, nullable=False)
    amount_inr: Mapped[float] = mapped_column(Float, nullable=False)
    cause: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    action: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[str | None] = mapped_column(String, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String, nullable=True)
    decided_at: Mapped[str | None] = mapped_column(String, nullable=True)


class JobState(Base):
    """Per-job health for the scheduler: failure streak and the next allowed run (A4)."""

    __tablename__ = "job_state"

    job_name: Mapped[str] = mapped_column(String, primary_key=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_allowed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_run_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class BatchRun(Base):
    """Progress of a long-running batch, read live by the progress dashboard (A5)."""

    __tablename__ = "batch_runs"
    __table_args__ = (Index("ix_batch_status", "status"),)

    batch_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)


class DunningTouch(Base):
    """One scheduled customer reminder in a dunning sequence (B5). Idempotent per (recovery, touch)."""

    __tablename__ = "dunning_touches"
    __table_args__ = (
        Index("ix_dunning_status_due", "status", "due_at"),
        Index("ix_dunning_recovery", "recovery_id"),
        UniqueConstraint("recovery_id", "touch_no", name="uq_dunning_touch"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recovery_id: Mapped[str] = mapped_column(
        String, ForeignKey("recovery_cases.id"), nullable=False
    )
    touch_no: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[str] = mapped_column(String, nullable=False)
    template: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    due_at: Mapped[str] = mapped_column(String, nullable=False)
    # scheduled | sent | failed | cancelled
    status: Mapped[str] = mapped_column(String, nullable=False, default="scheduled")
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    sent_at: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class GatewayObservation(Base):
    """One payment outcome per payment method, used for gateway health (B4)."""

    __tablename__ = "gateway_observations"
    __table_args__ = (Index("ix_gateway_method_time", "method", "observed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(String, nullable=False)
    success: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 captured, 0 failed
    recovery_id: Mapped[str | None] = mapped_column(String, nullable=True)
    observed_at: Mapped[str] = mapped_column(String, nullable=False)


class AbAssignment(Base):
    """Stable A/B assignment of one unit to one variant, with its first outcome (B6)."""

    __tablename__ = "ab_assignments"
    __table_args__ = (Index("ix_ab_experiment_variant", "experiment", "variant"),)

    experiment: Mapped[str] = mapped_column(String, primary_key=True)
    unit_id: Mapped[str] = mapped_column(String, primary_key=True)
    variant: Mapped[str] = mapped_column(String, nullable=False)
    assigned_at: Mapped[str] = mapped_column(String, nullable=False)
    outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 converted, 0 not
    outcome_at: Mapped[str | None] = mapped_column(String, nullable=True)

