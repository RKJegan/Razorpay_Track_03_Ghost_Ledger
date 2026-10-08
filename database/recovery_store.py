"""
Ghost Ledger v3 — event-sourced recovery tracking (A2).

Two tables describe one recovery:

* ``recovery_cases``  — the cached current state (status, timestamps). Fast to
  read for dashboards. Rebuildable from the events.
* ``recovery_events`` — the immutable history. Every stage transition is
  appended here and never edited (SQL triggers forbid UPDATE and DELETE).

The status of a case is a pure function of its event history (see
:data:`STAGE_STATUS`). Settled is terminal: once a payment has settled, no
later event can change the status. A late ``payment_failed`` webhook for an
earlier attempt therefore cannot undo a real settlement.

Trust boundary: this module records facts. It makes no financial decisions and
calls no LLM.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select, text

from database.engine import session_scope
from database.models import RECOVERY_STATUSES, Merchant, RecoveryCase, RecoveryEvent

logger = logging.getLogger(__name__)

#: Every stage a recovery can pass through. Unknown stages are rejected.
STAGES: tuple[str, ...] = (
    "diagnosis_created",
    "policy_check_passed",
    "playbook_selected",
    "payment_link_created",
    "customer_clicked_link",
    "payment_captured",
    "payment_failed",
    "settlement_confirmed",
    "recovery_complete",
    "recovery_failed",
    "stopping_rule_triggered",
    # A3 approvals and policy outcomes
    "policy_denied",
    "approval_requested",
    "approval_granted",
    "approval_rejected",
    # A4 scheduler outcomes
    "payment_link_expired",
    "reminder_queued",
    # A failed API call is an operational event, not the end of the recovery,
    # so it deliberately does NOT move the case to a terminal status.
    "link_call_failed",
    # B1-B8 strategy outcomes (only written when ENABLE_ADVANCED_STRATEGIES is on)
    "retry_scheduled",
    "method_suggested",
    "gateway_failover",
    "dunning_scheduled",
    "dunning_cancelled",
)

#: Case status each stage moves the recovery into. Stages not listed keep the
#: current status (for example ``payment_failed`` on one attempt can be
#: followed by a retry that succeeds).
STAGE_STATUS: dict[str, str] = {
    "payment_captured": "settled",
    "settlement_confirmed": "settled",
    "recovery_complete": "settled",
    "recovery_failed": "failed",
    "stopping_rule_triggered": "escalated",
    "approval_rejected": "escalated",
}

#: Statuses that cannot be left once reached.
TERMINAL_STATUSES: frozenset[str] = frozenset({"settled"})


class UnknownRecoveryError(LookupError):
    """Raised when an event or query names a recovery that does not exist."""


class UnknownStageError(ValueError):
    """Raised when an event names a stage outside :data:`STAGES`."""


def _now() -> str:
    """Return the current local timestamp in the project's canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def new_recovery_id() -> str:
    """
    Generate a new recovery identifier.

    Returns
    -------
    str
        ``rcv_`` followed by 16 hex characters.
    """
    return f"rcv_{uuid.uuid4().hex[:16]}"


def create_case(
    merchant_id: str,
    txn_id: str,
    amount_inr: float,
    cause: str | None = None,
    failure_id: str | None = None,
    recovery_id: str | None = None,
    customer_id: str | None = None,
) -> str:
    """
    Open a new recovery case in status ``pending``.

    Parameters
    ----------
    merchant_id : str
        Merchant that owns the failed transaction. Must exist.
    txn_id : str
        The failed transaction being recovered.
    amount_inr : float
        Amount at stake, in rupees. Must be positive.
    cause : str, optional
        Diagnosed root cause.
    failure_id : str, optional
        Link to the v2 ``failures`` row, when there is one.
    recovery_id : str, optional
        Explicit id. Generated when omitted.
    customer_id : str, optional
        Customer who owns the failed payment. Scopes the attempt counts.

    Returns
    -------
    str
        The recovery id.

    Raises
    ------
    ValueError
        If the amount is not positive.
    sqlalchemy.exc.IntegrityError
        If the merchant does not exist.
    """
    if amount_inr is None or amount_inr <= 0:
        raise ValueError(f"amount_inr must be positive, got {amount_inr!r}")
    rid = recovery_id or new_recovery_id()
    now = _now()
    with session_scope() as session:
        session.add(
            RecoveryCase(
                id=rid,
                merchant_id=merchant_id,
                txn_id=txn_id,
                customer_id=customer_id,
                failure_id=failure_id,
                cause=cause,
                amount_inr=float(amount_inr),
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
    logger.info("recovery %s opened for merchant %s, txn %s", rid, merchant_id, txn_id)
    return rid


def append_event(
    recovery_id: str,
    stage: str,
    detail: dict[str, Any] | None = None,
    created_by: str = "system",
) -> int:
    """
    Append one immutable stage event and update the cached case status.

    Parameters
    ----------
    recovery_id : str
        Recovery the event belongs to.
    stage : str
        One of :data:`STAGES`.
    detail : dict[str, Any], optional
        JSON-serialisable context (amounts, ids, reasons).
    created_by : str, optional
        Who or what produced the event, e.g. ``policy_engine`` or
        ``razorpay_webhook``.

    Returns
    -------
    int
        Id of the new event row.

    Raises
    ------
    UnknownStageError
        If ``stage`` is not a known stage.
    UnknownRecoveryError
        If the recovery does not exist.
    """
    if stage not in STAGES:
        raise UnknownStageError(f"unknown recovery stage {stage!r}")
    now = _now()
    with session_scope() as session:
        case = session.get(RecoveryCase, recovery_id)
        if case is None:
            raise UnknownRecoveryError(f"no recovery {recovery_id!r}")

        target = STAGE_STATUS.get(stage)
        if target and case.status not in TERMINAL_STATUSES:
            case.status = target
            if target == "settled" and case.settled_at is None:
                case.settled_at = now
        case.updated_at = now

        event = RecoveryEvent(
            recovery_id=recovery_id,
            timestamp=now,
            stage=stage,
            status=case.status,
            detail=json.dumps(detail, default=str, separators=(",", ":")) if detail is not None else None,
            created_by=created_by,
        )
        session.add(event)
        session.flush()
        event_id = int(event.id)
    logger.debug("event %s appended to %s: %s -> %s", event_id, recovery_id, stage, target or "(no status change)")
    return event_id


def has_payment_recorded(recovery_id: str, payment_id: str) -> bool:
    """
    Return True if this payment is already recorded as a capture or confirmation.

    Covers both the webhook path (``payment_captured``) and the confirmation
    of a provisional poll settlement (``settlement_confirmed``).
    """
    with session_scope() as session:
        row = session.execute(
            text(
                "SELECT 1 FROM recovery_events WHERE recovery_id = :rid "
                "AND stage IN ('payment_captured', 'settlement_confirmed') "
                "AND json_extract(detail, '$.payment_id') = :pid LIMIT 1"
            ),
            {"rid": recovery_id, "pid": payment_id},
        ).first()
    return row is not None


def has_provisional_settlement(recovery_id: str) -> bool:
    """
    Return True when the latest settlement was seen by a status poll and not yet
    confirmed by a webhook. A poll is provisional: it knows the link was paid,
    but not the payment id, so the webhook must still confirm it.
    """
    latest = latest_event(recovery_id, ("payment_captured", "settlement_confirmed"))
    if latest is None or latest["stage"] != "payment_captured":
        return False
    return (latest["detail"] or {}).get("source") == "status_poll"


def case_exists(recovery_id: str) -> bool:
    """Return True when a recovery with this id exists."""
    with session_scope() as session:
        return session.get(RecoveryCase, recovery_id) is not None


def has_event_for_payment(recovery_id: str, stage: str, payment_id: str) -> bool:
    """
    Return True if this recovery already has ``stage`` recorded for ``payment_id``.

    Used as a business-level duplicate guard: two different webhook deliveries
    describing the same payment must produce one settlement, not two.
    """
    with session_scope() as session:
        row = session.execute(
            text(
                "SELECT 1 FROM recovery_events WHERE recovery_id = :rid AND stage = :stage "
                "AND json_extract(detail, '$.payment_id') = :pid LIMIT 1"
            ),
            {"rid": recovery_id, "stage": stage, "pid": payment_id},
        ).first()
    return row is not None


def _case_to_dict(case: RecoveryCase) -> dict[str, Any]:
    """Convert a case row to a plain dict for callers and the API."""
    return {
        "recovery_id": case.id,
        "merchant_id": case.merchant_id,
        "txn_id": case.txn_id,
        "customer_id": case.customer_id,
        "failure_id": case.failure_id,
        "cause": case.cause,
        "amount_inr": case.amount_inr,
        "status": case.status,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "settled_at": case.settled_at,
    }


def get_recovery_status(recovery_id: str) -> dict[str, Any] | None:
    """
    Return the cached current state of one recovery.

    Returns
    -------
    dict[str, Any] | None
        The case as a dict, or None if it does not exist.
    """
    with session_scope() as session:
        case = session.get(RecoveryCase, recovery_id)
        return _case_to_dict(case) if case is not None else None


def get_recovery_timeline(recovery_id: str) -> list[dict[str, Any]]:
    """
    Return every event for a recovery, in the order they were appended.

    Returns
    -------
    list[dict[str, Any]]
        Oldest first. Empty if the recovery has no events or does not exist.
    """
    with session_scope() as session:
        rows = session.scalars(
            select(RecoveryEvent)
            .where(RecoveryEvent.recovery_id == recovery_id)
            .order_by(RecoveryEvent.id)
        ).all()
        return [
            {
                "event_id": r.id,
                "timestamp": r.timestamp,
                "stage": r.stage,
                "status": r.status,
                "detail": json.loads(r.detail) if r.detail else None,
                "created_by": r.created_by,
            }
            for r in rows
        ]


def get_active_recoveries(merchant_id: str | None = None) -> list[dict[str, Any]]:
    """
    Return recoveries still in status ``pending``, newest first.

    Parameters
    ----------
    merchant_id : str, optional
        Restrict to one merchant. None returns all merchants (internal view).
    """
    return _list_by_status("pending", merchant_id)


def get_settled_today(
    merchant_id: str | None = None, day: date | None = None
) -> list[dict[str, Any]]:
    """
    Return recoveries that settled on ``day`` (default: today, local time).

    Parameters
    ----------
    merchant_id : str, optional
        Restrict to one merchant.
    day : date, optional
        The calendar day to report on.
    """
    day = day or date.today()
    prefix = day.isoformat()
    rows = _list_by_status("settled", merchant_id)
    return [r for r in rows if (r["settled_at"] or "").startswith(prefix)]


def _list_by_status(status: str, merchant_id: str | None) -> list[dict[str, Any]]:
    """Return cases with ``status``, optionally for one merchant, newest first."""
    if status not in RECOVERY_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    stmt = select(RecoveryCase).where(RecoveryCase.status == status)
    if merchant_id is not None:
        stmt = stmt.where(RecoveryCase.merchant_id == merchant_id)
    stmt = stmt.order_by(RecoveryCase.updated_at.desc(), RecoveryCase.id)
    with session_scope() as session:
        return [_case_to_dict(c) for c in session.scalars(stmt).all()]


def count_events(recovery_id: str, stages: tuple[str, ...]) -> int:
    """Return how many events of the given stages a recovery has (for attempt counting)."""
    with session_scope() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(RecoveryEvent)
                .where(RecoveryEvent.recovery_id == recovery_id, RecoveryEvent.stage.in_(stages))
            )
            or 0
        )


def latest_event(recovery_id: str, stages: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the most recent event of the given stages, or None."""
    with session_scope() as session:
        row = session.scalars(
            select(RecoveryEvent)
            .where(RecoveryEvent.recovery_id == recovery_id, RecoveryEvent.stage.in_(stages))
            .order_by(RecoveryEvent.id.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        return {
            "event_id": row.id,
            "timestamp": row.timestamp,
            "stage": row.stage,
            "detail": json.loads(row.detail) if row.detail else None,
            "created_by": row.created_by,
        }


def list_cases(
    status: str | None = None,
    merchant_id: str | None = None,
    cause: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Return one page of cases, filtered, newest first, plus the total match count.

    Parameters
    ----------
    status, merchant_id, cause : str, optional
        Filters. None means no filter.
    limit, offset : int
        Pagination window.
    """
    conditions = []
    if status is not None:
        if status not in RECOVERY_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        conditions.append(RecoveryCase.status == status)
    if merchant_id is not None:
        conditions.append(RecoveryCase.merchant_id == merchant_id)
    if cause is not None:
        conditions.append(RecoveryCase.cause == cause)
    with session_scope() as session:
        total = int(session.scalar(select(func.count()).select_from(RecoveryCase).where(*conditions)) or 0)
        rows = session.scalars(
            select(RecoveryCase)
            .where(*conditions)
            .order_by(RecoveryCase.updated_at.desc(), RecoveryCase.id)
            .limit(limit)
            .offset(offset)
        ).all()
        return [_case_to_dict(r) for r in rows], total


def merchant_exists(merchant_id: str) -> bool:
    """Return True when the merchant row exists."""
    with session_scope() as session:
        return session.get(Merchant, merchant_id) is not None
