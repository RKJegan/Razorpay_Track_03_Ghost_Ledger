"""
Ghost Ledger v3 — approval records (A3, data layer).

An approval is created when a recovery is above the auto-approve ceiling and
needs a human decision. The decision itself is a single conditional UPDATE
(``WHERE status = 'pending'``), so two operators acting at the same moment
cannot both succeed: exactly one UPDATE changes a row, and the other gets
:class:`ApprovalAlreadyDecided` (HTTP 409).

This module stores and changes decisions. It does not execute recoveries; that
is :mod:`agents.approval_queue`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update

from database.engine import session_scope
from database.models import APPROVAL_STATUSES, Approval

logger = logging.getLogger(__name__)


class ApprovalNotFoundError(LookupError):
    """Raised when no approval has the requested id."""


class ApprovalAlreadyDecidedError(RuntimeError):
    """Raised when an approval has already been approved or rejected."""

    def __init__(self, approval_id: str, current_status: str) -> None:
        super().__init__(f"approval {approval_id} is already {current_status}")
        self.approval_id = approval_id
        self.current_status = current_status


def _now() -> str:
    """Return the current local timestamp in the project's canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _to_dict(row: Approval) -> dict[str, Any]:
    """Convert an approval row to a plain dict."""
    return {
        "approval_id": row.approval_id,
        "recovery_id": row.recovery_id,
        "merchant_id": row.merchant_id,
        "txn_id": row.txn_id,
        "amount_inr": row.amount_inr,
        "cause": row.cause,
        "confidence": row.confidence,
        "action": row.action,
        "status": row.status,
        "created_at": row.created_at,
        "approved_by": row.approved_by,
        "approved_at": row.approved_at,
        "rejection_reason": row.rejection_reason,
        "decided_by": row.decided_by,
        "decided_at": row.decided_at,
    }


def create_approval(
    recovery_id: str,
    merchant_id: str,
    txn_id: str,
    amount_inr: float,
    action: str,
    cause: str | None = None,
    confidence: float | None = None,
) -> str:
    """
    Open a pending approval for a recovery.

    Returns
    -------
    str
        The new approval id.
    """
    approval_id = f"apr_{uuid.uuid4().hex[:16]}"
    with session_scope() as session:
        session.add(
            Approval(
                approval_id=approval_id,
                recovery_id=recovery_id,
                merchant_id=merchant_id,
                txn_id=txn_id,
                amount_inr=float(amount_inr),
                cause=cause,
                confidence=confidence,
                action=action,
                status="pending",
                created_at=_now(),
            )
        )
    logger.info("approval %s requested for recovery %s (INR %.2f)", approval_id, recovery_id, amount_inr)
    return approval_id


def get_approval(approval_id: str) -> dict[str, Any] | None:
    """Return one approval as a dict, or None."""
    with session_scope() as session:
        row = session.get(Approval, approval_id)
        return _to_dict(row) if row is not None else None


def get_pending_approval_for(recovery_id: str) -> dict[str, Any] | None:
    """Return the open approval for a recovery, if one exists (prevents duplicates)."""
    with session_scope() as session:
        row = session.scalars(
            select(Approval)
            .where(Approval.recovery_id == recovery_id, Approval.status == "pending")
            .order_by(Approval.created_at)
            .limit(1)
        ).first()
        return _to_dict(row) if row is not None else None


def list_approvals(
    status: str | None = None,
    merchant_id: str | None = None,
    cause: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Return one page of approvals, newest first, plus the total match count.

    Raises
    ------
    ValueError
        If ``status`` is not a known approval status.
    """
    conditions = []
    if status is not None:
        if status not in APPROVAL_STATUSES:
            raise ValueError(f"unknown approval status {status!r}")
        conditions.append(Approval.status == status)
    if merchant_id is not None:
        conditions.append(Approval.merchant_id == merchant_id)
    if cause is not None:
        conditions.append(Approval.cause == cause)
    with session_scope() as session:
        total = int(session.scalar(select(func.count()).select_from(Approval).where(*conditions)) or 0)
        rows = session.scalars(
            select(Approval)
            .where(*conditions)
            .order_by(Approval.created_at.desc(), Approval.approval_id)
            .limit(limit)
            .offset(offset)
        ).all()
        return [_to_dict(r) for r in rows], total


def claim_decision(
    approval_id: str,
    new_status: str,
    actor: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """
    Move a pending approval to ``approved`` or ``rejected``, exactly once.

    Parameters
    ----------
    approval_id : str
        The approval to decide.
    new_status : str
        ``approved`` or ``rejected``.
    actor : str
        Who is deciding (free text from the operator).
    reason : str, optional
        Required for ``rejected``.

    Returns
    -------
    dict[str, Any]
        The approval after the decision.

    Raises
    ------
    ApprovalNotFoundError
        If the approval does not exist.
    ApprovalAlreadyDecidedError
        If it was already decided. Only the first decision succeeds.
    ValueError
        For an invalid status, empty actor, or a rejection without a reason.
    """
    if new_status not in ("approved", "rejected"):
        raise ValueError(f"decision must be approved or rejected, got {new_status!r}")
    if not actor or not actor.strip():
        raise ValueError("actor must be non-empty: say who made the decision")
    if new_status == "rejected" and not (reason and reason.strip()):
        raise ValueError("a rejection needs a reason")

    now = _now()
    values: dict[str, Any] = {"status": new_status, "decided_by": actor.strip(), "decided_at": now}
    if new_status == "approved":
        values.update(approved_by=actor.strip(), approved_at=now)
    else:
        values.update(rejection_reason=reason.strip())

    with session_scope() as session:
        result = session.execute(
            update(Approval)
            .where(Approval.approval_id == approval_id, Approval.status == "pending")
            .values(**values)
        )
        if result.rowcount == 1:
            row = session.get(Approval, approval_id)
            logger.info("approval %s %s by %s", approval_id, new_status, actor)
            return _to_dict(row)  # type: ignore[arg-type]
        current = session.get(Approval, approval_id)

    if current is None:
        raise ApprovalNotFoundError(f"no approval {approval_id!r}")
    raise ApprovalAlreadyDecidedError(approval_id, current.status)
