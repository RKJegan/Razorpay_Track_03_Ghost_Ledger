"""
Ghost Ledger v3 — approval routes (A3).

    GET   /api/approvals                    filter by status/merchant/cause, paginated
    GET   /api/approvals/{approval_id}      the approval plus its recovery timeline
    POST  /api/approvals/{approval_id}/approve   body: {"approved_by": "..."}
    POST  /api/approvals/{approval_id}/reject    body: {"rejected_by": "...", "reason": "..."}

Every route needs the operator key in the ``X-Operator-Key`` header. If
``OPERATOR_API_KEY`` is not configured the routes refuse everything (503), so a
misconfigured deployment cannot be approved by anonymous callers.

A second decision on the same approval returns 409. Only the first one counts.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

import config
from agents.approval_queue import approve_recovery, reject_recovery
from database import audit_trail, approvals, recovery_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


def require_operator(x_operator_key: str | None = Header(default=None)) -> str:
    """
    Check the operator key. Returns a fixed label for the audit trail.

    Raises
    ------
    HTTPException
        503 if no operator key is configured; 401 if the key is missing or wrong.
    """
    expected = config.OPERATOR_API_KEY
    if not expected:
        raise HTTPException(status_code=503, detail="operator access is not configured")
    if not x_operator_key or not hmac.compare_digest(x_operator_key, expected):
        audit_trail.log(
            component="policy_engine",
            action="operator_auth_rejected",
            input_data={"route": "approvals"},
            decision_reason="missing or wrong operator key",
            success=False,
        )
        raise HTTPException(status_code=401, detail="invalid operator key")
    return "operator"


class ApproveBody(BaseModel):
    """Who is approving."""

    approved_by: str = Field(min_length=1, max_length=120)


class RejectBody(BaseModel):
    """Who is rejecting, and why. The reason is required."""

    rejected_by: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=3, max_length=500)


@router.get("")
def list_all(
    status: Literal["pending", "approved", "rejected"] | None = Query(default=None),
    merchant_id: str | None = Query(default=None, max_length=120),
    cause: str | None = Query(default=None, max_length=60),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: str = Depends(require_operator),
) -> dict[str, Any]:
    """List approvals, newest first."""
    items, total = approvals.list_approvals(status, merchant_id, cause, limit, offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{approval_id}")
def get_one(approval_id: str, _: str = Depends(require_operator)) -> dict[str, Any]:
    """One approval with the full timeline of its recovery."""
    approval = approvals.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="approval not found")
    recovery = recovery_store.get_recovery_status(approval["recovery_id"])
    timeline = recovery_store.get_recovery_timeline(approval["recovery_id"])
    return {"approval": approval, "recovery": recovery, "timeline": timeline}


@router.post("/{approval_id}/approve")
def approve(approval_id: str, body: ApproveBody, _: str = Depends(require_operator)) -> dict[str, Any]:
    """Approve and execute. 409 if already decided."""
    try:
        return approve_recovery(approval_id, body.approved_by)
    except approvals.ApprovalNotFoundError:
        raise HTTPException(status_code=404, detail="approval not found") from None
    except approvals.ApprovalAlreadyDecidedError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "already_decided", "current_status": exc.current_status},
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post("/{approval_id}/reject")
def reject(approval_id: str, body: RejectBody, _: str = Depends(require_operator)) -> dict[str, Any]:
    """Reject and escalate. 409 if already decided."""
    try:
        return reject_recovery(approval_id, body.rejected_by, body.reason)
    except approvals.ApprovalNotFoundError:
        raise HTTPException(status_code=404, detail="approval not found") from None
    except approvals.ApprovalAlreadyDecidedError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "already_decided", "current_status": exc.current_status},
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
