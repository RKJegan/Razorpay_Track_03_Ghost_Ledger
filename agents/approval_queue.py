"""
Ghost Ledger v3 — human approval workflow (A3).

* :func:`approve_recovery` claims the decision atomically, records it, then
  runs the recovery through :func:`agents.recovery_executor.submit_recovery`
  with ``approved=True``. The approval lifts only the R1 amount ceiling. R2 (the
  attempt cap) and R3 (the stopping rule) still apply, so an approval cannot
  override them.
* :func:`reject_recovery` claims the rejection and escalates the recovery. Nothing
  is executed.

Every decision writes an audit record with who, what and when, and a recovery
event, so the timeline shows the human step.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.policy_engine import PolicyEngine
from agents.recovery_executor import SubmitResult, submit_recovery
from api.razorpay_client import LiveRazorpayClient, SimulatedRazorpayClient
from database import audit_trail, approvals, recovery_store

logger = logging.getLogger(__name__)


def approve_recovery(
    approval_id: str,
    approved_by: str,
    *,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    engine: PolicyEngine | None = None,
) -> dict[str, Any]:
    """
    Approve a pending recovery and execute it through the policy gate.

    Returns
    -------
    dict[str, Any]
        ``{"approval": <approval after decision>, "execution": <SubmitResult as dict>}``

    Raises
    ------
    approvals.ApprovalNotFoundError
        Unknown approval id.
    approvals.ApprovalAlreadyDecidedError
        Already approved or rejected (HTTP 409 at the API).
    ValueError
        Empty approver.
    """
    approval = approvals.claim_decision(approval_id, "approved", approved_by)
    recovery_id = approval["recovery_id"]

    recovery_store.append_event(
        recovery_id, "approval_granted",
        {"approval_id": approval_id, "approved_by": approved_by},
        created_by=approved_by,
    )
    audit_trail.log(
        component="policy_engine",
        action="approval_granted",
        input_data={"approval_id": approval_id, "recovery_id": recovery_id, "amount_inr": approval["amount_inr"]},
        output_data={"approved_by": approved_by},
        decision_reason="human approved an amount above the auto-approve ceiling",
        success=True,
        entity_id=recovery_id,
    )

    try:
        result: SubmitResult = submit_recovery(
            recovery_id, approved=True, approval_id=approval_id, client=client, engine=engine,
        )
    except Exception as exc:
        # The decision stands (it was made). Record the execution failure, do not hide it.
        logger.error("approval %s granted but execution failed: %s", approval_id, exc)
        recovery_store.append_event(
            recovery_id, "link_call_failed",
            {"stage": "execution_after_approval", "error": str(exc), "approval_id": approval_id},
            created_by="approval_queue",
        )
        raise

    return {"approval": approval, "execution": _as_dict(result)}


def reject_recovery(
    approval_id: str,
    rejected_by: str,
    reason: str,
) -> dict[str, Any]:
    """
    Reject a pending recovery and escalate it. Nothing is executed.

    Raises
    ------
    approvals.ApprovalNotFoundError
        Unknown approval id.
    approvals.ApprovalAlreadyDecidedError
        Already approved or rejected (HTTP 409 at the API).
    ValueError
        Empty rejector, or empty reason.
    """
    approval = approvals.claim_decision(approval_id, "rejected", rejected_by, reason=reason)
    recovery_id = approval["recovery_id"]

    recovery_store.append_event(
        recovery_id, "approval_rejected",
        {"approval_id": approval_id, "rejected_by": rejected_by, "reason": reason},
        created_by=rejected_by,
    )
    audit_trail.log(
        component="policy_engine",
        action="approval_rejected",
        input_data={"approval_id": approval_id, "recovery_id": recovery_id, "amount_inr": approval["amount_inr"]},
        output_data={"rejected_by": rejected_by, "reason": reason},
        decision_reason="human rejected the recovery; escalated, nothing executed",
        success=True,
        entity_id=recovery_id,
    )
    return {"approval": approval, "execution": None}


def _as_dict(result: SubmitResult) -> dict[str, Any]:
    """Convert a SubmitResult to a JSON-friendly dict."""
    return {
        "recovery_id": result.recovery_id,
        "outcome": result.outcome,
        "reason": result.reason,
        "attempt": result.attempt,
        "approval_id": result.approval_id,
        "link_id": result.link_id,
        "short_url": result.short_url,
    }
