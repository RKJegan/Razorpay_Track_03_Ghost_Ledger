"""
Ghost Ledger v3 — operator summary for the live dashboard (A5).

    GET /api/operator/summary      operator key required

One call gives the dashboard everything it shows: live recoveries and their
latest stage, failures in progress, the approval queue, and background-job
health. Read-only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query

from agents import scheduled_jobs
from api.approvals_api import require_operator
from database import approvals, recovery_store

router = APIRouter(prefix="/api/operator", tags=["operator"])

# Stages that mean something went wrong and is still being worked on.
FAILURE_STAGES: tuple[str, ...] = (
    "payment_failed",
    "link_call_failed",
    "payment_link_expired",
    "policy_denied",
    "stopping_rule_triggered",
    "recovery_failed",
)


def build_summary(limit: int = 50) -> dict[str, Any]:
    """
    Build the dashboard payload. Separate from the route so it can be tested directly.

    Parameters
    ----------
    limit : int
        Maximum rows per list (active, failures).
    """
    active_cases, active_total = recovery_store.list_cases(status="pending", limit=limit)
    active = []
    for case in active_cases:
        latest = recovery_store.latest_event(case["recovery_id"], recovery_store.STAGES)
        active.append({
            **case,
            "latest_stage": latest["stage"] if latest else None,
            "latest_at": latest["timestamp"] if latest else None,
        })

    failures: list[dict[str, Any]] = []
    candidates, _ = recovery_store.list_cases(status="pending", limit=200)
    escalated, _ = recovery_store.list_cases(status="escalated", limit=200)
    for case in candidates + escalated:
        failure = recovery_store.latest_event(case["recovery_id"], FAILURE_STAGES)
        if failure is None:
            continue
        failures.append({
            **case,
            "failure_stage": failure["stage"],
            "failure_at": failure["timestamp"],
            "failure_detail": failure["detail"],
        })
    failures.sort(key=lambda row: row["failure_at"] or "", reverse=True)

    pending_approvals, pending_total = approvals.list_approvals("pending", None, None, limit, 0)
    escalated_total = recovery_store.list_cases(status="escalated", limit=1)[1]

    return {
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "counts": {
            "active": active_total,
            "failures_in_progress": len(failures),
            "pending_approvals": pending_total,
            "escalated": escalated_total,
        },
        "active": active,
        "failures": failures[:limit],
        "approvals": pending_approvals,
        "jobs": scheduled_jobs.job_health(),
    }


@router.get("/summary")
def summary(
    limit: int = Query(default=50, ge=1, le=200),
    _: str = Depends(require_operator),
) -> dict[str, Any]:
    """Live state for the operator dashboard."""
    return build_summary(limit)
