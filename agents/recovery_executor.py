"""
Ghost Ledger v3 — recovery executor (A3, shared by approvals, jobs and the demo).

:func:`submit_recovery` is the single path by which a v3 recovery can create a
payment link. It is always:

1. **Gated** by :class:`agents.policy_engine.PolicyEngine` — the same rules as
   v2 (R1 ceiling, R2 attempt cap, R3 stopping rule). Nothing here can skip them.
2. **Logged before acting.** The policy ruling is written to the audit trail,
   pass or fail, before any Razorpay call.
3. **Recorded as events.** Every outcome is an immutable recovery event.

Outcomes
--------
``link_created``       a payment link was created; payment NOT yet confirmed
``awaiting_approval``  amount above the ceiling; an approval request was opened
``stopped``            R3 fired: escalated, no further automated attempts
``denied``             another policy rule blocked the attempt (for example R2)
``link_failed``        the Razorpay call failed; nothing was recorded as created
``not_pending``        the recovery is already settled, failed or escalated
``retry_scheduled``    (strategies on) the timing rule deferred the attempt; no link yet
``dunning_only``       (strategies on) the playbook sends reminders, no payment link

Trust boundary: the LLM is not involved. The decision is deterministic Python,
and the Razorpay response is a fact to record, not a settlement to assume. A
link response that says ``paid`` (the simulator does this) does not settle the
recovery. Only the verified webhook or a confirmed status read can do that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents.policy_engine import PolicyDecision, PolicyEngine, RecoveryAction
from api.razorpay_client import (
    LiveRazorpayClient,
    RazorpayResponse,
    SimulatedRazorpayClient,
    get_razorpay_client,
)
import config
from config import RAZORPAY_LIVE_TEST_MODE
from database import audit_trail, approvals, recovery_store

logger = logging.getLogger(__name__)

AGENT_NAME = "payment_failure_agent"
ACTION_TYPE = "create_payment_link"
DEFAULT_CUSTOMER: dict[str, str] = {
    "name": "Recovery customer",
    "email": "customer@example.in",
    "contact": "+919900000000",
}

#: Stages that count as one attempted payment link (the attempt number).
LINK_STAGES: tuple[str, ...] = ("payment_link_created",)
#: Stages that count as one failed attempt (R3 counts these).
FAILED_STAGES: tuple[str, ...] = ("payment_failed", "recovery_failed")


@dataclass(frozen=True)
class SubmitResult:
    """What happened when a recovery was submitted for execution."""

    recovery_id: str
    outcome: str
    reason: str
    attempt: int
    approval_id: str | None = None
    link_id: str | None = None
    short_url: str | None = None


def _attempt_count(recovery_id: str) -> int:
    """Return how many payment links this recovery has already had."""
    return recovery_store.count_events(recovery_id, LINK_STAGES)


@dataclass(frozen=True)
class Ruling:
    """The policy ruling for one recovery. Made and logged BEFORE anything runs (R4)."""

    recovery_id: str
    case: dict[str, Any]
    attempt: int
    decision: PolicyDecision
    approved: bool
    approval_id: str | None


def rule_on_recovery(
    recovery_id: str,
    *,
    approved: bool = False,
    approval_id: str | None = None,
    engine: PolicyEngine | None = None,
) -> "Ruling | SubmitResult":
    """
    Rule on the next attempt for one recovery. Performs NO side effects except the R4 audit entry.

    Returns a :class:`Ruling` to be executed with :func:`execute_ruling`, or a
    :class:`SubmitResult` (``not_pending``) when there is nothing to rule on.
    """
    case = recovery_store.get_recovery_status(recovery_id)
    if case is None:
        raise recovery_store.UnknownRecoveryError(f"no recovery {recovery_id!r}")
    engine = engine or PolicyEngine()

    if case["status"] != "pending":
        return SubmitResult(recovery_id, "not_pending", f"recovery is {case['status']}", 0)

    attempt = _attempt_count(recovery_id) + 1
    prior_failed = recovery_store.count_events(recovery_id, FAILED_STAGES)
    prior_total = _attempt_count(recovery_id)
    customer_id = case.get("customer_id") or "unknown_customer"

    action = RecoveryAction(
        failure_id=recovery_id,
        customer_id=customer_id,
        agent_name=AGENT_NAME,
        action_type=ACTION_TYPE,
        amount_inr=float(case["amount_inr"]),
        attempt_number=attempt,
        approved=approved,
        metadata={"merchant_id": case["merchant_id"], "txn_id": case["txn_id"]},
    )
    decision = engine.evaluate(
        action, prior_failed_attempts=prior_failed, prior_total_attempts=prior_total
    )
    # R4: the ruling is logged before anything is executed, pass or fail.
    audit_trail.log_policy_check(
        action=ACTION_TYPE,
        decision=decision,
        payload={
            "recovery_id": recovery_id,
            "merchant_id": case["merchant_id"],
            "amount_inr": case["amount_inr"],
            "attempt": attempt,
            "approved": approved,
            "approval_id": approval_id,
        },
        entity_id=recovery_id,
    )
    return Ruling(recovery_id, case, attempt, decision, approved, approval_id)


def execute_ruling(
    ruling: Ruling,
    *,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    customer: dict[str, str] | None = None,
) -> SubmitResult:
    """Act on a ruling made by :func:`rule_on_recovery`. Never re-rules."""
    recovery_id = ruling.recovery_id
    case = ruling.case
    decision = ruling.decision
    attempt = ruling.attempt
    client = client or get_razorpay_client()

    if decision.stopping_rule_triggered:
        recovery_store.append_event(
            recovery_id, "stopping_rule_triggered",
            {"reason": decision.reason, "rule": decision.rule}, created_by="policy_engine",
        )
        return SubmitResult(recovery_id, "stopped", decision.reason, attempt)

    if decision.requires_approval and not ruling.approved:
        return _request_approval(recovery_id, case, attempt, decision)

    if not decision.allowed:
        recovery_store.append_event(
            recovery_id, "policy_denied",
            {"reason": decision.reason, "rule": decision.rule, "reason_code": decision.reason_code},
            created_by="policy_engine",
        )
        return SubmitResult(recovery_id, "denied", decision.reason, attempt)

    recovery_store.append_event(
        recovery_id, "policy_check_passed",
        {"rule": decision.rule, "reason_code": decision.reason_code, "approval_id": ruling.approval_id},
        created_by="policy_engine",
    )
    return _create_link(recovery_id, case, attempt, client, customer, ruling.approval_id)


def submit_direct(
    recovery_id: str,
    *,
    approved: bool = False,
    approval_id: str | None = None,
    customer: dict[str, str] | None = None,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    engine: PolicyEngine | None = None,
) -> SubmitResult:
    """The v2-compatible path: rule, then execute. No strategy layer involved."""
    ruling = rule_on_recovery(recovery_id, approved=approved, approval_id=approval_id, engine=engine)
    if isinstance(ruling, SubmitResult):
        return ruling
    return execute_ruling(ruling, client=client, customer=customer)


def submit_recovery(
    recovery_id: str,
    *,
    approved: bool = False,
    approval_id: str | None = None,
    customer: dict[str, str] | None = None,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    engine: PolicyEngine | None = None,
) -> SubmitResult:
    """
    Single entry point for v3 recovery actions (approvals, jobs, demo).

    With ``ENABLE_ADVANCED_STRATEGIES`` off (the default) this is exactly
    :func:`submit_direct`. With it on, the strategy layer (B1-B8) decides
    whether to create a link now, schedule a retry, or send dunning only. The
    policy ruling still comes first in both cases.
    """
    if config.ENABLE_ADVANCED_STRATEGIES:
        from strategies.runner import handle_failure  # local import: avoids a cycle

        return handle_failure(
            recovery_id, approved=approved, approval_id=approval_id,
            customer=customer, client=client, engine=engine,
        )
    return submit_direct(
        recovery_id, approved=approved, approval_id=approval_id,
        customer=customer, client=client, engine=engine,
    )


def _request_approval(
    recovery_id: str, case: dict[str, Any], attempt: int, decision: PolicyDecision
) -> SubmitResult:
    """Open one pending approval for this recovery (never two at once)."""
    existing = approvals.get_pending_approval_for(recovery_id)
    if existing is not None:
        return SubmitResult(
            recovery_id, "awaiting_approval", "approval already pending",
            attempt, approval_id=existing["approval_id"],
        )
    approval_id = approvals.create_approval(
        recovery_id=recovery_id,
        merchant_id=case["merchant_id"],
        txn_id=case["txn_id"],
        amount_inr=float(case["amount_inr"]),
        action=ACTION_TYPE,
        cause=case.get("cause"),
        confidence=None,
    )
    recovery_store.append_event(
        recovery_id, "approval_requested",
        {"approval_id": approval_id, "amount_inr": case["amount_inr"], "reason": decision.reason},
        created_by="policy_engine",
    )
    return SubmitResult(
        recovery_id, "awaiting_approval", decision.reason, attempt, approval_id=approval_id,
    )


def _create_link(
    recovery_id: str,
    case: dict[str, Any],
    attempt: int,
    client: SimulatedRazorpayClient | LiveRazorpayClient,
    customer: dict[str, str] | None,
    approval_id: str | None,
) -> SubmitResult:
    """Call Razorpay for one payment link and record the outcome as events."""
    amount = float(case["amount_inr"])
    reference_id = f"{recovery_id}-a{attempt}"
    response: RazorpayResponse = client.create_payment_link(
        amount_inr=amount,
        description=f"Payment recovery - {case['txn_id']}",
        customer=customer or DEFAULT_CUSTOMER,
        notes={
            "recovery_id": recovery_id,
            "txn_id": case["txn_id"],
            "merchant_id": case["merchant_id"],
            "attempt": attempt,
            "cause": case.get("cause") or "",
            "agent": AGENT_NAME,
        },
        reference_id=reference_id,
    )

    if not response.ok:
        recovery_store.append_event(
            recovery_id, "link_call_failed",
            {"stage": "payment_link", "error": response.error, "status_code": response.status_code, "attempt": attempt},
            created_by="razorpay_client",
        )
        return SubmitResult(recovery_id, "link_failed", response.error or "link call failed", attempt)

    # In live mode a degraded response is the simulator standing in for a failed
    # real call. It must never be recorded as a real payment link.
    if response.degraded and RAZORPAY_LIVE_TEST_MODE:
        recovery_store.append_event(
            recovery_id, "link_call_failed",
            {"stage": "payment_link", "error": response.error, "attempt": attempt,
             "discarded": "simulated response discarded in live mode"},
            created_by="razorpay_client",
        )
        return SubmitResult(recovery_id, "link_failed", "live call failed; simulated link discarded", attempt)

    data = response.data
    link_id = str(data.get("id"))
    short_url = data.get("short_url")
    recovery_store.append_event(
        recovery_id, "payment_link_created",
        {
            "link_id": link_id,
            "short_url": short_url,
            "reference_id": reference_id,
            "amount_paise": int(round(amount * 100)),
            "attempt": attempt,
            "approval_id": approval_id,
            "link_status_reported": data.get("status"),
            "expire_by": data.get("expire_by"),
            "simulated": bool(data.get("simulated")),
            "settlement_confirmed": False,
        },
        created_by="payment_failure_agent",
    )
    logger.info("recovery %s: payment link %s created (attempt %d)", recovery_id, link_id, attempt)
    return SubmitResult(recovery_id, "link_created", "payment link created; awaiting payment confirmation",
                        attempt, approval_id=approval_id, link_id=link_id, short_url=short_url)
