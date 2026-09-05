"""
Shared recovery sequence for both Resurrector agents.

Additive helper module: it exists so the Payment Failure Agent and the
Subscription Agent cannot drift apart on the two things that must be identical
between them — the policy gate and the stopping rule. Both agents call
:func:`run_recovery_sequence` with their own ``execute`` callable.

The sequence, per failure:

    for attempt in 1 .. max_attempts + 1:
        count prior attempts from the recoveries table
        ask the policy engine            -> logged BEFORE any action, pass or fail
        if stopping rule fired           -> record STOPPED, halt, escalate
        if not allowed                   -> record BLOCKED, halt
        execute the action               -> record SUCCESS or FAIL
        if success                       -> halt, money recovered
    (falling off the end means the attempt cap stopped us)

Backoff is *simulated*, not slept: RETRY_BACKOFF_SECONDS is recorded in the
audit trail so the schedule is visible, but a demo must not block for 5
minutes between attempts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from agents.policy_engine import RecoveryAction, PolicyDecision, PolicyEngine
from api.razorpay_client import RazorpayResponse, response_is_recovered
from config import RETRY_BACKOFF_SECONDS
from database import audit_trail, db_client
from database.db_client import to_json


@dataclass
class RecoveryOutcome:
    """
    Result of running the recovery sequence for one failure.

    Attributes
    ----------
    failure_id : str
        Failure that was acted on.
    agent_name : str
        Agent that ran the sequence.
    outcome : str
        ``success`` | ``fail`` | ``stopped`` | ``blocked``
    recovered_amount : float
        Rupees actually recovered (0.0 unless outcome is ``success``).
    attempts : list[dict[str, Any]]
        One record per attempt, mirroring the ``recoveries`` rows.
    stopping_rule_triggered : bool
        True when the policy engine halted the sequence.
    stopping_reason : str | None
        Escalation text when halted.
    policy_reason : str | None
        Reason for the last policy decision.
    """

    failure_id: str
    agent_name: str
    outcome: str
    recovered_amount: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    stopping_rule_triggered: bool = False
    stopping_reason: str | None = None
    policy_reason: str | None = None


def _now() -> str:
    """Return the current timestamp in canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _count_prior(failure_id: str) -> tuple[int, int]:
    """
    Count prior attempts for a failure.

    Parameters
    ----------
    failure_id : str
        Failure identifier.

    Returns
    -------
    tuple[int, int]
        (prior failed-or-stopped attempts, prior total attempts).
    """
    total = int(
        db_client.scalar(
            "SELECT COUNT(*) FROM recoveries WHERE failure_id = ?", (failure_id,)
        )
        or 0
    )
    failed = int(
        db_client.scalar(
            "SELECT COUNT(*) FROM recoveries WHERE failure_id = ? "
            "AND outcome IN ('fail', 'stopped')",
            (failure_id,),
        )
        or 0
    )
    return failed, total


def _record_recovery(
    failure_id: str,
    agent_name: str,
    action_type: str,
    attempt_number: int,
    decision: PolicyDecision | None,
    outcome: str,
    recovered_amount: float,
    razorpay_response: RazorpayResponse | None,
) -> dict[str, Any]:
    """
    Insert one row into ``recoveries``.

    Parameters
    ----------
    failure_id : str
        Failure being recovered.
    agent_name : str
        Acting agent.
    action_type : str
        Action performed.
    attempt_number : int
        1-based attempt sequence number.
    decision : PolicyDecision | None
        Policy ruling, when one was taken.
    outcome : str
        ``success`` | ``fail`` | ``stopped`` | ``blocked``
    recovered_amount : float
        Rupees recovered by this attempt.
    razorpay_response : RazorpayResponse | None
        Raw API response, when an API call was made.

    Returns
    -------
    dict[str, Any]
        The row as written.
    """
    row = {
        "id": f"rec_{uuid.uuid4().hex[:18]}",
        "failure_id": failure_id,
        "agent_name": agent_name,
        "action_type": action_type,
        "attempt_number": attempt_number,
        "policy_check_passed": int(bool(decision and decision.allowed)),
        "policy_check_reason": decision.reason if decision else None,
        "stopping_rule_triggered": int(bool(decision and decision.stopping_rule_triggered)),
        "stopping_reason": decision.stopping_reason if decision else None,
        "executed_at": _now() if (decision and decision.allowed) else None,
        "outcome": outcome,
        "recovered_amount": float(recovered_amount),
        "razorpay_response": to_json(
            {
                **(razorpay_response.data if razorpay_response else {}),
                "status_code": razorpay_response.status_code if razorpay_response else None,
                "degraded": razorpay_response.degraded if razorpay_response else None,
                "error": razorpay_response.error if razorpay_response else None,
                "latency_ms": razorpay_response.latency_ms if razorpay_response else None,
            }
        ),
    }
    db_client.execute(
        "INSERT INTO recoveries (id, failure_id, agent_name, action_type, "
        "attempt_number, policy_check_passed, policy_check_reason, "
        "stopping_rule_triggered, stopping_reason, executed_at, outcome, "
        "recovered_amount, razorpay_response) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(row[k] for k in (
            "id", "failure_id", "agent_name", "action_type", "attempt_number",
            "policy_check_passed", "policy_check_reason", "stopping_rule_triggered",
            "stopping_reason", "executed_at", "outcome", "recovered_amount",
            "razorpay_response",
        )),
    )
    return row


def run_recovery_sequence(
    failure_id: str,
    customer_id: str,
    amount_inr: float,
    agent_name: str,
    action_type: str,
    cause: str,
    execute: Callable[[int], RazorpayResponse],
    engine: PolicyEngine | None = None,
    approved: bool = False,
    metadata: dict[str, Any] | None = None,
) -> RecoveryOutcome:
    """
    Run the bounded, policy-gated retry sequence for one failure.

    Parameters
    ----------
    failure_id : str
        Failure to recover.
    customer_id : str
        Customer the attempt key is scoped to.
    amount_inr : float
        Amount at stake.
    agent_name : str
        Acting agent (``payment_failure_agent`` or ``subscription_agent``).
    action_type : str
        Action label written to the audit trail.
    cause : str
        Diagnosed root cause; passed to the API as metadata.
    execute : Callable[[int], RazorpayResponse]
        Performs attempt ``n`` and returns the API response.
    engine : PolicyEngine, optional
        Policy engine; a default one is constructed if omitted.
    approved : bool, optional
        Explicit human approval flag for amounts above the ceiling.
    metadata : dict[str, Any], optional
        Extra context for the audit trail.

    Returns
    -------
    RecoveryOutcome
        Aggregated result, including whether the stopping rule fired.
    """
    engine = engine or PolicyEngine()
    outcome = RecoveryOutcome(failure_id=failure_id, agent_name=agent_name, outcome="fail")

    # One extra iteration so the STOP is actually observed and recorded on the
    # attempt AFTER the final permitted one - that is what makes the stopping
    # rule visible in the audit trail rather than merely implied.
    for attempt in range(1, engine.max_attempts + 2):
        prior_failed, prior_total = _count_prior(failure_id)

        action = RecoveryAction(
            failure_id=failure_id,
            customer_id=customer_id,
            agent_name=agent_name,
            action_type=action_type,
            amount_inr=amount_inr,
            attempt_number=attempt,
            approved=approved,
            metadata=metadata or {},
        )
        decision = engine.evaluate(
            action, prior_failed_attempts=prior_failed, prior_total_attempts=prior_total
        )
        outcome.policy_reason = decision.reason

        # R4: log the policy check BEFORE executing anything, pass or fail.
        audit_trail.log_policy_check(
            action=action_type,
            decision=decision,
            payload={
                "failure_id": failure_id,
                "customer_id": customer_id,
                "amount_inr": amount_inr,
                "attempt_number": attempt,
                "prior_failed_attempts": prior_failed,
                "prior_total_attempts": prior_total,
                "cause": cause,
                "approved": approved,
            },
            entity_id=failure_id,
        )

        # R3: stopping rule - halt permanently, escalate.
        if decision.stopping_rule_triggered:
            row = _record_recovery(
                failure_id, agent_name, action_type, attempt, decision,
                "stopped", 0.0, None,
            )
            outcome.attempts.append(row)
            outcome.outcome = "stopped"
            outcome.stopping_rule_triggered = True
            outcome.stopping_reason = decision.stopping_reason
            audit_trail.log(
                component=agent_name,
                action="escalate",
                input_data={"failure_id": failure_id, "attempt": attempt},
                output_data={"escalated_to": "manual_review"},
                decision_reason=decision.stopping_reason,
                success=True,
                entity_id=failure_id,
            )
            return outcome

        # Any other denial: halt without escalating.
        if not decision.allowed:
            row = _record_recovery(
                failure_id, agent_name, action_type, attempt, decision,
                "blocked", 0.0, None,
            )
            outcome.attempts.append(row)
            outcome.outcome = "blocked"
            audit_trail.log(
                component=agent_name,
                action="halt_policy_denial",
                input_data={"failure_id": failure_id, "attempt": attempt},
                output_data={"reason_code": decision.reason_code},
                decision_reason=decision.reason,
                success=True,
                entity_id=failure_id,
            )
            return outcome

        # --- execute ------------------------------------------------------
        resp = execute(attempt)
        recovered = amount_inr if response_is_recovered(resp) else 0.0
        attempt_outcome = "success" if recovered > 0 else "fail"
        row = _record_recovery(
            failure_id, agent_name, action_type, attempt, decision,
            attempt_outcome, recovered, resp,
        )
        outcome.attempts.append(row)

        audit_trail.log_action(
            agent_name=agent_name,
            action=action_type,
            payload={
                "failure_id": failure_id,
                "attempt": attempt,
                "amount_inr": amount_inr,
                "cause": cause,
                "backoff_seconds": RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                ],
            },
            result={
                "status_code": resp.status_code,
                "ok": resp.ok,
                "degraded": resp.degraded,
                "recovered": recovered > 0,
            },
            decision_reason=(
                f"Attempt {attempt}: {'recovered' if recovered else 'failed'}"
                + (" (served from fallback cache)" if resp.degraded else "")
            ),
            success=recovered > 0,
            entity_id=failure_id,
        )

        if recovered > 0:
            outcome.outcome = "success"
            outcome.recovered_amount = recovered
            return outcome

    # Fell through: attempt cap reached without a stop or a success.
    outcome.outcome = "fail"
    return outcome
