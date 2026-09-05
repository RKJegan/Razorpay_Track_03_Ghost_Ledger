"""
FR-004b — Subscription Recovery Agent.

Recovers failed subscription debits by triggering a mandate retry sequence
through the Razorpay Subscriptions API (test mode), behind the same policy
engine and with the same halting behaviour as the Payment Failure Agent.

The only differences from the Payment Failure Agent are the action performed
and the entity it acts on. Policy, stopping rule and audit behaviour are
shared via `agents.recovery_base` so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from agents.policy_engine import PolicyEngine
from agents.recovery_base import RecoveryOutcome, run_recovery_sequence
from api.razorpay_client import (
    get_razorpay_client,
    LiveRazorpayClient,
    RazorpayResponse,
    SimulatedRazorpayClient,
)

AGENT_NAME = "subscription_agent"
ACTION_TYPE = "retry_mandate"


def recover_subscription_failure(
    failure_id: str,
    customer_id: str,
    amount_inr: float,
    cause: str,
    confidence: float,
    transaction_id: str,
    subscription_id: str | None = None,
    mandate_id: str | None = None,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    engine: PolicyEngine | None = None,
    approved: bool = False,
) -> RecoveryOutcome:
    """
    Run the Subscription Recovery sequence for one failed mandate debit.

    Parameters
    ----------
    failure_id : str
        Identifier of the diagnosed failure.
    customer_id : str
        Customer who owns the mandate.
    amount_inr : float
        Plan amount to recover, in rupees.
    cause : str
        Diagnosed root cause.
    confidence : float
        Diagnoser confidence; recorded, never used to gate.
    transaction_id : str
        Originating subscription debit.
    subscription_id : str, optional
        Razorpay subscription id; derived from the customer when omitted.
    mandate_id : str, optional
        Mandate identifier.
    client : SimulatedRazorpayClient | LiveRazorpayClient, optional
        Injected client; defaults to the configured one.
    engine : PolicyEngine, optional
        Injected policy engine.
    approved : bool, optional
        Explicit human approval for amounts above the ceiling.

    Returns
    -------
    RecoveryOutcome
        Outcome including recovered amount and any stopping-rule trigger.
    """
    client = client or get_razorpay_client()
    subscription_id = subscription_id or f"sub_{customer_id.lower()}"
    mandate_id = mandate_id or f"mand_{customer_id.lower()}"

    def execute(attempt: int) -> RazorpayResponse:
        return client.retry_subscription_payment(
            subscription_id=subscription_id,
            mandate_id=mandate_id,
            amount_inr=amount_inr,
            notes={
                "failure_id": failure_id,
                "transaction_id": transaction_id,
                "cause": cause,
                "attempt": attempt,
                "agent": AGENT_NAME,
            },
        )

    return run_recovery_sequence(
        failure_id=failure_id,
        customer_id=customer_id,
        amount_inr=amount_inr,
        agent_name=AGENT_NAME,
        action_type=ACTION_TYPE,
        cause=cause,
        execute=execute,
        engine=engine,
        approved=approved,
        metadata={
            "confidence": confidence,
            "transaction_id": transaction_id,
            "subscription_id": subscription_id,
            "mandate_id": mandate_id,
        },
    )
