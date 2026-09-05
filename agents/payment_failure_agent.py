"""
FR-004a — Payment Failure Agent.

Recovers one-off payment failures by creating a Razorpay **test-mode**
Payment Link and retrying with backoff, all behind the policy engine.

The agent never decides whether it may act, how much to attempt, or when to
stop. It proposes; `agents.policy_engine` disposes. If the engine says STOP,
this function returns immediately and the failure is escalated.
"""

from __future__ import annotations

from typing import Any

from agents.policy_engine import PolicyEngine
from agents.recovery_base import RecoveryOutcome, run_recovery_sequence
from api.razorpay_client import (
    get_razorpay_client,
    RazorpayResponse,
    SimulatedRazorpayClient,
    LiveRazorpayClient,
)

AGENT_NAME = "payment_failure_agent"
ACTION_TYPE = "create_payment_link"


def recover_payment_failure(
    failure_id: str,
    customer_id: str,
    amount_inr: float,
    cause: str,
    confidence: float,
    transaction_id: str,
    customer: dict[str, Any] | None = None,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    engine: PolicyEngine | None = None,
    approved: bool = False,
) -> RecoveryOutcome:
    """
    Run the Payment Failure recovery sequence for one failure.

    Parameters
    ----------
    failure_id : str
        Identifier of the diagnosed failure.
    customer_id : str
        Customer who owns the failed payment.
    amount_inr : float
        Amount to recover, in rupees.
    cause : str
        Diagnosed root cause (drives the settlement model).
    confidence : float
        Diagnoser confidence; recorded in metadata, never used to gate.
    transaction_id : str
        Originating transaction, carried into the link notes.
    customer : dict[str, Any], optional
        ``{name, email, contact}`` for the Payment Link.
    client : SimulatedRazorpayClient | LiveRazorpayClient, optional
        Injected client; defaults to the configured one.
    engine : PolicyEngine, optional
        Injected policy engine; defaults to a standard one.
    approved : bool, optional
        Explicit human approval for amounts above the auto-approve ceiling.

    Returns
    -------
    RecoveryOutcome
        Outcome including recovered amount and any stopping-rule trigger.
    """
    client = client or get_razorpay_client()
    customer = customer or {
        "name": f"Customer {customer_id}",
        "email": f"{customer_id.lower()}@example.in",
        "contact": "+919900000000",
    }

    def execute(attempt: int) -> RazorpayResponse:
        return client.create_payment_link(
            amount_inr=amount_inr,
            description=f"Payment recovery - {transaction_id}",
            customer=customer,
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
        metadata={"confidence": confidence, "transaction_id": transaction_id},
    )
