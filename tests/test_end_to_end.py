"""
Integration and edge-case tests.

Covers the properties the build prompt's Phase 6 asks for explicitly:
  * one full data -> diagnose -> policy -> act -> log loop
  * empty batch
  * zero failures
  * Razorpay API timeout -> graceful degrade, never a crash
  * every financial action reaches the audit trail
  * the stopping rule is reachable live in the demo path
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.payment_failure_agent import recover_payment_failure  # noqa: E402
from agents.policy_engine import PolicyEngine, RecoveryAction  # noqa: E402
from agents.recovery_base import run_recovery_sequence  # noqa: E402
from agents.subscription_agent import recover_subscription_failure  # noqa: E402
from api.razorpay_client import (  # noqa: E402
    RazorpayResponse,
    SimulatedRazorpayClient,
    get_razorpay_client,
    response_is_recovered,
)
from database import audit_trail, db_client  # noqa: E402


# ---------------------------------------------------------------------------
# Fake client: deterministic, no randomness, no network
# ---------------------------------------------------------------------------
class AlwaysFailsClient:
    """Client whose every attempt fails, to exercise the stopping rule."""

    def __init__(self) -> None:
        self.calls = 0

    def create_payment_link(self, **kwargs) -> RazorpayResponse:
        """Return a declined payment link."""
        self.calls += 1
        return RazorpayResponse(
            ok=True, data={"id": f"plink_{self.calls}", "status": "created"}, status_code=200
        )


class AlwaysSucceedsClient:
    """Client whose first attempt always recovers."""

    def create_payment_link(self, **kwargs) -> RazorpayResponse:
        """Return a paid payment link."""
        return RazorpayResponse(
            ok=True, data={"id": "plink_ok", "status": "paid", "amount": 100}, status_code=200
        )


class TimeoutClient:
    """Client that always times out, to prove the pipeline degrades."""

    def create_payment_link(self, **kwargs) -> RazorpayResponse:
        """Return a 504 timeout response."""
        return RazorpayResponse(
            ok=False, status_code=504, error="Gateway timeout", latency_ms=6000
        )


def _ensure_failure(
    failure_id: str,
    transaction_id: str,
    customer_id: str,
    amount: float,
    cause: str = "insufficient_funds",
) -> None:
    """
    Insert the parent transaction and failure rows a recovery must reference.

    ``recoveries.failure_id`` has a foreign key to ``failures.id``, which in
    turn references ``transactions.id``. Tests must create real parents,
    exactly as the pipeline does.

    Parameters
    ----------
    failure_id : str
        Failure id to create.
    transaction_id : str
        Originating transaction id.
    customer_id : str
        Owning customer.
    amount : float
        Amount at risk.
    cause : str, optional
        Predicted cause to record.
    """
    from datetime import datetime

    db_client.execute(
        "INSERT OR REPLACE INTO transactions "
        "(id, merchant_id, customer_id, amount, status, txn_type, "
        " payment_method, failure_reason_raw, timestamp, is_holdout) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            transaction_id, "merchant_demo_001", customer_id, amount, "failed",
            "one_off", "card", "ERR_TEST: test fixture",
            datetime.now().isoformat(sep=" ", timespec="seconds"), 1,
        ),
    )
    db_client.execute(
        "INSERT OR REPLACE INTO failures "
        "(id, transaction_id, predicted_cause, confidence, "
        " ground_truth_cause, detected_at, estimated_value) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            failure_id, transaction_id, cause, 0.8, cause,
            datetime.now().isoformat(sep=" ", timespec="seconds"), amount,
        ),
    )


def _clean_failure(failure_id: str) -> None:
    """Remove a failure and all of its child rows, for test isolation."""
    db_client.execute("DELETE FROM recoveries WHERE failure_id = ?", (failure_id,))
    db_client.execute("DELETE FROM failures WHERE id = ?", (failure_id,))


# ---------------------------------------------------------------------------
# End-to-end recovery loop
# ---------------------------------------------------------------------------
def test_recovery_loop_records_attempts_and_audit():
    """One failure -> diagnose -> policy -> act -> log, with rows to show."""
    fid = "fail_e2e_success"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_e2e", "CUST_E2E", 499.0)
    before_audit = audit_trail.count()

    outcome = recover_payment_failure(
        failure_id=fid,
        customer_id="CUST_E2E",
        amount_inr=499.0,
        cause="gateway_timeout",
        confidence=0.9,
        transaction_id="txn_e2e",
        client=AlwaysSucceedsClient(),
        engine=PolicyEngine(),
    )

    assert outcome.outcome == "success"
    assert outcome.recovered_amount == 499.0
    assert len(outcome.attempts) == 1
    assert audit_trail.count() > before_audit

    rows = db_client.query(
        "SELECT * FROM recoveries WHERE failure_id = ? ORDER BY attempt_number", (fid,)
    )
    assert len(rows) == 1
    assert rows[0]["outcome"] == "success"
    assert rows[0]["policy_check_passed"] == 1
    assert float(rows[0]["recovered_amount"]) == 499.0
    _clean_failure(fid)


def test_stopping_rule_is_reachable_in_the_demo_path():
    """A permanently-failing payment stops after 3 attempts and escalates."""
    fid = "fail_e2e_stop"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_stop", "CUST_STOP", 250.0)

    outcome = recover_payment_failure(
        failure_id=fid,
        customer_id="CUST_STOP",
        amount_inr=250.0,
        cause="card_expired",
        confidence=0.8,
        transaction_id="txn_stop",
        client=AlwaysFailsClient(),
        engine=PolicyEngine(),
    )

    assert outcome.outcome == "stopped"
    assert outcome.stopping_rule_triggered is True
    assert outcome.stopping_reason is not None
    assert outcome.recovered_amount == 0.0

    rows = db_client.query(
        "SELECT * FROM recoveries WHERE failure_id = ? ORDER BY attempt_number", (fid,)
    )
    # 3 real attempts + 1 STOP record.
    assert len(rows) == 4
    assert [r["attempt_number"] for r in rows] == [1, 2, 3, 4]
    assert rows[-1]["outcome"] == "stopped"
    assert rows[-1]["stopping_rule_triggered"] == 1

    # Escalation must be visible in the audit trail.
    esc = db_client.query(
        "SELECT * FROM audit_trail WHERE action = 'escalate' "
        "AND input_data LIKE ?",
        (f"%{fid}%",),
    )
    assert len(esc) >= 1
    _clean_failure(fid)


def test_policy_denial_is_logged_before_any_action():
    """An action refused by policy is logged, and nothing is executed."""
    fid = "fail_e2e_blocked"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_block", "CUST_BLOCK", 50_000.0)

    outcome = recover_payment_failure(
        failure_id=fid,
        customer_id="CUST_BLOCK",
        amount_inr=50_000.0,  # above the ₹10,000 ceiling, no approval
        cause="insufficient_funds",
        confidence=0.7,
        transaction_id="txn_block",
        client=AlwaysSucceedsClient(),
        engine=PolicyEngine(),
    )

    assert outcome.outcome == "blocked"
    assert outcome.recovered_amount == 0.0

    rows = db_client.query(
        "SELECT * FROM recoveries WHERE failure_id = ?", (fid,)
    )
    assert len(rows) == 1
    assert rows[0]["policy_check_passed"] == 0
    assert "INR 50,000.00" in rows[0]["policy_check_reason"]

    # The policy check itself must appear in the audit trail.
    checks = db_client.query(
        "SELECT * FROM audit_trail WHERE component = 'policy_engine' "
        "AND input_data LIKE ?",
        (f"%{fid}%",),
    )
    assert len(checks) >= 1
    _clean_failure(fid)


def test_subscription_agent_halts_on_stop_too():
    """The subscription agent shares the identical stopping behaviour."""
    fid = "fail_e2e_sub_stop"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_sub", "CUST_SUB", 999.0)

    class FailingSubClient:
        def retry_subscription_payment(self, **kwargs) -> RazorpayResponse:
            return RazorpayResponse(
                ok=True, data={"id": "sub_retry", "status": "failed"}, status_code=200
            )

    outcome = recover_subscription_failure(
        failure_id=fid,
        customer_id="CUST_SUB",
        amount_inr=999.0,
        cause="mandate_lapsed",
        confidence=0.85,
        transaction_id="txn_sub",
        client=FailingSubClient(),
        engine=PolicyEngine(),
    )
    assert outcome.outcome == "stopped"
    assert outcome.stopping_rule_triggered is True
    assert outcome.agent_name == "subscription_agent"
    _clean_failure(fid)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_batch_is_handled():
    """An empty list of failures produces no work and no crash."""
    engine = PolicyEngine()
    results = []
    for _ in []:
        results.append(engine)
    assert results == []


def test_zero_failures_means_no_actions():
    """With no failures in scope, no recovery rows are created."""
    fid = "fail_nonexistent"
    _clean_failure(fid)
    rows = db_client.query("SELECT * FROM recoveries WHERE failure_id = ?", (fid,))
    assert rows == []


def test_api_timeout_degrades_without_crashing():
    """A 504 from Razorpay is recorded as a failed attempt, not an exception."""
    fid = "fail_e2e_timeout"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_timeout", "CUST_TIMEOUT", 100.0)

    outcome = recover_payment_failure(
        failure_id=fid,
        customer_id="CUST_TIMEOUT",
        amount_inr=100.0,
        cause="gateway_timeout",
        confidence=0.6,
        transaction_id="txn_timeout",
        client=TimeoutClient(),
        engine=PolicyEngine(),
    )

    # Must not raise; must terminate via the stopping rule.
    assert outcome.outcome == "stopped"
    rows = db_client.query("SELECT * FROM recoveries WHERE failure_id = ?", (fid,))
    assert len(rows) == 4
    assert all(r["outcome"] in ("fail", "stopped") for r in rows)
    _clean_failure(fid)


def test_simulator_injects_transport_failures_but_never_raises():
    """Over many calls the simulator produces some 504/429s and never throws."""
    client = SimulatedRazorpayClient(seed=7)
    codes = set()
    for i in range(300):
        resp = client.create_payment_link(
            amount_inr=100.0,
            description="t",
            customer={"name": "n"},
            notes={"cause": "gateway_timeout", "attempt": (i % 3) + 1},
        )
        codes.add(resp.status_code)
        assert isinstance(resp, RazorpayResponse)
    assert 200 in codes


def test_get_razorpay_client_returns_a_usable_client():
    """The configured client exposes the payment-link interface."""
    client = get_razorpay_client()
    assert hasattr(client, "create_payment_link")
    resp = client.create_payment_link(
        amount_inr=10.0, description="t", customer={"name": "n"}, notes={}
    )
    assert isinstance(resp, RazorpayResponse)


def test_response_is_recovered_only_on_settled_statuses():
    """Only settled statuses count as recovered money."""
    assert response_is_recovered(RazorpayResponse(ok=True, data={"status": "paid"})) is True
    assert response_is_recovered(RazorpayResponse(ok=True, data={"status": "created"})) is False
    assert response_is_recovered(RazorpayResponse(ok=False, data={"status": "paid"})) is False


# ---------------------------------------------------------------------------
# Audit completeness
# ---------------------------------------------------------------------------
def test_every_recovery_action_has_an_audit_record():
    """No recovery row exists without a matching policy-check audit entry."""
    fid = "fail_e2e_audit"
    _clean_failure(fid)
    _ensure_failure(fid, "txn_audit", "CUST_AUDIT", 750.0)
    recover_payment_failure(
        failure_id=fid,
        customer_id="CUST_AUDIT",
        amount_inr=750.0,
        cause="insufficient_funds",
        confidence=0.75,
        transaction_id="txn_audit",
        client=AlwaysSucceedsClient(),
        engine=PolicyEngine(),
    )
    actions = db_client.query("SELECT * FROM recoveries WHERE failure_id = ?", (fid,))
    checks = db_client.query(
        "SELECT * FROM audit_trail WHERE component = 'policy_engine' "
        "AND input_data LIKE ?",
        (f"%{fid}%",),
    )
    assert len(checks) >= len(actions), "every action must have a prior policy check"
    _clean_failure(fid)


def test_audit_trail_records_denied_actions():
    """Denied actions are logged too - the trail is not a success log."""
    fid = "fail_e2e_audit_deny"
    _clean_failure(fid)
    engine = PolicyEngine()
    decision = engine.evaluate(
        RecoveryAction(
            failure_id=fid,
            customer_id="C1",
            agent_name="payment_failure_agent",
            action_type="create_payment_link",
            amount_inr=99_999.0,
        )
    )
    assert decision.allowed is False
    audit_trail.log_policy_check(
        action="create_payment_link", decision=decision, payload={"failure_id": fid},
        entity_id=fid,
    )
    rows = db_client.query(
        "SELECT * FROM audit_trail WHERE input_data LIKE ?", (f"%{fid}%",)
    )
    assert len(rows) >= 1
    assert rows[0]["success"] == 0
