"""
Tests for the recovery policy engine (FR-003).

The 3-attempt stopping rule is the single most important behaviour in the
project: it is the guardrail that stops an autonomous agent from endlessly
hammering a customer. These tests are deliberately exhaustive about it.

The engine is pure and stateless, so every test is deterministic and needs no
fixtures, database, or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.policy_engine import (  # noqa: E402
    PolicyEngine,
    RecoveryAction,
    REASON_AMOUNT_LIMIT,
    REASON_ATTEMPT_LIMIT,
    REASON_INVALID_AMOUNT,
    REASON_OK,
    REASON_STOPPING_RULE,
    REASON_UNKNOWN_AGENT,
)
from config import (  # noqa: E402
    POLICY_MAX_ATTEMPTS_PER_FAILURE,
    POLICY_MAX_AUTO_APPROVE_INR,
)


def make_action(**overrides) -> RecoveryAction:
    """Build a valid, unremarkable action with optional overrides."""
    base = dict(
        failure_id="fail_001",
        customer_id="CUST_00001",
        agent_name="payment_failure_agent",
        action_type="create_payment_link",
        amount_inr=999.0,
        attempt_number=1,
        approved=False,
    )
    base.update(overrides)
    return RecoveryAction(**base)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_small_amount_first_attempt_is_allowed():
    """A normal action passes with no prior attempts."""
    d = PolicyEngine().evaluate(make_action(), prior_failed_attempts=0, prior_total_attempts=0)
    assert d.allowed is True
    assert d.reason_code == REASON_OK
    assert d.stopping_rule_triggered is False


def test_engine_is_deterministic():
    """Identical inputs produce identical decisions, every time."""
    engine = PolicyEngine()
    a = make_action()
    results = {engine.evaluate(a, 1, 1).reason_code for _ in range(50)}
    assert len(results) == 1


def test_attempt_key_is_customer_and_failure():
    """Attempts are counted per (customer, failure), not globally."""
    engine = PolicyEngine()
    a = make_action(customer_id="C1", failure_id="F1")
    assert engine.attempt_key(a) == ("C1", "F1")


# ---------------------------------------------------------------------------
# R1 — amount ceiling
# ---------------------------------------------------------------------------
def test_amount_above_ceiling_blocked_without_approval():
    """Over ₹10,000 with no approval flag: blocked, flagged as needing approval."""
    d = PolicyEngine().evaluate(make_action(amount_inr=10_000.01), 0, 0)
    assert d.allowed is False
    assert d.reason_code == REASON_AMOUNT_LIMIT
    assert d.requires_approval is True


def test_amount_at_ceiling_is_allowed():
    """Exactly ₹10,000 is within the ceiling (boundary is exclusive)."""
    d = PolicyEngine().evaluate(make_action(amount_inr=POLICY_MAX_AUTO_APPROVE_INR), 0, 0)
    assert d.allowed is True


def test_amount_above_ceiling_allowed_with_approval_flag():
    """With an explicit approval flag the same action is permitted."""
    d = PolicyEngine().evaluate(make_action(amount_inr=50_000.0, approved=True), 0, 0)
    assert d.allowed is True
    assert d.requires_approval is False


def test_very_large_amount_still_blocked_without_approval():
    """No amount is special-cased into being allowed."""
    d = PolicyEngine().evaluate(make_action(amount_inr=5_000_000.0), 0, 0)
    assert d.allowed is False
    assert d.reason_code == REASON_AMOUNT_LIMIT


# ---------------------------------------------------------------------------
# R2 — attempt cap
# ---------------------------------------------------------------------------
def test_attempts_1_2_and_3_are_all_allowed():
    """Three attempts are permitted when none have failed permanently."""
    engine = PolicyEngine()
    for prior_total in (0, 1, 2):
        d = engine.evaluate(
            make_action(attempt_number=prior_total + 1),
            prior_failed_attempts=0,
            prior_total_attempts=prior_total,
        )
        assert d.allowed is True, f"attempt {prior_total + 1} should be allowed"


def test_fourth_attempt_blocked_by_cap():
    """A 4th attempt is refused once 3 have already been recorded."""
    d = PolicyEngine().evaluate(make_action(attempt_number=4), 0, prior_total_attempts=3)
    assert d.allowed is False
    assert d.reason_code == REASON_ATTEMPT_LIMIT


def test_attempt_number_beyond_cap_is_refused():
    """An out-of-range attempt number is refused regardless of DB state."""
    d = PolicyEngine().evaluate(
        make_action(attempt_number=POLICY_MAX_ATTEMPTS_PER_FAILURE + 5), 0, 0
    )
    assert d.allowed is False


# ---------------------------------------------------------------------------
# R3 — the 3-failed-attempt stopping rule
# ---------------------------------------------------------------------------
def test_three_failed_attempts_trigger_stop():
    """THE critical test: 3 prior failures -> STOP, escalate, log reason."""
    d = PolicyEngine().evaluate(make_action(attempt_number=4), prior_failed_attempts=3)
    assert d.allowed is False
    assert d.stopping_rule_triggered is True
    assert d.reason_code == REASON_STOPPING_RULE
    assert d.stopping_reason is not None
    assert "escalat" in d.stopping_reason.lower()


def test_two_failed_attempts_do_not_stop():
    """Two failures still permit a third attempt."""
    d = PolicyEngine().evaluate(make_action(attempt_number=3), prior_failed_attempts=2)
    assert d.allowed is True
    assert d.stopping_rule_triggered is False


def test_stopping_rule_takes_precedence_over_amount_rule():
    """A stopped failure reports STOP, not a generic amount denial."""
    d = PolicyEngine().evaluate(
        make_action(amount_inr=99_999.0, attempt_number=4), prior_failed_attempts=3
    )
    assert d.stopping_rule_triggered is True
    assert d.reason_code == REASON_STOPPING_RULE


def test_stopping_rule_survives_an_approval_flag():
    """Human approval does not override the stopping rule."""
    d = PolicyEngine().evaluate(
        make_action(amount_inr=50_000.0, approved=True, attempt_number=4),
        prior_failed_attempts=3,
    )
    assert d.allowed is False
    assert d.stopping_rule_triggered is True


def test_stopping_rule_survives_more_than_three_failures():
    """Beyond 3 failures it still reports STOP, never silently allowed."""
    for n in (3, 4, 10, 99):
        d = PolicyEngine().evaluate(make_action(), prior_failed_attempts=n)
        assert d.allowed is False
        assert d.stopping_rule_triggered is True


def test_stop_reason_names_customer_and_failure():
    """The escalation text identifies the case for a human reviewer."""
    d = PolicyEngine().evaluate(
        make_action(customer_id="CUST_9", failure_id="F_9"), prior_failed_attempts=3
    )
    assert "CUST_9" in d.stopping_reason
    assert "F_9" in d.stopping_reason


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_amount", [0.0, -1.0, -9999.0, None])
def test_invalid_amounts_are_refused(bad_amount):
    """Non-positive or missing amounts never pass."""
    d = PolicyEngine().evaluate(make_action(amount_inr=bad_amount), 0, 0)
    assert d.allowed is False
    assert d.reason_code == REASON_INVALID_AMOUNT


def test_unknown_agent_is_refused():
    """An unregistered agent cannot act."""
    d = PolicyEngine().evaluate(make_action(agent_name="rogue_agent"), 0, 0)
    assert d.allowed is False
    assert d.reason_code == REASON_UNKNOWN_AGENT


def test_known_agents_are_accepted():
    """Both spec'd agents are recognised."""
    for agent in ("payment_failure_agent", "subscription_agent"):
        d = PolicyEngine().evaluate(make_action(agent_name=agent), 0, 0)
        assert d.allowed is True


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------
def test_limits_are_configurable():
    """Overriding limits changes behaviour, proving no hidden constants."""
    engine = PolicyEngine(max_attempts=1, stop_on_failed_attempt=1)
    d = engine.evaluate(make_action(), prior_failed_attempts=1)
    assert d.stopping_rule_triggered is True

    engine2 = PolicyEngine(max_auto_approve_inr=100.0)
    assert engine2.evaluate(make_action(amount_inr=500.0), 0, 0).allowed is False
    assert engine2.evaluate(make_action(amount_inr=50.0), 0, 0).allowed is True
