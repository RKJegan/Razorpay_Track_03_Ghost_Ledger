"""
Tests for A2 — event-sourced recovery tracking (database/recovery_store.py),
and for the append-only guarantee on recovery_events.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from database import merchants, recovery_store
from database.engine import session_scope


@pytest.fixture(scope="module")
def merchant_id() -> str:
    """One merchant shared by this module (the store holds at most 10)."""
    mid = "m_test_recovery_store"
    if not merchants.list_merchants() or mid not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(mid, "Recovery store tests")
    return mid


def _new_case(merchant_id: str, amount: float = 499.0) -> str:
    return recovery_store.create_case(
        merchant_id=merchant_id,
        txn_id=f"txn_{uuid.uuid4().hex[:8]}",
        amount_inr=amount,
        cause="card_expired",
    )


def test_new_case_starts_pending(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    status = recovery_store.get_recovery_status(rid)
    assert status is not None
    assert status["status"] == "pending"
    assert status["settled_at"] is None


def test_case_rejects_non_positive_amount(merchant_id: str) -> None:
    with pytest.raises(ValueError):
        recovery_store.create_case(merchant_id, "txn_bad", 0.0)


def test_unknown_stage_is_rejected(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    with pytest.raises(recovery_store.UnknownStageError):
        recovery_store.append_event(rid, "made_up_stage")


def test_unknown_recovery_is_rejected() -> None:
    with pytest.raises(recovery_store.UnknownRecoveryError):
        recovery_store.append_event("rcv_does_not_exist", "payment_captured")


def test_timeline_is_ordered_and_complete(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    stages = ["diagnosis_created", "policy_check_passed", "playbook_selected", "payment_link_created"]
    for stage in stages:
        recovery_store.append_event(rid, stage, {"stage_name": stage}, created_by="test")
    timeline = recovery_store.get_recovery_timeline(rid)
    assert [e["stage"] for e in timeline] == stages
    assert all(e["created_by"] == "test" for e in timeline)
    assert timeline[0]["detail"] == {"stage_name": "diagnosis_created"}


def test_capture_settles_and_is_terminal(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    recovery_store.append_event(rid, "payment_captured", {"payment_id": "pay_a"}, "razorpay_webhook")
    settled = recovery_store.get_recovery_status(rid)
    assert settled["status"] == "settled"
    assert settled["settled_at"] is not None

    # A late failure for an earlier attempt must NOT undo the settlement.
    recovery_store.append_event(rid, "payment_failed", {"payment_id": "pay_b"}, "razorpay_webhook")
    assert recovery_store.get_recovery_status(rid)["status"] == "settled"
    assert len(recovery_store.get_recovery_timeline(rid)) == 2  # the event is still recorded


def test_failure_then_recovery_can_still_settle(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    recovery_store.append_event(rid, "recovery_failed", {"reason": "attempt 1"})
    assert recovery_store.get_recovery_status(rid)["status"] == "failed"
    recovery_store.append_event(rid, "payment_captured", {"payment_id": "pay_c"})
    assert recovery_store.get_recovery_status(rid)["status"] == "settled"


def test_stopping_rule_escalates(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    recovery_store.append_event(rid, "stopping_rule_triggered", {"limit": 3})
    assert recovery_store.get_recovery_status(rid)["status"] == "escalated"


def test_active_and_settled_queries(merchant_id: str) -> None:
    pending = _new_case(merchant_id)
    settled = _new_case(merchant_id)
    recovery_store.append_event(settled, "payment_captured", {"payment_id": "pay_d"})

    active_ids = {r["recovery_id"] for r in recovery_store.get_active_recoveries(merchant_id)}
    assert pending in active_ids
    assert settled not in active_ids

    settled_ids = {r["recovery_id"] for r in recovery_store.get_settled_today(merchant_id)}
    assert settled in settled_ids
    assert pending not in settled_ids


def test_merchant_scope_filters_queries(merchant_id: str) -> None:
    other = "m_test_other_scope"
    if other not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(other, "Other merchant")
    mine = _new_case(merchant_id)
    theirs = recovery_store.create_case(other, "txn_other", 250.0)
    active = {r["recovery_id"] for r in recovery_store.get_active_recoveries(merchant_id)}
    assert mine in active
    assert theirs not in active


def test_recovery_events_are_append_only(merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    event_id = recovery_store.append_event(rid, "diagnosis_created", {"cause": "card_expired"})
    with pytest.raises(DBAPIError):
        with session_scope() as session:
            session.execute(
                text("UPDATE recovery_events SET stage = 'payment_captured' WHERE id = :i"),
                {"i": event_id},
            )
    with pytest.raises(DBAPIError):
        with session_scope() as session:
            session.execute(text("DELETE FROM recovery_events WHERE id = :i"), {"i": event_id})
    # The original row is untouched.
    assert recovery_store.get_recovery_timeline(rid)[0]["stage"] == "diagnosis_created"


def test_timeline_and_status_are_indexed() -> None:
    """NFR4: the hot query paths must have indexes."""
    with session_scope() as session:
        rows = session.execute(text("PRAGMA index_list('recovery_events')")).fetchall()
        names = {r[1] for r in rows}
        assert "ix_events_recovery_ts" in names
        rows = session.execute(text("PRAGMA index_list('recovery_cases')")).fetchall()
        names = {r[1] for r in rows}
        assert "ix_cases_merchant_status" in names
