"""
Tests for A4 — scheduled jobs: settlement polling, reconciliation, retries,
expired-link cleanup, backoff, and the scheduler wiring.

The Razorpay client is a fake with fixed answers, so every test is deterministic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

import config
from agents import scheduled_jobs as jobs
from agents.recovery_executor import submit_recovery
from api import scheduler as scheduler_mod
from api import webhooks
from api.main import app
from api.razorpay_client import RazorpayResponse
from database import merchants, recovery_store
from database.engine import session_scope
from database.models import JobState

SECRET = "whsec_test_only_jobs_not_real"
BELOW = 499.0


class FakeLinks:
    """Records link creation; answers fetches from a settable status."""

    def __init__(self) -> None:
        self.status = "created"
        self.paid_paise = 0
        self.payments: list[dict[str, Any]] = []
        self.degraded = False
        self.fail_fetch = False
        self.created: list[dict[str, Any]] = []
        self.expire_by: int | None = None

    def create_payment_link(self, amount_inr, description, customer, notes=None, reference_id=None):  # noqa: ANN001
        link_id = f"plink_job_{uuid.uuid4().hex[:10]}"
        self.created.append({"reference_id": reference_id, "link_id": link_id})
        return RazorpayResponse(ok=True, data={
            "id": link_id, "short_url": "https://rzp.io/i/job", "status": "created",
            "expire_by": self.expire_by, "simulated": True,
        })

    def fetch_payment_link(self, link_id: str) -> RazorpayResponse:
        if self.fail_fetch:
            return RazorpayResponse(ok=False, status_code=503, error="upstream unavailable (fake)")
        return RazorpayResponse(
            ok=True,
            data={"id": link_id, "status": self.status, "amount_paid": self.paid_paise,
                  "payments": self.payments},
            degraded=self.degraded,
        )


@pytest.fixture(scope="module")
def merchant_id() -> str:
    mid = "m_test_jobs"
    if mid not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(mid, "Job tests")
    return mid


@pytest.fixture
def fake() -> FakeLinks:
    return FakeLinks()


def _case(merchant_id: str, amount: float = BELOW) -> str:
    return recovery_store.create_case(merchant_id, f"txn_{uuid.uuid4().hex[:8]}", amount,
                                      cause="card_expired", customer_id="CUST_JOB")


def _open_link(rid: str, fake: FakeLinks) -> str:
    result = submit_recovery(rid, client=fake)
    assert result.outcome == "link_created", result
    return result.link_id


def _stages(rid: str) -> list[str]:
    return [e["stage"] for e in recovery_store.get_recovery_timeline(rid)]


def _job_name() -> str:
    return f"test_job_{uuid.uuid4().hex[:8]}"


# --- backoff and the job guard ----------------------------------------------

def test_backoff_doubles_and_is_capped() -> None:
    assert jobs.backoff_seconds(1, base=30, cap=1800) == 30
    assert jobs.backoff_seconds(2, base=30, cap=1800) == 60
    assert jobs.backoff_seconds(3, base=30, cap=1800) == 120
    assert jobs.backoff_seconds(10, base=30, cap=1800) == 1800
    assert jobs.backoff_seconds(0, base=30, cap=1800) == 0


def test_failing_job_backs_off_then_recovers_and_resets() -> None:
    name = _job_name()
    start = datetime(2026, 10, 8, 10, 0, 0)

    def boom() -> None:
        raise RuntimeError("razorpay down")

    first = jobs.run_guarded(name, boom, now=start)
    assert first["status"] == "failed" and first["streak"] == 1
    # Inside the backoff window the job does not run at all.
    skipped = jobs.run_guarded(name, lambda: "should not run", now=start + timedelta(seconds=10))
    assert skipped["status"] == "skipped"
    second = jobs.run_guarded(name, boom, now=start + timedelta(seconds=31))
    assert second["status"] == "failed" and second["streak"] == 2
    # After the window, a successful run clears the streak.
    ok = jobs.run_guarded(name, lambda: {"done": 1}, now=start + timedelta(seconds=31 + 61))
    assert ok["status"] == "ok"
    with session_scope() as session:
        row = session.get(JobState, name)
        assert row.consecutive_failures == 0 and row.next_allowed_at is None and row.last_status == "ok"
        session.delete(row)  # keep the shared DB free of test-only job rows


def test_failed_job_never_raises_out_of_the_guard() -> None:
    outcome = jobs.run_guarded(_job_name(), lambda: 1 / 0)
    assert outcome["status"] == "failed"


# --- settlement polling -----------------------------------------------------

def test_unpaid_link_is_left_alone(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    assert jobs.check_payment_settlement(rid, client=fake) == "not_paid"
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"


def test_paid_link_is_provisional_until_the_webhook_confirms(
        merchant_id: str, fake: FakeLinks, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks, "RAZORPAY_WEBHOOK_SECRET", SECRET)
    rid = _case(merchant_id, amount=BELOW)
    _open_link(rid, fake)
    fake.status = "paid"
    fake.paid_paise = 49900

    assert jobs.check_payment_settlement(rid, client=fake) == "provisional_settlement"
    assert recovery_store.get_recovery_status(rid)["status"] == "settled"
    assert recovery_store.has_provisional_settlement(rid)
    # A second poll must not record it again.
    assert jobs.check_payment_settlement(rid, client=fake) == "already_recorded"
    assert _stages(rid).count("payment_captured") == 1

    # The webhook (with the payment id) confirms it: one settlement, not two.
    payment_id = f"pay_{uuid.uuid4().hex[:14]}"
    body = json.dumps({
        "entity": "event", "event": "payment.captured", "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": payment_id, "amount": 49900, "currency": "INR", "status": "captured",
            "notes": {"recovery_id": rid, "txn_id": "txn_job"}}}},
        "created_at": 1760000000,
    }).encode()
    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    with TestClient(app) as c:
        resp = c.post("/webhooks/razorpay", content=body, headers={
            "Content-Type": "application/json", "X-Razorpay-Signature": sig,
            "X-Razorpay-Event-Id": f"evt_{uuid.uuid4().hex}"})
    assert resp.status_code == 200
    stages = _stages(rid)
    assert stages.count("payment_captured") == 1
    assert stages[-1] == "settlement_confirmed"
    assert not recovery_store.has_provisional_settlement(rid)


def test_degraded_or_failed_poll_changes_nothing(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    fake.status = "paid"
    fake.paid_paise = 49900
    fake.degraded = True
    with pytest.raises(jobs.TransientJobError):
        jobs.check_payment_settlement(rid, client=fake)
    fake.degraded = False
    fake.fail_fetch = True
    with pytest.raises(jobs.TransientJobError):
        jobs.check_payment_settlement(rid, client=fake)
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"
    assert "payment_captured" not in _stages(rid)


def test_poll_refuses_a_link_with_a_different_amount(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id, amount=BELOW)
    _open_link(rid, fake)
    fake.status = "paid"
    fake.paid_paise = 10000
    assert jobs.check_payment_settlement(rid, client=fake) == "not_paid"
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"


# --- reconciliation ---------------------------------------------------------

def test_reconcile_waits_for_a_confirmed_settlement(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    fake.status, fake.paid_paise = "paid", 49900
    jobs.check_payment_settlement(rid, client=fake)
    assert jobs.reconcile_recovery(rid) == "awaiting_confirmation"
    assert "settlement_confirmed" not in _stages(rid)


def test_reconcile_flags_an_amount_mismatch_without_closing(merchant_id: str) -> None:
    rid = _case(merchant_id, amount=BELOW)
    recovery_store.append_event(rid, "payment_captured",
                                {"payment_id": "pay_mismatch", "amount_paise": 10000, "source": "webhook"})
    assert jobs.reconcile_recovery(rid) == "mismatch"
    assert "settlement_confirmed" not in _stages(rid)


def test_reconcile_matches_and_is_idempotent(merchant_id: str) -> None:
    rid = _case(merchant_id, amount=BELOW)
    recovery_store.append_event(rid, "payment_captured",
                                {"payment_id": "pay_match", "amount_paise": 49900, "source": "webhook"})
    assert jobs.reconcile_recovery(rid) == "reconciled"
    assert jobs.reconcile_recovery(rid) == "already_reconciled"
    assert _stages(rid).count("settlement_confirmed") == 1


# --- retries ----------------------------------------------------------------

def test_active_link_is_never_replaced(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    assert jobs.retry_recovery(rid, client=fake) == "link_still_active"
    assert len(fake.created) == 1


def test_payment_failure_triggers_the_next_attempt(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    recovery_store.append_event(rid, "payment_failed", {"payment_id": "pay_failed_1"})
    assert jobs.retry_recovery(rid, client=fake) == "link_created"
    assert len(fake.created) == 2
    assert fake.created[1]["reference_id"] == f"{rid}-a2"  # new attempt, new reference


def test_expired_link_with_attempts_left_queues_a_reminder(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    _open_link(rid, fake)
    fake.status = "expired"
    # The sweep covers every open recovery in the shared test DB, so assert on this one only.
    jobs.clean_expired_links(client=fake)
    assert _stages(rid)[-2:] == ["payment_link_expired", "reminder_queued"]
    # The reminder does not repeat on the next run.
    jobs.clean_expired_links(client=fake)
    assert _stages(rid).count("reminder_queued") == 1


def test_expired_link_at_the_attempt_cap_escalates(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    for _ in range(3):
        assert submit_recovery(rid, client=fake).outcome == "link_created"
    fake.status = "expired"
    jobs.clean_expired_links(client=fake)
    assert recovery_store.get_recovery_status(rid)["status"] == "escalated"
    assert "stopping_rule_triggered" in _stages(rid)


def test_expiry_by_time_counts_only_when_unpaid(merchant_id: str, fake: FakeLinks) -> None:
    rid = _case(merchant_id)
    fake.expire_by = int((datetime(2026, 10, 1)).timestamp())  # long past
    _open_link(rid, fake)
    fake.status = "paid"  # Razorpay says paid: the clock must not expire it
    jobs.clean_expired_links(client=fake, now=datetime(2026, 10, 8))
    assert "payment_link_expired" not in _stages(rid)


# --- scheduler wiring -------------------------------------------------------

def test_scheduler_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    assert scheduler_mod.start_scheduler() is None
    assert not scheduler_mod.scheduler_running()


def test_scheduler_registers_all_four_jobs_with_safe_defaults() -> None:
    sched = scheduler_mod.build_scheduler()
    sched.start()  # job settings only become readable once the scheduler is running
    try:
        ids = sorted(job.id for job in sched.get_jobs())
        assert ids == ["expired_link_cleanup", "reconcile", "retry", "settlement_poll"]
        for job in sched.get_jobs():
            state = job.__getstate__()
            assert state["max_instances"] == 1
            assert state["coalesce"] is True
    finally:
        sched.shutdown(wait=False)


def test_scheduler_starts_and_stops_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    try:
        started = scheduler_mod.start_scheduler()
        assert started is not None and scheduler_mod.scheduler_running()
    finally:
        scheduler_mod.stop_scheduler()
    assert not scheduler_mod.scheduler_running()

