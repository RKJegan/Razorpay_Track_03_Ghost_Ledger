"""
Tests for A3 — the approval queue and the policy-gated recovery executor.

A fake Razorpay client with fixed responses keeps these deterministic. The
real simulator has random timeouts, which would make counts flaky.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import config
from agents import recovery_executor
from agents.approval_queue import approve_recovery, reject_recovery
from agents.recovery_executor import submit_recovery
from api.razorpay_client import RazorpayResponse
from api.main import app
from database import approvals, merchants, recovery_store
from database.engine import session_scope

ABOVE_CEILING = 15_000.0
BELOW_CEILING = 499.0


class FakeClient:
    """Deterministic stand-in for the Razorpay client."""

    def __init__(self, ok: bool = True, degraded: bool = False) -> None:
        self.ok = ok
        self.degraded = degraded
        self.calls: list[dict[str, Any]] = []

    def create_payment_link(self, amount_inr, description, customer, notes=None, reference_id=None):  # noqa: ANN001
        self.calls.append({"amount_inr": amount_inr, "reference_id": reference_id, "notes": notes})
        if not self.ok:
            return RazorpayResponse(ok=False, status_code=504, error="gateway timeout (fake)")
        n = len(self.calls)
        # A "paid" status here is what the real simulator returns. It must NOT settle anything.
        return RazorpayResponse(
            ok=True,
            data={"id": f"plink_fake_{n}_{uuid.uuid4().hex[:6]}", "short_url": "https://rzp.io/i/fake",
                  "status": "paid", "simulated": True},
            degraded=self.degraded,
        )


@pytest.fixture(scope="module")
def merchant_id() -> str:
    mid = "m_test_approvals"
    if mid not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(mid, "Approval tests")
    return mid


def _case(merchant_id: str, amount: float) -> str:
    return recovery_store.create_case(merchant_id, f"txn_{uuid.uuid4().hex[:8]}", amount,
                                      cause="card_expired", customer_id="CUST_TEST")


def _stages(recovery_id: str) -> list[str]:
    return [e["stage"] for e in recovery_store.get_recovery_timeline(recovery_id)]


# --- the executor and the policy gate ----------------------------------------

def test_small_recovery_creates_a_link_but_does_not_settle(merchant_id: str) -> None:
    rid = _case(merchant_id, BELOW_CEILING)
    client = FakeClient()
    result = submit_recovery(rid, client=client)
    assert result.outcome == "link_created"
    assert result.link_id and result.link_id.startswith("plink_fake_")
    status = recovery_store.get_recovery_status(rid)
    assert status["status"] == "pending"  # the link response said "paid"; still not settled
    assert "payment_link_created" in _stages(rid)
    assert "policy_check_passed" in _stages(rid)
    # The reference id makes a retried call safe at Razorpay (NFR2).
    assert client.calls[0]["reference_id"] == f"{rid}-a1"


def test_above_ceiling_waits_for_a_human_and_creates_no_link(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    client = FakeClient()
    result = submit_recovery(rid, client=client)
    assert result.outcome == "awaiting_approval"
    assert result.approval_id
    assert client.calls == []  # nothing executed
    pending = approvals.get_approval(result.approval_id)
    assert pending["status"] == "pending"
    assert "approval_requested" in _stages(rid)
    # Asking again returns the SAME pending approval, never a second one.
    again = submit_recovery(rid, client=client)
    assert again.approval_id == result.approval_id


def test_approval_executes_exactly_once_and_records_the_human(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    client = FakeClient()
    approval_id = submit_recovery(rid, client=client).approval_id
    out = approve_recovery(approval_id, "ops.priya", client=client)
    assert out["approval"]["status"] == "approved"
    assert out["approval"]["approved_by"] == "ops.priya"
    assert out["execution"]["outcome"] == "link_created"
    assert len(client.calls) == 1
    stages = _stages(rid)
    assert stages.index("approval_requested") < stages.index("approval_granted") < stages.index("payment_link_created")
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"


def test_second_decision_is_refused(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    client = FakeClient()
    approval_id = submit_recovery(rid, client=client).approval_id
    approve_recovery(approval_id, "ops.a", client=client)
    with pytest.raises(approvals.ApprovalAlreadyDecidedError) as excinfo:
        approve_recovery(approval_id, "ops.b", client=client)
    assert excinfo.value.current_status == "approved"
    with pytest.raises(approvals.ApprovalAlreadyDecidedError):
        reject_recovery(approval_id, "ops.b", "changed my mind")
    assert len(client.calls) == 1


def test_rejection_escalates_and_never_executes(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    client = FakeClient()
    approval_id = submit_recovery(rid, client=client).approval_id
    out = reject_recovery(approval_id, "ops.raj", "customer already paid by cheque")
    assert out["approval"]["status"] == "rejected"
    assert out["approval"]["rejection_reason"] == "customer already paid by cheque"
    assert recovery_store.get_recovery_status(rid)["status"] == "escalated"
    assert client.calls == []
    with pytest.raises(approvals.ApprovalAlreadyDecidedError):
        approve_recovery(approval_id, "ops.late", client=client)


def test_rejection_requires_a_reason(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    approval_id = submit_recovery(rid, client=FakeClient()).approval_id
    with pytest.raises(ValueError):
        approvals.claim_decision(approval_id, "rejected", "ops.x", reason="   ")
    assert approvals.get_approval(approval_id)["status"] == "pending"


def test_concurrent_approvals_execute_once(merchant_id: str) -> None:
    """Five operators click Approve at the same moment: exactly one execution."""
    rid = _case(merchant_id, ABOVE_CEILING)
    client = FakeClient()
    approval_id = submit_recovery(rid, client=client).approval_id
    outcomes: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(5)

    def click(name: str) -> None:
        barrier.wait()
        try:
            approve_recovery(approval_id, name, client=client)
            result = "won"
        except approvals.ApprovalAlreadyDecidedError:
            result = "409"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=click, args=(f"ops.{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(outcomes) == ["409", "409", "409", "409", "won"]
    assert len(client.calls) == 1
    assert _stages(rid).count("payment_link_created") == 1


def test_stopping_rule_cannot_be_overridden_by_approval(merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    for _ in range(3):
        recovery_store.append_event(rid, "payment_failed", {"payment_id": f"pay_{uuid.uuid4().hex[:6]}"})
    client = FakeClient()
    result = submit_recovery(rid, approved=True, client=client)
    assert result.outcome == "stopped"
    assert client.calls == []
    assert "stopping_rule_triggered" in _stages(rid)
    assert recovery_store.get_recovery_status(rid)["status"] == "escalated"


def test_attempt_cap_blocks_the_fourth_link(merchant_id: str) -> None:
    rid = _case(merchant_id, BELOW_CEILING)
    client = FakeClient()
    for _ in range(3):
        assert submit_recovery(rid, client=client).outcome == "link_created"
    fourth = submit_recovery(rid, client=client)
    assert fourth.outcome == "denied"
    assert "policy_denied" in _stages(rid)
    assert len(client.calls) == 3


def test_degraded_live_response_is_never_recorded_as_a_link(merchant_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recovery_executor, "RAZORPAY_LIVE_TEST_MODE", True)
    rid = _case(merchant_id, BELOW_CEILING)
    result = submit_recovery(rid, client=FakeClient(degraded=True))
    assert result.outcome == "link_failed"
    assert "payment_link_created" not in _stages(rid)
    assert "link_call_failed" in _stages(rid)
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"  # still retryable


def test_failed_link_call_is_recorded_and_retryable(merchant_id: str) -> None:
    rid = _case(merchant_id, BELOW_CEILING)
    assert submit_recovery(rid, client=FakeClient(ok=False)).outcome == "link_failed"
    assert submit_recovery(rid, client=FakeClient()).outcome == "link_created"


def test_non_pending_recovery_is_not_acted_on(merchant_id: str) -> None:
    rid = _case(merchant_id, BELOW_CEILING)
    recovery_store.append_event(rid, "payment_captured", {"payment_id": "pay_done"})
    client = FakeClient()
    assert submit_recovery(rid, client=client).outcome == "not_pending"
    assert client.calls == []


# --- the operator API ----------------------------------------------------------

@pytest.fixture
def operator(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(config, "OPERATOR_API_KEY", "op_test_key_123456")
    return {"X-Operator-Key": "op_test_key_123456"}


@pytest.fixture
def api() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_api_refuses_without_a_key(api: TestClient, operator: dict[str, str]) -> None:
    assert api.get("/api/approvals").status_code == 401
    assert api.get("/api/approvals", headers={"X-Operator-Key": "wrong"}).status_code == 401


def test_api_refuses_everything_when_unconfigured(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPERATOR_API_KEY", "")
    assert api.get("/api/approvals", headers={"X-Operator-Key": "anything"}).status_code == 503


def test_api_lists_with_filters_and_pagination(api: TestClient, operator: dict[str, str], merchant_id: str) -> None:
    for _ in range(3):
        submit_recovery(_case(merchant_id, ABOVE_CEILING), client=FakeClient())
    resp = api.get("/api/approvals", params={"status": "pending", "merchant_id": merchant_id, "limit": 2},
                   headers=operator)
    body = resp.json()
    assert resp.status_code == 200
    assert body["total"] >= 3
    assert len(body["items"]) == 2
    assert all(i["status"] == "pending" and i["merchant_id"] == merchant_id for i in body["items"])


def test_api_approve_then_conflict(api: TestClient, operator: dict[str, str], merchant_id: str,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    approval_id = submit_recovery(rid, client=FakeClient()).approval_id
    # Route the executor's client to the fake so no real call is made.
    monkeypatch.setattr(recovery_executor, "get_razorpay_client", lambda: FakeClient())
    first = api.post(f"/api/approvals/{approval_id}/approve", json={"approved_by": "ops.api"}, headers=operator)
    assert first.status_code == 200
    assert first.json()["execution"]["outcome"] == "link_created"
    second = api.post(f"/api/approvals/{approval_id}/approve", json={"approved_by": "ops.api"}, headers=operator)
    assert second.status_code == 409
    assert second.json()["detail"]["current_status"] == "approved"


def test_api_reject_needs_a_reason_and_detail_has_timeline(api: TestClient, operator: dict[str, str],
                                                           merchant_id: str) -> None:
    rid = _case(merchant_id, ABOVE_CEILING)
    approval_id = submit_recovery(rid, client=FakeClient()).approval_id
    assert api.post(f"/api/approvals/{approval_id}/reject", json={"rejected_by": "ops", "reason": ""},
                    headers=operator).status_code == 422
    ok = api.post(f"/api/approvals/{approval_id}/reject",
                  json={"rejected_by": "ops", "reason": "duplicate of case 17"}, headers=operator)
    assert ok.status_code == 200
    detail = api.get(f"/api/approvals/{approval_id}", headers=operator).json()
    assert detail["approval"]["status"] == "rejected"
    assert detail["recovery"]["status"] == "escalated"
    assert "approval_rejected" in [e["stage"] for e in detail["timeline"]]


def test_api_unknown_approval_is_404(api: TestClient, operator: dict[str, str]) -> None:
    assert api.get("/api/approvals/apr_nope", headers=operator).status_code == 404
    assert api.post("/api/approvals/apr_nope/approve", json={"approved_by": "ops"},
                    headers=operator).status_code == 404


def test_approval_status_is_checked_by_the_database(merchant_id: str) -> None:
    """A bad status written straight to SQL is refused by the CHECK constraint."""
    rid = _case(merchant_id, ABOVE_CEILING)
    approval_id = submit_recovery(rid, client=FakeClient()).approval_id
    with pytest.raises(Exception):
        with session_scope() as session:
            session.execute(text("UPDATE approvals SET status = 'maybe' WHERE approval_id = :a"),
                            {"a": approval_id})
