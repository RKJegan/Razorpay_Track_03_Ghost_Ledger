"""Operator summary (A5 data source): shape, failure filtering, and auth."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from api.operator_api import build_summary
from database import merchants, recovery_store


@pytest.fixture(scope="module")
def merchant_id() -> str:
    mid = "m_test_operator"
    if mid not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(mid, "Operator tests")
    return mid


def test_summary_lists_failures_in_progress_but_not_settled(merchant_id: str) -> None:
    failing = recovery_store.create_case(merchant_id, f"txn_{uuid.uuid4().hex[:8]}", 499.0, cause="card_expired")
    recovery_store.append_event(failing, "payment_failed", {"payment_id": f"pay_{uuid.uuid4().hex[:6]}"})
    settled = recovery_store.create_case(merchant_id, f"txn_{uuid.uuid4().hex[:8]}", 499.0, cause="card_expired")
    recovery_store.append_event(settled, "payment_failed", {"payment_id": f"pay_{uuid.uuid4().hex[:6]}"})
    recovery_store.append_event(settled, "payment_captured", {"payment_id": f"pay_{uuid.uuid4().hex[:6]}",
                                                              "amount_paise": 49900})

    summary = build_summary(limit=200)
    failure_ids = {row["recovery_id"] for row in summary["failures"]}
    assert failing in failure_ids
    assert settled not in failure_ids
    assert set(summary) >= {"counts", "active", "failures", "approvals", "jobs", "generated_at"}
    assert summary["counts"]["failures_in_progress"] >= 1


def test_summary_route_needs_the_operator_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPERATOR_API_KEY", "op_test_key_summary")
    with TestClient(app) as c:
        assert c.get("/api/operator/summary").status_code == 401
        ok = c.get("/api/operator/summary", headers={"X-Operator-Key": "op_test_key_summary"})
    assert ok.status_code == 200
    assert "counts" in ok.json()
