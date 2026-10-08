"""
Tests for A1 — the Razorpay webhook listener (api/webhooks.py, api/main.py).

Every test signs its payload with the same HMAC-SHA256 scheme Razorpay uses,
so the signature path is exercised for real, not mocked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from api import webhooks
from api.main import app
from database import db_client, merchants, recovery_store
from database.engine import session_scope

SECRET = "whsec_test_only_not_a_real_secret"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _captured_event(
    recovery_id: str | None,
    amount_paise: int,
    payment_id: str | None = None,
    event_name: str = "payment.captured",
    status: str = "captured",
) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    if recovery_id is not None:
        notes["recovery_id"] = recovery_id
        notes["txn_id"] = "txn_webhook_test"
    return {
        "entity": "event",
        "event": event_name,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id or f"pay_{uuid.uuid4().hex[:14]}",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": status,
                    "notes": notes,
                }
            }
        },
        "created_at": 1760000000,
    }


@pytest.fixture(scope="module")
def merchant_id() -> str:
    mid = "m_test_webhooks"
    if mid not in {m["id"] for m in merchants.list_merchants()}:
        merchants.create_merchant(mid, "Webhook tests")
    return mid


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(webhooks, "RAZORPAY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(webhooks, "WEBHOOK_MAX_BODY_BYTES", 1_000_000)
    with TestClient(app) as c:
        yield c


def _post(client: TestClient, payload: dict[str, Any] | bytes, *, event_id: str | None = None,
          signature: str | None = None) -> Any:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "X-Razorpay-Signature": signature or _sign(body)}
    if event_id is not None:
        headers["X-Razorpay-Event-Id"] = event_id
    return client.post("/webhooks/razorpay", content=body, headers=headers)


def _new_case(merchant_id: str, amount_inr: float = 499.0) -> str:
    return recovery_store.create_case(merchant_id, f"txn_{uuid.uuid4().hex[:8]}", amount_inr)


def _event_row(event_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.execute(
            text("SELECT event_id, process_status, recovery_id FROM webhook_events WHERE event_id = :e"),
            {"e": event_id},
        ).mappings().first()
    return dict(row) if row else None


def _audit_actions(action: str) -> int:
    """Count webhook audit records with this action (the trail is append-only)."""
    return int(
        db_client.scalar(
            "SELECT COUNT(*) FROM audit_trail WHERE component='webhook' AND action=?", (action,)
        )
        or 0
    )


# --- signature ----------------------------------------------------------------

def test_verify_signature_accepts_correct_and_rejects_wrong() -> None:
    body = b'{"event":"payment.captured"}'
    assert webhooks.verify_signature(body, _sign(body), SECRET)
    assert not webhooks.verify_signature(body, _sign(body, "other"), SECRET)
    assert not webhooks.verify_signature(body + b" ", _sign(body), SECRET)
    assert not webhooks.verify_signature(body, None, SECRET)
    assert not webhooks.verify_signature(body, _sign(body), "")


def test_invalid_signature_is_refused_and_logged(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    before = _audit_actions("signature_rejected")
    event_id = f"evt_{uuid.uuid4().hex}"
    body = json.dumps(_captured_event(rid, 49900)).encode()
    response = _post(client, body, event_id=event_id, signature="0" * 64)
    assert response.status_code == 401
    assert response.json()["status"] == "invalid_signature"
    assert _event_row(event_id) is None  # nothing stored
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"
    assert _audit_actions("signature_rejected") == before + 1


def test_missing_signature_header_is_refused(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id)
    body = json.dumps(_captured_event(rid, 49900)).encode()
    response = client.post("/webhooks/razorpay", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 401


def test_secret_not_configured_refuses_everything(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                  merchant_id: str) -> None:
    monkeypatch.setattr(webhooks, "RAZORPAY_WEBHOOK_SECRET", "")
    rid = _new_case(merchant_id)
    response = _post(client, _captured_event(rid, 49900))
    assert response.status_code == 503


# --- happy path and idempotency -----------------------------------------------

def test_capture_settles_the_recovery(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id, amount_inr=499.0)
    event_id = f"evt_{uuid.uuid4().hex}"
    payment_id = f"pay_{uuid.uuid4().hex[:14]}"
    response = _post(client, _captured_event(rid, 49900, payment_id), event_id=event_id)
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    assert recovery_store.get_recovery_status(rid)["status"] == "settled"
    stages = [e["stage"] for e in recovery_store.get_recovery_timeline(rid)]
    assert stages == ["payment_captured"]
    timeline = recovery_store.get_recovery_timeline(rid)[0]
    assert timeline["detail"]["payment_id"] == payment_id
    assert timeline["created_by"] == "razorpay_webhook"
    assert _event_row(event_id)["process_status"] == "processed"


def test_duplicate_delivery_is_not_processed_twice(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id, amount_inr=250.0)
    event_id = f"evt_{uuid.uuid4().hex}"
    payload = _captured_event(rid, 25000)
    first = _post(client, payload, event_id=event_id)
    second = _post(client, payload, event_id=event_id)
    assert first.json()["status"] == "accepted"
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert len(recovery_store.get_recovery_timeline(rid)) == 1


def test_same_payment_under_two_event_ids_settles_once(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id, amount_inr=100.0)
    payment_id = f"pay_{uuid.uuid4().hex[:14]}"
    _post(client, _captured_event(rid, 10000, payment_id), event_id=f"evt_{uuid.uuid4().hex}")
    _post(client, _captured_event(rid, 10000, payment_id), event_id=f"evt_{uuid.uuid4().hex}")
    assert len(recovery_store.get_recovery_timeline(rid)) == 1


def test_failed_payment_is_recorded_without_settling(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id, amount_inr=300.0)
    response = _post(client, _captured_event(rid, 30000, event_name="payment.failed", status="failed"))
    assert response.status_code == 200
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"
    assert [e["stage"] for e in recovery_store.get_recovery_timeline(rid)] == ["payment_failed"]


def test_unhandled_event_type_is_stored_and_ignored(client: TestClient) -> None:
    event_id = f"evt_{uuid.uuid4().hex}"
    body = {"event": "refund.processed", "payload": {}}
    response = _post(client, body, event_id=event_id)
    assert response.status_code == 200
    assert _event_row(event_id)["process_status"] == "ignored"


# --- anomalies never crash and never settle -----------------------------------

def test_missing_recovery_id_is_anomaly_but_still_200(client: TestClient) -> None:
    event_id = f"evt_{uuid.uuid4().hex}"
    response = _post(client, _captured_event(None, 49900), event_id=event_id)
    assert response.status_code == 200
    assert _event_row(event_id)["process_status"] == "anomaly"


def test_unknown_recovery_id_is_anomaly(client: TestClient) -> None:
    event_id = f"evt_{uuid.uuid4().hex}"
    response = _post(client, _captured_event("rcv_not_real_0000", 49900), event_id=event_id)
    assert response.status_code == 200
    assert _event_row(event_id)["process_status"] == "anomaly"


def test_amount_mismatch_never_settles(client: TestClient, merchant_id: str) -> None:
    rid = _new_case(merchant_id, amount_inr=499.0)
    event_id = f"evt_{uuid.uuid4().hex}"
    response = _post(client, _captured_event(rid, 100), event_id=event_id)  # Rs 1, not Rs 499
    assert response.status_code == 200
    assert _event_row(event_id)["process_status"] == "anomaly"
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"


def test_malformed_json_with_valid_signature_is_400(client: TestClient) -> None:
    response = _post(client, b"{not json")
    assert response.status_code == 400


def test_non_object_json_is_400(client: TestClient) -> None:
    response = _post(client, b"[1, 2, 3]")
    assert response.status_code == 400


def test_oversized_body_is_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks, "WEBHOOK_MAX_BODY_BYTES", 50)
    response = _post(client, {"event": "payment.captured", "padding": "x" * 200})
    assert response.status_code == 413


# --- storage failure asks Razorpay to retry ------------------------------------

def test_storage_failure_returns_500_so_razorpay_retries(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                                         merchant_id: str) -> None:
    rid = _new_case(merchant_id)

    def broken_store(*_args: Any, **_kwargs: Any) -> bool:
        raise OperationalError("INSERT", {}, Exception("disk I/O error (simulated)"))

    monkeypatch.setattr(webhooks, "store_webhook_event", broken_store)
    response = _post(client, _captured_event(rid, 49900))
    assert response.status_code == 500
    assert response.json()["status"] == "storage_unavailable"
    assert recovery_store.get_recovery_status(rid)["status"] == "pending"


# --- parsing ------------------------------------------------------------------

def test_parse_reads_notes_from_payment_link_when_payment_has_none() -> None:
    payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {"entity": {"id": "pay_x", "amount": 100, "status": "captured", "notes": []}},
            "payment_link": {"entity": {"notes": {"recovery_id": "rcv_from_link", "merchant_id": "m1"}}},
        },
    }
    parsed = webhooks.parse_razorpay_event(payload)
    assert parsed.recovery_id == "rcv_from_link"
    assert parsed.merchant_id == "m1"
    assert parsed.payment_id == "pay_x"
    assert parsed.amount_paise == 100


def test_parse_tolerates_garbage_without_raising() -> None:
    parsed = webhooks.parse_razorpay_event({"event": None, "payload": "not a dict"})
    assert parsed.event_type == ""
    assert parsed.recovery_id is None
    assert parsed.amount_paise is None
