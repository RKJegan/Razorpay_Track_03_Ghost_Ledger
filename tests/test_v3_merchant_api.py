"""
Tests for A6 — the merchant-scoped API, the sanitised timeline, and the web page.

The core promise: merchant A's key can never read merchant B's data, and the
data a merchant sees contains no operator-only fields.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agents.recovery_executor import submit_recovery
from api.main import app
from database import merchants, recovery_store
from database.engine import session_scope
from database.models import Merchant

A_ID = "m_test_api_a"
B_ID = "m_test_api_b"


def _ensure_merchant(mid: str, name: str) -> str:
    """Return a fresh plain-text key for the merchant, creating it if needed."""
    existing = {m["id"] for m in merchants.list_merchants()}
    if mid in existing:
        return merchants.rotate_api_key(mid)
    return merchants.create_merchant(mid, name)


@pytest.fixture(scope="module")
def keys() -> dict[str, str]:
    return {A_ID: _ensure_merchant(A_ID, "Store A"), B_ID: _ensure_merchant(B_ID, "Store B")}


@pytest.fixture(scope="module")
def data(keys: dict[str, str]) -> dict[str, str]:
    """Two recoveries for A (one settled, one failing), one for B, one awaiting approval for A."""
    a_settled = recovery_store.create_case(A_ID, f"txn_{uuid.uuid4().hex[:8]}", 499.0, cause="card_expired",
                                           customer_id="CUST_A1")
    recovery_store.append_event(a_settled, "payment_captured",
                                {"payment_id": "pay_a_ok", "amount_paise": 49900, "source": "webhook"},
                                created_by="operator_secret_user")
    a_failing = recovery_store.create_case(A_ID, f"txn_{uuid.uuid4().hex[:8]}", 250.0, cause="insufficient_funds",
                                           customer_id="CUST_A2")
    recovery_store.append_event(a_failing, "payment_failed", {"payment_id": "pay_a_fail"})
    b_case = recovery_store.create_case(B_ID, f"txn_{uuid.uuid4().hex[:8]}", 499.0, cause="card_expired",
                                        customer_id="CUST_B1")
    recovery_store.append_event(b_case, "payment_failed", {"payment_id": "pay_b_fail"})

    a_big = recovery_store.create_case(A_ID, f"txn_{uuid.uuid4().hex[:8]}", 15_000.0, cause="card_expired",
                                       customer_id="CUST_A3")

    class _MustNotBeCalled:
        def create_payment_link(self, *a: Any, **k: Any) -> None:
            raise AssertionError("no link may be created above the ceiling without approval")

    result = submit_recovery(a_big, client=_MustNotBeCalled())
    assert result.outcome == "awaiting_approval"
    return {"a_settled": a_settled, "a_failing": a_failing, "b_case": b_case, "a_big": a_big,
            "approval_id": result.approval_id}


def _client() -> TestClient:
    return TestClient(app)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --- keys -------------------------------------------------------------------

def test_key_resolves_to_its_merchant_and_only_while_active(keys: dict[str, str]) -> None:
    assert merchants.merchant_for_api_key(keys[A_ID]) == A_ID
    assert merchants.merchant_for_api_key(keys[B_ID]) == B_ID
    assert merchants.merchant_for_api_key("not-a-key") is None
    assert merchants.merchant_for_api_key("") is None
    assert merchants.merchant_for_api_key(None) is None


def test_rotating_the_key_stops_the_old_one(keys: dict[str, str]) -> None:
    old = keys[A_ID]
    new = merchants.rotate_api_key(A_ID)
    try:
        assert merchants.merchant_for_api_key(old) is None
        assert merchants.merchant_for_api_key(new) == A_ID
    finally:
        keys[A_ID] = new


def test_only_a_hash_is_stored(keys: dict[str, str]) -> None:
    with session_scope() as session:
        stored = session.scalars(select(Merchant.api_key_hash).where(Merchant.id == A_ID)).one()
    assert stored != keys[A_ID]
    assert stored == hashlib.sha256(keys[A_ID].encode()).hexdigest()


# --- authentication -----------------------------------------------------------

def test_requests_without_a_valid_key_are_refused(keys: dict[str, str]) -> None:
    with _client() as c:
        assert c.get("/api/merchant/summary").status_code == 401
        assert c.get("/api/merchant/summary", headers={"Authorization": "Basic abc"}).status_code == 401
        assert c.get("/api/merchant/summary", headers=_auth("wrong-key")).status_code == 401
        resp = c.get("/api/merchant/summary", headers=_auth("wrong-key"))
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert "wrong-key" not in resp.text  # the key is never echoed back


def test_me_identifies_the_merchant(keys: dict[str, str]) -> None:
    with _client() as c:
        body = c.get("/api/merchant/me", headers=_auth(keys[A_ID])).json()
    assert body == {"merchant_id": A_ID, "name": "Store A"}


def test_responses_are_not_cached(keys: dict[str, str]) -> None:
    with _client() as c:
        resp = c.get("/api/merchant/me", headers=_auth(keys[A_ID]))
    assert resp.headers.get("cache-control") == "no-store"


# --- scoping ------------------------------------------------------------------

def test_list_returns_only_this_merchants_recoveries(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        body = c.get("/api/merchant/recoveries?limit=200", headers=_auth(keys[A_ID])).json()
    ids = {r["recovery_id"] for r in body["items"]}
    assert {data["a_settled"], data["a_failing"], data["a_big"]} <= ids
    assert data["b_case"] not in ids
    assert all(r["merchant_id"] == A_ID for r in body["items"])


def test_another_merchants_recovery_is_indistinguishable_from_missing(
        keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        theirs = c.get(f"/api/merchant/recoveries/{data['b_case']}", headers=_auth(keys[A_ID]))
        missing = c.get("/api/merchant/recoveries/rcv_does_not_exist", headers=_auth(keys[A_ID]))
    assert theirs.status_code == 404 and missing.status_code == 404
    assert theirs.json() == missing.json()


def test_summary_counts_only_this_merchant(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        a = c.get("/api/merchant/summary", headers=_auth(keys[A_ID])).json()
        b = c.get("/api/merchant/summary", headers=_auth(keys[B_ID])).json()
    a_failing_ids = {f["recovery_id"] for f in a["failures"]}
    assert data["a_failing"] in a_failing_ids
    assert data["b_case"] not in a_failing_ids
    assert all(f["merchant_id"] == A_ID for f in a["failures"])
    assert all(x["merchant_id"] == A_ID for x in a["active"])
    assert data["b_case"] in {f["recovery_id"] for f in b["failures"]}
    assert data["a_settled"] not in {x["recovery_id"] for x in b["active"]}
    assert a["counts"]["pending_approvals"] >= 1
    assert data["approval_id"] in {ap["approval_id"] for ap in a["approvals"]}
    assert data["approval_id"] not in {ap["approval_id"] for ap in b["approvals"]}


def test_approvals_are_listed_without_operator_fields(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        body = c.get("/api/merchant/summary", headers=_auth(keys[A_ID])).json()
    match = [ap for ap in body["approvals"] if ap["approval_id"] == data["approval_id"]]
    assert match, "the pending approval for A should be visible to A"
    assert set(match[0]) <= {"approval_id", "recovery_id", "amount_inr", "cause", "status", "created_at"}
    assert "approved_by" not in match[0] and "rejection_reason" not in match[0]


# --- sanitised timeline -------------------------------------------------------

def test_timeline_hides_operator_identity_and_internal_fields(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        body = c.get(f"/api/merchant/recoveries/{data['a_settled']}", headers=_auth(keys[A_ID])).json()
    assert body["recovery"]["recovery_id"] == data["a_settled"]
    text = str(body)
    assert "operator_secret_user" not in text  # created_by is never returned
    for event in body["timeline"]:
        assert set(event) == {"stage", "timestamp", "detail"}
        assert set(event["detail"]) <= {
            "payment_id", "amount_paise", "attempt", "short_url", "expire_by", "link_id"}
    captured = [e for e in body["timeline"] if e["stage"] == "payment_captured"][0]
    assert captured["detail"]["payment_id"] == "pay_a_ok"
    assert "source" not in captured["detail"]  # internal bookkeeping stays internal


def test_customer_id_is_visible_only_to_its_merchant(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        mine = c.get(f"/api/merchant/recoveries/{data['a_settled']}", headers=_auth(keys[A_ID])).json()
        theirs = c.get(f"/api/merchant/recoveries/{data['b_case']}", headers=_auth(keys[A_ID]))
    assert mine["recovery"]["customer_id"] == "CUST_A1"
    assert theirs.status_code == 404


# --- filters and paging -------------------------------------------------------

def test_filters_and_paging_are_validated(keys: dict[str, str]) -> None:
    with _client() as c:
        h = _auth(keys[A_ID])
        assert c.get("/api/merchant/recoveries?status=settled", headers=h).status_code == 200
        assert c.get("/api/merchant/recoveries?status=bogus", headers=h).status_code == 422
        assert c.get("/api/merchant/recoveries?limit=0", headers=h).status_code == 422
        assert c.get("/api/merchant/recoveries?limit=201", headers=h).status_code == 422
        assert c.get("/api/merchant/recoveries?offset=-1", headers=h).status_code == 422


def test_status_filter_only_returns_that_status(keys: dict[str, str], data: dict[str, str]) -> None:
    with _client() as c:
        body = c.get("/api/merchant/recoveries?status=settled&limit=200", headers=_auth(keys[A_ID])).json()
    assert data["a_settled"] in {r["recovery_id"] for r in body["items"]}
    assert all(r["status"] == "settled" for r in body["items"])


# --- the page -----------------------------------------------------------------

def test_dashboard_page_is_served_with_a_strict_policy() -> None:
    with _client() as c:
        page = c.get("/merchant")
        js = c.get("/merchant/app.js")
        css = c.get("/merchant/app.css")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert '<script src="/merchant/app.js">' in page.text
    assert js.status_code == 200 and css.status_code == 200


def test_page_and_script_use_relative_urls_only() -> None:
    from api.merchant_api import WEB_DIR

    for name in ("index.html", "app.js"):
        text = (WEB_DIR / name).read_text(encoding="utf-8")
        assert "localhost" not in text and "127.0.0.1" not in text
        assert "http://" not in text and "https://" not in text
        assert ".innerHTML" not in text and "insertAdjacentHTML" not in text  # API data: textContent only
    assert "/api/merchant/summary" in (WEB_DIR / "app.js").read_text(encoding="utf-8")
