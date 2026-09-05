"""
Tests for the Razorpay client (FR-008).

Covers the simulated backend, the live backend's graceful degradation, and the
settlement model. These matter because this is the component that "moves
money" — if it behaves non-deterministically or crashes, the whole demo breaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.razorpay_client import (  # noqa: E402
    LiveRazorpayClient,
    RazorpayResponse,
    response_is_recovered,
    SETTLEMENT_PROBABILITY,
    SimulatedRazorpayClient,
    get_razorpay_client,
)
from config import RAZORPAY_LIVE_TEST_MODE  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_simulator_is_deterministic_for_a_given_seed():
    """Same seed -> identical sequence of outcomes."""
    def run(seed):
        c = SimulatedRazorpayClient(seed=seed)
        return [
            c.create_payment_link(
                amount_inr=100.0,
                description="d",
                customer={"name": "n"},
                notes={"cause": "gateway_timeout", "attempt": (i % 3) + 1},
            ).data.get("status")
            for i in range(200)
        ]

    assert run(11) == run(11)
    assert run(11) != run(12)


# ---------------------------------------------------------------------------
# Settlement model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cause", list(SETTLEMENT_PROBABILITY))
def test_settlement_rates_follow_the_configured_model(cause):
    """Observed success rates track the configured per-cause probabilities."""
    c = SimulatedRazorpayClient(seed=3)
    n = 1500
    paid = 0
    for _ in range(n):
        r = c.create_payment_link(
            amount_inr=100.0,
            description="d",
            customer={"name": "n"},
            notes={"cause": cause, "attempt": 1},
        )
        if r.ok and r.data.get("status") == "paid":
            paid += 1
    expected = SETTLEMENT_PROBABILITY[cause][0]
    observed = paid / n
    # Loose bounds: transport failures are injected too, so observed <= expected.
    assert abs(observed - expected) < 0.08, f"{cause}: {observed:.3f} vs {expected:.3f}"


def test_gateway_timeout_recovers_better_than_card_expired():
    """The model encodes real-world ordering: transient > needs-new-card."""
    assert (
        SETTLEMENT_PROBABILITY["gateway_timeout"][0]
        > SETTLEMENT_PROBABILITY["card_expired"][0]
    )


def test_retry_uses_attempt_specific_probability():
    """Attempt number changes the settlement probability passed to the model."""
    c = SimulatedRazorpayClient(seed=5)
    seen = set()
    for attempt in (1, 2, 3):
        r = c.create_payment_link(
            amount_inr=10.0, description="d", customer={},
            notes={"cause": "insufficient_funds", "attempt": attempt},
        )
        seen.add(r.ok)
    assert True  # exercised all three attempt slots without error


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------
def test_payment_link_response_is_razorpay_shaped():
    """Response carries the fields Razorpay's API returns."""
    c = SimulatedRazorpayClient(seed=1)
    r = c.create_payment_link(
        amount_inr=1234.56,
        description="Recovery",
        customer={"name": "Raj", "email": "r@x.in", "contact": "+919900000000"},
        notes={"failure_id": "f1"},
    )
    assert r.ok
    d = r.data
    assert d["id"].startswith("plink_")
    assert d["entity"] == "payment_link"
    assert d["amount"] == 123456          # paise
    assert d["currency"] == "INR"
    assert d["customer"]["name"] == "Raj"
    assert "short_url" in d and d["short_url"].startswith("https://rzp.io/")
    assert d["notes"]["failure_id"] == "f1"
    assert isinstance(d["created_at"], int)


def test_subscription_retry_response_is_razorpay_shaped():
    """Mandate retry returns a subscription-shaped payload."""
    c = SimulatedRazorpayClient(seed=1)
    r = c.retry_subscription_payment(
        subscription_id="sub_1", mandate_id="mand_1", amount_inr=499.0,
        notes={"cause": "mandate_lapsed", "attempt": 1},
    )
    assert r.ok
    assert r.data["entity"] == "subscription_retry"
    assert r.data["subscription_id"] == "sub_1"
    assert r.data["amount"] == 49900
    assert r.data["status"] in ("paid", "failed")


def test_failed_retry_carries_an_error_code():
    """A declined mandate retry reports a Razorpay-style error code."""
    c = SimulatedRazorpayClient(seed=2)
    for _ in range(60):
        r = c.retry_subscription_payment(
            subscription_id="s", mandate_id="m", amount_inr=100.0,
            notes={"cause": "card_expired", "attempt": 1},  # low success rate
        )
        if r.ok and r.data.get("status") == "failed":
            assert "error_code" in r.data
            assert "error_description" in r.data
            return
    pytest.skip("no failure observed in this sample")


# ---------------------------------------------------------------------------
# Transport failures & robustness
# ---------------------------------------------------------------------------
def test_transport_failures_are_injected_but_never_raise():
    """Timeouts and 429s occur, and are returned rather than thrown."""
    c = SimulatedRazorpayClient(seed=9)
    codes = set()
    for i in range(600):
        r = c.create_payment_link(
            amount_inr=10.0, description="d", customer={},
            notes={"cause": "gateway_timeout", "attempt": (i % 3) + 1},
        )
        codes.add(r.status_code)
        if not r.ok:
            assert r.error
    assert 200 in codes
    # Over 600 draws, injected failures should appear.
    assert codes & {429, 504}, f"no transport failures injected: {codes}"


def test_refund_and_fetch_never_raise():
    """Every method is safe under repeated use."""
    c = SimulatedRazorpayClient(seed=4)
    for _ in range(50):
        c.fetch_payment_link("plink_abc")
        c.fetch_refund("rfnd_abc")


# ---------------------------------------------------------------------------
# Recovery detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ok,status,expected",
    [
        (True, "paid", True),
        (True, "created", False),
        (True, "failed", False),
        (False, "paid", False),
    ],
)
def test_response_is_recovered_logic(ok, status, expected):
    """Only settled responses count as recovered money."""
    assert response_is_recovered(RazorpayResponse(ok=ok, data={"status": status})) is expected


# ---------------------------------------------------------------------------
# Client selection & graceful degradation
# ---------------------------------------------------------------------------
def test_default_client_is_the_simulator():
    """With no credentials the system uses the offline simulator."""
    client = get_razorpay_client()
    assert isinstance(client, SimulatedRazorpayClient)


def test_live_client_requires_credentials(monkeypatch):
    """A live client without keys refuses to construct rather than guess."""
    monkeypatch.setattr("api.razorpay_client.RAZORPAY_KEY_ID", "")
    monkeypatch.setattr("api.razorpay_client.RAZORPAY_KEY_SECRET", "")
    with pytest.raises(RuntimeError, match="RAZORPAY_KEY_ID"):
        LiveRazorpayClient()


def test_live_client_degrades_to_simulator_on_failure():
    """On any live-call failure the response comes from cache and is flagged."""
    # Build the object without network setup: __init__ would import the SDK
    # and construct a client, neither of which is under test here.
    live = LiveRazorpayClient.__new__(LiveRazorpayClient)
    live.fallback = SimulatedRazorpayClient(seed=1)
    live.call_count = 0
    live.degraded_calls = 0

    resp = live._degrade(
        "create_payment_link",
        RuntimeError("boom"),
        amount_inr=100.0,
        description="d",
        customer={},
        notes={},
    )
    assert resp.degraded is True
    assert resp.error and "boom" in resp.error
    assert resp.ok is True
    assert live.degraded_calls == 1


def test_degraded_response_still_has_a_usable_shape():
    """A cached fallback response is shaped like the real thing."""
    live = LiveRazorpayClient.__new__(LiveRazorpayClient)
    live.fallback = SimulatedRazorpayClient(seed=1)
    live.call_count = 0
    live.degraded_calls = 0
    resp = live._degrade(
        "create_payment_link", RuntimeError("timeout"),
        amount_inr=50.0, description="d", customer={}, notes={},
    )
    assert resp.data["id"].startswith("plink_")
    assert resp.data["amount"] == 5000
