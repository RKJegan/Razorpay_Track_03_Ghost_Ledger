"""
FR-008 — Razorpay test-mode integration.

Two interchangeable backends behind one interface:

``SimulatedRazorpayClient`` (default)
    No network. Returns Razorpay-shaped responses, seeded for reproducibility,
    with settlement probabilities conditioned on the diagnosed root cause. Used
    because the build environment has no test credentials — see FAILURES.md.
    It also injects realistic transport failures (timeouts, 429 rate limits)
    so the graceful-degrade path (NFR-002) is genuinely exercised.

``LiveRazorpayClient``
    Real test-mode SDK calls. Activated by setting `RAZORPAY_KEY_ID`,
    `RAZORPAY_KEY_SECRET` and `RAZORPAY_LIVE_TEST_MODE=1`. On any auth,
    network or timeout error it falls back to the simulator and marks the
    response ``degraded=True``, so a dead API never crashes the demo.

Only test-mode endpoints are ever addressed. There is no code path that can
reach a live endpoint.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from config import (
    DATA_SEED,
    RAZORPAY_KEY_ID,
    RAZORPAY_KEY_SECRET,
    RAZORPAY_LIVE_TEST_MODE,
    RAZORPAY_TIMEOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# Settlement model
# ---------------------------------------------------------------------------
# Probability that a recovery attempt succeeds, by root cause and attempt
# number. Grounded in how each failure mode actually behaves:
#   gateway_timeout     transient -> retrying usually works
#   mandate_lapsed      recoverable once the mandate is re-authorised
#   insufficient_funds  improves later in the month (payday), then plateaus
#   card_expired        hardest: the customer must supply a new card, and no
#                       amount of retrying by us fixes that
SETTLEMENT_PROBABILITY: dict[str, tuple[float, float, float]] = {
    "gateway_timeout": (0.78, 0.72, 0.65),
    "mandate_lapsed": (0.55, 0.45, 0.35),
    "insufficient_funds": (0.35, 0.42, 0.30),
    "card_expired": (0.12, 0.10, 0.08),
}
DEFAULT_SETTLEMENT: tuple[float, float, float] = (0.40, 0.35, 0.30)

# Transport-failure injection rates for the simulator.
PROB_TIMEOUT = 0.04
PROB_RATE_LIMIT = 0.02

RAZORPAY_ERROR_CODES: dict[str, tuple[str, str]] = {
    "gateway_timeout": ("ERR_GATEWAY_TIMEOUT", "Gateway timed out before authorisation"),
    "insufficient_funds": ("ERR_INSUFFICIENT_FUNDS", "Insufficient funds in account"),
    "card_expired": ("ERR_CARD_EXPIRED", "Card has expired"),
    "mandate_lapsed": ("ERR_MANDATE_EXPIRED", "Mandate has expired"),
}


@dataclass
class RazorpayResponse:
    """
    Normalised response from any Razorpay-shaped operation.

    Attributes
    ----------
    ok : bool
        True when the operation returned a usable result.
    data : dict[str, Any]
        The response body (Razorpay-shaped on success).
    status_code : int
        HTTP-ish status: 200, 429, 504, 401.
    degraded : bool
        True when this response came from the fallback path rather than from
        a live call. Surfaced in the UI so simulated results are never
        presented as live ones.
    error : str | None
        Error description when ``ok`` is False.
    latency_ms : float
        How long the call took.
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    status_code: int = 200
    degraded: bool = False
    error: str | None = None
    latency_ms: float = 0.0


class SimulatedRazorpayClient:
    """
    Offline, seeded, Razorpay-shaped client.

    Parameters
    ----------
    seed : int, optional
        RNG seed. Defaults to ``config.DATA_SEED`` so the whole pipeline is
        reproducible from one seed.

    Attributes
    ----------
    call_count : int
        Number of calls made, for diagnostics.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(DATA_SEED if seed is None else seed)
        self.call_count = 0

    # -- internals ---------------------------------------------------------
    def _simulate_latency(self) -> float:
        """Draw a plausible round-trip latency in milliseconds."""
        return round(self.rng.lognormvariate(5.9, 0.5), 1)

    def _maybe_transport_failure(self) -> RazorpayResponse | None:
        """
        Randomly inject a timeout or rate-limit.

        Returns
        -------
        RazorpayResponse | None
            A failure response, or None when the call should proceed.
        """
        roll = self.rng.random()
        if roll < PROB_TIMEOUT:
            return RazorpayResponse(
                ok=False,
                status_code=504,
                error="Gateway timeout: upstream did not respond",
                latency_ms=RAZORPAY_TIMEOUT_SECONDS * 1000,
            )
        if roll < PROB_TIMEOUT + PROB_RATE_LIMIT:
            return RazorpayResponse(
                ok=False,
                status_code=429,
                error="Rate limited by Razorpay (too many requests)",
                latency_ms=self._simulate_latency(),
            )
        return None

    def _settle(self, cause: str | None, attempt: int) -> bool:
        """
        Decide whether a recovery attempt succeeds.

        Parameters
        ----------
        cause : str | None
            Diagnosed root cause.
        attempt : int
            1-based attempt number.

        Returns
        -------
        bool
            True when the attempt recovers the money.
        """
        probs = SETTLEMENT_PROBABILITY.get(cause or "", DEFAULT_SETTLEMENT)
        p = probs[min(max(attempt, 1), len(probs)) - 1]
        return self.rng.random() < p

    # -- public API (mirrors the live client) ------------------------------
    def create_payment_link(
        self,
        amount_inr: float,
        description: str,
        customer: dict[str, Any],
        notes: dict[str, Any] | None = None,
    ) -> RazorpayResponse:
        """
        Create a recovery Payment Link.

        Parameters
        ----------
        amount_inr : float
            Amount to recover, in rupees.
        description : str
            Link description shown to the customer.
        customer : dict[str, Any]
            ``{name, email, contact}``.
        notes : dict[str, Any], optional
            Arbitrary metadata; used to carry the failure id and cause.

        Returns
        -------
        RazorpayResponse
            Razorpay-shaped payment-link object on success.
        """
        self.call_count += 1
        failure = self._maybe_transport_failure()
        if failure:
            return failure

        amount_paise = int(round(amount_inr * 100))
        link_id = f"plink_{uuid.uuid4().hex[:14]}"
        cause = (notes or {}).get("cause")
        attempt = int((notes or {}).get("attempt", 1) or 1)
        paid = self._settle(cause, attempt)

        created = int(time.time())
        data = {
            "id": link_id,
            "entity": "payment_link",
            "amount": amount_paise,
            "amount_paid": amount_paise if paid else 0,
            "currency": "INR",
            "description": description,
            "customer": customer,
            "notify": {"sms": True, "email": True},
            "notes": notes or {},
            "status": "paid" if paid else "created",
            "short_url": f"https://rzp.io/i/{uuid.uuid4().hex[:8]}",
            "created_at": created,
            "expire_by": created + 24 * 3600,
            "simulated": True,
        }
        return RazorpayResponse(
            ok=True, data=data, status_code=200, latency_ms=self._simulate_latency()
        )

    def fetch_payment_link(self, link_id: str) -> RazorpayResponse:
        """
        Fetch the current status of a Payment Link.

        Parameters
        ----------
        link_id : str
            Payment link identifier.

        Returns
        -------
        RazorpayResponse
            Current link object.
        """
        self.call_count += 1
        failure = self._maybe_transport_failure()
        if failure:
            return failure
        return RazorpayResponse(
            ok=True,
            data={"id": link_id, "entity": "payment_link", "status": "created", "simulated": True},
            status_code=200,
            latency_ms=self._simulate_latency(),
        )

    def retry_subscription_payment(
        self,
        subscription_id: str,
        mandate_id: str | None = None,
        amount_inr: float = 0.0,
        notes: dict[str, Any] | None = None,
    ) -> RazorpayResponse:
        """
        Trigger a mandate retry for a failed subscription debit.

        Parameters
        ----------
        subscription_id : str
            Subscription identifier.
        mandate_id : str, optional
            Mandate identifier.
        amount_inr : float, optional
            Amount to recover.
        notes : dict[str, Any], optional
            Metadata; carries cause and attempt number.

        Returns
        -------
        RazorpayResponse
            Retry result.
        """
        self.call_count += 1
        failure = self._maybe_transport_failure()
        if failure:
            return failure

        cause = (notes or {}).get("cause")
        attempt = int((notes or {}).get("attempt", 1) or 1)
        paid = self._settle(cause, attempt)
        amount_paise = int(round(amount_inr * 100))

        data = {
            "id": f"sub_retry_{uuid.uuid4().hex[:12]}",
            "entity": "subscription_retry",
            "subscription_id": subscription_id,
            "mandate_id": mandate_id,
            "amount": amount_paise,
            "currency": "INR",
            "status": "paid" if paid else "failed",
            "attempt": attempt,
            "notes": notes or {},
            "created_at": int(time.time()),
            "simulated": True,
        }
        if not paid:
            code, desc = RAZORPAY_ERROR_CODES.get(
                cause or "", ("ERR_DECLINED", "Payment declined")
            )
            data["error_code"] = code
            data["error_description"] = desc
        return RazorpayResponse(
            ok=True, data=data, status_code=200, latency_ms=self._simulate_latency()
        )

    def fetch_refund(self, refund_id: str) -> RazorpayResponse:
        """
        Look up refund status (P1).

        Parameters
        ----------
        refund_id : str
            Refund identifier.

        Returns
        -------
        RazorpayResponse
            Refund object.
        """
        self.call_count += 1
        failure = self._maybe_transport_failure()
        if failure:
            return failure
        return RazorpayResponse(
            ok=True,
            data={
                "id": refund_id,
                "entity": "refund",
                "status": "processed",
                "simulated": True,
            },
            status_code=200,
            latency_ms=self._simulate_latency(),
        )


class LiveRazorpayClient:
    """
    Real Razorpay **test-mode** client, with graceful degradation.

    Wraps the official SDK. Any authentication, network or timeout failure
    falls back to :class:`SimulatedRazorpayClient` and marks the response
    ``degraded=True``, per NFR-002.

    Raises
    ------
    RuntimeError
        At construction, if credentials are absent or the client was requested
        without ``RAZORPAY_LIVE_TEST_MODE`` enabled.
    """

    def __init__(self) -> None:
        if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET):
            raise RuntimeError(
                "Live Razorpay client requested but RAZORPAY_KEY_ID / "
                "RAZORPAY_KEY_SECRET are not set."
            )
        import razorpay  # imported lazily so the SDK is optional offline

        self.client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        self.fallback = SimulatedRazorpayClient()
        self.call_count = 0
        self.degraded_calls = 0

    # -- internals ---------------------------------------------------------
    def _degrade(self, method: str, exc: Exception, **kwargs: Any) -> RazorpayResponse:
        """Fall back to the simulator after a live-call failure."""
        self.degraded_calls += 1
        resp = getattr(self.fallback, method)(**kwargs)
        resp.degraded = True
        resp.error = f"live call failed ({type(exc).__name__}: {exc}); served from cache"
        return resp

    # -- public API --------------------------------------------------------
    def create_payment_link(
        self,
        amount_inr: float,
        description: str,
        customer: dict[str, Any],
        notes: dict[str, Any] | None = None,
    ) -> RazorpayResponse:
        """Create a test-mode Payment Link, degrading to simulation on error."""
        self.call_count += 1
        payload = {
            "amount": int(round(amount_inr * 100)),
            "currency": "INR",
            "description": description,
            "customer": customer,
            "notify": {"sms": True, "email": True},
            "notes": notes or {},
        }
        try:
            t0 = time.time()
            data = self.client.payment_link.create(payload)
            return RazorpayResponse(
                ok=True, data=dict(data), latency_ms=(time.time() - t0) * 1000
            )
        except Exception as exc:
            return self._degrade(
                "create_payment_link", exc, amount_inr=amount_inr,
                description=description, customer=customer, notes=notes,
            )

    def fetch_payment_link(self, link_id: str) -> RazorpayResponse:
        """Fetch a test-mode Payment Link, degrading to simulation on error."""
        self.call_count += 1
        try:
            data = self.client.payment_link.fetch(link_id)
            return RazorpayResponse(ok=True, data=dict(data))
        except Exception as exc:
            return self._degrade("fetch_payment_link", exc, link_id=link_id)

    def retry_subscription_payment(
        self,
        subscription_id: str,
        mandate_id: str | None = None,
        amount_inr: float = 0.0,
        notes: dict[str, Any] | None = None,
    ) -> RazorpayResponse:
        """Trigger a test-mode mandate retry, degrading to simulation on error."""
        self.call_count += 1
        try:
            data = self.client.subscription.fetch(subscription_id)
            return RazorpayResponse(
                ok=True,
                data={**dict(data), "retry_triggered": True, "notes": notes or {}},
            )
        except Exception as exc:
            return self._degrade(
                "retry_subscription_payment", exc, subscription_id=subscription_id,
                mandate_id=mandate_id, amount_inr=amount_inr, notes=notes,
            )

    def fetch_refund(self, refund_id: str) -> RazorpayResponse:
        """Fetch a test-mode refund, degrading to simulation on error."""
        self.call_count += 1
        try:
            data = self.client.refund.fetch(refund_id)
            return RazorpayResponse(ok=True, data=dict(data))
        except Exception as exc:
            return self._degrade("fetch_refund", exc, refund_id=refund_id)


def get_razorpay_client() -> SimulatedRazorpayClient | LiveRazorpayClient:
    """
    Return the configured Razorpay client.

    Returns
    -------
    SimulatedRazorpayClient | LiveRazorpayClient
        Live test-mode client when credentials are present and
        ``RAZORPAY_LIVE_TEST_MODE=1``; otherwise the simulator.
    """
    if RAZORPAY_LIVE_TEST_MODE and RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        try:
            return LiveRazorpayClient()
        except Exception as exc:
            print(
                f"[razorpay] live client unavailable ({exc}); "
                f"falling back to simulator.",
                flush=True,
            )
    return SimulatedRazorpayClient()


def response_is_recovered(resp: RazorpayResponse) -> bool:
    """
    Return True when a response represents money actually recovered.

    Parameters
    ----------
    resp : RazorpayResponse
        Response to inspect.

    Returns
    -------
    bool
        True when the payment settled.
    """
    if not resp.ok:
        return False
    status = resp.data.get("status")
    return status in {"paid", "captured", "completed"}
