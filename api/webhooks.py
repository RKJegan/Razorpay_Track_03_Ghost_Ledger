"""
Ghost Ledger v3 — Razorpay webhook listener (A1).

``POST /webhooks/razorpay`` is the only way a recovery is confirmed as paid.
A payment link being created is NOT treated as success.

Request flow
------------
1. Read the raw body. The signature is computed over these exact bytes.
2. Check the body size (413 if over ``WEBHOOK_MAX_BODY_BYTES``).
3. Verify ``X-Razorpay-Signature`` = HMAC-SHA256(body, webhook secret).
   Failure -> 401, plus a security record in the audit trail. The body is
   not processed.
4. Parse the JSON. Malformed JSON with a valid signature -> 400.
5. Store the event keyed by its id (``X-Razorpay-Event-Id``, or the body's
   SHA-256 if the header is absent). A duplicate delivery is recognised by the
   primary key and answered 200 without re-processing. A storage failure
   answers 500 so Razorpay retries the delivery.
6. Return 200 immediately. The heavy work (matching the recovery, appending
   events, anomaly checks) runs in a background task after the response.

Trust boundary: this module only records verified facts. It moves no money and
calls no LLM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

import config
from database import audit_trail, recovery_store
from database.engine import session_scope
from database.models import WebhookEvent

logger = logging.getLogger(__name__)

router = APIRouter()

#: Razorpay header carrying the HMAC-SHA256 signature of the raw body.
SIGNATURE_HEADER = "X-Razorpay-Signature"
#: Razorpay header carrying a unique id per event delivery (idempotency key).
EVENT_ID_HEADER = "X-Razorpay-Event-Id"

#: Events this listener acts on. Everything else is stored and marked ignored.
HANDLED_EVENTS: dict[str, str] = {
    "payment.captured": "payment_captured",
    "payment.failed": "payment_failed",
}

# Read at call time (not import time) so tests and operators can change it.
RAZORPAY_WEBHOOK_SECRET: str = config.RAZORPAY_WEBHOOK_SECRET
WEBHOOK_MAX_BODY_BYTES: int = config.WEBHOOK_MAX_BODY_BYTES


def _now() -> str:
    """Return the current local timestamp in the project's canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """
    Check a Razorpay webhook signature in constant time.

    Parameters
    ----------
    body : bytes
        The raw request body, exactly as received.
    signature : str | None
        Value of the ``X-Razorpay-Signature`` header.
    secret : str
        The webhook secret configured on the Razorpay dashboard.

    Returns
    -------
    bool
        True only when the signature matches. An empty secret or missing
        signature always returns False.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


@dataclass(frozen=True)
class ParsedWebhook:
    """The fields of a Razorpay event that Ghost Ledger uses."""

    event_type: str
    payment_id: str | None
    amount_paise: int | None
    payment_status: str | None
    recovery_id: str | None
    txn_id: str | None
    merchant_id: str | None


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it is a dict, otherwise an empty dict.

    Razorpay sends ``notes`` as an object, but an empty one can arrive as a
    list. Treat anything that is not a dict as "no notes".
    """
    return value if isinstance(value, dict) else {}


def parse_razorpay_event(payload: dict[str, Any]) -> ParsedWebhook:
    """
    Extract the fields Ghost Ledger needs from a Razorpay event body.

    Notes are read from the payment entity first, then the payment-link entity.
    Missing fields become None; this function never raises on a well-formed
    JSON object.

    Parameters
    ----------
    payload : dict[str, Any]
        Parsed JSON body.

    Returns
    -------
    ParsedWebhook
        The extracted fields.
    """
    event_type = str(payload.get("event") or "")
    body = _as_dict(payload.get("payload"))
    payment = _as_dict(_as_dict(body.get("payment")).get("entity"))
    link = _as_dict(_as_dict(body.get("payment_link")).get("entity"))

    notes = {**_as_dict(link.get("notes")), **_as_dict(payment.get("notes"))}
    amount = payment.get("amount")
    return ParsedWebhook(
        event_type=event_type,
        payment_id=str(payment["id"]) if payment.get("id") else None,
        amount_paise=int(amount) if isinstance(amount, (int, float)) else None,
        payment_status=str(payment["status"]) if payment.get("status") else None,
        recovery_id=str(notes["recovery_id"]) if notes.get("recovery_id") else None,
        txn_id=str(notes.get("txn_id") or notes.get("transaction_id") or "") or None,
        merchant_id=str(notes["merchant_id"]) if notes.get("merchant_id") else None,
    )


def store_webhook_event(
    event_id: str,
    event_type: str,
    body_text: str,
    recovery_id: str | None,
) -> bool:
    """
    Persist a verified webhook. Returns False when this event id was seen before.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        On any storage failure other than a duplicate key. The caller answers
        500 so Razorpay retries.
    """
    try:
        with session_scope() as session:
            session.add(
                WebhookEvent(
                    event_id=event_id,
                    event_type=event_type or None,
                    received_at=_now(),
                    signature_valid=1,
                    process_status="received",
                    recovery_id=recovery_id,
                    payload=body_text,
                )
            )
            session.flush()
    except IntegrityError:
        return False
    return True


def _mark(event_id: str, status: str) -> None:
    """Set the processing status of a stored webhook event."""
    with session_scope() as session:
        row = session.get(WebhookEvent, event_id)
        if row is not None:
            row.process_status = status
            row.processed_at = _now()


def process_webhook_event(event_id: str) -> str:
    """
    Apply one stored webhook to its recovery. Runs after the HTTP response.

    Outcomes recorded in ``webhook_events.process_status``:

    * ``processed`` — a stage event was appended to the recovery.
    * ``ignored``   — an event type this listener does not act on.
    * ``anomaly``   — could not be matched safely (missing or unknown
      recovery id, or amount mismatch). Logged, nothing changed.
    * ``failed``    — unexpected error. A later reprocessing job can retry it.

    Returns
    -------
    str
        The final process status.
    """
    with session_scope() as session:
        row = session.get(WebhookEvent, event_id)
        if row is None:
            logger.error("webhook %s vanished before processing", event_id)
            return "failed"
        body_text = row.payload or "{}"
        recovery_id = row.recovery_id

    try:
        payload = json.loads(body_text)
        parsed = parse_razorpay_event(payload if isinstance(payload, dict) else {})
    except Exception as exc:  # defensive: stored JSON was validated on receipt
        logger.error("webhook %s: stored body unreadable: %s", event_id, exc)
        _mark(event_id, "failed")
        return "failed"

    stage = HANDLED_EVENTS.get(parsed.event_type)
    if stage is None:
        _mark(event_id, "ignored")
        audit_trail.log(
            component="webhook",
            action="event_ignored",
            input_data={"event_id": event_id, "event_type": parsed.event_type},
            decision_reason="event type not handled by Ghost Ledger",
            success=True,
        )
        return "ignored"

    recovery_id = recovery_id or parsed.recovery_id
    if not recovery_id:
        logger.warning("webhook %s (%s) carries no recovery_id — anomaly", event_id, parsed.event_type)
        _mark(event_id, "anomaly")
        audit_trail.log(
            component="webhook",
            action="anomaly_missing_recovery_id",
            input_data={"event_id": event_id, "event_type": parsed.event_type, "payment_id": parsed.payment_id},
            decision_reason="payment notes carry no recovery_id; nothing was changed",
            success=False,
        )
        return "anomaly"

    try:
        if not recovery_store.case_exists(recovery_id):
            logger.warning("webhook %s names unknown recovery %s — anomaly", event_id, recovery_id)
            _mark(event_id, "anomaly")
            audit_trail.log(
                component="webhook",
                action="anomaly_unknown_recovery",
                input_data={"event_id": event_id, "recovery_id": recovery_id},
                decision_reason="recovery_id does not match any recovery case; nothing was changed",
                success=False,
            )
            return "anomaly"

        case = recovery_store.get_recovery_status(recovery_id) or {}
        if parsed.amount_paise is not None and parsed.event_type == "payment.captured":
            expected_paise = int(round(float(case.get("amount_inr", 0.0)) * 100))
            if parsed.amount_paise != expected_paise:
                logger.error(
                    "webhook %s: amount mismatch on %s (paid %s paise, expected %s)",
                    event_id, recovery_id, parsed.amount_paise, expected_paise,
                )
                _mark(event_id, "anomaly")
                audit_trail.log(
                    component="webhook",
                    action="anomaly_amount_mismatch",
                    input_data={
                        "event_id": event_id,
                        "recovery_id": recovery_id,
                        "paid_paise": parsed.amount_paise,
                        "expected_paise": expected_paise,
                    },
                    decision_reason="captured amount differs from the recovery amount; not settled",
                    success=False,
                )
                return "anomaly"

        if parsed.payment_id and recovery_store.has_payment_recorded(recovery_id, parsed.payment_id):
            # A different delivery id for a payment already recorded.
            _mark(event_id, "processed")
            audit_trail.log(
                component="webhook",
                action="duplicate_payment_ignored",
                input_data={"event_id": event_id, "recovery_id": recovery_id, "payment_id": parsed.payment_id},
                decision_reason="this payment is already recorded for the recovery",
                success=True,
            )
            return "processed"

        detail = {
            "event_id": event_id,
            "event_type": parsed.event_type,
            "payment_id": parsed.payment_id,
            "amount_paise": parsed.amount_paise,
            "payment_status": parsed.payment_status,
            "txn_id": parsed.txn_id,
        }
        if stage == "payment_captured" and case.get("status") == "settled":
            if recovery_store.has_provisional_settlement(recovery_id):
                # A status poll saw the link paid; this webhook confirms it with the payment id.
                recovery_store.append_event(
                    recovery_id, "settlement_confirmed", detail=detail, created_by="razorpay_webhook",
                )
                _mark(event_id, "processed")
                audit_trail.log(
                    component="webhook",
                    action="settlement_confirmed",
                    input_data=detail,
                    output_data={"recovery_id": recovery_id},
                    decision_reason="webhook confirmed a provisional settlement seen by status poll",
                    success=True,
                    entity_id=recovery_id,
                )
                return "processed"
            _mark(event_id, "anomaly")
            audit_trail.log(
                component="webhook",
                action="anomaly_second_payment_for_settled_recovery",
                input_data=detail,
                decision_reason="a different payment arrived for a recovery that already settled; "
                                "not counted, flagged for an operator",
                success=False,
                entity_id=recovery_id,
            )
            return "anomaly"
        recovery_store.append_event(recovery_id, stage, detail=detail, created_by="razorpay_webhook")
        _mark(event_id, "processed")
        audit_trail.log(
            component="webhook",
            action=stage,
            input_data=detail,
            output_data={"recovery_id": recovery_id},
            decision_reason=f"verified Razorpay {parsed.event_type} recorded for recovery",
            success=True,
            entity_id=recovery_id,
        )
        return "processed"
    except SQLAlchemyError as exc:
        logger.error("webhook %s: database error while processing: %s", event_id, exc)
        _mark(event_id, "failed")
        return "failed"


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, background: BackgroundTasks) -> JSONResponse:
    """
    Receive a Razorpay webhook delivery.

    Status codes: 200 accepted or duplicate; 400 malformed JSON; 401 bad
    signature; 413 body too large; 500 storage failure (Razorpay will retry);
    503 webhook secret not configured.
    """
    body = await request.body()
    if len(body) > WEBHOOK_MAX_BODY_BYTES:
        return JSONResponse({"status": "payload_too_large"}, status_code=413)

    secret = RAZORPAY_WEBHOOK_SECRET
    if not secret:
        logger.error("webhook received but RAZORPAY_WEBHOOK_SECRET is not set; refusing")
        return JSONResponse({"status": "webhook_secret_not_configured"}, status_code=503)

    signature = request.headers.get(SIGNATURE_HEADER)
    if not verify_signature(body, signature, secret):
        logger.warning("webhook rejected: invalid signature")
        audit_trail.log(
            component="webhook",
            action="signature_rejected",
            input_data={
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "signature_present": signature is not None,
                "client": request.client.host if request.client else None,
            },
            decision_reason="HMAC-SHA256 signature did not match; body not processed",
            success=False,
        )
        return JSONResponse({"status": "invalid_signature"}, status_code=401)

    try:
        payload = json.loads(body)
    except ValueError:
        return JSONResponse({"status": "malformed_json"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"status": "malformed_json"}, status_code=400)

    parsed = parse_razorpay_event(payload)
    event_id = request.headers.get(EVENT_ID_HEADER) or hashlib.sha256(body).hexdigest()
    body_text = body.decode("utf-8", errors="replace")

    try:
        first_time = store_webhook_event(event_id, parsed.event_type, body_text, parsed.recovery_id)
    except SQLAlchemyError as exc:
        logger.error("webhook %s: could not store event: %s", event_id, exc)
        return JSONResponse({"status": "storage_unavailable"}, status_code=500)

    if not first_time:
        logger.info("webhook %s: duplicate delivery, already stored", event_id)
        return JSONResponse({"status": "duplicate", "event_id": event_id}, status_code=200)

    background.add_task(process_webhook_event, event_id)
    return JSONResponse({"status": "accepted", "event_id": event_id}, status_code=200)
