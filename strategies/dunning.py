"""
Ghost Ledger v3 — dunning sequencer (B5).

A dunning sequence is a fixed list of customer reminders for one failed
payment, for example: email at 0 minutes, SMS at 24 hours, a final notice at 72
hours. Each touch is one row in ``dunning_touches``.

Rules
-----
* **Idempotent scheduling.** ``(recovery_id, touch_no)`` is unique, so
  scheduling the same sequence twice creates no duplicates.
* **No sends after settlement.** Before each send, the recovery is checked. If it
  is no longer ``pending``, the touch is marked ``cancelled`` and nothing is sent.
* **Sending is mocked.** :func:`send_via_channel` does not contact any provider.
  It returns a simulated receipt. Replace that one function to connect a real
  email, SMS, or WhatsApp provider.
* **Recipient by reference.** A touch carries the customer id, never a phone
  number or email address. Those are not stored in Ghost Ledger.
* **Text from templates only.** Messages come from the playbook templates with
  three fields filled in. No LLM generates or edits customer-facing text.

The job that calls :func:`run_due` runs every ``JOB_DUNNING_SECONDS`` (default 900)
when ``ENABLE_ADVANCED_STRATEGIES`` is on.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable

from sqlalchemy import select

from database import recovery_store
from database.engine import session_scope
from database.models import DunningTouch
from strategies.playbooks import CHANNELS

logger = logging.getLogger(__name__)

STATUS_SCHEDULED = "scheduled"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

NEXT_STEP_TEXT: dict[str, str] = {
    "create_link": "Please complete the payment using the secure link we sent you.",
    "schedule_retry": "We will retry the payment automatically.",
    "dunning_only": "Please renew your mandate or update your payment method.",
}


def send_via_channel(channel: str, recipient_ref: str, message: str) -> dict[str, Any]:
    """
    MOCK sender. Validates the channel and returns a simulated receipt.

    No network call is made. Tests replace this function. A real provider goes
    here, behind the same signature.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}")
    logger.info("simulated %s send to %s (%d chars)", channel, recipient_ref, len(message))
    return {"ok": True, "provider": "simulated", "provider_ref": f"sim_{uuid.uuid4().hex[:12]}"}


def render_message(template_text: str, *, amount_inr: float, txn_id: str, next_step: str) -> str:
    """Fill the three allowed template fields. Unknown fields are rejected by the loader."""
    return template_text.format(
        amount_inr=f"{float(amount_inr):,.2f}", txn_id=txn_id, next_step=next_step,
    )


def next_step_for(action: str, suggested_methods: Iterable[str] = ()) -> str:
    """Return the fixed customer-facing sentence for an action, plus any method suggestion."""
    text = NEXT_STEP_TEXT.get(action, NEXT_STEP_TEXT["dunning_only"])
    methods = [m for m in suggested_methods]
    if methods:
        names = " or ".join(m.upper() if m == "upi" else m for m in methods)
        text = f"{text} You can also pay by {names}."
    return text


def schedule_touches(
    recovery_id: str,
    touches: Iterable[dict[str, Any]],
    *,
    base: datetime,
    messages: dict[int, str],
) -> int:
    """
    Insert the touches that do not exist yet. Returns how many were inserted.

    ``touches`` are dicts with ``touch_no``, ``offset_minutes``, ``channel``, ``template``.
    ``messages`` maps ``touch_no`` to the rendered text.
    """
    created_at = base.isoformat(sep=" ", timespec="seconds")
    inserted = 0
    with session_scope() as session:
        existing = {
            n for (n,) in session.execute(
                select(DunningTouch.touch_no).where(DunningTouch.recovery_id == recovery_id)
            ).all()
        }
        for touch in touches:
            number = int(touch["touch_no"])
            if number in existing:
                continue
            due = base + timedelta(minutes=int(touch["offset_minutes"]))
            session.add(
                DunningTouch(
                    recovery_id=recovery_id,
                    touch_no=number,
                    channel=str(touch["channel"]),
                    template=str(touch["template"]),
                    message=messages[number],
                    due_at=due.isoformat(sep=" ", timespec="seconds"),
                    status=STATUS_SCHEDULED,
                    created_at=created_at,
                )
            )
            inserted += 1
    return inserted


def _mark(touch_id: int, status: str, *, sent_at: str | None = None,
          provider_ref: str | None = None, note: str | None = None) -> None:
    with session_scope() as session:
        row = session.get(DunningTouch, touch_id)
        if row is None:
            return
        row.status = status
        if sent_at is not None:
            row.sent_at = sent_at
        if provider_ref is not None:
            row.provider_ref = provider_ref
        if note is not None:
            row.note = note


def cancel_for_recovery(recovery_id: str, reason: str) -> int:
    """Cancel every scheduled touch for a recovery (for example, on settlement)."""
    with session_scope() as session:
        rows = session.scalars(
            select(DunningTouch).where(
                DunningTouch.recovery_id == recovery_id,
                DunningTouch.status == STATUS_SCHEDULED,
            )
        ).all()
        for row in rows:
            row.status = STATUS_CANCELLED
            row.note = reason
        count = len(rows)
    if count:
        recovery_store.append_event(
            recovery_id, "dunning_cancelled", {"count": count, "reason": reason},
            created_by="dunning_sequencer",
        )
    return count


def _due_rows(now: datetime) -> list[dict[str, Any]]:
    stamp = now.isoformat(sep=" ", timespec="seconds")
    with session_scope() as session:
        rows = session.scalars(
            select(DunningTouch)
            .where(DunningTouch.status == STATUS_SCHEDULED, DunningTouch.due_at <= stamp)
            .order_by(DunningTouch.id)
        ).all()
        return [
            {
                "id": r.id,
                "recovery_id": r.recovery_id,
                "touch_no": r.touch_no,
                "channel": r.channel,
                "message": r.message,
            }
            for r in rows
        ]


def run_due(
    now: datetime | None = None,
    sender: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, int]:
    """
    Send every touch that is due. Never raises for a single bad touch.

    Returns counts: ``sent``, ``failed``, ``cancelled``.
    """
    now = now or datetime.now()
    send = sender or send_via_channel  # look up at call time so tests can replace it
    counts = {"sent": 0, "failed": 0, "cancelled": 0}
    for touch in _due_rows(now):
        rid = touch["recovery_id"]
        case = recovery_store.get_recovery_status(rid)
        if case is None or case["status"] != "pending":
            _mark(touch["id"], STATUS_CANCELLED, note="recovery no longer pending")
            counts["cancelled"] += 1
            continue
        try:
            receipt = send(touch["channel"], case.get("customer_id") or "unknown_customer", touch["message"])
        except Exception as exc:  # one failed send must not stop the sequence
            _mark(touch["id"], STATUS_FAILED, note=f"send error: {type(exc).__name__}")
            recovery_store.append_event(
                rid, "reminder_queued",
                {"touch_no": touch["touch_no"], "channel": touch["channel"], "status": STATUS_FAILED},
                created_by="dunning_sequencer",
            )
            counts["failed"] += 1
            continue
        ok = bool(receipt.get("ok"))
        status = STATUS_SENT if ok else STATUS_FAILED
        _mark(
            touch["id"], status,
            sent_at=now.isoformat(sep=" ", timespec="seconds") if ok else None,
            provider_ref=receipt.get("provider_ref"),
        )
        recovery_store.append_event(
            rid, "reminder_queued",
            {"touch_no": touch["touch_no"], "channel": touch["channel"], "status": status},
            created_by="dunning_sequencer",
        )
        counts["sent" if ok else "failed"] += 1
    return counts


def list_touches(recovery_id: str) -> list[dict[str, Any]]:
    """Return every touch for a recovery, in sequence order."""
    with session_scope() as session:
        rows = session.scalars(
            select(DunningTouch)
            .where(DunningTouch.recovery_id == recovery_id)
            .order_by(DunningTouch.touch_no)
        ).all()
        return [
            {
                "touch_no": r.touch_no, "channel": r.channel, "status": r.status,
                "due_at": r.due_at, "sent_at": r.sent_at, "note": r.note,
            }
            for r in rows
        ]
