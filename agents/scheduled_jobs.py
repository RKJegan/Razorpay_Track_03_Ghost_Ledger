"""
Ghost Ledger v3 — background jobs (A4).

Jobs
----
check_payment_settlement   read each open payment link's status from Razorpay.
                           A paid link is recorded as a PROVISIONAL settlement;
                           the webhook confirms it with the payment id.
reconcile_recovery         for webhook-confirmed settlements, check the captured
                           amount against the recovery amount and record the result.
retry_recovery             when the customer's last link expired or a payment
                           failed, ask the policy gate for the next attempt.
clean_expired_links        mark expired links; queue a reminder if attempts
                           remain, otherwise escalate.

Rules that hold for every job
-----------------------------
* A degraded or failed Razorpay response is a TRANSIENT error. It changes no
  recovery state. A simulated response in live mode is never treated as truth.
* Every state change goes through :mod:`database.recovery_store` (append-only)
  and is audited.
* Repeated failures of a job back off exponentially (see :func:`run_guarded`).
  While backing off the job is skipped and the skip is visible in ``job_state``.

Trust boundary: no LLM, no randomness. Every decision is a comparison or a
policy call.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy import select

import config
from agents import recovery_executor
from api.razorpay_client import LiveRazorpayClient, RazorpayResponse, SimulatedRazorpayClient, get_razorpay_client
from database import audit_trail, recovery_store
from database.engine import session_scope
from database.models import JobState, RecoveryCase, RecoveryEvent
from agents.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

LINK_STAGE = "payment_link_created"
PAID_STATUSES = frozenset({"paid"})
EXPIRED_STATUSES = frozenset({"expired", "cancelled"})


class TransientJobError(RuntimeError):
    """A recoverable failure, such as a degraded or failed Razorpay response."""


def _now() -> datetime:
    """Return the current local time. Isolated so tests can reason about it."""
    return datetime.now()


def _ts(value: datetime) -> str:
    """Format a datetime in the project's canonical format."""
    return value.isoformat(sep=" ", timespec="seconds")


# --- backoff ------------------------------------------------------------------

def backoff_seconds(streak: int, base: int | None = None, cap: int | None = None) -> int:
    """
    Return the wait after ``streak`` consecutive failures: base * 2^(streak-1), capped.

    ``streak`` is 1 for the first failure. Pure function, so it is unit-tested directly.
    """
    base = config.JOB_BACKOFF_BASE_SECONDS if base is None else base
    cap = config.JOB_BACKOFF_MAX_SECONDS if cap is None else cap
    if streak < 1:
        return 0
    return int(min(cap, base * (2 ** (streak - 1))))


def _load_state(job_name: str) -> dict[str, Any]:
    """Return the health record for a job (defaults if it has never run)."""
    with session_scope() as session:
        row = session.get(JobState, job_name)
        if row is None:
            return {"consecutive_failures": 0, "next_allowed_at": None}
        return {
            "consecutive_failures": row.consecutive_failures,
            "next_allowed_at": row.next_allowed_at,
        }


def _save_state(job_name: str, **fields: Any) -> None:
    """Create or update a job's health record."""
    with session_scope() as session:
        row = session.get(JobState, job_name)
        if row is None:
            row = JobState(job_name=job_name)
            session.add(row)
        for key, value in fields.items():
            setattr(row, key, value)


def run_guarded(job_name: str, job: Callable[[], Any], now: datetime | None = None) -> dict[str, Any]:
    """
    Run a job, honouring its backoff window, and record the outcome.

    Returns
    -------
    dict[str, Any]
        ``{"job": name, "status": "ok" | "skipped" | "failed", ...}``.
        Never raises: the scheduler must keep running after a job fails.
    """
    now = now or _now()
    state = _load_state(job_name)
    if state["next_allowed_at"] and now < datetime.fromisoformat(state["next_allowed_at"]):
        return {"job": job_name, "status": "skipped", "until": state["next_allowed_at"]}

    try:
        result = job()
    except Exception as exc:  # the scheduler must survive any single job failure
        streak = int(state["consecutive_failures"]) + 1
        wait = backoff_seconds(streak)
        next_at = now + timedelta(seconds=wait)
        _save_state(
            job_name,
            consecutive_failures=streak,
            next_allowed_at=_ts(next_at),
            last_run_at=_ts(now),
            last_status="failed",
            last_error=f"{type(exc).__name__}: {exc}"[:500],
        )
        logger.error("job %s failed (streak %d); next run after %s: %s", job_name, streak, _ts(next_at), exc)
        audit_trail.log(
            component="pipeline",
            action=f"job_{job_name}_failed",
            output_data={"streak": streak, "next_allowed_at": _ts(next_at), "error": str(exc)[:300]},
            decision_reason="background job failed; backing off",
            success=False,
        )
        return {"job": job_name, "status": "failed", "streak": streak, "next_allowed_at": _ts(next_at)}

    _save_state(
        job_name,
        consecutive_failures=0,
        next_allowed_at=None,
        last_run_at=_ts(now),
        last_status="ok",
        last_error=None,
    )
    return {"job": job_name, "status": "ok", "result": result}


# --- helpers ------------------------------------------------------------------

def _client(client: SimulatedRazorpayClient | LiveRazorpayClient | None) -> SimulatedRazorpayClient | LiveRazorpayClient:
    return client or get_razorpay_client()


def _checked(response: RazorpayResponse, what: str) -> dict[str, Any]:
    """
    Turn a Razorpay response into data, or raise a transient error.

    A degraded response in live mode is the simulator standing in for a failed
    call. It must not drive any state change.
    """
    if not response.ok:
        raise TransientJobError(f"{what}: {response.error or 'call failed'}")
    if response.degraded:
        raise TransientJobError(f"{what}: degraded response, not trusted")
    return dict(response.data)


def job_health() -> list[dict[str, Any]]:
    """Return the health record of every job that has run at least once."""
    with session_scope() as session:
        rows = session.scalars(select(JobState).order_by(JobState.job_name)).all()
        return [
            {
                "job": row.job_name,
                "last_status": row.last_status,
                "last_run_at": row.last_run_at,
                "consecutive_failures": row.consecutive_failures,
                "next_allowed_at": row.next_allowed_at,
                "last_error": row.last_error,
            }
            for row in rows
        ]


def _open_link_recoveries() -> list[str]:
    """Return ids of pending recoveries that have a payment link and no capture yet."""
    with session_scope() as session:
        rows = session.scalars(
            select(RecoveryEvent.recovery_id)
            .where(RecoveryEvent.stage == LINK_STAGE)
            .distinct()
        ).all()
    open_ids: list[str] = []
    for rid in rows:
        case = recovery_store.get_recovery_status(rid)
        if case and case["status"] == "pending":
            open_ids.append(rid)
    return sorted(open_ids)


def _latest_link(recovery_id: str) -> dict[str, Any] | None:
    """Return the most recent payment-link event for a recovery, or None."""
    return recovery_store.latest_event(recovery_id, (LINK_STAGE,))


# --- jobs ---------------------------------------------------------------------

def check_payment_settlement(
    recovery_id: str,
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
) -> str:
    """
    Poll one recovery's payment link. Record a paid link as a provisional settlement.

    Returns
    -------
    str
        ``no_link`` | ``not_paid`` | ``provisional_settlement`` | ``already_recorded``

    Raises
    ------
    TransientJobError
        When Razorpay could not give a trusted answer.
    """
    link = _latest_link(recovery_id)
    if link is None:
        return "no_link"
    link_id = (link["detail"] or {}).get("link_id")
    if not link_id:
        return "no_link"

    data = _checked(_client(client).fetch_payment_link(link_id), f"fetch link {link_id}")
    status = str(data.get("status") or "")
    if status not in PAID_STATUSES:
        return "not_paid"

    case = recovery_store.get_recovery_status(recovery_id) or {}
    paid_paise = int(data.get("amount_paid") or 0)
    expected_paise = int(round(float(case.get("amount_inr", 0.0)) * 100))
    if paid_paise and paid_paise != expected_paise:
        audit_trail.log(
            component="pipeline", action="poll_amount_mismatch",
            input_data={"recovery_id": recovery_id, "link_id": link_id},
            output_data={"paid_paise": paid_paise, "expected_paise": expected_paise},
            decision_reason="link reports a different amount than the recovery; not settled",
            success=False, entity_id=recovery_id,
        )
        return "not_paid"

    if recovery_store.has_provisional_settlement(recovery_id) or case.get("settled_at"):
        return "already_recorded"

    payments = data.get("payments") or []
    payment_id = None
    if isinstance(payments, list) and payments and isinstance(payments[0], dict):
        payment_id = payments[0].get("id")
    detail = {
        "source": "status_poll",
        "link_id": link_id,
        "payment_id": payment_id,
        "amount_paise": expected_paise,
        "link_status": status,
        "confirmed_by_webhook": False,
    }
    recovery_store.append_event(recovery_id, "payment_captured", detail=detail, created_by="settlement_poll")
    audit_trail.log(
        component="pipeline", action="provisional_settlement",
        input_data={"recovery_id": recovery_id, "link_id": link_id},
        output_data=detail,
        decision_reason="status poll reports the link paid; awaiting webhook confirmation",
        success=True, entity_id=recovery_id,
    )
    return "provisional_settlement"


def poll_open_links(client: SimulatedRazorpayClient | LiveRazorpayClient | None = None) -> dict[str, int]:
    """Run :func:`check_payment_settlement` on every open link. Raises if any call was transient."""
    counts: dict[str, int] = {}
    transient: list[str] = []
    for rid in _open_link_recoveries():
        try:
            outcome = check_payment_settlement(rid, client=client)
        except TransientJobError as exc:
            transient.append(f"{rid}: {exc}")
            outcome = "transient_error"
        counts[outcome] = counts.get(outcome, 0) + 1
    if transient:
        raise TransientJobError(f"{len(transient)} link(s) could not be checked; first: {transient[0]}")
    return counts


def reconcile_recovery(recovery_id: str) -> str:
    """
    Compare a webhook-confirmed capture with the recovery amount, and record the result.

    Only a payment with a payment id (webhook-backed) is reconciled. A provisional
    poll settlement is left alone until its webhook arrives.

    Returns
    -------
    str
        ``reconciled`` | ``mismatch`` | ``awaiting_confirmation`` | ``already_reconciled`` | ``not_settled``
    """
    case = recovery_store.get_recovery_status(recovery_id)
    if case is None or case["status"] != "settled":
        return "not_settled"
    timeline = recovery_store.get_recovery_timeline(recovery_id)
    if any(e["stage"] == "settlement_confirmed" and (e["detail"] or {}).get("reconciled") is not None
           for e in timeline):
        return "already_reconciled"
    captures = [e for e in timeline if e["stage"] in ("payment_captured", "settlement_confirmed")]
    confirmed = [e for e in captures if (e["detail"] or {}).get("payment_id")]
    if not confirmed:
        return "awaiting_confirmation"

    expected_paise = int(round(float(case["amount_inr"]) * 100))
    paid_paise = int((confirmed[-1]["detail"] or {}).get("amount_paise") or 0)
    matched = paid_paise == expected_paise
    detail = {
        "reconciled": matched,
        "expected_paise": expected_paise,
        "paid_paise": paid_paise,
        "payment_id": (confirmed[-1]["detail"] or {}).get("payment_id"),
    }
    if matched:
        recovery_store.append_event(recovery_id, "settlement_confirmed", detail=detail, created_by="reconciler")
        return "reconciled"
    audit_trail.log(
        component="pipeline", action="reconciliation_mismatch",
        input_data={"recovery_id": recovery_id}, output_data=detail,
        decision_reason="paid amount differs from the recovery amount; flagged, not closed",
        success=False, entity_id=recovery_id,
    )
    return "mismatch"


def reconcile_all() -> dict[str, int]:
    """Reconcile every settled recovery that is not yet reconciled."""
    with session_scope() as session:
        ids = session.scalars(
            select(RecoveryCase.id).where(RecoveryCase.status == "settled")
        ).all()
    counts: dict[str, int] = {}
    for rid in ids:
        outcome = reconcile_recovery(rid)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def retry_recovery(recovery_id: str, client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
                   engine: PolicyEngine | None = None) -> str:
    """
    Create the next attempt if the last link is dead (expired or the payment failed).

    An active, unpaid link is never replaced. The policy gate decides the rest.

    Returns
    -------
    str
        ``link_still_active`` | ``not_pending`` | the executor's outcome.
    """
    case = recovery_store.get_recovery_status(recovery_id)
    if case is None or case["status"] != "pending":
        return "not_pending"
    link = _latest_link(recovery_id)
    failure = recovery_store.latest_event(recovery_id, ("payment_failed", "payment_link_expired"))
    if link is None:
        return "no_link"
    if failure is None or failure["event_id"] < link["event_id"]:
        return "link_still_active"
    result = recovery_executor.submit_recovery(recovery_id, client=_client(client), engine=engine)
    return result.outcome


def retry_due_recoveries(client: SimulatedRazorpayClient | LiveRazorpayClient | None = None) -> dict[str, int]:
    """Run :func:`retry_recovery` over every open recovery that has a link."""
    counts: dict[str, int] = {}
    for rid in _open_link_recoveries():
        outcome = retry_recovery(rid, client=client)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _link_is_expired(link: dict[str, Any], remote_status: str | None, now: datetime) -> bool:
    """A link is expired when Razorpay says so, or its own expiry time has passed (unpaid)."""
    if remote_status in EXPIRED_STATUSES:
        return True
    expire_by = (link["detail"] or {}).get("expire_by")
    return bool(expire_by) and now.timestamp() >= float(expire_by)


def clean_expired_links(
    client: SimulatedRazorpayClient | LiveRazorpayClient | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """
    Handle expired payment links.

    An expired link with attempts left gets a ``reminder_queued`` event, and the
    retry job then makes the next attempt. Once the attempt limit is reached the
    recovery is escalated through the stopping rule.
    """
    now = now or _now()
    counts: dict[str, int] = {}
    transient: list[str] = []
    for rid in _open_link_recoveries():
        link = _latest_link(rid)
        if link is None:
            continue
        expired_mark = recovery_store.latest_event(rid, ("payment_link_expired",))
        if expired_mark is not None and expired_mark["event_id"] > link["event_id"]:
            counts["already_expired"] = counts.get("already_expired", 0) + 1
            continue
        link_id = (link["detail"] or {}).get("link_id")
        remote_status = None
        if link_id:
            try:
                data = _checked(_client(client).fetch_payment_link(link_id), f"fetch link {link_id}")
                remote_status = str(data.get("status") or "")
            except TransientJobError as exc:
                transient.append(f"{rid}: {exc}")
                counts["transient_error"] = counts.get("transient_error", 0) + 1
                continue
        if not _link_is_expired(link, remote_status, now) or remote_status in PAID_STATUSES:
            counts["still_active"] = counts.get("still_active", 0) + 1
            continue

        recovery_store.append_event(
            rid, "payment_link_expired",
            {"link_id": link_id, "remote_status": remote_status, "checked_at": _ts(now)},
            created_by="expiry_job",
        )
        attempts = recovery_store.count_events(rid, (LINK_STAGE,))
        if attempts < config.POLICY_MAX_ATTEMPTS_PER_FAILURE:
            recovery_store.append_event(
                rid, "reminder_queued",
                {"reason": "payment link expired", "attempts_so_far": attempts},
                created_by="expiry_job",
            )
            counts["reminder_queued"] = counts.get("reminder_queued", 0) + 1
        else:
            recovery_store.append_event(
                rid, "stopping_rule_triggered",
                {"reason": f"payment link expired after {attempts} attempts; attempt limit reached"},
                created_by="expiry_job",
            )
            counts["escalated"] = counts.get("escalated", 0) + 1

    if transient:
        raise TransientJobError(f"{len(transient)} link(s) could not be checked; first: {transient[0]}")
    return counts
