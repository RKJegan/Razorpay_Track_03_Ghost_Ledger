"""
Ghost Ledger v3 — strategy runner (B8 glue).

This module is the only place where the strategy layer touches recovery
actions. It is reached from :func:`agents.recovery_executor.submit_recovery`
only when ``ENABLE_ADVANCED_STRATEGIES`` is on. With the flag off, nothing in
this module runs.

Sequence for one attempt (:func:`handle_failure`):

1. ``rule_on_recovery``: the policy engine rules (R1-R4). The ruling is
   logged here, before anything else.
2. If the ruling blocks (stop, denial, or approval needed), that outcome is
   returned exactly as v2 would return it. The router does not run.
3. Only if the ruling is allowed: the playbook for the cause is loaded (with
   hot reload). If it is missing or invalid, the attempt falls back to the v2
   behaviour (immediate link, still policy-gated) and records why.
4. The deterministic router builds a plan. The executor records it. The action
   is then taken: a link through ``execute_ruling``, a scheduled retry, or
   dunning only.

Also here: :func:`retry_pass` (the scheduled retry job), :func:`observe_payment`
(the webhook hook for gateway health and dunning cancellation), and
:func:`dry_run_summary` (the startup check in ``main.py``).

Trust boundary: no LLM. Every amount, approval, stop, and timing decision is
deterministic Python. Text comes from playbook templates.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import config
from agents.recovery_executor import (
    LINK_STAGES,
    Ruling,
    SubmitResult,
    execute_ruling,
    rule_on_recovery,
)
from database import recovery_store
from strategies import dunning
from strategies.ab_test import ABTestManager
from strategies.executor import PlaybookExecutor
from strategies.gateway import GatewayHealthMonitor
from strategies.playbooks import PlaybookError, PlaybookLoader, default_loader
from strategies.router import (
    ACTION_DUNNING_ONLY,
    ACTION_SCHEDULE_RETRY,
    RouteContext,
    StrategyPlan,
    route,
)

logger = logging.getLogger(__name__)

AB_EXPERIMENT = "retry_timing_v1"
DEFAULT_ROUTE = "card"
FAILURE_STAGES: tuple[str, ...] = ("payment_failed", "payment_link_expired")


def _ab_variant(recovery_id: str, now: datetime) -> str | None:
    if not config.AB_RETRY_TIMING_EXPERIMENT:
        return None
    return ABTestManager().assign(AB_EXPERIMENT, recovery_id, split_percent=50.0, now=now)


def _card_failures_and_last_method(recovery_id: str) -> tuple[int, str | None]:
    """Count card failures on this recovery and return the last failed method."""
    card_failures = 0
    last_method: str | None = None
    for event in recovery_store.get_recovery_timeline(recovery_id):
        if event["stage"] != "payment_failed":
            continue
        method = (event.get("detail") or {}).get("payment_method")
        if method == "card":
            card_failures += 1
        if method:
            last_method = method
    return card_failures, last_method


def build_context(ruling: Ruling, playbook_methods: set[str], now: datetime,
                  monitor: GatewayHealthMonitor | None = None) -> RouteContext:
    """Collect the facts the router may use. Reads the database; the router does not."""
    recovery_id = ruling.recovery_id
    card_failures, failed_method = _card_failures_and_last_method(recovery_id)
    # The first route is a card payment link, so the current route is the last
    # failed method, or "card" before any failure has been seen.
    current = failed_method or DEFAULT_ROUTE
    monitor = monitor or GatewayHealthMonitor()
    methods = set(playbook_methods) | {DEFAULT_ROUTE, current}
    health = {m: monitor.snapshot(m, now) for m in sorted(methods)}
    return RouteContext(
        cause=ruling.case.get("cause") or "unknown",
        now=now,
        attempt_number=ruling.attempt,
        card_failures=card_failures,
        last_method=current,
        method_health=health,
        ab_variant=_ab_variant(recovery_id, now),
    )


def handle_failure(
    recovery_id: str,
    *,
    approved: bool = False,
    approval_id: str | None = None,
    customer: dict[str, str] | None = None,
    client: Any = None,
    engine: Any = None,
    now: datetime | None = None,
    loader: PlaybookLoader | None = None,
) -> SubmitResult:
    """Run one recovery attempt through the policy gate, then the strategy layer."""
    now = now or datetime.now()
    ruling = rule_on_recovery(recovery_id, approved=approved, approval_id=approval_id, engine=engine)
    if isinstance(ruling, SubmitResult):
        return ruling

    decision = ruling.decision
    blocked = decision.stopping_rule_triggered or (decision.requires_approval and not approved)
    if blocked or not decision.allowed:
        # Blocked by policy: the v2 outcome, unchanged. The router never runs.
        return execute_ruling(ruling, client=client, customer=customer)

    cause = ruling.case.get("cause") or ""
    try:
        book = (loader or default_loader()).get(cause)
    except PlaybookError as exc:
        logger.error("no usable playbook for %s, falling back to v2 path: %s", cause, exc)
        recovery_store.append_event(
            recovery_id, "playbook_selected",
            {"fallback": "v2_immediate", "error": str(exc)[:300]}, created_by="strategy_router",
        )
        return execute_ruling(ruling, client=client, customer=customer)

    ctx = build_context(ruling, set(book.method_candidates) | set(book.failover_candidates), now)
    plan: StrategyPlan = route(book, ctx)
    PlaybookExecutor().apply(
        recovery_id, plan, book,
        amount_inr=float(ruling.case["amount_inr"]),
        txn_id=str(ruling.case.get("txn_id") or recovery_id),
        now=now,
    )

    reason = "; ".join(plan.reasons)
    if plan.action == ACTION_SCHEDULE_RETRY:
        return SubmitResult(recovery_id, "retry_scheduled", reason, ruling.attempt)
    if plan.action == ACTION_DUNNING_ONLY:
        return SubmitResult(recovery_id, "dunning_only", reason, ruling.attempt)
    return execute_ruling(ruling, client=client, customer=customer)


def _latest_link(recovery_id: str) -> dict[str, Any] | None:
    return recovery_store.latest_event(recovery_id, LINK_STAGES)


def retry_one(recovery_id: str, client: Any = None, now: datetime | None = None) -> str:
    """Decide whether one open recovery needs an attempt now. Returns an outcome label."""
    now = now or datetime.now()
    scheduled = recovery_store.latest_event(recovery_id, ("retry_scheduled",))
    link = _latest_link(recovery_id)
    if scheduled is not None and (link is None or scheduled["event_id"] > link["event_id"]):
        run_at = datetime.fromisoformat(scheduled["detail"]["run_at"])
        if run_at > now:
            return "not_due"
        return handle_failure(recovery_id, client=client, now=now).outcome
    if link is None:
        return "no_link"
    failure = recovery_store.latest_event(recovery_id, FAILURE_STAGES)
    if failure is None or failure["event_id"] < link["event_id"]:
        return "link_still_active"
    return handle_failure(recovery_id, client=client, now=now).outcome


def retry_pass(client: Any = None, now: datetime | None = None) -> dict[str, int]:
    """Run :func:`retry_one` over every pending recovery. Used by the retry job."""
    now = now or datetime.now()
    counts: dict[str, int] = {}
    for case in recovery_store.get_active_recoveries():
        outcome = retry_one(case["recovery_id"], client=client, now=now)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def observe_payment(recovery_id: str | None, stage: str, payment_method: str | None,
                    now: datetime | None = None) -> None:
    """
    Webhook hook, called after a payment event is recorded. Never raises.

    * ``payment_captured`` / ``payment_failed`` feed gateway health.
    * ``payment_captured`` cancels the rest of the dunning sequence.
    """
    try:
        if stage in ("payment_captured", "payment_failed"):
            GatewayHealthMonitor().record(
                payment_method, success=(stage == "payment_captured"),
                recovery_id=recovery_id, now=now,
            )
        if stage == "payment_captured" and recovery_id:
            dunning.cancel_for_recovery(recovery_id, "customer paid")
    except Exception:  # the webhook path must never fail because of analytics
        logger.exception("strategy observation failed for %s (%s)", recovery_id, stage)


def dry_run_summary(loader: PlaybookLoader | None = None) -> list[str]:
    """Return one readable line per cause, from the validated playbooks. No writes."""
    loader = loader or default_loader()
    lines: list[str] = []
    for cause, book in sorted(loader.all().items()):
        touches = ", ".join(f"{t.channel}@{t.offset_minutes}m" for t in book.touches) or "none"
        lines.append(
            f"{cause}: v{book.version} route={book.route} timing={book.timing_rule} "
            f"failover={'on' if book.failover_enabled else 'off'} dunning=[{touches}]"
        )
    return lines


__all__ = [
    "AB_EXPERIMENT",
    "build_context",
    "dry_run_summary",
    "handle_failure",
    "observe_payment",
    "retry_one",
    "retry_pass",
]
