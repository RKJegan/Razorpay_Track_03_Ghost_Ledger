"""
Ghost Ledger v3 — deterministic strategy router (B2).

:func:`route` takes a validated playbook and a :class:`RouteContext` and returns
a :class:`StrategyPlan`. It is a pure function of its inputs: no database, no
clock, no randomness. The caller supplies ``now`` and the health snapshots.
The same inputs always produce the same plan and the same reasons.

Order of operations in the runner (see :mod:`strategies.runner`):

1. The policy engine rules on the next attempt (R1-R4). Nothing else happens yet.
2. Only if that ruling is ``allowed`` (and not blocked on approval) does the
   router run. A policy denial, a stop, or an approval request never reaches here.
3. The router can only choose among actions the policy already allowed:
   create a payment link now, schedule a retry later, or send dunning only. It
   cannot raise an amount, skip an approval, or override a stop.

Actions
-------
``create_link``     create a payment link now (through the policy-gated executor)
``schedule_retry``  the timing rule says wait; the retry runs at ``retry_at``
``dunning_only``    no link (for example, a mandate that must be renewed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategies.gateway import DEGRADED, HealthSnapshot, decide_failover
from strategies.methods import suggest_methods
from strategies.playbooks import Playbook
from strategies.timing import apply_rule

ACTION_CREATE_LINK = "create_link"
ACTION_SCHEDULE_RETRY = "schedule_retry"
ACTION_DUNNING_ONLY = "dunning_only"
ACTIONS: tuple[str, ...] = (ACTION_CREATE_LINK, ACTION_SCHEDULE_RETRY, ACTION_DUNNING_ONLY)

VARIANT_CONTROL = "control"


@dataclass(frozen=True)
class RouteContext:
    """Everything the router may look at. Built by the runner from the database."""

    cause: str
    now: datetime
    attempt_number: int
    card_failures: int
    last_method: str | None
    method_health: dict[str, HealthSnapshot] = field(default_factory=dict)
    ab_variant: str | None = None  # None = no experiment; "control" = v2 immediate timing


@dataclass(frozen=True)
class StrategyPlan:
    """The routed decision for one attempt, with the reason for each step."""

    cause: str
    playbook_version: int
    action: str
    retry_at: datetime | None
    timing_rule: str
    failover_to: str | None
    suggested_methods: tuple[str, ...]
    touches: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe copy for the audit trail and the event log."""
        return {
            "cause": self.cause,
            "playbook_version": self.playbook_version,
            "action": self.action,
            "retry_at": self.retry_at.isoformat(sep=" ", timespec="seconds") if self.retry_at else None,
            "timing_rule": self.timing_rule,
            "failover_to": self.failover_to,
            "suggested_methods": list(self.suggested_methods),
            "touches": [dict(t) for t in self.touches],
            "reasons": list(self.reasons),
        }


def _health(ctx: RouteContext, method: str) -> HealthSnapshot:
    """Snapshot for ``method``, or an empty unknown snapshot if none was supplied."""
    if method in ctx.method_health:
        return ctx.method_health[method]
    return HealthSnapshot(method, 0, 0, None, "unknown", 0)


def route(playbook: Playbook, ctx: RouteContext) -> StrategyPlan:
    """
    Choose the action for one attempt. Pure and deterministic.

    Raises
    ------
    ValueError
        If the playbook is for a different cause than the context.
    """
    if playbook.cause != ctx.cause:
        raise ValueError(f"playbook is for {playbook.cause!r}, context is {ctx.cause!r}")

    reasons: list[str] = [f"playbook {playbook.cause} v{playbook.version}"]
    action = ACTION_CREATE_LINK
    retry_at: datetime | None = None

    if playbook.route == "dunning_only":
        action = ACTION_DUNNING_ONLY
        reasons.append("playbook route is dunning_only: no payment link for this cause")
    elif ctx.ab_variant == VARIANT_CONTROL:
        reasons.append("A/B control arm: immediate attempt (v2 timing)")
    elif playbook.timing_rule != "none":
        run_at = apply_rule(playbook.timing_rule, ctx.now)
        if run_at is not None and run_at > ctx.now:
            action = ACTION_SCHEDULE_RETRY
            retry_at = run_at
            reasons.append(
                f"timing rule {playbook.timing_rule} defers the attempt to "
                f"{run_at.isoformat(sep=' ', timespec='minutes')}"
            )
        else:
            reasons.append(f"timing rule {playbook.timing_rule}: run now")
    else:
        reasons.append("no timing rule: immediate attempt")

    failover_to: str | None = None
    if playbook.failover_enabled and ctx.last_method:
        decision = decide_failover(
            _health(ctx, ctx.last_method),
            [_health(ctx, m) for m in playbook.failover_candidates],
        )
        reasons.append(decision.reason)
        failover_to = decision.to

    suggested: tuple[str, ...] = ()
    if playbook.method_candidates and playbook.method_after_card_failures is not None:
        degraded = [m for m, h in ctx.method_health.items() if h.state == DEGRADED]
        suggested = suggest_methods(
            ctx.card_failures,
            playbook.method_after_card_failures,
            playbook.method_candidates,
            degraded,
        )
        if suggested:
            reasons.append(
                f"{ctx.card_failures} card failure(s) >= {playbook.method_after_card_failures}: "
                f"suggest {', '.join(suggested)}"
            )

    touches = tuple(
        {
            "touch_no": t.touch_no,
            "offset_minutes": t.offset_minutes,
            "channel": t.channel,
            "template": t.template,
        }
        for t in playbook.touches
    )

    return StrategyPlan(
        cause=playbook.cause,
        playbook_version=playbook.version,
        action=action,
        retry_at=retry_at,
        timing_rule=playbook.timing_rule,
        failover_to=failover_to,
        suggested_methods=suggested,
        touches=touches,
        reasons=tuple(reasons),
    )
