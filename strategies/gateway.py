"""
Ghost Ledger v3 — gateway health and failover (B4).

"Gateway" here means a Razorpay payment method route (card, upi, netbanking,
wallet, emi): the thing a customer can pay through. Every verified
``payment.captured`` or ``payment.failed`` webhook records one observation.

:class:`GatewayHealthMonitor`
    Rolling-window success rate per method. Below ``GATEWAY_HEALTH_MIN_SAMPLE``
    observations the state is ``unknown``. Unknown is never treated as healthy
    *or* degraded, so a quiet route cannot trigger a switch.

    ``healthy``   success rate >= GATEWAY_DEGRADED_BELOW
    ``degraded``  success rate <  GATEWAY_DEGRADED_BELOW
    ``unknown``   fewer than GATEWAY_HEALTH_MIN_SAMPLE observations in the window

:class:`GatewayFailoverEngine`
    Chooses an alternative route only when the current route is ``degraded``
    and a candidate is ``healthy`` (not merely unknown). Candidates keep the
    playbook order, so the choice is deterministic.

Trust boundary: no LLM. Counts come from the database, and the decision is a
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy import func, select

import config
from database.engine import session_scope
from database.models import GatewayObservation

HEALTHY = "healthy"
DEGRADED = "degraded"
UNKNOWN = "unknown"


def _stamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="seconds")


@dataclass(frozen=True)
class HealthSnapshot:
    """The health of one payment method at one moment."""

    method: str
    samples: int
    successes: int
    success_rate: float | None
    state: str
    window_minutes: int

    def explain(self) -> str:
        if self.state == UNKNOWN:
            return (
                f"{self.method}: {self.samples} observation(s) in the last "
                f"{self.window_minutes} min, below the minimum sample; state unknown"
            )
        return (
            f"{self.method}: {self.successes}/{self.samples} captured in the last "
            f"{self.window_minutes} min ({self.success_rate:.0%}); state {self.state}"
        )


class GatewayHealthMonitor:
    """Records payment outcomes and reports per-method health over a rolling window."""

    def __init__(
        self,
        window_minutes: int | None = None,
        min_sample: int | None = None,
        degraded_below: float | None = None,
    ) -> None:
        self.window_minutes = int(window_minutes or config.GATEWAY_HEALTH_WINDOW_MINUTES)
        self.min_sample = int(min_sample or config.GATEWAY_HEALTH_MIN_SAMPLE)
        self.degraded_below = float(
            degraded_below if degraded_below is not None else config.GATEWAY_DEGRADED_BELOW
        )

    def record(
        self,
        method: str | None,
        success: bool,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Store one outcome. A missing method is recorded as ``unknown``."""
        when = _stamp(now or datetime.now())
        with session_scope() as session:
            session.add(
                GatewayObservation(
                    method=(method or "unknown"),
                    success=1 if success else 0,
                    recovery_id=recovery_id,
                    observed_at=when,
                )
            )

    def snapshot(self, method: str | None, now: datetime | None = None) -> HealthSnapshot:
        """Return the health of ``method`` over the window ending at ``now``."""
        method = method or "unknown"
        end = now or datetime.now()
        since = _stamp(end - timedelta(minutes=self.window_minutes))
        with session_scope() as session:
            samples, successes = session.execute(
                select(
                    func.count(GatewayObservation.id),
                    func.coalesce(func.sum(GatewayObservation.success), 0),
                ).where(
                    GatewayObservation.method == method,
                    GatewayObservation.observed_at >= since,
                    GatewayObservation.observed_at <= _stamp(end),
                )
            ).one()
        samples = int(samples or 0)
        successes = int(successes or 0)
        if samples < self.min_sample:
            return HealthSnapshot(method, samples, successes, None, UNKNOWN, self.window_minutes)
        rate = successes / samples
        state = HEALTHY if rate >= self.degraded_below else DEGRADED
        return HealthSnapshot(method, samples, successes, rate, state, self.window_minutes)


@dataclass(frozen=True)
class FailoverDecision:
    """Whether to move a payment to another route, and why."""

    switch: bool
    current: str
    to: str | None
    reason: str


def decide_failover(
    current: HealthSnapshot, candidates: Iterable[HealthSnapshot]
) -> FailoverDecision:
    """
    Pure failover rule over health snapshots. Never switches on unknown data.

    Switch only when ``current`` is degraded and a candidate is verified healthy.
    Candidates are tried in the order given.
    """
    if current.state != DEGRADED:
        return FailoverDecision(False, current.method, None, f"no failover: {current.explain()}")
    for health in candidates:
        if health.method == current.method:
            continue
        if health.state == HEALTHY:
            return FailoverDecision(
                True, current.method, health.method,
                f"{current.method} is degraded ({current.success_rate:.0%}); "
                f"{health.method} is healthy ({health.success_rate:.0%})",
            )
    return FailoverDecision(
        False, current.method, None,
        f"{current.method} is degraded but no candidate is verified healthy",
    )


class GatewayFailoverEngine:
    """Reads live health from the monitor, then applies :func:`decide_failover`."""

    def __init__(self, monitor: GatewayHealthMonitor | None = None) -> None:
        self.monitor = monitor or GatewayHealthMonitor()

    def choose(
        self,
        current: str | None,
        candidates: Iterable[str],
        now: datetime | None = None,
    ) -> FailoverDecision:
        """Return the failover decision for ``current`` over ``candidates``."""
        current_health = self.monitor.snapshot(current, now)
        candidate_health = [self.monitor.snapshot(c, now) for c in candidates]
        return decide_failover(current_health, candidate_health)
