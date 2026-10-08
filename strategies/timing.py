"""
Ghost Ledger v3 — rule-based retry timing (B3).

A timing rule answers one question: *given the time now, should the retry run
now, or later?* It is a pure function. The same input always gives the same
answer, and nothing is random or learned.

Rules
-----
``none``
    Retry immediately (the v2 behaviour).
``cycle_aware_funds``
    For ``insufficient_funds``. The generator models an end-of-cycle cash crunch
    from day 26 onward (``data/synthetic_generator.py``, insufficient_funds
    weight). A retry in that window is deferred to the 2nd of the next month at
    10:00 local time.
``peak_avoidance``
    For ``gateway_timeout``. The generator's peak hours are 11:00-14:59 and
    19:00-22:59. A retry inside a peak window is pushed forward in 30-minute
    steps until it is outside one (at most 12 steps, so at most 6 hours).

Honesty note: these rules encode the generator's assumptions, not measured
real-world recovery data. The holdout comparison script
(``scripts/retry_timing_holdout.py``) states this and labels its outcomes as
simulated.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

#: Day of month from which the insufficient-funds cash crunch applies.
CRUNCH_START_DAY: int = 26
#: Day of the next month to retry on, after a crunch deferral.
POST_CRUNCH_RETRY_DAY: int = 2
#: Local hour at which a deferred funds retry runs.
POST_CRUNCH_RETRY_HOUR: int = 10
#: Peak windows, in hours (inclusive), matching the generator.
PEAK_WINDOWS: tuple[tuple[int, int], ...] = ((11, 14), (19, 22))
PEAK_STEP_MINUTES: int = 30
PEAK_MAX_STEPS: int = 12


def is_peak_hour(hour: int) -> bool:
    """True when ``hour`` falls in a generator peak window."""
    return any(low <= hour <= high for low, high in PEAK_WINDOWS)


def _next_month_day(now: datetime, day: int, hour: int) -> datetime:
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, day, hour, 0, 0)


def cycle_aware_funds(now: datetime) -> datetime | None:
    """Defer funds retries inside the end-of-cycle window. None means run now."""
    if now.day >= CRUNCH_START_DAY:
        return _next_month_day(now, POST_CRUNCH_RETRY_DAY, POST_CRUNCH_RETRY_HOUR)
    return None


def peak_avoidance(now: datetime) -> datetime | None:
    """Step a gateway retry forward out of peak hours. None means run now."""
    t = now
    for _ in range(PEAK_MAX_STEPS):
        if not is_peak_hour(t.hour):
            break
        t = t + timedelta(minutes=PEAK_STEP_MINUTES)
    if t == now:
        return None
    return t


def no_timing(now: datetime) -> datetime | None:  # noqa: ARG001 - signature is shared
    """Retry immediately."""
    return None


#: Registry used by the playbook loader (names must match playbook YAML).
TIMING_RULES: dict[str, Callable[[datetime], datetime | None]] = {
    "none": no_timing,
    "cycle_aware_funds": cycle_aware_funds,
    "peak_avoidance": peak_avoidance,
}


def apply_rule(rule: str, now: datetime) -> datetime | None:
    """
    Run a named rule. Returns the time to run at, or None to run now.

    Raises
    ------
    KeyError
        If the rule name is not registered.
    """
    return TIMING_RULES[rule](now)
