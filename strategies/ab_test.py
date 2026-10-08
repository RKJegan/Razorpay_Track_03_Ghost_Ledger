"""
Ghost Ledger v3 — A/B tests for strategies (B6).

Assignment
----------
``variant = bucket(experiment, unit) < split`` where ``bucket`` is a SHA-256 hash
of ``"<experiment>:<unit>"`` mapped to 0-9999 (two decimals of a percent). The
same unit always gets the same bucket, on any machine, in any process. It does
not depend on Python's randomised ``hash()``.

The first assignment is also stored in ``ab_assignments`` and wins. If the
split is later changed, existing units keep their variant, so an experiment
never switches a customer's arm mid-flight.

Evaluation
----------
A two-sided two-proportion z-test on the first recorded outcome per unit:

    pooled p = (x1 + x2) / (n1 + n2)
    z = (p2 - p1) / sqrt(p (1 - p) (1/n1 + 1/n2))
    p-value = erfc(|z| / sqrt(2))

The verdict is ``insufficient_sample`` unless both arms have at least
``AB_MIN_SAMPLE_PER_ARM`` outcomes. Then it is ``significant`` at ``AB_ALPHA``
or ``no_significant_difference``. Peeking early and stopping on a
first-significant result is not supported.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select

import config
from database.engine import session_scope
from database.models import AbAssignment

CONTROL = "control"
TREATMENT = "treatment"
VERDICT_INSUFFICIENT = "insufficient_sample"
VERDICT_SIGNIFICANT = "significant"
VERDICT_NO_DIFFERENCE = "no_significant_difference"


def bucket_of(experiment: str, unit_id: str) -> int:
    """Stable bucket in 0..9999 for a (experiment, unit) pair."""
    digest = hashlib.sha256(f"{experiment}:{unit_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def variant_for_bucket(bucket: int, split_percent: float) -> str:
    """Control gets the first ``split_percent`` of buckets, treatment the rest."""
    return CONTROL if bucket < round(float(split_percent) * 100) else TREATMENT


@dataclass(frozen=True)
class ZTestResult:
    """Output of :func:`two_proportion_z`."""

    x1: int
    n1: int
    x2: int
    n2: int
    p1: float
    p2: float
    z: float
    p_value: float


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> ZTestResult:
    """Two-sided two-proportion z-test. Returns z=0, p=1 when the pooled rate is 0 or 1."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both arms need at least one observation")
    if not (0 <= x1 <= n1 and 0 <= x2 <= n2):
        raise ValueError("successes must be between 0 and n")
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if denom == 0.0:
        return ZTestResult(x1, n1, x2, n2, p1, p2, 0.0, 1.0)
    z = (p2 - p1) / denom
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return ZTestResult(x1, n1, x2, n2, p1, p2, z, p_value)


@dataclass(frozen=True)
class ExperimentReport:
    """Result of evaluating one experiment. z and p_value are None until both arms have data."""

    experiment: str
    x_control: int
    n_control: int
    x_treatment: int
    n_treatment: int
    z: float | None
    p_value: float | None
    verdict: str
    winner: str | None
    alpha: float
    min_sample_per_arm: int

    @property
    def rate_control(self) -> float | None:
        return self.x_control / self.n_control if self.n_control else None

    @property
    def rate_treatment(self) -> float | None:
        return self.x_treatment / self.n_treatment if self.n_treatment else None

    def explain(self) -> str:
        head = (
            f"{self.experiment}: control {self.x_control}/{self.n_control}, "
            f"treatment {self.x_treatment}/{self.n_treatment}"
        )
        if self.verdict == VERDICT_INSUFFICIENT:
            return f"{head}; need {self.min_sample_per_arm} per arm before a verdict"
        winner = f" (winner: {self.winner})" if self.winner else ""
        return f"{head}; z={self.z:.3f}, p={self.p_value:.4f}; {self.verdict}{winner}"


class ABTestManager:
    """Assigns units to variants and evaluates outcomes."""

    def __init__(self, alpha: float | None = None, min_sample_per_arm: int | None = None) -> None:
        self.alpha = float(alpha if alpha is not None else config.AB_ALPHA)
        self.min_sample_per_arm = int(
            min_sample_per_arm if min_sample_per_arm is not None else config.AB_MIN_SAMPLE_PER_ARM
        )

    def assign(self, experiment: str, unit_id: str, split_percent: float = 50.0,
               now: datetime | None = None) -> str:
        """Return the stored variant for this unit, creating it on first call."""
        candidate = variant_for_bucket(bucket_of(experiment, unit_id), split_percent)
        stamp = (now or datetime.now()).isoformat(sep=" ", timespec="seconds")
        with session_scope() as session:
            row = session.get(AbAssignment, (experiment, unit_id))
            if row is None:
                session.add(AbAssignment(
                    experiment=experiment, unit_id=unit_id, variant=candidate, assigned_at=stamp,
                ))
                return candidate
            return row.variant

    def record_outcome(self, experiment: str, unit_id: str, converted: bool,
                       now: datetime | None = None) -> bool:
        """Record the first outcome for a unit. Returns False if it was already recorded."""
        stamp = (now or datetime.now()).isoformat(sep=" ", timespec="seconds")
        with session_scope() as session:
            row = session.get(AbAssignment, (experiment, unit_id))
            if row is None:
                raise KeyError(f"unit {unit_id!r} is not assigned in experiment {experiment!r}")
            if row.outcome is not None:
                return False
            row.outcome = 1 if converted else 0
            row.outcome_at = stamp
            return True

    def counts(self, experiment: str) -> dict[str, tuple[int, int]]:
        """Return {variant: (conversions, observations)} for recorded outcomes."""
        with session_scope() as session:
            rows = session.execute(
                select(
                    AbAssignment.variant,
                    func.coalesce(func.sum(AbAssignment.outcome), 0),
                    func.count(AbAssignment.outcome),
                )
                .where(AbAssignment.experiment == experiment)
                .group_by(AbAssignment.variant)
            ).all()
        return {variant: (int(conv), int(n)) for variant, conv, n in rows}

    def evaluate(self, experiment: str) -> ExperimentReport:
        """Evaluate control against treatment on recorded outcomes."""
        counts = self.counts(experiment)
        cx, cn = counts.get(CONTROL, (0, 0))
        tx, tn = counts.get(TREATMENT, (0, 0))

        def report(verdict: str, z: float | None = None, p: float | None = None,
                   winner: str | None = None) -> ExperimentReport:
            return ExperimentReport(
                experiment, cx, cn, tx, tn, z, p, verdict, winner,
                self.alpha, self.min_sample_per_arm,
            )

        if cn == 0 or tn == 0 or min(cn, tn) < self.min_sample_per_arm:
            return report(VERDICT_INSUFFICIENT)
        result = two_proportion_z(cx, cn, tx, tn)
        if result.p_value < self.alpha:
            winner = TREATMENT if result.p2 > result.p1 else CONTROL
            return report(VERDICT_SIGNIFICANT, result.z, result.p_value, winner)
        return report(VERDICT_NO_DIFFERENCE, result.z, result.p_value)
