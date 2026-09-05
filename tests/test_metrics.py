"""
Tests for the metrics module — the single source of truth for the dashboard.

Includes an explicit regression test for the attempt-multiplication bug: a
failure with 3 attempts was counted 3 times in the per-cause and per-day
breakdowns, making those tables disagree with the headline by ~2.1x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (  # noqa: E402
    autopsy_stats,
    cause_breakdown,
    compute_headline_metrics,
    diagnoser_metrics,
    recovery_over_time,
    stopping_rule_events,
)

TOL = 0.01  # rupees


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
def test_headline_has_the_required_fields():
    """Every field the dashboard renders is present and typed."""
    m = compute_headline_metrics()
    for key in (
        "n_transactions", "n_failures", "amount_at_risk_inr",
        "amount_recovered_inr", "recovery_rate", "stopping_rule_events",
        "recovery_actions", "audit_records", "reconciled",
    ):
        assert key in m, f"missing {key}"
    assert isinstance(m["n_transactions"], int)
    assert isinstance(m["amount_recovered_inr"], float)


def test_headline_states_its_basis():
    """NFR-004: no metric without its evaluation set and split method."""
    m = compute_headline_metrics()
    assert m["basis"] == "held-out batch"
    assert len(m["split_method"]) > 40
    assert "[" in m["split_method"] or "random" in m["split_method"]


def test_recovery_rate_is_consistent():
    """Recovery rate equals recovered / at risk."""
    m = compute_headline_metrics()
    if m["amount_at_risk_inr"]:
        # recovery_rate is stored rounded to 4dp, so compare at that precision.
        expected = m["amount_recovered_inr"] / m["amount_at_risk_inr"]
        assert abs(m["recovery_rate"] - expected) < 1e-4


def test_headline_reconciles_with_audit_trail():
    """The headline equals the sum over recovery actions (build prompt 13)."""
    m = compute_headline_metrics()
    assert m["reconciled"] is True


# ---------------------------------------------------------------------------
# Regression: attempt multiplication
# ---------------------------------------------------------------------------
def test_per_cause_totals_match_the_headline():
    """REGRESSION: per-cause sums must equal headline, not exceed it.

    A failure with N attempts used to be counted N times because `failures`
    was joined straight to `recoveries`.
    """
    m = compute_headline_metrics()
    causes = cause_breakdown()
    if not causes:
        pytest.skip("no recovery activity")

    assert (
        abs(sum(c["at_risk"] for c in causes) - m["amount_at_risk_inr"]) < TOL
    ), "per-cause at-risk does not match headline"
    assert (
        abs(sum(c["recovered"] for c in causes) - m["amount_recovered_inr"]) < TOL
    ), "per-cause recovered does not match headline"
    assert sum(c["failures"] for c in causes) == m["n_failures"]


def test_daily_totals_match_the_headline():
    """REGRESSION: the trend chart must not inflate at-risk either."""
    m = compute_headline_metrics()
    series = recovery_over_time()
    if not series:
        pytest.skip("no recovery activity")

    assert abs(sum(d["at_risk"] for d in series) - m["amount_at_risk_inr"]) < TOL
    assert abs(sum(d["recovered"] for d in series) - m["amount_recovered_inr"]) < TOL
    assert sum(d["failures"] for d in series) == m["n_failures"]


def test_cause_breakdown_covers_every_failure_exactly_once():
    """No failure is dropped or double-counted across buckets."""
    m = compute_headline_metrics()
    causes = cause_breakdown()
    if causes:
        assert sum(c["failures"] for c in causes) == m["n_failures"]


def test_recovery_rates_are_within_bounds():
    """No bucket reports a nonsensical rate."""
    for c in cause_breakdown():
        assert 0.0 <= c["recovery_rate"] <= 1.0
        assert c["recovered"] <= c["at_risk"] + TOL


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------
def test_stopping_events_carry_a_reason():
    """Every stop is explainable, not just counted."""
    events = stopping_rule_events(limit=20)
    if not events:
        pytest.skip("no stopping events in this batch")
    for e in events:
        assert e["stopping_reason"], "stop without a reason"
        assert "escalat" in e["stopping_reason"].lower()
        assert e["predicted_cause"]
        assert e["customer_id"]


def test_stopping_event_count_matches_headline():
    """The panel count and the headline count agree."""
    m = compute_headline_metrics()
    # stopping_rule_events() is capped; headline is total.
    if m["stopping_rule_events"] <= 200:
        assert len(stopping_rule_events(limit=200)) == m["stopping_rule_events"]


# ---------------------------------------------------------------------------
# Diagnoser metrics
# ---------------------------------------------------------------------------
def test_diagnoser_metrics_carry_n_and_split_method():
    """NFR-004 applies to the precision/recall panel too."""
    d = diagnoser_metrics()
    assert d, "no held-out evaluation found - run diagnoser/train.py"
    assert d["n"] > 0
    assert len(d["split_method"]) > 40
    assert d["evaluation_set"] == "held-out split only"


def test_diagnoser_per_class_supports_sum_to_n():
    """Per-class counts account for every evaluated example."""
    d = diagnoser_metrics()
    assert sum(v["support"] for v in d["per_class"].values()) == d["n"]


def test_diagnoser_metrics_are_in_bounds():
    """No impossible precision/recall values."""
    d = diagnoser_metrics()
    assert 0.0 <= d["accuracy"] <= 1.0
    for v in d["per_class"].values():
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= v[key] <= 1.0


def test_attribution_is_present_and_ordered():
    """The decomposition is stored: lookup floor, full model, ablated."""
    d = diagnoser_metrics()
    a = d.get("attribution")
    assert a and len(a) == 3
    lookup, full, ablated = a
    assert "lookup" in lookup["name"].lower() or "no ML" in lookup["name"]
    assert "WITHOUT" in ablated["name"]
    # The full model must beat the lookup floor, else the model is worthless.
    assert full["macro_f1"] > lookup["macro_f1"]
    # Ablation must cost something, else the error code is the only signal.
    assert ablated["macro_f1"] < full["macro_f1"]


# ---------------------------------------------------------------------------
# Autopsy stats
# ---------------------------------------------------------------------------
def test_autopsy_stats_report_counts():
    """Autopsy coverage is measurable from the metrics module."""
    s = autopsy_stats()
    assert "total" in s and s["total"] >= 0
    if s["total"]:
        assert s["models"]
        assert s["sample"] is None or "report_text" in s["sample"]
