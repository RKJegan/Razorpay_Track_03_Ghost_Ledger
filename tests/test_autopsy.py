"""
Tests for the autopsy reporter (FR-005).

The critical property: this component is **text-only and never in the money
path**. These tests assert that an unreachable or misbehaving LLM degrades to
a template instead of raising, and that hallucinated numbers are detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.autopsy_reporter import (  # noqa: E402
    AutopsyReport,
    build_facts,
    build_prompt,
    check_hallucinated_numbers,
    generate_autopsy,
    persist,
    SYSTEM_PROMPT,
    _template_report,
)
from database import db_client  # noqa: E402


def _ensure_failure(failure_id: str, transaction_id: str) -> None:
    """
    Create the parent transaction/failure rows an autopsy row references.

    ``autopsy_reports.failure_id`` has a foreign key to ``failures.id``.

    Parameters
    ----------
    failure_id : str
        Failure id to create.
    transaction_id : str
        Originating transaction id.
    """
    from datetime import datetime

    db_client.execute(
        "INSERT OR REPLACE INTO transactions "
        "(id, merchant_id, customer_id, amount, status, txn_type, "
        " payment_method, failure_reason_raw, timestamp, is_holdout) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (transaction_id, "merchant_demo_001", "CUST_AUT", 1499.0, "failed",
         "one_off", "card", "ERR_CARD_EXPIRED: test",
         datetime.now().isoformat(sep=" ", timespec="seconds"), 1),
    )
    db_client.execute(
        "INSERT OR REPLACE INTO failures "
        "(id, transaction_id, predicted_cause, confidence, ground_truth_cause, "
        " detected_at, estimated_value) VALUES (?,?,?,?,?,?,?)",
        (failure_id, transaction_id, "card_expired", 0.91, "card_expired",
         datetime.now().isoformat(sep=" ", timespec="seconds"), 1499.0),
    )


def _drop_failure(failure_id: str) -> None:
    """Remove a failure and its children."""
    db_client.execute("DELETE FROM autopsy_reports WHERE failure_id = ?", (failure_id,))
    db_client.execute("DELETE FROM recoveries WHERE failure_id = ?", (failure_id,))
    db_client.execute("DELETE FROM failures WHERE id = ?", (failure_id,))


@pytest.fixture
def facts():
    """A representative fact block."""
    return build_facts(
        failure_id="fail_test_001",
        transaction_id="txn_test_001",
        cause="card_expired",
        confidence=0.9132,
        amount_inr=1499.0,
        txn_type="one_off",
        payment_method="card",
        gateway="hdfc",
        error_code="ERR_CARD_EXPIRED",
        timestamp="2026-08-14 09:31:00",
        attempt_count=2,
        recovered_amount=0.0,
    )


# ---------------------------------------------------------------------------
# Fact construction
# ---------------------------------------------------------------------------
def test_facts_contain_all_grounding_values(facts):
    """Everything the text may reference is in the fact block."""
    for key in (
        "amount_inr", "root_cause", "confidence_pct", "payment_method",
        "gateway", "error_code", "failure_timestamp",
    ):
        assert key in facts


def test_facts_round_amounts(facts):
    """Amounts are rounded so text matching is stable."""
    assert facts["amount_inr"] == 1499.0
    assert facts["confidence_pct"] == 91.3


def test_prompt_injects_the_data_not_prose(facts):
    """The prompt is structured JSON, not a free-text description."""
    prompt = build_prompt(facts)
    assert "FACTS" in prompt
    assert "1499.0" in prompt
    assert "ERR_CARD_EXPIRED" in prompt
    assert SYSTEM_PROMPT  # system prompt exists separately


# ---------------------------------------------------------------------------
# Template backend
# ---------------------------------------------------------------------------
def test_template_report_mentions_the_real_values(facts):
    """The deterministic fallback uses the supplied numbers."""
    text = _template_report(facts)
    assert "1,499.00" in text
    assert "ERR_CARD_EXPIRED" in text
    assert "card expired" in text.lower()


def test_template_report_is_two_to_three_sentences(facts):
    """Output respects the 2-3 sentence contract."""
    text = _template_report(facts)
    assert 2 <= text.count(".") <= 6


def test_template_backend_never_degrades(facts):
    """The deterministic path reports itself as healthy."""
    report = generate_autopsy("fail_t", facts, backend="template")
    assert report.degraded is False
    assert report.model == "template-v1"
    assert report.hallucination_flags == []


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------
def test_clean_text_produces_no_flags(facts):
    """Text using only supplied numbers is not flagged."""
    text = "A payment of 1,499.00 failed with ERR_CARD_EXPIRED at 2026-08-14."
    assert check_hallucinated_numbers(text, facts) == []


def test_invented_numbers_are_flagged(facts):
    """A number absent from the facts is flagged as invented."""
    text = "A payment of 8,742.50 failed 17 times over 93 days."
    flags = check_hallucinated_numbers(text, facts)
    assert flags
    assert "8,742.50" in flags


def test_generated_template_output_has_no_flags(facts):
    """End to end, the template output passes its own guard."""
    report = generate_autopsy("fail_t2", facts, backend="template")
    assert report.hallucination_flags == []


# ---------------------------------------------------------------------------
# Degradation — the LLM must never break the pipeline
# ---------------------------------------------------------------------------
def test_unreachable_llm_falls_back_to_template(facts, monkeypatch):
    """An Ollama outage degrades to the template instead of raising."""

    def boom(_prompt):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("agents.autopsy_reporter._call_ollama", boom)
    report = generate_autopsy("fail_t3", facts, backend="ollama")
    assert report.degraded is True
    assert "template" in report.model
    assert report.text


def test_empty_llm_completion_falls_back(facts, monkeypatch):
    """An empty completion is treated as a failure, not accepted."""

    def empty(_prompt):
        return "", "ollama:fake"

    monkeypatch.setattr("agents.autopsy_reporter._call_ollama", empty)
    report = generate_autopsy("fail_t4", facts, backend="ollama")
    assert report.degraded is True
    assert report.text.strip()


def test_openai_without_key_falls_back(facts, monkeypatch):
    """A misconfigured OpenAI backend degrades rather than raising."""
    monkeypatch.setattr("agents.autopsy_reporter.OPENAI_API_KEY", "")
    report = generate_autopsy("fail_t5", facts, backend="openai")
    assert report.degraded is True
    assert report.text


def test_successful_llm_output_is_used(facts, monkeypatch):
    """When the model responds, its text is returned verbatim."""
    monkeypatch.setattr(
        "agents.autopsy_reporter._call_ollama",
        lambda _p: ("The card on file expired. The customer must update it.", "ollama:test"),
    )
    report = generate_autopsy("fail_t6", facts, backend="ollama")
    assert report.degraded is False
    assert report.model == "ollama:test"
    assert "expired" in report.text


def test_hallucinating_llm_is_flagged_not_silently_accepted(facts, monkeypatch):
    """A model that invents numbers is flagged, and the text is still stored."""
    monkeypatch.setattr(
        "agents.autopsy_reporter._call_ollama",
        lambda _p: ("The payment of 55,321.99 failed after 42 retries.", "ollama:test"),
    )
    report = generate_autopsy("fail_t7", facts, backend="ollama")
    assert report.hallucination_flags, "invented numbers must be flagged"
    assert "55,321.99" in report.text  # stored, not hidden


# ---------------------------------------------------------------------------
# Provenance & persistence
# ---------------------------------------------------------------------------
def test_report_carries_provenance(facts):
    """Every report records model and basis."""
    report = generate_autopsy("fail_t8", facts, backend="template")
    assert isinstance(report, AutopsyReport)
    assert report.basis
    assert report.failure_id == "fail_t8"
    assert report.latency_ms >= 0


def test_persist_and_retrieve(facts):
    """Reports round-trip through the database."""
    _ensure_failure("fail_persist_001", "txn_persist_001")
    report = generate_autopsy("fail_persist_001", facts, backend="template")
    persist(report)
    row = db_client.query_one(
        "SELECT * FROM autopsy_reports WHERE failure_id = ?", ("fail_persist_001",)
    )
    assert row is not None
    assert row["report_text"] == report.text
    assert row["model"] == "template-v1"
    assert row["basis"]
    _drop_failure("fail_persist_001")


def test_persist_is_idempotent(facts):
    """Re-generating for the same failure replaces, not duplicates."""
    _ensure_failure("fail_persist_002", "txn_persist_002")
    for _ in range(3):
        persist(generate_autopsy("fail_persist_002", facts, backend="template"))
    n = db_client.scalar(
        "SELECT COUNT(*) FROM autopsy_reports WHERE failure_id = ?", ("fail_persist_002",)
    )
    assert n == 1
    _drop_failure("fail_persist_002")
