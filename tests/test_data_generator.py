"""
Tests for the synthetic data generator (FR-001).

Focus: reproducibility, split integrity, and the leakage guarantees that every
downstream metric depends on. If these break, no precision/recall figure in
the project can be trusted.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CAUSE_BUCKETS  # noqa: E402
from data.synthetic_generator import (  # noqa: E402
    SyntheticDataGenerator,
    verify,
)


@pytest.fixture(scope="module")
def small_dataset():
    """Generate a small dataset once for the module (fast)."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = SyntheticDataGenerator(
            seed=42, days=30, n_customers=200, avg_daily_txns=60, profile="demo"
        )
        gen.run(outdir=Path(tmp))
        yield gen, Path(tmp)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_produces_identical_output():
    """Seed is the only source of randomness: same seed -> same data."""
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        g1 = SyntheticDataGenerator(seed=42, days=20, n_customers=100, avg_daily_txns=40)
        g1.run(outdir=Path(t1))
        g2 = SyntheticDataGenerator(seed=42, days=20, n_customers=100, avg_daily_txns=40)
        g2.run(outdir=Path(t2))

        a = json.loads((Path(t1) / "failure_contexts.json").read_text())
        b = json.loads((Path(t2) / "failure_contexts.json").read_text())
        assert a == b
        assert g1._train_labels == g2._train_labels
        assert g1._holdout_labels == g2._holdout_labels


def test_different_seed_produces_different_output():
    """A different seed must actually change the data."""
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        g1 = SyntheticDataGenerator(seed=42, days=20, n_customers=100, avg_daily_txns=40)
        g1.run(outdir=Path(t1))
        g2 = SyntheticDataGenerator(seed=43, days=20, n_customers=100, avg_daily_txns=40)
        g2.run(outdir=Path(t2))

        a = json.loads((Path(t1) / "failure_contexts.json").read_text())
        b = json.loads((Path(t2) / "failure_contexts.json").read_text())
        assert a != b


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_output_files_are_all_written(small_dataset):
    """Every documented artefact is produced."""
    _, tmp = small_dataset
    for name in (
        "failure_contexts.json",
        "holdout_set.json",
        "demo_window.json",
        "subscriptions.json",
        "train_labels.json",
        "ground_truth_holdout.json",
        "generation_manifest.json",
    ):
        assert (tmp / name).exists(), f"{name} missing"


def test_transactions_have_spec_schema_columns(small_dataset):
    """Each record carries the ten columns from spec §11."""
    gen, _ = small_dataset
    required = {
        "id", "merchant_id", "customer_id", "amount", "status",
        "txn_type", "payment_method", "failure_reason_raw", "timestamp", "is_holdout",
    }
    for t in gen.transactions[:50]:
        assert required <= set(t.keys())


def test_manifest_records_seed_and_split_method(small_dataset):
    """Provenance travels with the data, not with a module default."""
    gen, _ = small_dataset
    m = gen.manifest
    assert m["seed"] == 42
    assert "split" in m and len(m["split"]["method"]) > 40
    assert m["split"]["strategy"] == "random_grouped"
    assert m["counts"]["transactions"] > 0
    assert m["counts"]["failed_holdout"] > 0


def test_manifest_states_assumptions(small_dataset):
    """Modelling assumptions are recorded, not buried."""
    gen, _ = small_dataset
    assert len(gen.manifest["assumptions"]) >= 3


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------
def test_grouped_split_has_no_customer_overlap(small_dataset):
    """No customer appears on both sides of the split."""
    gen, _ = small_dataset
    tr = {t["customer_id"] for t in gen.transactions if t["is_holdout"] == 0}
    ho = {t["customer_id"] for t in gen.transactions if t["is_holdout"] == 1}
    assert tr & ho == set()


def test_labels_are_partitioned_by_split(small_dataset):
    """Every failure is labelled exactly once, in exactly one split."""
    gen, _ = small_dataset
    failed_ids = {t["id"] for t in gen.transactions if t["status"] == "failed"}
    assert set(gen._train_labels) | set(gen._holdout_labels) == failed_ids
    assert set(gen._train_labels) & set(gen._holdout_labels) == set()


def test_holdout_fraction_is_approximately_target(small_dataset):
    """The holdout is ~20% of transactions, not a rounding accident."""
    gen, _ = small_dataset
    n_hold = sum(t["is_holdout"] for t in gen.transactions)
    frac = n_hold / len(gen.transactions)
    assert 0.15 < frac < 0.25


def test_feature_context_contains_no_label(small_dataset):
    """Ground truth must never be present in the feature block."""
    gen, _ = small_dataset
    for t in gen.transactions[:200]:
        assert "cause" not in t["context"]
        for v in t["context"].values():
            assert v not in CAUSE_BUCKETS


def test_all_cause_buckets_are_represented(small_dataset):
    """Every bucket appears, so per-class metrics are definable."""
    gen, _ = small_dataset
    labels = {**gen._train_labels, **gen._holdout_labels}
    assert set(labels.values()) == set(CAUSE_BUCKETS)


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["random_grouped", "random_rows", "temporal"])
def test_all_split_strategies_produce_valid_partitions(strategy):
    """Each strategy yields two non-empty, disjoint, fully-covering splits."""
    gen = SyntheticDataGenerator(
        seed=7, days=30, n_customers=150, avg_daily_txns=50, split_strategy=strategy
    )
    gen.generate()
    hold = [t for t in gen.transactions if t["is_holdout"] == 1]
    train = [t for t in gen.transactions if t["is_holdout"] == 0]
    assert len(hold) > 0 and len(train) > 0
    assert len(hold) + len(train) == len(gen.transactions)
    assert strategy in gen.split_method


def test_temporal_split_is_time_ordered():
    """Under `temporal`, every holdout row is at least as recent as every train row."""
    gen = SyntheticDataGenerator(
        seed=7, days=30, n_customers=150, avg_daily_txns=50, split_strategy="temporal"
    )
    gen.generate()
    tr_max = max(t["timestamp"] for t in gen.transactions if t["is_holdout"] == 0)
    ho_min = min(t["timestamp"] for t in gen.transactions if t["is_holdout"] == 1)
    assert tr_max <= ho_min


def test_random_split_is_not_time_ordered():
    """Under random strategies the holdout spans the whole window."""
    gen = SyntheticDataGenerator(
        seed=7, days=30, n_customers=150, avg_daily_txns=50, split_strategy="random_grouped"
    )
    gen.generate()
    tr_max = max(t["timestamp"] for t in gen.transactions if t["is_holdout"] == 0)
    ho_min = min(t["timestamp"] for t in gen.transactions if t["is_holdout"] == 1)
    assert ho_min < tr_max


def test_split_method_text_describes_the_actual_run():
    """The reported split method matches the run, not a module default."""
    gen = SyntheticDataGenerator(
        seed=7, days=60, n_customers=100, avg_daily_txns=40,
        split_strategy="random_grouped", holdout_fraction=0.3,
    )
    gen.generate()
    assert "60" in gen.split_method or "0.3" in gen.split_method or "30" in gen.split_method
    assert gen.manifest["split"]["method"] == gen.split_method


# ---------------------------------------------------------------------------
# Built-in verifier
# ---------------------------------------------------------------------------
def test_builtin_verify_passes(small_dataset):
    """The generator's own leakage checks pass on a fresh dataset."""
    gen, _ = small_dataset
    lines = verify(gen.transactions, gen._train_labels, gen._holdout_labels, gen.split_strategy)
    assert lines
    assert all("PASS" in line for line in lines), [l for l in lines if "FAIL" in l]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_tiny_dataset_does_not_crash():
    """A degenerate dataset still produces a valid split."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = SyntheticDataGenerator(seed=1, days=5, n_customers=10, avg_daily_txns=5)
        gen.run(outdir=Path(tmp))
        assert len(gen.transactions) > 0
        m = gen.manifest
        assert m["counts"]["transactions"] == len(gen.transactions)


def test_zero_failures_is_handled():
    """If nothing fails, label sets are empty and nothing raises."""
    gen = SyntheticDataGenerator(seed=2, days=5, n_customers=10, avg_daily_txns=5)
    gen.generate()
    for t in gen.transactions:
        t["status"] = "success"
    gen._cause_by_txn = {}
    gen._train_labels = {}
    gen._holdout_labels = {}
    lines = verify(gen.transactions, {}, {}, gen.split_strategy)
    # Coverage check legitimately fails when labels were stripped; the point is
    # that verification runs and reports rather than throwing.
    assert any("labels cover all failures" in line for line in lines)


def test_error_codes_overlap_across_causes():
    """The task is genuinely ambiguous: shared codes exist in the data."""
    gen = SyntheticDataGenerator(seed=42, days=30, n_customers=300, avg_daily_txns=80)
    gen.generate()
    code_to_causes: dict[str, set[str]] = {}
    for tid, cause in gen._cause_by_txn.items():
        code = next(
            (t["context"]["error_code"] for t in gen.transactions if t["id"] == tid), None
        )
        if code:
            code_to_causes.setdefault(code, set()).add(cause)
    ambiguous = {c: v for c, v in code_to_causes.items() if len(v) > 1}
    assert ambiguous, "no shared error codes - the task would be trivially solvable"
