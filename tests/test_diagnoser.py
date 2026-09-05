"""
Tests for the root-cause diagnoser (FR-002).

Covers output shape, the NFR-004 provenance requirement (n + split_method
must travel with every metric), explainability, and the leakage guarantee
that no customer straddles the split boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CAUSE_BUCKETS  # noqa: E402
from diagnoser.eval_holdout import (  # noqa: E402
    compute_metrics,
    evaluate_holdout,
    load_split_method,
)
from diagnoser.root_cause_classifier import (  # noqa: E402
    build_feature_matrix,
    load_failure_dataset,
    RootCauseClassifier,
)

# Fit once for the whole module: training on the full train split takes ~2s.
_FIXTURE = {}


def _get_fitted():
    """Return (clf, X_train, y_train, X_hold, y_hold, df), fitted once."""
    if not _FIXTURE:
        df = load_failure_dataset()
        X_all, _ = build_feature_matrix(df)
        train = df[df["is_holdout"] == 0]
        hold = df[df["is_holdout"] == 1]
        clf = RootCauseClassifier().fit(
            X_all.loc[train.index], train["cause"], params={"n_estimators": 60, "max_depth": 4}
        )
        _FIXTURE.update(
            clf=clf, X_all=X_all, df=df, train=train, hold=hold
        )
    return (
        _FIXTURE["clf"], _FIXTURE["X_all"], _FIXTURE["train"], _FIXTURE["hold"], _FIXTURE["df"]
    )


# ---------------------------------------------------------------------------
# Dataset / split
# ---------------------------------------------------------------------------
def test_dataset_loads_with_expected_columns():
    """Every failure row carries a cause, a customer id and a split flag."""
    df = load_failure_dataset()
    assert not df.empty
    for col in ("cause", "customer_id", "is_holdout", "error_code", "txn_type"):
        assert col in df.columns, f"missing column {col}"


def test_labels_are_valid_cause_buckets():
    """Ground-truth labels are drawn from the declared taxonomy."""
    df = load_failure_dataset()
    assert set(df["cause"].dropna().unique()) <= set(CAUSE_BUCKETS)


def test_no_customer_straddles_the_split():
    """Under the grouped strategy no customer appears on both sides."""
    df = load_failure_dataset()
    strategy = load_split_method()
    if "random_rows" in strategy:
        pytest.skip("row-level split intentionally allows customer overlap")
    tr = set(df.loc[df["is_holdout"] == 0, "customer_id"])
    ho = set(df.loc[df["is_holdout"] == 1, "customer_id"])
    assert tr & ho == set(), "customer appears in both splits"


def test_split_method_is_non_empty_and_descriptive():
    """The split basis must exist and be long enough to be meaningful."""
    method = load_split_method()
    assert isinstance(method, str)
    assert len(method) > 40, "split description too short to be useful"


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def test_feature_matrix_is_numeric_and_aligned():
    """Design matrix is all-numeric, NaN-preserving, and row-aligned."""
    df = load_failure_dataset()
    X, names = build_feature_matrix(df)
    assert len(X) == len(df)
    assert list(X.index) == list(df.index)
    assert all(np.issubdtype(X[c].dtype, np.number) for c in X.columns)
    assert names == list(X.columns)


def test_customer_id_is_never_a_feature():
    """customer_id is a CV grouping key; it must not leak into the design matrix."""
    df = load_failure_dataset()
    X, _ = build_feature_matrix(df)
    assert "customer_id" not in X.columns
    assert not any("customer_id" in c for c in X.columns)


# ---------------------------------------------------------------------------
# Model output shape
# ---------------------------------------------------------------------------
def test_predict_returns_valid_labels():
    """Every prediction is one of the four declared cause buckets."""
    clf, X_all, train, hold, _ = _get_fitted()
    pred = clf.predict(X_all.loc[hold.index])
    assert len(pred) == len(hold)
    assert set(pred.unique()) <= set(CAUSE_BUCKETS)


def test_confidence_is_a_valid_probability():
    """Confidence lies in [0, 1] and equals the winning class probability."""
    clf, X_all, train, hold, _ = _get_fitted()
    pred, conf = clf.predict_with_confidence(X_all.loc[hold.index])
    assert conf.between(0.0, 1.0).all()
    proba = clf.predict_proba(X_all.loc[hold.index])
    assert np.allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(proba.to_numpy().max(axis=1), conf.to_numpy(), atol=1e-6)


def test_model_beats_majority_baseline():
    """Sanity: the fitted model must clearly beat predicting one class."""
    clf, X_all, train, hold, _ = _get_fitted()
    pred, _ = clf.predict_with_confidence(X_all.loc[hold.index])
    acc = (pred.to_numpy() == hold["cause"].to_numpy()).mean()
    majority = hold["cause"].value_counts(normalize=True).max()
    assert acc > majority + 0.10, f"accuracy {acc:.3f} vs majority {majority:.3f}"


# ---------------------------------------------------------------------------
# Explainability (NFR-003)
# ---------------------------------------------------------------------------
def test_explain_returns_top_k_features_per_row():
    """One attribution list per row, each of length top_k, names resolvable."""
    clf, X_all, train, hold, _ = _get_fitted()
    sample = X_all.loc[hold.index].head(10)
    expl = clf.explain(sample, top_k=3)
    assert len(expl) == len(sample)
    for row in expl:
        assert len(row) == 3
        for name, value in row:
            assert name in clf.feature_names
            assert np.isfinite(value)


def test_explain_is_row_specific_not_constant():
    """Attributions must vary by row; a constant 'reason' would be useless."""
    clf, X_all, train, hold, _ = _get_fitted()
    sample = X_all.loc[hold.index].head(20)
    expl = clf.explain(sample, top_k=1)
    top_features = {row[0][0] for row in expl}
    assert len(top_features) > 1, "every row got the same top feature"


# ---------------------------------------------------------------------------
# Metrics provenance (NFR-004)
# ---------------------------------------------------------------------------
def test_metrics_carry_n_and_split_method():
    """A metric dict without n and split_method is a bug, not a metric."""
    y_true = ["card_expired", "gateway_timeout", "insufficient_funds", "mandate_lapsed"]
    y_pred = ["card_expired", "gateway_timeout", "card_expired", "mandate_lapsed"]
    report = evaluate_holdout(y_true, y_pred)
    assert report["n"] == len(y_true)
    assert "split_method" in report and len(report["split_method"]) > 40
    assert report["evaluation_set"] == "held-out split only"


def test_per_class_metrics_sum_supports_to_n():
    """Per-class supports must account for every evaluated example."""
    clf, X_all, train, hold, _ = _get_fitted()
    pred, _ = clf.predict_with_confidence(X_all.loc[hold.index])
    m = compute_metrics(hold["cause"].tolist(), pred.tolist())
    total = sum(v["support"] for v in m["per_class"].values())
    assert total == m["n"] == len(hold)


def test_evaluation_uses_holdout_only():
    """The evaluated set is the held-out split, not the training split."""
    clf, X_all, train, hold, _ = _get_fitted()
    pred, _ = clf.predict_with_confidence(X_all.loc[hold.index])
    report = evaluate_holdout(hold["cause"].tolist(), pred.tolist())
    assert report["n"] == len(hold)
    assert report["n"] != len(train)
