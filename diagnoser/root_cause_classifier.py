"""
FR-002 — Root Cause Diagnoser
=============================

Classifies each failed transaction into one of four cause buckets
(``card_expired``, ``insufficient_funds``, ``gateway_timeout``,
``mandate_lapsed``) and returns a confidence score plus a human-readable
reason for the prediction.

Where AI is and is not used
---------------------------
* **Used here:** XGBoost for classification, and tree-SHAP for the
  human-readable reason required by NFR-003.
* **Not used here:** anything that moves money. This module reads context and
  emits a label. It never decides an amount, never triggers a retry, never
  decides to stop. Those are the policy engine's job.

Honesty constraints honoured
----------------------------
* Hyperparameters are selected by **cross-validation on the training split
  only**. The held-out batch is never touched during tuning.
* CV folds are **grouped by customer**, matching the outer split's guarantee.
  A customer's rows never straddle a fold boundary, so CV scores are not
  inflated by the same customer appearing on both sides of a fold.
* No metric is emitted bare. Every figure returned by
  ``diagnoser.eval_holdout`` carries ``n`` and ``split_method`` (NFR-004).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from config import (
    CAUSE_BUCKETS,
    CV_FOLDS,
    CV_RANDOM_STATE,
    MODELS_DIR,
    SAMPLE_OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# Numeric features are passed through as-is; NaN is meaningful and is handled
# natively by XGBoost (a null days_to_card_expiry means "not a card payment",
# which is information, not a gap to impute).
NUMERIC_FEATURES: tuple[str, ...] = (
    "amount",
    "amount_vs_customer_median",
    "customer_median_amount",
    "customer_tenure_days",
    "customer_prior_txn_count",
    "customer_prior_failure_count",
    "customer_prior_success_rate",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_peak_hour",
    "gateway_latency_ms",
    "days_to_card_expiry",
    "mandate_age_days",
    "mandate_validity_days",
    "mandate_days_overdue",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "payment_method",
    "gateway",
    "txn_type",
    "error_code",
)

# customer_id is a GROUPING KEY for cross-validation, never a model feature.
# Including it would let the model memorise customers instead of causes.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("customer_id", "is_holdout", "cause")

SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0.0, 0.1, 0.5],
}

N_SEARCH_ITER: int = 40


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_failure_dataset(
    sample_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Assemble the modelling table: features + ground-truth cause + split flag.

    Feature vectors come from ``failure_contexts.json``; labels from the two
    per-split label files; ``customer_id`` (the CV grouping key) from SQLite.

    Parameters
    ----------
    sample_dir : Path, optional
        Directory holding the generator artefacts. Defaults to
        ``config.SAMPLE_OUTPUT_DIR``.

    Returns
    -------
    pd.DataFrame
        One row per failed transaction, indexed by transaction id, with
        columns: the feature columns, ``cause``, ``customer_id``, ``is_holdout``.

    Raises
    ------
    FileNotFoundError
        If the generator has not been run yet.
    """
    sample_dir = Path(sample_dir) if sample_dir else SAMPLE_OUTPUT_DIR
    ctx_path = sample_dir / "failure_contexts.json"
    if not ctx_path.exists():
        raise FileNotFoundError(
            f"{ctx_path} not found. Run: python data/synthetic_generator.py --profile train"
        )

    contexts: dict[str, dict[str, Any]] = json.loads(ctx_path.read_text(encoding="utf-8"))
    train_labels: dict[str, str] = json.loads(
        (sample_dir / "train_labels.json").read_text(encoding="utf-8")
    )
    holdout_labels: dict[str, str] = json.loads(
        (sample_dir / "ground_truth_holdout.json").read_text(encoding="utf-8")
    )

    df = pd.DataFrame.from_dict(contexts, orient="index")
    df.index.name = "transaction_id"

    labels = {**train_labels, **holdout_labels}
    df["cause"] = pd.Series(labels)
    df["is_holdout"] = df.index.map(
        lambda t: 1 if t in holdout_labels else 0
    ).astype(int)

    # customer_id is required as the CV grouping key. Sourced from the ledger,
    # not from the feature block, so it can never be mistaken for a feature.
    from database import db_client

    ids = list(df.index)
    placeholders = ",".join("?" * len(ids))
    rows = db_client.query(
        f"SELECT id, customer_id FROM transactions WHERE id IN ({placeholders})",
        ids,
    )
    cust_map = {r["id"]: r["customer_id"] for r in rows}
    df["customer_id"] = df.index.map(cust_map)

    missing = df["customer_id"].isna().sum()
    if missing:
        raise RuntimeError(
            f"{missing} failed transactions have no matching row in the ledger. "
            "Re-run the generator with --load-db."
        )
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Turn the modelling table into a numeric design matrix.

    Categorical columns are one-hot encoded (with a ``_nan`` column so that a
    missing value is itself a signal). Numeric columns keep their NaN, which
    XGBoost's histogram learner handles natively.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_failure_dataset`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (design matrix indexed like ``df``, feature names in column order).
    """
    parts = [df[list(NUMERIC_FEATURES)].astype(float)]
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col].astype("object"), prefix=col, dummy_na=True)
        parts.append(dummies.astype(float))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class RootCauseClassifier:
    """
    XGBoost root-cause classifier with grouped-CV tuning and SHAP reasons.

    Attributes
    ----------
    model : xgboost.XGBClassifier | None
        Fitted estimator.
    classes_ : list[str]
        Cause buckets in the model's internal label order.
    feature_names : list[str]
        Design-matrix column order the model was fitted on.
    best_params_ : dict[str, Any]
        Hyperparameters chosen by cross-validation.
    cv_results_ : dict[str, Any] | None
        Best CV score and the fold scores behind it.
    """

    model: xgb.XGBClassifier | None = None
    classes_: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    best_params_: dict[str, Any] = field(default_factory=dict)
    cv_results_: dict[str, Any] | None = None

    # -- tuning ------------------------------------------------------------
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Sequence[str],
        n_iter: int = N_SEARCH_ITER,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Select hyperparameters by randomised search over grouped CV.

        The CV splitter is :class:`~sklearn.model_selection.StratifiedGroupKFold`
        with ``groups=customer_id``, so folds are class-balanced *and* no
        customer straddles a fold boundary. This mirrors the outer split's
        guarantee; a plain KFold here would report optimistic scores.

        Parameters
        ----------
        X : pd.DataFrame
            Training design matrix.
        y : pd.Series
            Ground-truth cause labels (strings).
        groups : Sequence[str]
            Customer id per row.
        n_iter : int, optional
            Number of sampled hyperparameter combinations.
        verbose : bool, optional
            Print progress and the selected configuration.

        Returns
        -------
        dict[str, Any]
            ``cv_results_``: best params, best macro-F1, per-fold scores.
        """
        self.classes_ = sorted(set(y))
        y_codes = pd.Series(
            pd.Categorical(y, categories=self.classes_).codes, index=y.index
        )
        groups_arr = np.asarray(groups)

        cv = StratifiedGroupKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE
        )
        base = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2,
            verbosity=0,
            random_state=CV_RANDOM_STATE,
        )
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=SEARCH_SPACE,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=CV_RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            verbose=0,
        )
        if verbose:
            print(f"[diagnoser] tuning: {n_iter} configs x {CV_FOLDS} grouped folds ...")
        t0 = time.time()
        search.fit(X, y_codes, groups=groups_arr)
        elapsed = time.time() - t0

        self.best_params_ = dict(search.best_params_)
        # cv_results_["split{i}_test_score"] is an array of length n_iter; the
        # per-fold scores of the SELECTED config sit at best_index_.
        best_idx = int(search.best_index_)
        fold_scores = [
            float(search.cv_results_[f"split{i}_test_score"][best_idx])
            for i in range(CV_FOLDS)
        ]
        self.cv_results_ = {
            "n_configs": n_iter,
            "n_folds": CV_FOLDS,
            "cv_type": "StratifiedGroupKFold(groups=customer_id)",
            "scoring": "f1_macro",
            "best_score": float(search.best_score_),
            "best_score_std": float(np.std(fold_scores)),
            "fold_scores": [round(v, 4) for v in fold_scores],
            "best_params": self.best_params_,
            "elapsed_seconds": round(elapsed, 2),
        }
        self.model = search.best_estimator_
        self.feature_names = list(X.columns)
        if verbose:
            print(
                f"[diagnoser] best macro-F1 (CV, train split only) = "
                f"{search.best_score_:.4f} in {elapsed:.1f}s"
            )
            print(f"[diagnoser] params: {self.best_params_}")
        return self.cv_results_

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> "RootCauseClassifier":
        """
        Fit on the full training split.

        Parameters
        ----------
        X : pd.DataFrame
            Training design matrix.
        y : pd.Series
            Ground-truth cause labels.
        params : dict[str, Any], optional
            Hyperparameters; defaults to the CV-selected ``best_params_``.

        Returns
        -------
        RootCauseClassifier
            Self, for chaining.
        """
        self.classes_ = sorted(set(y))
        y_codes = pd.Series(
            pd.Categorical(y, categories=self.classes_).codes, index=y.index
        )
        cfg = dict(params or self.best_params_)
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2,
            verbosity=0,
            random_state=CV_RANDOM_STATE,
            **cfg,
        )
        self.model.fit(X, y_codes)
        self.feature_names = list(X.columns)
        return self

    # -- inference ---------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the most likely cause for each row.

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix.

        Returns
        -------
        pd.Series
            Predicted cause label per row, indexed like ``X``.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        codes = self.model.predict(X)
        return pd.Series(
            [self.classes_[int(c)] for c in codes], index=X.index, name="predicted_cause"
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Class probabilities per row.

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix.

        Returns
        -------
        pd.DataFrame
            Columns are the cause buckets, rows indexed like ``X``.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        probs = self.model.predict_proba(X)
        return pd.DataFrame(probs, index=X.index, columns=self.classes_)

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """
        Predict cause and confidence (probability of the winning class).

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            (predicted cause, confidence in [0, 1]) both indexed like ``X``.
        """
        proba = self.predict_proba(X)
        codes = proba.values.argmax(axis=1)
        conf = proba.values.max(axis=1)
        pred = pd.Series(
            [self.classes_[int(c)] for c in codes], index=X.index, name="predicted_cause"
        )
        return pred, pd.Series(conf, index=X.index, name="confidence")

    # -- explainability ----------------------------------------------------
    def explain(
        self, X: pd.DataFrame, top_k: int = 3
    ) -> list[list[tuple[str, float]]]:
        """
        Per-row tree-SHAP attribution, returning the top contributing features.

        Uses XGBoost's built-in ``pred_contribs`` (exact tree SHAP), so no
        external ``shap`` dependency is required — see FAILURES.md.

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix.
        top_k : int, optional
            Number of top contributors to return per row.

        Returns
        -------
        list[list[tuple[str, float]]]
            For each row, up to ``top_k`` (feature, shap_value) pairs sorted by
            descending absolute contribution.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        booster = self.model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        contrib = booster.predict(dmat, pred_contribs=True)
        # Multiclass pred_contribs is 3-D: (n_samples, n_classes, n_features+1).
        # We want the attributions for the class the model actually chose.
        if contrib.ndim == 3:
            pred_codes = np.asarray(self.model.predict(X)).astype(int)
            contrib = contrib[np.arange(contrib.shape[0]), pred_codes, :]
        # Last column is the bias term; drop it.
        contrib = contrib[:, :-1]

        out: list[list[tuple[str, float]]] = []
        for row in contrib:
            order = np.argsort(-np.abs(row))[:top_k]
            out.append([(self.feature_names[int(i)], float(row[i])) for i in order])
        return out

    def global_feature_importance(self, top_k: int = 12) -> list[tuple[str, float]]:
        """
        Model-wide feature importance by total gain.

        Parameters
        ----------
        top_k : int, optional
            Number of features to return.

        Returns
        -------
        list[tuple[str, float]]
            (feature, gain) pairs sorted by descending gain.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        scores = self.model.get_booster().get_score(importance_type="gain")
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, float(v)) for k, v in items[:top_k]]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """
        Persist the fitted model and its label/feature metadata.

        Parameters
        ----------
        path : Path, optional
            Destination. Defaults to ``models/root_cause_classifier.json``.

        Returns
        -------
        Path
            Path of the written model file.
        """
        if self.model is None:
            raise RuntimeError("Nothing to save: model is not fitted.")
        path = Path(path) if path else MODELS_DIR / "root_cause_classifier.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta = {
            "classes_": self.classes_,
            "feature_names": self.feature_names,
            "best_params_": self.best_params_,
            "cv_results_": self.cv_results_,
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "RootCauseClassifier":
        """
        Load a persisted model.

        Parameters
        ----------
        path : Path, optional
            Source. Defaults to ``models/root_cause_classifier.json``.

        Returns
        -------
        RootCauseClassifier
            Instance with the fitted booster and metadata restored.
        """
        path = Path(path) if path else MODELS_DIR / "root_cause_classifier.json"
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(
            classes_=meta["classes_"],
            feature_names=meta["feature_names"],
            best_params_=meta.get("best_params_", {}),
            cv_results_=meta.get("cv_results_"),
        )
        obj.model = xgb.XGBClassifier()
        obj.model.load_model(str(path))
        return obj


if __name__ == "__main__":
    df = load_failure_dataset()
    print(f"[diagnoser] loaded {len(df):,} failure rows")
    print(f"[diagnoser] train={int((df.is_holdout == 0).sum()):,}  "
          f"holdout={int((df.is_holdout == 1).sum()):,}")
