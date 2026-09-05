"""
Train the root-cause diagnoser end to end.

Steps
-----
1. Load the failure dataset (features + ground truth + split flag).
2. Build the design matrix.
3. Tune hyperparameters by grouped cross-validation **on the training split**.
4. Refit on the full training split.
5. Evaluate once, on the held-out split, and print with n + split method.
6. Persist the model and the evaluation record.

The held-out batch is not read at any point before step 5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from diagnoser.eval_holdout import (  # noqa: E402
    evaluate_holdout,
    print_report,
)
from diagnoser.root_cause_classifier import (  # noqa: E402
    build_feature_matrix,
    load_failure_dataset,
    N_SEARCH_ITER,
    RootCauseClassifier,
)


def evaluate_model(
    clf: RootCauseClassifier,
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    extra: dict | None = None,
) -> tuple[dict, pd.Series, pd.Series]:
    """
    Evaluate a fitted model on the held-out split and build the report.

    Shared by ``diagnoser/train.py`` and ``main.py`` so the dashboard's
    diagnoser panel always describes the model actually in use, never a stale
    one from an earlier run.

    Parameters
    ----------
    clf : RootCauseClassifier
        Fitted model.
    df : pd.DataFrame
        Full failure frame with ``is_holdout`` and ``cause``.
    X_all : pd.DataFrame
        Design matrix aligned to ``df``.
    extra : dict, optional
        Additional fields to merge into the report.

    Returns
    -------
    tuple[dict, pd.Series, pd.Series]
        (report, predictions, confidences) for the held-out split.
    """
    holdout_df = df[df["is_holdout"] == 1]
    train_df = df[df["is_holdout"] == 0]
    pred, conf = clf.predict_with_confidence(X_all.loc[holdout_df.index])
    cv = clf.cv_results_ or {}

    # Attribution is computed here rather than in main() so that the stored
    # report is identical whichever entry point produced it. Previously
    # `main.py` overwrote holdout_metrics.json without the attribution study,
    # silently dropping the decomposition from the dashboard.
    attribution = run_attribution_checks(
        clf,
        X_all.loc[train_df.index],
        train_df["cause"],
        df,
        X_all.loc[holdout_df.index],
        holdout_df["cause"],
        list(X_all.columns),
    )

    report = evaluate_holdout(
        holdout_df["cause"].tolist(),
        pred.tolist(),
        extra={
            "cv_macro_f1": cv.get("best_score"),
            "cv_macro_f1_std": cv.get("best_score_std"),
            "cv_fold_scores": cv.get("fold_scores"),
            "cv_type": cv.get("cv_type"),
            "cv_folds": cv.get("n_folds"),
            "cv_n_configs": cv.get("n_configs"),
            "best_params": clf.best_params_,
            "training_n": int(len(train_df)),
            "mean_confidence": float(conf.mean()),
            "feature_count": len(X_all.columns),
            **(extra or {}),
        },
    )
    report["attribution"] = attribution
    return report, pred, conf


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point for diagnoser training.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Train the Ghost Ledger root-cause diagnoser")
    parser.add_argument(
        "--n-iter", type=int, default=N_SEARCH_ITER, help="CV hyperparameter configs to sample"
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="skip CV tuning and use defaults"
    )
    args = parser.parse_args(argv)

    t0 = time.time()

    # --- 1. data ----------------------------------------------------------
    df = load_failure_dataset()
    train_df = df[df["is_holdout"] == 0]
    holdout_df = df[df["is_holdout"] == 1]
    print("=" * 78)
    print("  GHOST LEDGER v2 — ROOT CAUSE DIAGNOSER TRAINING")
    print("=" * 78)
    print(f"  failure rows      : {len(df):,}")
    print(f"  training split    : {len(train_df):,} rows")
    print(f"  held-out split    : {len(holdout_df):,} rows  (not touched until eval)")
    print(f"  split method      : {_split_line()}")
    print("-" * 78)

    # --- 2. design matrix -------------------------------------------------
    X_all, feature_names = build_feature_matrix(df)
    X_train = X_all.loc[train_df.index]
    y_train = train_df["cause"]
    X_hold = X_all.loc[holdout_df.index]
    y_hold = holdout_df["cause"]
    print(f"  features          : {len(feature_names)} "
          f"({len(X_all.columns) - len(feature_names) == 0 and 'numeric + one-hot'})")

    # --- 3. tune (train split only) ---------------------------------------
    clf = RootCauseClassifier()
    if args.no_tune:
        clf.fit(X_train, y_train)
        print("[diagnoser] tuning skipped (--no-tune)")
    else:
        clf.tune(X_train, y_train, groups=train_df["customer_id"], n_iter=args.n_iter)
        # RandomizedSearchCV refits on all of X_train with refit=True,
        # so the held-out batch has still not been used.
        clf.fit(X_train, y_train)

    print("-" * 78)

    # --- 4. evaluate on holdout -------------------------------------------
    report, pred, conf = evaluate_model(clf, df, X_all)
    print_report(report)

    print_report(report)

    # --- 5. persist -------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = clf.save()

    imp = clf.global_feature_importance(top_k=12)
    print("  top features by gain:")
    for name, gain in imp:
        print(f"      {name:<38}{gain:>12,.1f}")
    print("-" * 78)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "holdout_metrics.json"

    # --- 6. is the model doing real work, or reading the error code? -------
    # A 97%-looking score is only meaningful next to a baseline. If a plain
    # lookup on error_code alone gets most of the way there, the honest thing
    # is to say so rather than bank the headline number.
    ablations = run_attribution_checks(
        clf, X_train, y_train, df, X_hold, y_hold, feature_names
    )
    print_ablation_table(ablations, report)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Sample explanations, to eyeball that reasons are specific not generic.
    print("  sample per-prediction reasons (tree-SHAP, top 3):")
    expl = clf.explain(X_hold.head(3), top_k=3)
    for (txn_id, _), feats in zip(X_hold.head(3).iterrows(), expl):
        reasons = ", ".join(f"{f}={v:+.3f}" for f, v in feats)
        print(f"      {txn_id}: true={y_hold.loc[txn_id]:<20} pred={pred.loc[txn_id]:<20}")
        print(f"          {reasons}")
    print("-" * 78)
    print(f"  model  -> {model_path}")
    print(f"  report -> {report_path}")
    print(f"  total  -> {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0


def _split_line() -> str:
    """Load the split method string for console output."""
    from diagnoser.eval_holdout import load_split_method

    return load_split_method()


def error_code_lookup_baseline(
    train_df: pd.DataFrame,
    y_hold: pd.Series,
    holdout_codes: pd.Series,
) -> dict[str, Any]:
    """
    Baseline with no model at all: look up the most likely cause per error code.

    The lookup table is built from the TRAINING split's empirical distribution
    only. It answers: "how much of the diagnoser's score is just reading the
    error code?"

    Parameters
    ----------
    train_df : pd.DataFrame
        Training rows, carrying raw ``error_code`` and ``cause`` columns.
    y_hold : pd.Series
        Held-out ground-truth causes.
    holdout_codes : pd.Series
        Held-out error codes, aligned to ``y_hold``.

    Returns
    -------
    dict[str, Any]
        Accuracy and macro-F1 of the lookup, plus a note on interpretation.
    """
    from diagnoser.eval_holdout import compute_metrics

    # P(cause | error_code) estimated on the training split only.
    lookup = train_df.groupby("error_code")["cause"].agg(
        lambda s: s.value_counts().idxmax()
    )
    global_default = train_df["cause"].value_counts().idxmax()
    pred = holdout_codes.map(lookup).fillna(global_default)

    m = compute_metrics(y_hold.tolist(), pred.tolist())
    return {
        "name": "baseline: error_code -> majority cause (no ML)",
        "accuracy": m["accuracy"],
        "macro_f1": m["macro"]["f1"],
        "n": m["n"],
        "lookup_table": {str(k): str(v) for k, v in lookup.items()},
        "note": (
            "Pure lookup table built from training-split frequencies. Any score "
            "the model achieves above this is genuinely attributable to the "
            "model rather than to the error code."
        ),
    }


def run_attribution_checks(
    clf: RootCauseClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """
    Quantify how much of the score comes from the error code vs. the model.

    Three rows:
      1. no-ML lookup on error_code  -> the floor
      2. full model                  -> the headline
      3. model with all error_code
         features removed            -> what the model contributes alone

    Parameters
    ----------
    clf : RootCauseClassifier
        Fitted full model.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    train_df : pd.DataFrame
        ALL rows (train + holdout) with raw columns including ``error_code``.
        Only the train portion is used to build the lookup table.
    X_hold, y_hold : pd.DataFrame, pd.Series
        Held-out data.
    feature_names : list[str]
        Full feature list.

    Returns
    -------
    list[dict[str, Any]]
        One result dict per condition.
    """
    from diagnoser.eval_holdout import compute_metrics

    out: list[dict[str, Any]] = []

    holdout_codes = train_df.loc[X_hold.index, "error_code"].astype(str)
    train_only = train_df[train_df["is_holdout"] == 0]
    out.append(error_code_lookup_baseline(train_only, y_hold, holdout_codes))

    # Full model.
    pred_full, _ = clf.predict_with_confidence(X_hold)
    m_full = compute_metrics(y_hold.tolist(), pred_full.tolist())
    out.append(
        {
            "name": "full model (all features)",
            "accuracy": m_full["accuracy"],
            "macro_f1": m_full["macro"]["f1"],
            "n": m_full["n"],
        }
    )

    # Ablation: drop every error_code column.
    keep = [c for c in feature_names if not c.startswith("error_code_")]
    X_train_a, X_hold_a = X_train[keep], X_hold[keep]
    abl = RootCauseClassifier().fit(X_train_a, y_train, params=clf.best_params_)
    pred_a, _ = abl.predict_with_confidence(X_hold_a)
    m_a = compute_metrics(y_hold.tolist(), pred_a.tolist())
    out.append(
        {
            "name": f"model WITHOUT error_code ({len(keep)} features)",
            "accuracy": m_a["accuracy"],
            "macro_f1": m_a["macro"]["f1"],
            "n": m_a["n"],
            "note": "Reuses the full model's tuned hyperparameters.",
        }
    )
    return out


def print_ablation_table(rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    """
    Print the attribution comparison.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Output of :func:`run_attribution_checks`.
    report : dict[str, Any]
        Main evaluation report, for the reference n.
    """
    W = 78
    print("=" * W)
    print("  ATTRIBUTION — how much is the model actually contributing?")
    print("=" * W)
    print(f"  {'condition':<44}{'accuracy':>11}{'macro-F1':>11}")
    print("-" * W)
    for r in rows:
        print(f"  {r['name']:<44}{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}")
    print("-" * W)
    floor = rows[0]["macro_f1"]
    full = rows[1]["macro_f1"]
    abl = rows[2]["macro_f1"]
    print(f"  lookup floor (no ML)          : {floor:.4f} macro-F1")
    print(f"  model lift over lookup        : {full - floor:+.4f} macro-F1")
    print(f"  model with error code removed : {abl:.4f} macro-F1")
    print("-" * W)
    print("  Read this before quoting the headline number. If the lift over the")
    print("  lookup floor is small, the error code is doing the work and the")
    print("  classifier is largely a decoder — say so in the pitch rather than")
    print("  letting a high accuracy imply more intelligence than it does.")
    print("=" * W)


if __name__ == "__main__":
    raise SystemExit(main())
