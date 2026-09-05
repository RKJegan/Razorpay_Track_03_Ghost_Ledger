"""
Split-strategy leakage experiment.

Answers one question with a measurement rather than an assertion:

    How much does the choice of split inflate the reported score?

The experiment holds everything constant — same corpus, same feature matrix,
same hyperparameters — and varies only how the held-out batch is selected:

    random_grouped  whole customers held out (the default, leakage-safe)
    random_rows     individual transactions held out (customers recur)
    temporal        trailing N days held out

Any difference in score is therefore attributable to the split alone.

Usage
-----
    python diagnoser/split_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DATA_SEED, HOLDOUT_DAYS, SAMPLE_OUTPUT_DIR  # noqa: E402
from diagnoser.eval_holdout import compute_metrics, print_leakage_comparison  # noqa: E402
from diagnoser.root_cause_classifier import (  # noqa: E402
    build_feature_matrix,
    load_failure_dataset,
    RootCauseClassifier,
)


def assign_split(
    df: pd.DataFrame, strategy: str, seed: int = DATA_SEED
) -> pd.Series:
    """
    Return a boolean "is holdout" mask for a given strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Failure rows; requires ``customer_id`` and ``timestamp`` columns.
    strategy : str
        One of ``random_grouped``, ``random_rows``, ``temporal``.
    seed : int, optional
        RNG seed for the random strategies.

    Returns
    -------
    pd.Series
        Boolean mask indexed like ``df``; True marks held-out rows.
    """
    rng = np.random.default_rng(seed)

    if strategy == "random_grouped":
        by_customer = df.groupby("customer_id").indices
        customers = list(by_customer.keys())
        rng.shuffle(customers)
        target = int(round(0.20 * len(df)))
        mask = pd.Series(False, index=df.index)
        assigned = 0
        for cid in customers:
            if assigned >= target:
                break
            idx = by_customer[cid]
            mask.iloc[idx] = True
            assigned += len(idx)
        return mask

    if strategy == "random_rows":
        target = int(round(0.20 * len(df)))
        idx = rng.permutation(len(df))[:target]
        mask = pd.Series(False, index=df.index)
        mask.iloc[idx] = True
        return mask

    if strategy == "temporal":
        cutoff = df["timestamp"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        return df["timestamp"] >= cutoff

    raise ValueError(f"unknown split strategy: {strategy}")


def main() -> int:
    """
    Run the three-way split comparison and print the result.

    Returns
    -------
    int
        Process exit code.
    """
    df = load_failure_dataset()

    # Timestamps are needed for the temporal arm; not a model feature.
    from database import db_client

    ids = list(df.index)
    placeholders = ",".join("?" * len(ids))
    ts_map = {
        r["id"]: r["timestamp"]
        for r in db_client.query(
            f"SELECT id, timestamp FROM transactions WHERE id IN ({placeholders})", ids
        )
    }
    df["timestamp"] = pd.to_datetime(df.index.map(ts_map))

    X_all, feature_names = build_feature_matrix(df)
    y = df["cause"]

    # Fixed hyperparameters from the tuned run, so every arm is identical
    # except for the split.
    metrics_path = SAMPLE_OUTPUT_DIR.parent.parent / "reports" / "holdout_metrics.json"
    best_params: dict = {}
    if metrics_path.exists():
        best_params = json.loads(metrics_path.read_text()).get("best_params", {})
    print(f"[experiment] using fixed params from tuned run: {best_params or 'defaults'}")

    rows: list[dict] = []
    for strategy in ("random_grouped", "random_rows", "temporal"):
        mask = assign_split(df, strategy)
        train_idx = df.index[~mask]
        hold_idx = df.index[mask]

        # Customer overlap is the quantity that actually drives inflation.
        tr_c = set(df.loc[train_idx, "customer_id"])
        ho_c = set(df.loc[hold_idx, "customer_id"])
        overlap = len(tr_c & ho_c)

        clf = RootCauseClassifier().fit(
            X_all.loc[train_idx], y.loc[train_idx], params=best_params
        )
        pred, _ = clf.predict_with_confidence(X_all.loc[hold_idx])
        m = compute_metrics(y.loc[hold_idx].tolist(), pred.tolist())

        rows.append(
            {
                "strategy": strategy,
                "n": int(len(hold_idx)),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro"]["f1"],
                "customers_overlapping": overlap,
            }
        )
        print(
            f"[experiment] {strategy:<16} n={len(hold_idx):>5,}  "
            f"acc={m['accuracy']:.4f}  macro-F1={m['macro']['f1']:.4f}  "
            f"customers in both splits={overlap:,}"
        )

    print()
    print_leakage_comparison(rows)

    out = SAMPLE_OUTPUT_DIR.parent.parent / "reports" / "split_experiment.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
