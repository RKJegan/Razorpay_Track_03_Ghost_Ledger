"""
Ghost Ledger v2 — one-command pipeline.

    python main.py

Runs the complete loop, in build order:

    generate -> diagnose -> policy check -> agent acts -> audit -> metrics

Nothing here makes a financial decision. Every action passes through
`agents.policy_engine`, and the LLM (if enabled) only writes prose.

Options
-------
--profile NAME       dataset size preset (demo | train | max)
--regen              regenerate the dataset from the seed
--retrain            force retraining the diagnoser (slow: CV tuning)
--limit N            process only the first N held-out failures
--autopsy-limit N    generate at most N LLM autopsies (default 50)
--llm-backend NAME   ollama | openai | template
--no-autopsy         skip autopsy generation entirely
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from agents.autopsy_reporter import build_facts, generate_autopsy, persist  # noqa: E402
from agents.payment_failure_agent import recover_payment_failure  # noqa: E402
from agents.policy_engine import PolicyEngine  # noqa: E402
from agents.subscription_agent import recover_subscription_failure  # noqa: E402
from api.razorpay_client import get_razorpay_client  # noqa: E402
from config import (  # noqa: E402
    DATASET_PROFILE,
    MODELS_DIR,
    REPORTS_DIR,
    SAMPLE_OUTPUT_DIR,
)
from database import audit_trail, db_client  # noqa: E402
from metrics import compute_headline_metrics  # noqa: E402
from data.synthetic_generator import SyntheticDataGenerator  # noqa: E402
from diagnoser.train import evaluate_model  # noqa: E402
from diagnoser.root_cause_classifier import (  # noqa: E402
    build_feature_matrix,
    load_failure_dataset,
    RootCauseClassifier,
)

W = 80


def _header(title: str) -> None:
    """Print a section header."""
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def step_generate(regen: bool, profile: str) -> None:
    """
    Ensure a dataset exists, regenerating from the seed when asked.

    Parameters
    ----------
    regen : bool
        Force regeneration.
    profile : str
        Dataset size preset.
    """
    manifest_path = SAMPLE_OUTPUT_DIR / "generation_manifest.json"
    db_client.init_db()
    ledger_rows = int(db_client.scalar("SELECT COUNT(*) FROM transactions") or 0)

    # Regenerate whenever the artefacts OR the ledger are missing. Checking
    # only the manifest is not enough: on a clean checkout the JSON files can
    # exist (committed fixtures) while the SQLite ledger does not, which left
    # the diagnoser with feature rows but no customer ids.
    # Regeneration is safe because the generator is seeded - it reproduces the
    # identical dataset byte for byte.
    need_data = regen or not manifest_path.exists() or ledger_rows == 0

    if need_data:
        _header("STEP 1/6 — SYNTHETIC DATA GENERATION")
        if ledger_rows == 0 and manifest_path.exists() and not regen:
            print("  ledger is empty - regenerating from seed (deterministic)")
        gen = SyntheticDataGenerator(profile=profile)
        manifest, paths = gen.run()
        from database.seed_db import load_transactions

        n, mb = load_transactions(gen.transactions)
        print(f"  {n:,} transactions -> SQLite ({mb:.2f} MB)")
    else:
        manifest = json.loads(manifest_path.read_text())
        print(f"[skip] dataset present: profile={manifest['profile']}, "
              f"seed={manifest['seed']}, {ledger_rows:,} rows in ledger; "
              f"use --regen to rebuild")


def step_diagnose(retrain: bool) -> tuple[RootCauseClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Load or train the diagnoser and score every failed transaction.

    Parameters
    ----------
    retrain : bool
        Force CV tuning instead of loading the persisted model.

    Returns
    -------
    tuple[RootCauseClassifier, pd.DataFrame, pd.DataFrame]
        (classifier, full failure frame, design matrix)
    """
    _header("STEP 2/6 — ROOT CAUSE DIAGNOSIS")
    model_path = MODELS_DIR / "root_cause_classifier.json"

    # Hard constraint 7: one command must start the system on a clean
    # checkout. So a missing model is trained here rather than fatal.
    # First run uses a reduced search (~1 min) to stay inside the 5-minute
    # demo budget; `python diagnoser/train.py` does the full 40-config run.
    if retrain or not model_path.exists():
        print("  no trained model found - training now (reduced CV search) ...")
        df_all = load_failure_dataset()
        X_all, _ = build_feature_matrix(df_all)
        train_mask = df_all["is_holdout"] == 0
        clf = RootCauseClassifier()
        clf.tune(
            X_all[train_mask],
            df_all.loc[train_mask, "cause"],
            groups=df_all.loc[train_mask, "customer_id"],
            n_iter=15,
        )
        clf.save(model_path)
        print(f"  saved -> {model_path}")

    clf = RootCauseClassifier.load(model_path)
    df = load_failure_dataset()
    X, _ = build_feature_matrix(df)
    pred, conf = clf.predict_with_confidence(X)
    df["predicted_cause"] = pred
    df["confidence"] = conf
    print(f"  model        : {model_path.name}")
    print(f"  diagnosed    : {len(df):,} failed transactions")
    print(f"  mean conf    : {conf.mean():.4f}")

    # Always refresh the stored held-out evaluation, so the dashboard's
    # diagnoser panel describes THIS model rather than one from an earlier
    # run. Without this, auto-training on a clean checkout would leave metrics
    # on screen that belong to a different model than the one taking actions.
    report, _, _ = evaluate_model(clf, df, X)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "holdout_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  held-out     : macro-F1 {report['macro']['f1']:.4f} "
          f"(n={report['n']:,}) -> reports/holdout_metrics.json")
    return clf, df, X


def step_persist_failures(df: pd.DataFrame) -> int:
    """
    Write one ``failures`` row per failed transaction.

    Parameters
    ----------
    df : pd.DataFrame
        Diagnosed failures.

    Returns
    -------
    int
        Rows written.
    """
    rows = []
    for txn_id, r in df.iterrows():
        rows.append(
            (
                f"fail_{txn_id}",
                txn_id,
                r["predicted_cause"],
                float(r["confidence"]),
                r["cause"],
                datetime.now().isoformat(sep=" ", timespec="seconds"),
                float(r["amount"]),
            )
        )
    db_client.executemany(
        "INSERT OR REPLACE INTO failures (id, transaction_id, predicted_cause, "
        "confidence, ground_truth_cause, detected_at, estimated_value) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    audit_trail.log(
        component="diagnoser",
        action="diagnose_batch",
        input_data={"n_failures": len(rows)},
        output_data={
            "cause_distribution": df["predicted_cause"].value_counts().to_dict()
        },
        decision_reason=f"Diagnosed {len(rows):,} failures with the trained XGBoost model.",
        success=True,
    )
    return len(rows)


def step_recover(
    df: pd.DataFrame, holdout_only: bool = True, limit: int | None = None
) -> dict[str, Any]:
    """
    Run the policy-gated recovery agents over the batch.

    Parameters
    ----------
    df : pd.DataFrame
        Diagnosed failures.
    holdout_only : bool, optional
        Restrict to the held-out batch so the headline matches the eval set.
    limit : int, optional
        Cap the number of failures processed.

    Returns
    -------
    dict[str, Any]
        Aggregate counters for the run.
    """
    _header("STEP 4/6 — POLICY-GATED RECOVERY ACTIONS")
    work = df[df["is_holdout"] == 1] if holdout_only else df
    if limit:
        work = work.head(limit)

    client = get_razorpay_client()
    engine = PolicyEngine()
    print(f"  policy       :\n{_indent(engine.describe())}")
    print(f"  razorpay     : {type(client).__name__}")
    print(f"  failures     : {len(work):,}")

    counts = {"success": 0, "fail": 0, "stopped": 0, "blocked": 0}
    recovered_total = 0.0
    t0 = time.time()

    for i, (txn_id, r) in enumerate(work.iterrows(), start=1):
        failure_id = f"fail_{txn_id}"
        is_sub = r["txn_type"] == "subscription"
        kwargs = dict(
            failure_id=failure_id,
            customer_id=r["customer_id"],
            amount_inr=float(r["amount"]),
            cause=r["predicted_cause"],
            confidence=float(r["confidence"]),
            transaction_id=txn_id,
            client=client,
            engine=engine,
        )
        if is_sub:
            outcome = recover_subscription_failure(
                mandate_id=r.get("mandate_id"), **kwargs
            )
        else:
            outcome = recover_payment_failure(**kwargs)

        counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        recovered_total += outcome.recovered_amount

        if i % 400 == 0:
            print(f"    ... {i:,}/{len(work):,} processed "
                  f"({time.time() - t0:.1f}s, recovered INR {recovered_total:,.0f})")

    print(f"  outcomes     : {counts}")
    print(f"  recovered    : INR {recovered_total:,.2f}")
    print(f"  elapsed      : {time.time() - t0:.1f}s")
    return {
        "counts": counts,
        "recovered_total": recovered_total,
        "processed": len(work),
    }


def reset_action_state() -> None:
    """
    Clear the results of any previous pipeline run.

    Without this, running the pipeline twice stacks a second set of recovery
    actions on top of the first: the attempt counter would see prior attempts,
    the policy engine would immediately hit its stopping rule, and every
    headline figure would describe the union of two runs instead of one.

    The transaction ledger is deliberately left alone — it is regenerated only
    with ``--regen`` — and generator records in the audit trail are preserved.
    """
    # Children before parents: `recoveries` and `autopsy_reports` both carry
    # foreign keys to `failures`, and foreign key enforcement is on.
    db_client.execute("DELETE FROM recoveries")
    db_client.execute("DELETE FROM autopsy_reports")
    db_client.execute("DELETE FROM failures")
    db_client.execute(
        "DELETE FROM audit_trail WHERE component != 'synthetic_generator'"
    )


def _indent(text: str, spaces: int = 6) -> str:
    """Indent each line of a block."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def step_autopsy(
    df: pd.DataFrame, limit: int, backend: str | None
) -> dict[str, Any]:
    """
    Generate autopsy explanations.

    Parameters
    ----------
    df : pd.DataFrame
        Diagnosed failures.
    limit : int
        Maximum number of reports to generate with a live model.
    backend : str | None
        LLM backend override.

    Returns
    -------
    dict[str, Any]
        Counters: generated, hallucination-flagged, degraded.
    """
    _header("STEP 5/6 — AUTOPSY REPORTS (text only)")
    from config import LLM_BACKEND

    effective = backend or LLM_BACKEND
    work = df[df["is_holdout"] == 1]
    # A live model costs ~1s per report, so the batch is sampled unless the
    # deterministic template is in use. The sample size is recorded in the
    # provenance string so the dashboard can state its basis.
    if effective == "template":
        sample = work
    else:
        sample = work.head(limit)

    generated = flagged = degraded = 0
    t0 = time.time()
    for txn_id, r in sample.iterrows():
        facts = build_facts(
            failure_id=f"fail_{txn_id}",
            transaction_id=txn_id,
            cause=r["predicted_cause"],
            confidence=float(r["confidence"]),
            amount_inr=float(r["amount"]),
            txn_type=r["txn_type"],
            payment_method=r["payment_method"],
            gateway=r["gateway"],
            error_code=r.get("error_code"),
            timestamp=str(r.get("timestamp", "")),
        )
        report = generate_autopsy(f"fail_{txn_id}", facts, backend=backend)
        persist(report)
        generated += 1
        flagged += bool(report.hallucination_flags)
        degraded += report.degraded

    print(f"  backend      : {effective}")
    print(f"  generated    : {generated:,} of {len(work):,} held-out failures")
    if generated < len(work):
        print(f"  NOTE         : sampled; a live model costs ~1s per report.")
    print(f"  hallucination flags: {flagged}")
    print(f"  degraded     : {degraded}")
    print(f"  elapsed      : {time.time() - t0:.1f}s")
    return {"generated": generated, "flagged": flagged, "degraded": degraded}


def step_metrics(recovery_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the headline figures straight from the database.

    Delegates to ``metrics.compute_headline_metrics`` - the same function the
    dashboard calls - so the headline cannot drift from the audit trail.

    Parameters
    ----------
    recovery_stats : dict[str, Any]
        Output of :func:`step_recover`.

    Returns
    -------
    dict[str, Any]
        Headline metrics, with n and split method attached.
    """
    _header("STEP 6/6 — HEADLINE METRICS (held-out batch)")
    metrics = compute_headline_metrics()
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    # Sanity check, not a source of truth: the live per-failure counters this
    # run just produced must agree with what metrics.py independently
    # recomputes from the database.
    if metrics["outcomes"] != recovery_stats["counts"]:
        print(
            "  WARNING: live outcome counters "
            f"{recovery_stats['counts']} do not match metrics.py's "
            f"recomputation {metrics['outcomes']} — investigate before "
            f"trusting this run's headline."
        )

    print(f"  N transactions      : {metrics['n_transactions']:,}")
    print(f"  Failures            : {metrics['n_failures']:,}")
    print(f"  Amount at risk      : INR {metrics['amount_at_risk_inr']:,.2f}")
    print(f"  Amount recovered    : INR {metrics['amount_recovered_inr']:,.2f}")
    print(f"  Recovery rate       : {metrics['recovery_rate']:.2%}")
    print(f"  Stopping-rule stops : {metrics['stopping_rule_events']:,}")
    print(f"  Recovery actions    : {metrics['recovery_actions']:,}")
    print(f"  Audit records       : {metrics['audit_records']:,}")
    print(f"  Reconciles with audit trail: {metrics['reconciled']}")
    print("-" * W)
    print(f"  basis               : {metrics['basis']}")
    print(f"  split method        : {metrics['split_method']}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """
    Run the whole pipeline.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Ghost Ledger v2 — full pipeline")
    parser.add_argument("--profile", default=DATASET_PROFILE, choices=["demo", "train", "max"])
    parser.add_argument("--regen", action="store_true", help="regenerate the dataset")
    parser.add_argument("--retrain", action="store_true", help="force diagnoser retraining")
    parser.add_argument("--reset-db", action="store_true", help="drop and recreate all tables")
    parser.add_argument("--limit", type=int, default=None, help="cap failures processed")
    parser.add_argument("--autopsy-limit", type=int, default=50)
    parser.add_argument("--llm-backend", default=None, choices=["ollama", "openai", "template"])
    parser.add_argument("--no-autopsy", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.time()

    if args.reset_db:
        db_client.reset_db()
        print("[db] reset complete")

    _header("GHOST LEDGER v2 — FULL PIPELINE")
    print(f"  started    : {datetime.now().isoformat(timespec='seconds')}")
    print(f"  profile    : {args.profile}")

    step_generate(args.regen, args.profile)
    db_client.init_db()
    clf, df, X = step_diagnose(args.retrain)

    # Make the run idempotent: clear any results from a previous run so the
    # figures describe exactly one pass, not the union of several.
    reset_action_state()

    _header("STEP 3/6 — PERSIST DIAGNOSES")
    n = step_persist_failures(df)
    print(f"  {n:,} failure records written")

    recovery_stats = step_recover(df, limit=args.limit)

    if args.no_autopsy:
        print("\n[skip] autopsy generation disabled")
    else:
        step_autopsy(df, limit=args.autopsy_limit, backend=args.llm_backend)

    metrics = step_metrics(recovery_stats)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "headline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    _header("PIPELINE COMPLETE")
    print(f"  total elapsed : {time.time() - t_start:.1f}s")
    print(f"  headline      : {REPORTS_DIR / 'headline_metrics.json'}")
    print(f"  next          : streamlit run dashboard/app.py")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
