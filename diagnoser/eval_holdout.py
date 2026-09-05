"""
FR-002 evaluation — held-out precision / recall.

Computes the diagnoser's precision, recall and F1 **against ground-truth
causes on the held-out split only**, and never emits a metric without the two
things NFR-004 requires beside it:

    n             the number of held-out examples the figure came from
    split_method  exactly how the held-out batch was selected

A bare accuracy number is treated as a bug. Both travel inside the same dict
as the scores, and every printer in this module renders them.

The split method is read from ``generation_manifest.json`` — the artefact the
generator wrote for *this exact corpus* — never from a module-level default.
That was the subject of a real bug; see FAILURES.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import CAUSE_BUCKETS, SAMPLE_OUTPUT_DIR


def load_split_method(sample_dir: Path | None = None) -> str:
    """
    Read the split description written by the generator for this corpus.

    Parameters
    ----------
    sample_dir : Path, optional
        Directory holding generator artefacts. Defaults to
        ``config.SAMPLE_OUTPUT_DIR``.

    Returns
    -------
    str
        The exact split-method string, or an explicit warning if absent.

    Raises
    ------
    FileNotFoundError
        If the manifest is missing (generator not yet run).
    """
    sample_dir = Path(sample_dir) if sample_dir else SAMPLE_OUTPUT_DIR
    manifest_path = sample_dir / "generation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run the generator before evaluating."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["split"]["method"]


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Per-class and aggregate precision / recall / F1.

    Parameters
    ----------
    y_true : Sequence[str]
        Ground-truth cause labels.
    y_pred : Sequence[str]
        Predicted cause labels.
    labels : Sequence[str], optional
        Label set to report on; defaults to ``config.CAUSE_BUCKETS`` order.

    Returns
    -------
    dict[str, Any]
        ``per_class``: dict of cause -> {precision, recall, f1, support}.
        ``macro`` / ``weighted``: aggregate precision, recall, f1.
        ``accuracy``, ``n``, ``confusion_matrix`` (with row/col labels).
    """
    lab = list(labels) if labels is not None else list(CAUSE_BUCKETS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=lab, zero_division=0
    )

    per_class: dict[str, dict[str, float]] = {}
    for i, c in enumerate(lab):
        per_class[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=lab)
    return {
        "per_class": per_class,
        "macro": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(f1_score(y_true, y_pred, labels=lab, average="macro", zero_division=0)),
        },
        "weighted": {
            "precision": float(
                np.average(precision, weights=support) if support.sum() else 0.0
            ),
            "recall": float(
                np.average(recall, weights=support) if support.sum() else 0.0
            ),
            "f1": float(
                f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0)
            ),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "confusion_matrix": {
            "labels": lab,
            "rows": [[int(v) for v in row] for row in cm],
        },
    }


def evaluate_holdout(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sample_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the complete, self-describing held-out evaluation record.

    Parameters
    ----------
    y_true : Sequence[str]
        Ground-truth causes for the held-out batch.
    y_pred : Sequence[str]
        Predicted causes for the held-out batch.
    sample_dir : Path, optional
        Directory holding generator artefacts.
    extra : dict[str, Any], optional
        Additional fields to merge in (e.g. CV score, model params).

    Returns
    -------
    dict[str, Any]
        Metrics plus the mandatory ``n`` and ``split_method`` provenance.
    """
    metrics = compute_metrics(y_true, y_pred)
    report: dict[str, Any] = {
        "evaluation_set": "held-out split only",
        "n": metrics["n"],
        "split_method": load_split_method(sample_dir),
        **metrics,
    }
    if extra:
        report.update(extra)
    return report


def print_report(report: dict[str, Any], title: str = "HELD-OUT EVALUATION") -> None:
    """
    Render an evaluation report, with n and split method always visible.

    Parameters
    ----------
    report : dict[str, Any]
        Output of :func:`evaluate_holdout`.
    title : str, optional
        Section heading.
    """
    W = 78
    print("=" * W)
    print(f"  {title}")
    print("=" * W)
    print(f"  evaluation set : {report['evaluation_set']}")
    print(f"  n              : {report['n']:,}")
    print(f"  split method   : {report['split_method']}")
    if report.get("cv_macro_f1") is not None:
        std = report.get("cv_macro_f1_std")
        std_txt = f" +/- {std:.4f}" if std is not None else ""
        print(
            f"  CV (train only): macro-F1 {report['cv_macro_f1']:.4f}{std_txt} "
            f"({report.get('cv_type', 'grouped CV')}, "
            f"{report.get('cv_folds', '?')} folds, "
            f"{report.get('cv_n_configs', '?')} configs)"
        )
        if report.get("cv_fold_scores"):
            print(f"  CV fold scores : {report['cv_fold_scores']}")
    print("-" * W)
    print(f"  {'cause':<20}{'precision':>10}{'recall':>10}{'F1':>10}{'n':>8}")
    print("-" * W)
    for cause, m in report["per_class"].items():
        print(
            f"  {cause:<20}{m['precision']:>10.4f}{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}{m['support']:>8,}"
        )
    print("-" * W)
    print(
        f"  {'macro avg':<20}{report['macro']['precision']:>10.4f}"
        f"{report['macro']['recall']:>10.4f}{report['macro']['f1']:>10.4f}"
        f"{report['n']:>8,}"
    )
    print(
        f"  {'weighted avg':<20}{report['weighted']['precision']:>10.4f}"
        f"{report['weighted']['recall']:>10.4f}{report['weighted']['f1']:>10.4f}"
        f"{report['n']:>8,}"
    )
    print(f"  {'accuracy':<20}{report['accuracy']:>31.4f}")
    print("-" * W)

    cm = report["confusion_matrix"]
    labels = [c[:11] for c in cm["labels"]]
    print("  confusion matrix (rows = truth, cols = predicted):")
    print(f"  {'':<14}" + "".join(f"{l:>13}" for l in labels))
    for lab, row in zip(cm["labels"], cm["rows"]):
        print(f"  {lab[:13]:<14}" + "".join(f"{v:>13,}" for v in row))
    print("=" * W)


def print_leakage_comparison(rows: list[dict[str, Any]]) -> None:
    """
    Print a side-by-side comparison of the same model across split strategies.

    Exists to make the effect of customer-level leakage measurable rather than
    a matter of opinion.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One dict per strategy with keys ``strategy``, ``n``, ``accuracy``,
        ``macro_f1``.
    """
    W = 78
    print("=" * W)
    print("  SPLIT-STRATEGY COMPARISON — leakage sensitivity")
    print("=" * W)
    print(f"  {'strategy':<22}{'n':>8}{'accuracy':>12}{'macro-F1':>12}{'delta':>10}")
    print("-" * W)
    base = rows[0]["macro_f1"] if rows else 0.0
    for r in rows:
        delta = r["macro_f1"] - base
        print(
            f"  {r['strategy']:<22}{r['n']:>8,}{r['accuracy']:>12.4f}"
            f"{r['macro_f1']:>12.4f}{delta:>+10.4f}"
        )
    print("-" * W)
    print("  Identical model and features in every row. The only difference is")
    print("  how the held-out batch was selected, so any delta is attributable")
    print("  to the split, not to the model.")
    print("=" * W)
