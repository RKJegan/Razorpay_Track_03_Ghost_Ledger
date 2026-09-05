"""
FR-009 — Weekly Recovery Report (P1).

Generates a plain-text/markdown summary of recovery activity: rupees
recovered, cases stopped by policy, and the top cause buckets.

Every figure states its basis. The report is generated from ``metrics.py``,
the same source of truth as the dashboard, so the numbers cannot disagree.

Usage
-----
    python weekly_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPORTS_DIR  # noqa: E402
from metrics import (  # noqa: E402
    autopsy_stats,
    cause_breakdown,
    compute_headline_metrics,
    diagnoser_metrics,
    recovery_over_time,
    stopping_rule_events,
)


def _inr(value: float) -> str:
    """Format a float as an Indian-rupee string."""
    return f"₹{value:,.2f}"


def build_report() -> str:
    """
    Assemble the weekly report as markdown.

    Returns
    -------
    str
        The full report text.
    """
    m = compute_headline_metrics()
    causes = cause_breakdown()
    series = recovery_over_time()
    stops = stopping_rule_events(limit=5)
    diag = diagnoser_metrics()
    autopsy = autopsy_stats()

    lines: list[str] = []
    lines.append("# Ghost Ledger v2 — Weekly Recovery Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    # --- headline ---------------------------------------------------------
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Recovered:** {_inr(m['amount_recovered_inr'])}")
    lines.append(f"- **At risk:** {_inr(m['amount_at_risk_inr'])}")
    lines.append(f"- **Recovery rate:** {m['recovery_rate']:.2%}")
    lines.append(f"- **Transactions in batch:** {m['n_transactions']:,}")
    lines.append(f"- **Failures in batch:** {m['n_failures']:,}")
    lines.append(f"- **Recovery actions taken:** {m['recovery_actions']:,}")
    lines.append(f"- **Audit records written:** {m['audit_records']:,}")
    lines.append("")
    lines.append(f"**Basis:** {m['basis']} · N = {m['n_transactions']:,} transactions")
    lines.append("")
    lines.append(f"**Split method:** {m['split_method']}")
    lines.append("")
    lines.append(
        f"**Reconciliation:** {'PASS' if m['reconciled'] else 'FAIL'} — headline "
        f"figures match the audit-trail totals."
    )
    lines.append("")

    # --- outcomes ---------------------------------------------------------
    lines.append("## Action outcomes")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("|---|---:|")
    for k, v in sorted(m["outcomes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} |")
    lines.append("")

    # --- stopping rules ---------------------------------------------------
    lines.append("## Stopping-rule events (cases escalated, not retried)")
    lines.append("")
    lines.append(f"**{m['stopping_rule_events']:,}** cases hit the 3-failed-attempt rule.")
    lines.append("")
    if stops:
        lines.append("Most recent:")
        lines.append("")
        for s in stops:
            lines.append(
                f"- `{s['failure_id']}` — customer `{s['customer_id']}`, "
                f"cause `{s['predicted_cause']}`, "
                f"{_inr(float(s['estimated_value']))} at risk"
            )
        lines.append("")
        lines.append(f"_Reason recorded:_ {stops[0]['stopping_reason']}")
        lines.append("")

    # --- causes -----------------------------------------------------------
    lines.append("## Top cause buckets")
    lines.append("")
    lines.append("| Cause | Failures | At risk | Recovered | Rate | Stopped |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in causes:
        lines.append(
            f"| {c['cause']} | {c['failures']:,} | {_inr(c['at_risk'])} | "
            f"{_inr(c['recovered'])} | {c['recovery_rate']:.1%} | {c['stops']:,} |"
        )
    lines.append("")

    # --- diagnoser --------------------------------------------------------
    lines.append("## Diagnoser performance")
    lines.append("")
    if diag:
        lines.append(
            f"Held-out **n = {diag['n']:,}** failed transactions · "
            f"macro-F1 **{diag['macro']['f1']:.4f}** · "
            f"accuracy **{diag['accuracy']:.4f}**"
        )
        lines.append("")
        lines.append("| Cause | Precision | Recall | F1 | n |")
        lines.append("|---|---:|---:|---:|---:|")
        for cause, v in diag["per_class"].items():
            lines.append(
                f"| {cause} | {v['precision']:.4f} | {v['recall']:.4f} | "
                f"{v['f1']:.4f} | {v['support']:,} |"
            )
        lines.append("")
        if diag.get("attribution"):
            a = diag["attribution"]
            lines.append(
                f"_Attribution:_ model lift over a no-ML error-code lookup is "
                f"**{a[1]['macro_f1'] - a[0]['macro_f1']:+.4f}** macro-F1; "
                f"with the error code withheld the model still reaches "
                f"**{a[2]['macro_f1']:.4f}**."
            )
            lines.append("")
    else:
        lines.append("_No held-out evaluation found. Run `python diagnoser/train.py`._")
        lines.append("")

    # --- activity ---------------------------------------------------------
    if series:
        lines.append("## Activity window")
        lines.append("")
        lines.append(
            f"{len(series)} days covered "
            f"({series[0]['day']} → {series[-1]['day']})."
        )
        lines.append("")

    lines.append(f"## Autopsy reports")
    lines.append("")
    lines.append(f"{autopsy['total']:,} explanations generated "
                 f"({', '.join(f'{k}: {v:,}' for k, v in autopsy['models'].items())}).")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_All figures are computed from the same source of truth as the "
        "dashboard (`metrics.py`). Recovery settlement in this environment is "
        "simulated unless Razorpay test credentials are configured._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """
    Write the weekly report to ``reports/weekly_recovery_report.md``.

    Returns
    -------
    int
        Exit code.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "weekly_recovery_report.md"
    text = build_report()
    path.write_text(text, encoding="utf-8")
    print(f"[report] wrote {path} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
