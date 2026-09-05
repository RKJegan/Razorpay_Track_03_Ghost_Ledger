"""
Diagnoser precision / recall panel.

Renders the held-out evaluation produced by ``diagnoser/train.py``. Per NFR-004
the evaluation set size (n) and the split method are rendered as part of the
panel — a precision figure without them is treated as a bug, not a metric.

It also surfaces the attribution study, which decomposes how much of the score
comes from the model versus from the raw error code.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metrics import diagnoser_metrics


def render() -> None:
    """Render the diagnoser evaluation and attribution panels."""
    st.subheader("🎯 Diagnoser precision / recall")
    report = diagnoser_metrics()

    if report is None:
        st.warning(
            "No held-out evaluation found. Run `python diagnoser/train.py` "
            "to produce it. Refusing to show an unmeasured accuracy claim."
        )
        return

    n = report["n"]
    st.caption(
        f"**Evaluation set:** held-out split only · **n = {n:,}** failed "
        f"transactions · **Method:** {report['split_method']}"
    )

    rows = []
    for cause, m in report["per_class"].items():
        rows.append(
            {
                "Cause": cause,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "n": m["support"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Macro F1", f"{report['macro']['f1']:.4f}", help=f"On n={n:,} held-out failures")
    c2.metric("Accuracy", f"{report['accuracy']:.4f}", help=f"On n={n:,} held-out failures")
    c3.metric(
        "CV (train only)",
        f"{report.get('cv_macro_f1', 0):.4f}" if report.get("cv_macro_f1") else "n/a",
        help=report.get("cv_type", ""),
    )

    # --- attribution ------------------------------------------------------
    attribution = report.get("attribution")
    if attribution:
        st.markdown("#### How much of that is actually the model?")
        st.write(
            "The error code Razorpay returns is highly informative. This "
            "decomposition shows what the model adds on top of it."
        )
        rows = [
            {
                "Condition": r["name"],
                "Accuracy": round(r["accuracy"], 4),
                "Macro-F1": round(r["macro_f1"], 4),
            }
            for r in attribution
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        lift = attribution[1]["macro_f1"] - attribution[0]["macro_f1"]
        st.caption(
            f"Model lift over a no-ML lookup table: **{lift:+.4f} macro-F1**. "
            f"With the error code withheld entirely the model still reaches "
            f"{attribution[2]['macro_f1']:.4f} from transaction context alone."
        )
