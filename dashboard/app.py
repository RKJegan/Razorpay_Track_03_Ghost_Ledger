"""
FR-007 — Recovery Dashboard.

    streamlit run dashboard/app.py

Reads exclusively from SQLite via ``metrics.py``, the same module the pipeline
uses, so the headline figures cannot drift from the audit trail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from dashboard.components import (  # noqa: E402
    audit_trail_table,
    headline_metric,
    precision_recall_panel,
    stopping_rule_panel,
)
from metrics import (  # noqa: E402
    autopsy_stats,
    cause_breakdown,
    recovery_over_time,
)

st.set_page_config(
    page_title="Ghost Ledger v2 — AI Revenue Recovery",
    page_icon="👻",
    layout="wide",
)


def _simulated_banner() -> None:
    """Warn clearly when settlement came from the simulator, not live Razorpay."""
    from config import RAZORPAY_LIVE_TEST_MODE

    if not RAZORPAY_LIVE_TEST_MODE:
        st.warning(
            "**Simulated settlement.** No Razorpay test credentials are "
            "configured, so recovery outcomes come from the offline simulator "
            "(`SimulatedRazorpayClient`). Response shapes, cause-dependent "
            "success rates and injected timeouts are modelled, but the rupee "
            "figures are **not** live Razorpay results. Set "
            "`RAZORPAY_LIVE_TEST_MODE=1` with `rzp_test_*` keys to switch.",
            icon="⚠️",
        )


def main() -> None:
    """Render the dashboard."""
    st.title("👻 Ghost Ledger v2")
    st.subheader("Detect revenue at risk → diagnose → act → report the recovered ₹, honestly.")

    _simulated_banner()

    # --- headline ---------------------------------------------------------
    st.markdown("---")
    headline_metric.render()

    # --- recovery trend ---------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Recovery over time")
    series = recovery_over_time()
    if series:
        df = pd.DataFrame(series)
        chart = (
            df.set_index("day")[["recovered", "at_risk"]]
            .rename(columns={"recovered": "Recovered ₹", "at_risk": "At risk ₹"})
        )
        st.line_chart(chart)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Recovery rate by day")
            st.bar_chart(df.set_index("day")["recovery_rate"])
        with col_b:
            st.caption("Attempts and policy stops by day")
            st.bar_chart(df.set_index("day")[["attempts", "stops"]])
    else:
        st.info("No recovery activity yet. Run `python main.py` first.")

    # --- cause breakdown --------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Recovery by diagnosed cause")
    causes = cause_breakdown()
    if causes:
        cdf = pd.DataFrame(causes)
        st.dataframe(
            cdf.rename(
                columns={
                    "cause": "Cause",
                    "failures": "Failures",
                    "at_risk": "At risk ₹",
                    "recovered": "Recovered ₹",
                    "recovery_rate": "Recovery rate",
                    "stops": "Stopped",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- stopping rules ---------------------------------------------------
    st.markdown("---")
    stopping_rule_panel.render()

    # --- diagnoser metrics ------------------------------------------------
    st.markdown("---")
    precision_recall_panel.render()

    # --- autopsy ----------------------------------------------------------
    st.markdown("---")
    st.subheader("🧾 Autopsy reports")
    stats = autopsy_stats()
    if stats["total"]:
        st.caption(
            f"{stats['total']:,} reports stored · models: "
            + ", ".join(f"{k} ({v:,})" for k, v in stats["models"].items())
        )
        if stats["sample"]:
            st.markdown(f"**Sample — {stats['sample']['failure_id']}**")
            st.write(stats["sample"]["report_text"])
            st.caption(f"Basis: {stats['sample']['basis']}")
    else:
        st.info("No autopsy reports yet.")

    # --- audit ------------------------------------------------------------
    st.markdown("---")
    audit_trail_table.render()

    st.markdown("---")
    st.caption(
        "Ghost Ledger v2 · Razorpay AI Buildathon — Track 03: AI Revenue "
        "Recovery · Every figure states its basis; every financial action is "
        "policy-checked and logged before execution."
    )


if __name__ == "__main__":
    main()
