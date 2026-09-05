"""
Stopping-rule panel.

Makes the system's self-limitation visible. A demo that only shows successful
recoveries is showing half a system; the point of this panel is that the
agent knows when to stop.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metrics import stopping_rule_events, count_stopping_rule_events


def render(limit: int = 100) -> None:
    """
    Render the stopping-rule events panel.

    Parameters
    ----------
    limit : int, optional
        Maximum events to display.
    """
    total = count_stopping_rule_events()
    events = stopping_rule_events(limit=limit)
    st.subheader("🛑 Stopping-rule events")
    st.write(
        "Cases where the policy engine halted recovery after **3 failed attempts**. "
        "The agent does not retry a 4th time, does not escalate to itself, and "
        "cannot be overridden — the STOP is final and logged."
    )

    if not total:
        st.info(
            "No stopping-rule events in this batch. Every failure either "
            "recovered or was blocked before exhausting its attempts."
        )
        return

    df = pd.DataFrame(events)
    st.metric("Cases stopped by policy", f"{total:,}")
    if total > len(events):
        st.caption(f"Showing the latest {len(events):,} of {total:,} in the table below.")

    reason = (
        df.iloc[0]["stopping_reason"]
        if "stopping_reason" in df.columns and df.iloc[0]["stopping_reason"]
        else "No reason recorded."
    )
    st.code(reason, language=None)

    show = df[
        [
            c
            for c in (
                "failure_id",
                "customer_id",
                "agent_name",
                "predicted_cause",
                "attempt_number",
                "estimated_value",
                "txn_type",
            )
            if c in df.columns
        ]
    ].rename(
        columns={
            "failure_id": "Failure",
            "customer_id": "Customer",
            "agent_name": "Agent",
            "predicted_cause": "Cause",
            "attempt_number": "Attempt",
            "estimated_value": "At risk (₹)",
            "txn_type": "Type",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)
