"""
Filterable audit-trail table.

Shows every decision the system made — including the ones that were refused.
The point of the audit trail is that denied actions are as visible as executed
ones, so the filter defaults to showing all components rather than only
successes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.audit_trail import COMPONENTS, recent


def render(limit: int = 400) -> None:
    """
    Render the audit trail with component and outcome filters.

    Parameters
    ----------
    limit : int, optional
        Maximum rows to load.
    """
    st.subheader("📜 Audit trail")

    c1, c2 = st.columns([2, 1])
    with c1:
        component = st.selectbox(
            "Component",
            options=["(all)"] + list(COMPONENTS),
            index=0,
            key="audit_component",
        )
    with c2:
        success_filter = st.selectbox(
            "Outcome",
            options=["(all)", "success only", "denied / failed only"],
            index=0,
            key="audit_outcome",
        )

    rows = recent(limit=limit, component=None if component == "(all)" else component)
    if not rows:
        st.info("No audit records yet. Run `python main.py` first.")
        return

    df = pd.DataFrame(rows)
    if success_filter == "success only":
        df = df[df["success"] == 1]
    elif success_filter == "denied / failed only":
        df = df[df["success"] == 0]

    display = df[
        [c for c in ("timestamp", "component", "action", "decision_reason", "success")
         if c in df.columns]
    ].rename(
        columns={
            "timestamp": "Timestamp",
            "component": "Component",
            "action": "Action",
            "decision_reason": "Decision / reason",
            "success": "OK",
        }
    )
    st.caption(f"Showing {len(display):,} of {len(df):,} loaded records (newest first).")
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("Inspect a record's full payload"):
        if len(df):
            idx = st.number_input(
                "Row index", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1,
                key="audit_row_idx",
            )
            st.json(df.iloc[int(idx)].to_dict(), expanded=False)
