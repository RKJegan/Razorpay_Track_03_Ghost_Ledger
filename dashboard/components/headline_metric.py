"""
Headline figures: Recovered / At Risk / N transactions, on the held-out batch.

Every number comes from ``metrics.compute_headline_metrics`` — the same call
the pipeline makes — and the basis ("held-out batch", n, split method) is
rendered directly beneath the figures, because NFR-004 forbids a bare metric.
"""

from __future__ import annotations

import streamlit as st

from metrics import compute_headline_metrics


def _inr(value: float) -> str:
    """Format a float as an Indian-rupee string with thousands separators."""
    return f"₹{value:,.0f}"


def render() -> dict:
    """
    Render the headline metric row.

    Returns
    -------
    dict
        The metrics dict, so callers can reuse it without a second query.
    """
    m = compute_headline_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "₹ Recovered",
        _inr(m["amount_recovered_inr"]),
        help="Sum of recovered_amount across all recovery actions on the held-out batch.",
    )
    col2.metric(
        "₹ At Risk",
        _inr(m["amount_at_risk_inr"]),
        help="Sum of estimated_value for every failed transaction in the held-out batch.",
    )
    col3.metric(
        "N Transactions",
        f"{m['n_transactions']:,}",
        help="All transactions (successful and failed) in the held-out batch.",
    )
    col4.metric(
        "Recovery Rate",
        f"{m['recovery_rate']:.1%}",
        help="Recovered / At Risk. Money actually recovered, not attempts made.",
    )

    st.caption(
        f"**Basis:** {m['basis']} · N transactions = {m['n_transactions']:,} · "
        f"N failures = {m['n_failures']:,} · "
        f"recovery actions = {m['recovery_actions']:,} · "
        f"audit records = {m['audit_records']:,}"
    )

    if m["reconciled"]:
        st.caption(
            "✅ Headline reconciles with the audit-trail totals "
            "(both recomputed from the same source of truth)."
        )
    else:
        st.error(
            "❌ Headline does NOT reconcile with the audit trail. "
            "Do not present these numbers until this is resolved."
        )

    with st.expander("What is the held-out batch, and how was it chosen?"):
        st.write(m["split_method"])
    return m
