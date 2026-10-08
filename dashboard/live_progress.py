"""
Ghost Ledger v3 — live progress dashboard (A5).

Run (after the API is up)::

    streamlit run dashboard/live_progress.py --server.address 0.0.0.0 --server.port 8501

Tabs
----
Live Progress         counts, active recoveries, job health
Approval Queue        pending approvals; approve or reject (operator key, server-side)
Failures in Progress  recoveries that hit a failure and are not yet settled

The page refreshes every 2 seconds without a full reload. The Streamlit process
calls the API with the operator key read from the server environment, so the key
is never sent to the browser. If the API is down the page says so and keeps polling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REFRESH_SECONDS: int = 2
REQUEST_TIMEOUT_SECONDS: float = 3.0


def _headers() -> dict[str, str]:
    return {"X-Operator-Key": config.OPERATOR_API_KEY or ""}


def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET from the API. Raises requests.RequestException on failure."""
    resp = requests.get(f"{API_BASE_URL}{path}", params=params, headers=_headers(),
                        timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    """POST to the API. Returns (status code, parsed body) and does not raise on 4xx."""
    resp = requests.post(f"{API_BASE_URL}{path}", json=body, headers=_headers(),
                         timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        payload = resp.json()
    except ValueError:
        payload = {"detail": resp.text}
    return resp.status_code, payload


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """A DataFrame with the chosen columns, present even when there are no rows."""
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def _live_progress_tab(data: dict[str, Any]) -> None:
    counts = data.get("counts", {})
    cols = st.columns(4)
    cols[0].metric("Active recoveries", counts.get("active", 0))
    cols[1].metric("Failures in progress", counts.get("failures_in_progress", 0))
    cols[2].metric("Pending approvals", counts.get("pending_approvals", 0))
    cols[3].metric("Escalated", counts.get("escalated", 0))
    st.caption(f"Updated {data.get('generated_at', '—')} · refresh every {REFRESH_SECONDS}s")

    st.subheader("Active recoveries")
    st.dataframe(
        _frame(data.get("active", []),
               ["recovery_id", "merchant_id", "amount_inr", "cause", "latest_stage", "latest_at"]),
        hide_index=True,
    )

    st.subheader("Background jobs")
    st.dataframe(
        _frame(data.get("jobs", []),
               ["job", "last_status", "last_run_at", "consecutive_failures", "next_allowed_at", "last_error"]),
        hide_index=True,
    )


def _approval_tab(data: dict[str, Any]) -> None:
    rows = data.get("approvals", [])
    st.subheader("Pending approvals")
    st.caption("Recoveries above the auto-approve ceiling wait here for a human decision.")
    st.dataframe(
        _frame(rows, ["approval_id", "recovery_id", "merchant_id", "amount_inr", "cause", "confidence", "created_at"]),
        hide_index=True,
    )
    if not rows:
        st.info("No approvals waiting.")
        return

    st.markdown("**Decide**")
    ids = [r["approval_id"] for r in rows]
    chosen = st.selectbox("Approval", ids, key="approval_choice")
    operator = st.text_input("Your name (recorded on the decision)", key="operator_name", max_chars=120)
    reason = st.text_input("Rejection reason (required to reject)", key="reject_reason", max_chars=500)
    left, right = st.columns(2)
    if left.button("Approve", key="btn_approve", disabled=not operator.strip()):
        _decide(chosen, "approve", {"approved_by": operator.strip()})
    if right.button("Reject", key="btn_reject", disabled=not (operator.strip() and len(reason.strip()) >= 3)):
        _decide(chosen, "reject", {"rejected_by": operator.strip(), "reason": reason.strip()})


def _decide(approval_id: str, verb: str, body: dict[str, Any]) -> None:
    try:
        status, payload = api_post(f"/api/approvals/{approval_id}/{verb}", body)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if status == 200:
        st.success(f"{verb.capitalize()}d {approval_id}.")
    elif status == 409:
        st.warning(f"{approval_id} was already decided. Nothing changed.")
    else:
        st.error(f"Request failed ({status}): {payload.get('detail', payload) if isinstance(payload, dict) else payload}")


def _failures_tab(data: dict[str, Any]) -> None:
    st.subheader("Failures in progress")
    st.caption("Recoveries that hit a failure and are not settled yet. The scheduler retries these with backoff.")
    st.dataframe(
        _frame(data.get("failures", []),
               ["recovery_id", "merchant_id", "amount_inr", "status", "failure_stage", "failure_at"]),
        hide_index=True,
    )


@st.fragment(run_every=REFRESH_SECONDS)
def live_panel() -> None:
    """The auto-refreshing part of the page."""
    try:
        data = api_get("/api/operator/summary")
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_BASE_URL}. Retrying every {REFRESH_SECONDS}s. ({exc})")
        return
    except ValueError:
        st.error("The API returned something that is not JSON. Check the API log.")
        return

    live_tab, approval_tab, failure_tab = st.tabs(["Live Progress", "Approval Queue", "Failures in Progress"])
    with live_tab:
        _live_progress_tab(data)
    with approval_tab:
        _approval_tab(data)
    with failure_tab:
        _failures_tab(data)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Ghost Ledger · Live", layout="wide")
    st.title("Ghost Ledger — live recovery progress")
    if not config.OPERATOR_API_KEY:
        st.warning("OPERATOR_API_KEY is not set. The API will refuse operator requests (503).")
    live_panel()


main()
