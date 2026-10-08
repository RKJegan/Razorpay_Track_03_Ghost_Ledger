"""
Smoke tests for the A5 live dashboard using Streamlit's AppTest.

No browser and no running API: requests is replaced with a fake, so the page
logic runs in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

import config

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "live_progress.py")


class _Resp:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


SUMMARY = {
    "generated_at": "2026-10-08 12:00:00",
    "counts": {"active": 2, "failures_in_progress": 1, "pending_approvals": 1, "escalated": 0},
    "active": [{"recovery_id": "rcv_1", "merchant_id": "m1", "amount_inr": 499.0, "cause": "card_expired",
                "latest_stage": "payment_link_created", "latest_at": "2026-10-08 11:59:00"}],
    "failures": [{"recovery_id": "rcv_2", "merchant_id": "m1", "amount_inr": 900.0, "status": "pending",
                  "failure_stage": "payment_failed", "failure_at": "2026-10-08 11:58:00"}],
    "approvals": [{"approval_id": "apr_1", "recovery_id": "rcv_3", "merchant_id": "m1", "amount_inr": 15000.0,
                   "cause": "card_expired", "confidence": 0.9, "created_at": "2026-10-08 11:50:00"}],
    "jobs": [{"job": "settlement_poll", "last_status": "ok", "last_run_at": "2026-10-08 11:55:00",
              "consecutive_failures": 0, "next_allowed_at": None, "last_error": None}],
}


def test_page_shows_a_message_when_the_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_a: Any, **_k: Any) -> None:
        raise requests.ConnectionError("connection refused (fake)")

    monkeypatch.setattr(requests, "get", down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_page_renders_all_three_tabs_from_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, params=None, headers=None, timeout=None):  # noqa: ANN001
        calls.append(url)
        assert headers and "X-Operator-Key" in headers  # the key is sent server-side
        return _Resp(200, SUMMARY)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(config, "OPERATOR_API_KEY", "op_test_key_for_dashboard")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert calls and calls[0].endswith("/api/operator/summary")
    labels = [t.label for t in at.tabs]
    assert labels == ["Live Progress", "Approval Queue", "Failures in Progress"]
    assert [m.label for m in at.metric][:2] == ["Active recoveries", "Failures in progress"]


def test_approval_decision_handles_a_conflict_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, SUMMARY))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(409, {"detail": "already decided"}))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="operator_name").input("ops.test").run()
    at.button(key="btn_approve").click().run()
    assert not at.exception
    assert any("already decided" in w.value for w in at.warning)
