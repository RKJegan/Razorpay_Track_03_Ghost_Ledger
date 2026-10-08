"""
Tests for the A7 end-to-end script.

* The preflight refuses live keys and missing setup (no network needed).
* The OFFLINE mode runs the full flow for real: a uvicorn server in a subprocess,
  the simulator for links, and correctly signed webhooks posted over HTTP.
  This proves the wiring. The live Razorpay Test Mode run is done by the user.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

import config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "e2e_test_mode.py"
WEBHOOK_SECRET = "whsec_test_only_e2e_not_real"


def _load_script():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("e2e_test_mode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --- preflight -----------------------------------------------------------------

def test_live_mode_refuses_a_live_key(monkeypatch: pytest.MonkeyPatch) -> None:
    e2e = _load_script()
    monkeypatch.setattr(config, "RAZORPAY_WEBHOOK_SECRET", "whsec_x")
    monkeypatch.setattr(config, "RAZORPAY_LIVE_TEST_MODE", True)
    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "rzp_live_should_never_be_used")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "s")
    monkeypatch.setattr(e2e.requests, "get", lambda *a, **k: type("R", (), {"status_code": 200})())
    problems = e2e.preflight("live")
    assert any("not a test key" in p for p in problems)


def test_live_mode_needs_the_flag_and_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    e2e = _load_script()
    monkeypatch.setattr(config, "RAZORPAY_WEBHOOK_SECRET", "")
    monkeypatch.setattr(config, "RAZORPAY_LIVE_TEST_MODE", False)
    monkeypatch.setattr(config, "RAZORPAY_KEY_ID", "")
    monkeypatch.setattr(config, "RAZORPAY_KEY_SECRET", "")
    monkeypatch.setattr(e2e.requests, "get", lambda *a, **k: type("R", (), {"status_code": 200})())
    problems = e2e.preflight("live")
    assert any("RAZORPAY_WEBHOOK_SECRET" in p for p in problems)
    assert any("RAZORPAY_LIVE_TEST_MODE" in p for p in problems)
    assert any("RAZORPAY_KEY_ID" in p for p in problems)


def test_refuses_an_amount_that_would_need_approval() -> None:
    result = subprocess_run(["--mode", "offline", "--amount", "15000"])
    assert result.returncode == 2
    assert "auto-approve ceiling" in result.stdout


def subprocess_run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env,
    )


# --- the offline flow, for real -------------------------------------------------

@pytest.fixture(scope="module")
def server() -> str:
    """Start the API in a subprocess on the test database, with the test webhook secret."""
    port = _free_port()
    env = dict(os.environ)
    env.update({
        "DB_PATH": "data/test_ghost_ledger.db",
        "RAZORPAY_WEBHOOK_SECRET": WEBHOOK_SECRET,
        "RAZORPAY_LIVE_TEST_MODE": "0",
        "SCHEDULER_ENABLED": "0",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.3)
        else:
            raise RuntimeError("API did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_offline_flow_passes_end_to_end(server: str, tmp_path: Path) -> None:
    report_path = tmp_path / "e2e.json"
    env = dict(os.environ)
    env.update({
        "DB_PATH": "data/test_ghost_ledger.db",
        "RAZORPAY_WEBHOOK_SECRET": WEBHOOK_SECRET,
        "API_BASE_URL": server,
    })
    result = subprocess_run(["--mode", "offline", "--report", str(report_path)], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["failure reached the webhook and was recorded"]
    assert names["next attempt created"]
    assert names["recovery settled"]
    assert names["paid amount matches recovery amount"]
    assert names["timeline hides internal fields"]
    assert names["wrong key cannot read it"]
    # The run must never print the webhook secret or a merchant key.
    assert WEBHOOK_SECRET not in result.stdout
    assert "Bearer" not in result.stdout
