"""
B9 — end-to-end tests for the strategy engine with ENABLE_ADVANCED_STRATEGIES on.

These go through the real entry points: ``submit_recovery`` (the dispatch point),
the retry job, the webhook endpoint, and the scheduler wiring. Razorpay is a
fake client, so the tests are deterministic and make no network calls.

Each test uses its own recovery, its own customer id, and its own far-future
clock window for gateway observations, so tests cannot see each other's data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import config
from agents import scheduled_jobs
from agents.recovery_executor import submit_recovery
from api import scheduler as scheduler_mod
from api import webhooks
from api.main import app
from api.razorpay_client import RazorpayResponse
from database import merchants, recovery_store
from database.engine import session_scope
from database.models import GatewayObservation
from strategies import dunning, runner
from strategies.playbooks import PlaybookLoader
from strategies.runner import handle_failure, retry_one

SECRET = "whsec_test_only_strategies_not_real"
ROOT = Path(config.PROJECT_ROOT)


class FakeRazorpay:
    """Records payment links. Never touches the network."""

    def __init__(self) -> None:
        self.links: list[dict[str, Any]] = []

    def create_payment_link(self, amount_inr, description, customer, notes=None, reference_id=None):  # noqa: ANN001
        link_id = f"plink_s_{uuid.uuid4().hex[:10]}"
        self.links.append({"link_id": link_id, "reference_id": reference_id, "amount": amount_inr})
        return RazorpayResponse(ok=True, data={
            "id": link_id, "short_url": "https://rzp.io/i/strategy", "status": "created",
            "expire_by": None, "simulated": True,
        })

    def fetch_payment_link(self, link_id: str) -> RazorpayResponse:
        return RazorpayResponse(ok=True, data={"id": link_id, "status": "created", "amount_paid": 0,
                                               "payments": []})


@pytest.fixture
def strategies_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ENABLE_ADVANCED_STRATEGIES", True)


@pytest.fixture
def fake() -> FakeRazorpay:
    return FakeRazorpay()


@pytest.fixture(scope="module")
def merchant_id() -> str:
    active = [m["id"] for m in merchants.list_merchants() if m["is_active"]]
    if active:
        return str(active[0])
    merchants.create_merchant("m_test_e2e_strategies", "Strategy e2e")
    return "m_test_e2e_strategies"


def _case(merchant: str, cause: str, amount: float = 499.0) -> str:
    return recovery_store.create_case(
        merchant, f"txn_{uuid.uuid4().hex[:8]}", amount,
        cause=cause, customer_id=f"CUST_E_{uuid.uuid4().hex[:6]}",
    )


def _stages(rid: str) -> list[str]:
    return [e["stage"] for e in recovery_store.get_recovery_timeline(rid)]


def _fail(rid: str, method: str = "card") -> None:
    recovery_store.append_event(rid, "payment_failed", {"payment_method": method}, created_by="test")


def _window(year: int, month: int, day: int, hour: int = 9) -> datetime:
    return datetime(year, month, day, hour, 0, 0)


# --- flag off: the v2 path, unchanged ----------------------------------------------

def test_flag_off_keeps_the_v2_path(merchant_id: str, fake: FakeRazorpay,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ENABLE_ADVANCED_STRATEGIES", False)
    rid = _case(merchant_id, "insufficient_funds")
    result = submit_recovery(rid, client=fake)
    assert result.outcome == "link_created"
    assert "playbook_selected" not in _stages(rid)
    assert "dunning_scheduled" not in _stages(rid)


def test_flag_off_registers_only_the_four_v2_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ENABLE_ADVANCED_STRATEGIES", False)
    assert [name for name, _, _ in scheduler_mod.job_table()] == [
        "settlement_poll", "reconcile", "retry", "expired_link_cleanup",
    ]


def test_flag_on_adds_the_dunning_job(strategies_on: None) -> None:
    names = [name for name, _, _ in scheduler_mod.job_table()]
    assert names[-1] == "dunning" and len(names) == 5


# --- B2/B3: timing decided by the router, before any link ---------------------------

def test_funds_failure_in_the_cash_crunch_waits_and_then_retries(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "insufficient_funds", amount=300.0)
    result = handle_failure(rid, client=fake, now=_window(2026, 10, 27, 10))
    assert result.outcome == "retry_scheduled"
    assert fake.links == [], "no link may be created while the retry is deferred"

    stages = _stages(rid)
    assert "playbook_selected" in stages and "retry_scheduled" in stages
    assert "payment_link_created" not in stages
    assert stages.count("dunning_scheduled") == 1
    assert len(dunning.list_touches(rid)) == 3

    assert retry_one(rid, client=fake, now=_window(2026, 10, 27, 12)) == "not_due"
    assert fake.links == []

    # Due on 2 Nov at 10:00: the rule runs now and the attempt is allowed.
    assert retry_one(rid, client=fake, now=_window(2026, 11, 2, 10)) == "link_created"
    assert len(fake.links) == 1


def test_gateway_timeout_in_peak_hours_is_pushed_out_of_the_window(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "gateway_timeout", amount=250.0)
    result = handle_failure(rid, client=fake, now=_window(2033, 3, 4, 12))
    assert result.outcome == "retry_scheduled"
    timeline = recovery_store.get_recovery_timeline(rid)
    scheduled = next(e for e in timeline if e["stage"] == "retry_scheduled")
    assert scheduled["detail"]["run_at"] == "2033-03-04 15:00:00"


# --- B4/B7: failover and method suggestion -----------------------------------------

def _seed_health(method: str, successes: int, failures: int, at: datetime) -> None:
    mon = runner.GatewayHealthMonitor()
    for _ in range(successes):
        mon.record(method, True, now=at)
    for _ in range(failures):
        mon.record(method, False, now=at)


def test_degraded_card_route_fails_over_to_a_healthy_method(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    at = _window(2044, 5, 1, 8).replace(minute=55)  # inside the 60-minute window before 09:00
    _seed_health("card", successes=0, failures=25, at=at)
    _seed_health("upi", successes=25, failures=0, at=at)

    rid = _case(merchant_id, "gateway_timeout", amount=180.0)
    result = handle_failure(rid, client=fake, now=_window(2044, 5, 1, 9))
    assert result.outcome == "link_created"
    failover = next(e for e in recovery_store.get_recovery_timeline(rid) if e["stage"] == "gateway_failover")
    assert failover["detail"]["to"] == "upi"


def test_card_failures_trigger_a_method_suggestion_on_the_next_attempt(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "card_expired", amount=640.0)
    assert handle_failure(rid, client=fake, now=_window(2045, 6, 1, 9)).outcome == "link_created"
    _fail(rid, "card")
    assert "method_suggested" not in _stages(rid)  # one failure: below the threshold of two

    _fail(rid, "card")
    assert handle_failure(rid, client=fake, now=_window(2045, 6, 2, 9)).outcome == "link_created"
    suggestion = next(e for e in recovery_store.get_recovery_timeline(rid) if e["stage"] == "method_suggested")
    assert suggestion["detail"]["methods"] == ["upi", "netbanking"]
    assert len(fake.links) == 2


# --- B1/B2: dunning-only causes and the policy gate ----------------------------------

def test_lapsed_mandate_sends_reminders_and_never_creates_a_link(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "mandate_lapsed", amount=999.0)
    result = handle_failure(rid, client=fake, now=_window(2046, 7, 1, 9))
    assert result.outcome == "dunning_only"
    assert fake.links == []
    assert len(dunning.list_touches(rid)) == 2


def test_policy_stop_is_returned_before_the_router_runs(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "insufficient_funds", amount=400.0)
    for _ in range(3):
        _fail(rid, "card")
    result = handle_failure(rid, client=fake, now=_window(2047, 8, 1, 9))
    assert result.outcome == "stopped"
    stages = _stages(rid)
    assert "playbook_selected" not in stages
    assert "stopping_rule_triggered" in stages
    assert fake.links == []


def test_amount_above_the_ceiling_needs_approval_before_any_strategy(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "card_expired", amount=15_000.0)
    result = handle_failure(rid, client=fake, now=_window(2048, 9, 1, 9))
    assert result.outcome == "awaiting_approval"
    assert "playbook_selected" not in _stages(rid)
    assert fake.links == []

    # An approval lifts only R1. The strategy then runs as normal.
    approved = handle_failure(rid, client=fake, approved=True, approval_id="apr_test_1",
                              now=_window(2048, 9, 1, 10))
    assert approved.outcome == "link_created"
    assert "playbook_selected" in _stages(rid)


def test_submit_recovery_dispatches_to_the_strategy_layer(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "card_expired", amount=350.0)
    result = submit_recovery(rid, client=fake)
    assert result.outcome == "link_created"
    assert "playbook_selected" in _stages(rid)


# --- fallback: a broken playbook never blocks recovery --------------------------------

def test_a_broken_playbook_falls_back_to_the_v2_path(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None, tmp_path: Path
) -> None:
    empty = tmp_path / "no_playbooks"
    empty.mkdir()
    rid = _case(merchant_id, "card_expired", amount=220.0)
    result = handle_failure(rid, client=fake, loader=PlaybookLoader(empty), now=_window(2049, 10, 1, 9))
    assert result.outcome == "link_created"
    selected = next(e for e in recovery_store.get_recovery_timeline(rid) if e["stage"] == "playbook_selected")
    assert selected["detail"]["fallback"] == "v2_immediate"


# --- B8: the retry job and the webhook hook -----------------------------------------

def test_retry_job_uses_the_strategy_pass_when_on(
    merchant_id: str, fake: FakeRazorpay, strategies_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _case(merchant_id, "insufficient_funds", amount=310.0)
    handle_failure(rid, client=fake, now=_window(2050, 11, 27, 10))  # schedules a retry for 2050-12-02
    # Limit the pass to this recovery so the test does not touch other tests' data.
    only_this = [c for c in recovery_store.get_active_recoveries() if c["recovery_id"] == rid]
    monkeypatch.setattr(recovery_store, "get_active_recoveries", lambda merchant_id=None: only_this)
    counts = scheduled_jobs.retry_due_recoveries(client=fake)
    assert counts == {"not_due": 1}, "the scheduled retry is in the future relative to the wall clock"


def _signed(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    return body, hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _captured(recovery_id: str, amount_paise: int, method: str) -> dict[str, Any]:
    return {
        "entity": "event", "event": "payment.captured", "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": f"pay_{uuid.uuid4().hex[:14]}", "amount": amount_paise, "currency": "INR",
            "status": "captured", "method": method,
            "notes": {"recovery_id": recovery_id, "txn_id": "txn_strategy_webhook"},
        }}},
        "created_at": 1760000000,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(webhooks, "RAZORPAY_WEBHOOK_SECRET", SECRET)
    with TestClient(app) as c:
        yield c


def _post_capture(client: TestClient, rid: str, method: str) -> Any:
    body, signature = _signed(_captured(rid, 49900, method))
    return client.post("/webhooks/razorpay", content=body, headers={
        "Content-Type": "application/json", "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": f"evt_{uuid.uuid4().hex[:12]}",
    })


def test_capture_webhook_settles_cancels_dunning_and_records_gateway_health(
    client: TestClient, merchant_id: str, fake: FakeRazorpay, strategies_on: None
) -> None:
    rid = _case(merchant_id, "card_expired", amount=499.0)
    handle_failure(rid, client=fake)  # real clock: touches are due at once, not yet sent
    assert len(dunning.list_touches(rid)) == 3

    response = _post_capture(client, rid, "upi")
    assert response.status_code == 200

    assert {t["status"] for t in dunning.list_touches(rid)} == {"cancelled"}
    with session_scope() as session:
        observed = session.scalar(
            select(func.count()).select_from(GatewayObservation)
            .where(GatewayObservation.recovery_id == rid, GatewayObservation.method == "upi",
                   GatewayObservation.success == 1)
        )
    assert observed == 1
    captured = next(e for e in recovery_store.get_recovery_timeline(rid) if e["stage"] == "payment_captured")
    assert captured["detail"]["payment_method"] == "upi"


def test_capture_webhook_with_strategies_off_records_no_health(
    client: TestClient, merchant_id: str, fake: FakeRazorpay, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENABLE_ADVANCED_STRATEGIES", False)
    rid = _case(merchant_id, "card_expired", amount=499.0)
    assert _post_capture(client, rid, "upi").status_code == 200
    with session_scope() as session:
        observed = session.scalar(
            select(func.count()).select_from(GatewayObservation)
            .where(GatewayObservation.recovery_id == rid)
        )
    assert observed == 0
    assert recovery_store.get_recovery_status(rid)["status"] == "settled"


# --- trust boundary and wiring --------------------------------------------------------

LLM_MODULES = ("openai", "ollama", "anthropic", "langchain", "google.generativeai", "groq")


def test_no_strategy_module_imports_an_llm_client() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "strategies").glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*(from|import)\s+([\w\.]+)", line)
            if match and any(match.group(2).startswith(m) for m in LLM_MODULES):
                offenders.append(f"{path.name}: {line.strip()}")
    assert offenders == []


def test_dry_run_summary_describes_every_cause() -> None:
    lines = runner.dry_run_summary()
    assert len(lines) == len(config.CAUSE_BUCKETS)
    assert any(line.startswith("mandate_lapsed: ") and "route=dunning_only" in line for line in lines)
    assert any("timing=cycle_aware_funds" in line for line in lines)


def test_playbook_templates_only_use_the_three_allowed_fields() -> None:
    allowed = {"amount_inr", "txn_id", "next_step"}
    for path in (ROOT / "playbooks").glob("*.yaml"):
        for field in re.findall(r"\{(\w+)\}", path.read_text(encoding="utf-8")):
            assert field in allowed, f"{path.name} uses {field}"
