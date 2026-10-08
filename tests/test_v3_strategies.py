"""
Unit tests for Track B (B1-B7): playbooks, router, timing, gateway health,
dunning, A/B tests, method suggestion, and migration re-runs.

Every test is deterministic: fixed clock values, unique ids per test, and no
network. Database-backed tests use the copied test database.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import config
from database import merchants, recovery_store
from database.migrations import run_migrations
from strategies import dunning
from strategies.ab_test import (
    CONTROL,
    TREATMENT,
    VERDICT_INSUFFICIENT,
    VERDICT_NO_DIFFERENCE,
    VERDICT_SIGNIFICANT,
    ABTestManager,
    bucket_of,
    two_proportion_z,
    variant_for_bucket,
)
from strategies.gateway import (
    DEGRADED,
    HEALTHY,
    UNKNOWN,
    GatewayHealthMonitor,
    HealthSnapshot,
    decide_failover,
)
from strategies.methods import suggest_methods
from strategies.playbooks import PlaybookError, PlaybookLoader, load_playbook_set
from strategies.router import ACTION_CREATE_LINK, ACTION_DUNNING_ONLY, ACTION_SCHEDULE_RETRY, RouteContext, route
from strategies.timing import apply_rule, cycle_aware_funds, is_peak_hour, peak_avoidance

SHIPPED = Path(config.PLAYBOOK_DIR)


def _merchant_id() -> str:
    active = [m["id"] for m in merchants.list_merchants() if m["is_active"]]
    if active:
        return str(active[0])
    merchants.create_merchant("m_test_strategy", "Strategy tests")  # returns the API key: not kept
    return "m_test_strategy"


def _recovery(cause: str, amount: float = 499.0) -> str:
    return recovery_store.create_case(
        _merchant_id(), f"txn_{uuid.uuid4().hex[:8]}", amount,
        cause=cause, customer_id=f"CUST_U_{uuid.uuid4().hex[:6]}",
    )


def _copy_playbooks(tmp_path: Path) -> Path:
    target = tmp_path / "playbooks"
    shutil.copytree(SHIPPED, target)
    return target


# --- B1: playbooks -------------------------------------------------------------

def test_shipped_playbooks_cover_every_cause_and_validate() -> None:
    books = load_playbook_set(SHIPPED)
    assert set(books) == set(config.CAUSE_BUCKETS)
    assert books["mandate_lapsed"].route == "dunning_only"
    assert books["insufficient_funds"].timing_rule == "cycle_aware_funds"
    assert books["gateway_timeout"].failover_enabled is True


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        ("card_expired.yaml", "route: create_link", "route: magic"),
        ("card_expired.yaml", "timing_rule: none", "timing_rule: vibes"),
        ("card_expired.yaml", "{txn_id}", "{customer_name}"),
        ("card_expired.yaml", "offset_minutes: 4320", "offset_minutes: 10"),
        ("card_expired.yaml", "template: card_final}", "template: nope}"),
        ("card_expired.yaml", "channel: email, template: card_first", "channel: pigeon, template: card_first"),
        ("card_expired.yaml", "version: 1\n", "version: 1\nsurprise: 1\n"),
        ("card_expired.yaml", "candidates: [upi, netbanking]", "candidates: [bitcoin]"),
        ("card_expired.yaml", "cause: card_expired", "cause: made_up_cause"),
        ("gateway_timeout.yaml", "candidates: [upi, netbanking]\nmethod_suggestion", "candidates: []\nmethod_suggestion"),
    ],
)
def test_invalid_playbook_is_rejected(tmp_path: Path, file: str, old: str, new: str) -> None:
    directory = _copy_playbooks(tmp_path)
    text = (directory / file).read_text(encoding="utf-8")
    assert old in text, f"test setup: {old!r} not in {file}"
    (directory / file).write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(PlaybookError):
        load_playbook_set(directory)


def test_invalid_yaml_and_missing_cause_are_rejected(tmp_path: Path) -> None:
    directory = _copy_playbooks(tmp_path)
    (directory / "card_expired.yaml").write_text("cause: [unclosed", encoding="utf-8")
    with pytest.raises(PlaybookError, match="invalid YAML"):
        load_playbook_set(directory)

    directory = _copy_playbooks(tmp_path / "b")
    (directory / "mandate_lapsed.yaml").unlink()
    with pytest.raises(PlaybookError, match="no playbook for causes"):
        load_playbook_set(directory)


def test_hot_reload_applies_valid_edit_and_keeps_last_good_on_invalid(tmp_path: Path) -> None:
    directory = _copy_playbooks(tmp_path)
    loader = PlaybookLoader(directory)
    loader.load()
    assert loader.get("card_expired").version == 1

    path = directory / "card_expired.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "version: 1", "version: 2\n# edited with a longer line so the file size changes too"),
        encoding="utf-8")
    assert loader.get("card_expired").version == 2

    path.write_text("cause: card_expired\nversion: 3\nroute: nonsense\n", encoding="utf-8")
    assert loader.get("card_expired").version == 2, "last good set must keep running"
    assert loader.last_error is not None


# --- B2: router ----------------------------------------------------------------

def _ctx(cause: str, when: datetime, **kw: Any) -> RouteContext:
    return RouteContext(cause=cause, now=when, attempt_number=1, card_failures=kw.pop("card_failures", 0),
                        last_method=kw.pop("last_method", None), **kw)


def test_router_is_deterministic_and_explains_itself() -> None:
    book = load_playbook_set(SHIPPED)["insufficient_funds"]
    ctx = _ctx("insufficient_funds", datetime(2026, 10, 27, 10, 0))
    first, second = route(book, ctx), route(book, ctx)
    assert first == second
    assert first.reasons and any("timing rule" in r for r in first.reasons)


def test_router_refuses_a_playbook_for_another_cause() -> None:
    book = load_playbook_set(SHIPPED)["card_expired"]
    with pytest.raises(ValueError):
        route(book, _ctx("insufficient_funds", datetime(2026, 10, 1, 9)))


def test_mandate_lapsed_never_creates_a_link() -> None:
    book = load_playbook_set(SHIPPED)["mandate_lapsed"]
    plan = route(book, _ctx("mandate_lapsed", datetime(2026, 10, 1, 9)))
    assert plan.action == ACTION_DUNNING_ONLY
    assert len(plan.touches) == 2


def test_funds_retry_is_deferred_in_the_cash_crunch_and_immediate_otherwise() -> None:
    book = load_playbook_set(SHIPPED)["insufficient_funds"]
    late = route(book, _ctx("insufficient_funds", datetime(2026, 10, 27, 10, 0)))
    assert late.action == ACTION_SCHEDULE_RETRY
    assert late.retry_at == datetime(2026, 11, 2, 10, 0)
    early = route(book, _ctx("insufficient_funds", datetime(2026, 10, 10, 10, 0)))
    assert early.action == ACTION_CREATE_LINK and early.retry_at is None


def test_ab_control_arm_ignores_the_timing_rule() -> None:
    book = load_playbook_set(SHIPPED)["insufficient_funds"]
    plan = route(book, _ctx("insufficient_funds", datetime(2026, 10, 27, 10, 0), ab_variant=CONTROL))
    assert plan.action == ACTION_CREATE_LINK
    assert any("A/B control" in r for r in plan.reasons)


def test_failover_needs_a_degraded_current_route_and_a_healthy_candidate() -> None:
    book = load_playbook_set(SHIPPED)["gateway_timeout"]
    when = datetime(2026, 10, 8, 9, 0)
    healthy_card = {"card": HealthSnapshot("card", 50, 48, 0.96, HEALTHY, 60)}
    assert route(book, _ctx("gateway_timeout", when, last_method="card",
                            method_health=healthy_card)).failover_to is None

    degraded = {
        "card": HealthSnapshot("card", 50, 20, 0.40, DEGRADED, 60),
        "upi": HealthSnapshot("upi", 50, 48, 0.96, HEALTHY, 60),
        "netbanking": HealthSnapshot("netbanking", 3, 0, None, UNKNOWN, 60),
    }
    plan = route(book, _ctx("gateway_timeout", when, last_method="card", method_health=degraded))
    assert plan.failover_to == "upi"


def test_failover_never_switches_on_unknown_data() -> None:
    current = HealthSnapshot("card", 50, 20, 0.40, DEGRADED, 60)
    unknown = HealthSnapshot("upi", 2, 2, None, UNKNOWN, 60)
    decision = decide_failover(current, [unknown])
    assert decision.switch is False and decision.to is None


# --- B3: timing rules and the holdout script ------------------------------------

def test_cycle_aware_funds_defers_from_day_26_only() -> None:
    assert cycle_aware_funds(datetime(2026, 10, 25, 12)) is None
    assert cycle_aware_funds(datetime(2026, 10, 26, 8)) == datetime(2026, 11, 2, 10, 0)
    assert cycle_aware_funds(datetime(2026, 12, 30, 8)) == datetime(2027, 1, 2, 10, 0)


def test_peak_avoidance_steps_out_of_the_peak_window() -> None:
    assert is_peak_hour(12) and not is_peak_hour(15) and is_peak_hour(20) and not is_peak_hour(23)
    assert peak_avoidance(datetime(2026, 10, 8, 12, 0)) == datetime(2026, 10, 8, 15, 0)
    assert peak_avoidance(datetime(2026, 10, 8, 9, 0)) is None
    assert peak_avoidance(datetime(2026, 10, 8, 22, 0)) == datetime(2026, 10, 8, 23, 0)


def test_unknown_rule_name_is_an_error() -> None:
    with pytest.raises(KeyError):
        apply_rule("does_not_exist", datetime(2026, 10, 1))


def test_holdout_comparison_is_reproducible_and_labelled_simulated() -> None:
    sys.path.insert(0, str(Path(config.PROJECT_ROOT) / "scripts"))
    import retry_timing_holdout as holdout  # noqa: PLC0415

    first = holdout.run(limit=400)
    second = holdout.run(limit=400)
    assert first["causes"] == second["causes"]
    assert "SIMULATED" in first["basis"]
    assert set(first["causes"]) == {"card_expired", "gateway_timeout", "insufficient_funds", "mandate_lapsed"}
    assert "uplift_points" not in first["causes"]["card_expired"]
    funds = first["causes"]["insufficient_funds"]
    assert 0.0 <= funds["p_value"] <= 1.0


def test_paired_uniform_is_stable_and_in_range() -> None:
    sys.path.insert(0, str(Path(config.PROJECT_ROOT) / "scripts"))
    import retry_timing_holdout as holdout  # noqa: PLC0415

    u = holdout.paired_uniform("txn_000088")
    assert 0.0 <= u < 1.0 and u == holdout.paired_uniform("txn_000088")


# --- B4: gateway health ---------------------------------------------------------

def test_health_is_unknown_below_the_minimum_sample() -> None:
    mon = GatewayHealthMonitor(window_minutes=60, min_sample=5, degraded_below=0.8)
    method = f"t_unknown_{uuid.uuid4().hex[:6]}"
    t0 = datetime(2041, 1, 1, 10, 0)
    for _ in range(4):
        mon.record(method, True, now=t0)
    assert mon.snapshot(method, now=t0).state == UNKNOWN


def test_health_turns_healthy_or_degraded_from_the_rolling_window() -> None:
    mon = GatewayHealthMonitor(window_minutes=60, min_sample=5, degraded_below=0.8)
    t0 = datetime(2041, 2, 1, 10, 0)
    ok = f"t_ok_{uuid.uuid4().hex[:6]}"
    for _ in range(5):
        mon.record(ok, True, now=t0)
    assert mon.snapshot(ok, now=t0).state == HEALTHY

    bad = f"t_bad_{uuid.uuid4().hex[:6]}"
    for _ in range(2):
        mon.record(bad, True, now=t0)
    for _ in range(4):
        mon.record(bad, False, now=t0)
    snap = mon.snapshot(bad, now=t0)
    assert snap.state == DEGRADED and snap.samples == 6


def test_observations_outside_the_window_are_ignored() -> None:
    mon = GatewayHealthMonitor(window_minutes=60, min_sample=1, degraded_below=0.8)
    method = f"t_old_{uuid.uuid4().hex[:6]}"
    t0 = datetime(2041, 3, 1, 10, 0)
    mon.record(method, False, now=datetime(2041, 3, 1, 7, 0))  # three hours earlier
    assert mon.snapshot(method, now=t0).samples == 0


# --- B5: dunning ---------------------------------------------------------------

def _schedule(rid: str, base: datetime, touches: list[dict[str, Any]]) -> int:
    messages = {t["touch_no"]: f"touch {t['touch_no']} for {rid}" for t in touches}
    return dunning.schedule_touches(rid, touches, base=base, messages=messages)


def _touches(n: int = 3) -> list[dict[str, Any]]:
    return [{"touch_no": i, "offset_minutes": (i - 1) * 60, "channel": "email", "template": "t"}
            for i in range(1, n + 1)]


def test_scheduling_is_idempotent_per_recovery_and_touch() -> None:
    rid = _recovery("card_expired")
    base = datetime(2041, 4, 1, 9, 0)
    assert _schedule(rid, base, _touches(3)) == 3
    assert _schedule(rid, base, _touches(3)) == 0
    assert len(dunning.list_touches(rid)) == 3


def test_due_touches_are_sent_through_the_mock_with_a_customer_reference_only() -> None:
    rid = _recovery("insufficient_funds")
    base = datetime(2041, 4, 2, 9, 0)
    _schedule(rid, base, _touches(2))
    calls: list[tuple[str, str, str]] = []

    def fake(channel: str, recipient: str, message: str) -> dict[str, Any]:
        calls.append((channel, recipient, message))
        return {"ok": True, "provider_ref": "sim_1"}

    # Other tests leave due touches behind in the shared database, so only count this recovery.
    dunning.run_due(now=datetime(2041, 4, 2, 12, 0), sender=fake)
    mine = [c for c in calls if rid in c[2]]
    assert len(mine) == 2
    assert all(c[1].startswith("CUST_U_") for c in mine), "recipient must be a reference, not contact data"
    assert [t["status"] for t in dunning.list_touches(rid)] == ["sent", "sent"]


def test_nothing_is_sent_after_settlement() -> None:
    rid = _recovery("card_expired")
    base = datetime(2041, 5, 1, 9, 0)
    _schedule(rid, base, _touches(3))
    recovery_store.append_event(rid, "payment_captured", {"payment_id": "pay_x"}, created_by="test")
    calls: list[Any] = []
    dunning.run_due(now=datetime(2041, 5, 1, 13, 0),
                    sender=lambda *a: calls.append(a) or {"ok": True})
    assert [c for c in calls if rid in c[2]] == []
    assert {t["status"] for t in dunning.list_touches(rid)} == {"cancelled"}


def test_a_failed_send_is_recorded_and_the_sequence_continues() -> None:
    rid = _recovery("card_expired")
    base = datetime(2041, 6, 1, 9, 0)
    _schedule(rid, base, _touches(3))
    state = {"n": 0}

    def flaky(channel: str, recipient: str, message: str) -> dict[str, Any]:
        if rid not in message:
            return {"ok": True}  # another test's touch
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("provider down")
        return {"ok": True}

    dunning.run_due(now=datetime(2041, 6, 1, 12, 0), sender=flaky)
    statuses = [t["status"] for t in dunning.list_touches(rid)]
    assert statuses == ["failed", "sent", "sent"], "one failure must not stop the sequence"


def test_the_default_sender_is_mocked_and_validates_channels() -> None:
    receipt = dunning.send_via_channel("sms", "CUST_X", "hello")
    assert receipt["ok"] is True and receipt["provider"] == "simulated"
    with pytest.raises(ValueError):
        dunning.send_via_channel("carrier_pigeon", "CUST_X", "hello")


# --- B6: A/B tests --------------------------------------------------------------

def test_bucket_is_a_stable_sha256_value() -> None:
    assert bucket_of("e", "u1") == bucket_of("e", "u1")
    assert bucket_of("e", "u1") == 1296  # fixed: sha256("e:u1") mod 10000


def test_split_extremes_and_boundary() -> None:
    assert variant_for_bucket(0, 100) == CONTROL
    assert variant_for_bucket(9999, 0) == TREATMENT
    assert variant_for_bucket(4999, 50) == CONTROL
    assert variant_for_bucket(5000, 50) == TREATMENT


def test_assignment_is_stored_and_survives_a_split_change() -> None:
    mgr = ABTestManager()
    exp = f"exp_{uuid.uuid4().hex[:6]}"
    unit = f"unit_{uuid.uuid4().hex[:6]}"
    first = mgr.assign(exp, unit, split_percent=50)
    assert mgr.assign(exp, unit, split_percent=50) == first
    assert mgr.assign(exp, unit, split_percent=0 if first == CONTROL else 100) == first


def test_z_test_matches_the_worked_value() -> None:
    result = two_proportion_z(50, 100, 70, 100)
    assert round(result.z, 3) == 2.887
    assert round(result.p_value, 4) == 0.0039


def test_z_test_degenerate_pool_gives_no_signal() -> None:
    result = two_proportion_z(0, 100, 0, 100)
    assert result.z == 0.0 and result.p_value == 1.0
    with pytest.raises(ValueError):
        two_proportion_z(5, 0, 1, 10)


def _seed_arms(exp: str, control: tuple[int, int], treatment: tuple[int, int]) -> None:
    """Insert assignments with known outcomes: (conversions, units) per arm."""
    from database.models import AbAssignment  # noqa: PLC0415

    stamp = "2041-01-01 00:00:00"
    with recovery_store.session_scope() as session:
        for arm, (conv, units) in (
            (CONTROL, control), (TREATMENT, treatment),
        ):
            for i in range(units):
                session.add(AbAssignment(
                    experiment=exp, unit_id=f"{arm}_{i}", variant=arm, assigned_at=stamp,
                    outcome=1 if i < conv else 0, outcome_at=stamp,
                ))


def test_evaluation_finds_the_winner_once_both_arms_have_enough_units() -> None:
    mgr = ABTestManager(alpha=0.05, min_sample_per_arm=100)
    exp = f"exp_eval_{uuid.uuid4().hex[:6]}"
    _seed_arms(exp, control=(36, 120), treatment=(72, 120))
    report = mgr.evaluate(exp)
    assert report.verdict == VERDICT_SIGNIFICANT
    assert report.winner == TREATMENT
    assert report.p_value is not None and report.p_value < 0.05


def test_too_few_units_gives_no_verdict() -> None:
    mgr = ABTestManager(alpha=0.05, min_sample_per_arm=100)
    exp = f"exp_small_{uuid.uuid4().hex[:6]}"
    for i in range(10):
        mgr.assign(exp, f"s{i}", split_percent=50)
        mgr.record_outcome(exp, f"s{i}", converted=bool(i % 2))
    report = mgr.evaluate(exp)
    assert report.verdict == VERDICT_INSUFFICIENT and report.z is None


def test_the_first_outcome_wins() -> None:
    mgr = ABTestManager()
    exp = f"exp_once_{uuid.uuid4().hex[:6]}"
    mgr.assign(exp, "u", split_percent=50)
    assert mgr.record_outcome(exp, "u", converted=True) is True
    assert mgr.record_outcome(exp, "u", converted=False) is False
    with pytest.raises(KeyError):
        mgr.record_outcome(exp, "never_assigned", converted=True)


def test_no_difference_is_reported_when_rates_match() -> None:
    mgr = ABTestManager(alpha=0.05, min_sample_per_arm=50)
    exp = f"exp_flat_{uuid.uuid4().hex[:6]}"
    _seed_arms(exp, control=(50, 100), treatment=(50, 100))
    assert mgr.evaluate(exp).verdict == VERDICT_NO_DIFFERENCE


# --- B7: method suggestion ------------------------------------------------------

def test_method_suggestion_needs_repeated_card_failures() -> None:
    assert suggest_methods(1, 2, ("upi", "netbanking")) == ()
    assert suggest_methods(2, 2, ("upi", "netbanking")) == ("upi", "netbanking")


def test_method_suggestion_skips_degraded_and_card() -> None:
    assert suggest_methods(3, 2, ("card", "upi", "netbanking"), degraded=("upi",)) == ("netbanking",)
    assert suggest_methods(3, None, ("upi",)) == ()  # disabled when no threshold


# --- migrations -----------------------------------------------------------------

def test_migrations_rerun_safely_and_create_the_track_b_tables() -> None:
    first = run_migrations()
    second = run_migrations()
    assert first == second >= 4
    from database.engine import get_engine  # noqa: PLC0415
    from sqlalchemy import inspect  # noqa: PLC0415

    tables = set(inspect(get_engine()).get_table_names())
    assert {"dunning_touches", "gateway_observations", "ab_assignments",
            "recovery_cases", "recovery_events"} <= tables
