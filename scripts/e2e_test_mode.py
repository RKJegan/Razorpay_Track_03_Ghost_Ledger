#!/usr/bin/env python3
"""
Ghost Ledger v3 — end-to-end check (A7). Plain Python, no test framework.

What it proves, in order
------------------------
1. A failed payment becomes a recovery case for a dedicated test merchant.
2. The system creates a payment link (policy-gated, reference id attached).
3. The payment FAILS. The failure reaches the webhook and is recorded.
4. The system creates the next attempt (a second link).
5. The second payment SUCCEEDS. The capture is recorded and the recovery settles.
6. Reconciliation matches the paid amount with the recovery amount.
7. The merchant API (HTTP, merchant key) shows that merchant's own timeline only.

Modes
-----
--mode live     Real Razorpay TEST MODE. Requires RAZORPAY_LIVE_TEST_MODE=1 and
                keys starting with ``rzp_test_``. Razorpay sends the webhooks to the
                public URL you configured (for example an ngrok tunnel to port 8000).
                The script prints payment links; YOU pay them in the browser: first
                with a method that fails, then with a test card that succeeds.
                Live keys are refused.

--mode offline  Same flow with the simulator. The script posts correctly signed
                webhooks to the local API itself. Use it to check the wiring without
                Razorpay or a browser.

Prerequisites
-------------
* The API is running and uses the same database:
      python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
* RAZORPAY_WEBHOOK_SECRET is set in .env (never printed by this script).
* In live mode: RAZORPAY_LIVE_TEST_MODE=1, RAZORPAY_KEY_ID=rzp_test_..., RAZORPAY_KEY_SECRET.

Usage
-----
    python scripts/e2e_test_mode.py --mode offline
    python scripts/e2e_test_mode.py --mode live --timeout 900 --report data/e2e_live.json

Exit code 0 when every check passes, 1 otherwise, 2 when preflight fails.
The merchant API key is generated in memory for this run and never written out.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

import config  # noqa: E402
from agents import scheduled_jobs  # noqa: E402
from agents.recovery_executor import submit_recovery  # noqa: E402
from api.razorpay_client import LiveRazorpayClient, SimulatedRazorpayClient  # noqa: E402
from database import db_client, merchants, recovery_store  # noqa: E402

E2E_MERCHANT_ID = "m_e2e_test_mode"
E2E_MERCHANT_NAME = "E2E Test Mode"
DEFAULT_AMOUNT_INR = 499.0
CUSTOMER_ID = "CUST_E2E_TEST_MODE"
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    mode: str
    started_at: str
    recovery_id: str = ""
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, ok, detail))
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)


# --- preflight ---------------------------------------------------------------

def preflight(mode: str) -> list[str]:
    """Return a list of problems. An empty list means the run may start."""
    problems: list[str] = []
    if not config.RAZORPAY_WEBHOOK_SECRET:
        problems.append("RAZORPAY_WEBHOOK_SECRET is not set in .env")
    if mode == "live":
        if not config.RAZORPAY_LIVE_TEST_MODE:
            problems.append("set RAZORPAY_LIVE_TEST_MODE=1 for live mode")
        if not config.RAZORPAY_KEY_ID or not config.RAZORPAY_KEY_SECRET:
            problems.append("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set")
        elif not config.RAZORPAY_KEY_ID.startswith("rzp_test_"):
            problems.append("refusing to run: RAZORPAY_KEY_ID is not a test key (rzp_test_...)")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code != 200:
            problems.append(f"API at {API_BASE_URL} answered /health with {health.status_code}")
    except requests.RequestException as exc:
        problems.append(f"API not reachable at {API_BASE_URL} ({type(exc).__name__}); start it first")
    return problems


# --- helpers -----------------------------------------------------------------

def ensure_merchant() -> str:
    """Create the e2e merchant if needed, rotate its key, and return the key in memory only."""
    existing = {m["id"] for m in merchants.list_merchants()}
    if E2E_MERCHANT_ID in existing:
        return merchants.rotate_api_key(E2E_MERCHANT_ID)
    return merchants.create_merchant(E2E_MERCHANT_ID, E2E_MERCHANT_NAME)


def new_case(amount_inr: float) -> str:
    txn = f"txn_e2e_{uuid.uuid4().hex[:10]}"
    return recovery_store.create_case(E2E_MERCHANT_ID, txn, amount_inr,
                                      cause="card_expired", customer_id=CUSTOMER_ID)


def open_link(rid: str, client: Any, attempts: int = 3) -> dict[str, Any] | None:
    """Ask the executor for a link, retrying transient call failures. None if all fail."""
    for _ in range(attempts):
        result = submit_recovery(rid, client=client)
        if result.outcome == "link_created":
            return {"link_id": result.link_id, "short_url": result.short_url, "attempt": result.attempt}
        if result.outcome != "link_failed":
            print(f"    executor said {result.outcome}: {result.reason}")
            return None
        time.sleep(1)
    return None


def stages(rid: str) -> list[str]:
    return [e["stage"] for e in recovery_store.get_recovery_timeline(rid)]


def wait_until(predicate: Callable[[], bool], timeout_s: float, interval_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def post_signed_webhook(event: str, rid: str, amount_paise: int, status: str) -> requests.Response:
    """Send a webhook exactly as Razorpay would, signed with the configured secret."""
    payment_id = f"pay_e2e_{uuid.uuid4().hex[:14]}"
    body = json.dumps({
        "entity": "event",
        "event": event,
        "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": payment_id, "amount": amount_paise, "currency": "INR", "status": status,
            "notes": {"recovery_id": rid, "txn_id": "txn_e2e_signed"},
        }}},
        "created_at": int(time.time()),
    }).encode()
    signature = hmac.new(config.RAZORPAY_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return requests.post(
        f"{API_BASE_URL}/webhooks/razorpay",
        data=body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature,
                 "X-Razorpay-Event-Id": f"evt_e2e_{uuid.uuid4().hex}"},
        timeout=10,
    )


# --- the flow ----------------------------------------------------------------

def run(mode: str, amount_inr: float, timeout_s: float) -> Report:
    report = Report(mode=mode, started_at=datetime.now().isoformat(sep=" ", timespec="seconds"))
    live = mode == "live"
    client: Any = LiveRazorpayClient() if live else SimulatedRazorpayClient()
    amount_paise = int(round(amount_inr * 100))

    print(f"\n1. Recovery for ₹{amount_inr:.2f} on merchant {E2E_MERCHANT_ID}")
    key = ensure_merchant()
    rid = new_case(amount_inr)
    report.recovery_id = rid
    report.add("case created", recovery_store.get_recovery_status(rid) is not None, rid)

    print("\n2. First payment link")
    link1 = open_link(rid, client)
    if not report.add("first link created", link1 is not None,
                      "" if link1 else "Razorpay did not return a link (see the API log)"):
        return report
    if live:
        print(f"    >>> Open this link and pay with a method that FAILS (e.g. UPI failure@razorpay,")
        print(f"        or the Failure option in the test checkout):\n        {link1['short_url']}")
    else:
        print("    (offline) the script will report a failed payment itself")

    print("\n3. The payment fails")
    if live:
        failed = wait_until(lambda: "payment_failed" in stages(rid), timeout_s)
    else:
        resp = post_signed_webhook("payment.failed", rid, amount_paise, "failed")
        failed = resp.status_code == 200 and wait_until(lambda: "payment_failed" in stages(rid), 10, 0.5)
    report.add("failure reached the webhook and was recorded", failed,
               "" if failed else "no payment.failed webhook within the timeout; check the public webhook URL")
    if not failed:
        return report

    print("\n4. Next attempt")
    retry_outcome = scheduled_jobs.retry_recovery(rid, client=client)
    report.add("next attempt created", retry_outcome == "link_created", f"retry outcome: {retry_outcome}")
    if retry_outcome != "link_created":
        return report
    latest = recovery_store.latest_event(rid, ("payment_link_created",))
    link2_url = (latest or {}).get("detail", {}).get("short_url")
    if live:
        print(f"    >>> Open this link and pay with a test card that SUCCEEDS")
        print(f"        (4111 1111 1111 1111, any future expiry, any CVV):\n        {link2_url}")

    print("\n5. The second payment succeeds")
    if live:
        def settled_or_checked() -> bool:
            try:
                scheduled_jobs.check_payment_settlement(rid, client=client)
            except scheduled_jobs.TransientJobError:
                pass
            return "payment_captured" in stages(rid) or "settlement_confirmed" in stages(rid)

        captured = wait_until(settled_or_checked, timeout_s)
    else:
        resp = post_signed_webhook("payment.captured", rid, amount_paise, "captured")
        captured = resp.status_code == 200 and wait_until(
            lambda: "payment_captured" in stages(rid), 10, 0.5)
    report.add("capture recorded", captured,
               "" if captured else "no captured payment within the timeout")
    if not captured:
        return report
    status = recovery_store.get_recovery_status(rid)["status"]
    report.add("recovery settled", status == "settled", f"status={status}")

    print("\n6. Reconciliation")
    reconciled_states = ("reconciled", "already_reconciled")
    outcome_rec = scheduled_jobs.reconcile_recovery(rid)
    if live and outcome_rec not in reconciled_states:
        # The webhook may still be on its way to confirm the provisional settlement.
        wait_until(lambda: scheduled_jobs.reconcile_recovery(rid) in reconciled_states, timeout_s, 5.0)
        outcome_rec = scheduled_jobs.reconcile_recovery(rid)
    report.add("paid amount matches recovery amount", outcome_rec in reconciled_states,
               f"outcome={outcome_rec}")

    print("\n7. Merchant API view")
    resp = requests.get(f"{API_BASE_URL}/api/merchant/recoveries/{rid}",
                        headers={"Authorization": f"Bearer {key}"}, timeout=10)
    body = resp.json() if resp.ok else {}
    timeline = body.get("timeline", [])
    seen = [e["stage"] for e in timeline]
    report.add("merchant API returns the recovery", resp.status_code == 200, f"HTTP {resp.status_code}")
    report.add("two payment links on the timeline", seen.count("payment_link_created") == 2,
               f"{seen.count('payment_link_created')} link(s)")
    report.add("failure and capture both on the timeline",
               "payment_failed" in seen and ("payment_captured" in seen or "settlement_confirmed" in seen))
    report.add("timeline hides internal fields",
               all("created_by" not in e and set(e) == {"stage", "timestamp", "detail"} for e in timeline))
    other = requests.get(f"{API_BASE_URL}/api/merchant/recoveries/{rid}",
                         headers={"Authorization": "Bearer not-the-key"}, timeout=10)
    report.add("wrong key cannot read it", other.status_code == 401, f"HTTP {other.status_code}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ghost Ledger end-to-end check")
    parser.add_argument("--mode", choices=["live", "offline"], required=True)
    parser.add_argument("--amount", type=float, default=DEFAULT_AMOUNT_INR,
                        help="recovery amount in INR (default 499; must stay under the ₹10,000 ceiling)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for each live payment step (default 900)")
    parser.add_argument("--report", type=Path, help="write the result as JSON to this path")
    args = parser.parse_args(argv)

    if args.amount <= 0 or args.amount >= config.POLICY_MAX_AUTO_APPROVE_INR:
        print("amount must be above 0 and below the auto-approve ceiling, so no approval is needed")
        return 2

    db_client.init_db()
    problems = preflight(args.mode)
    if problems:
        print("Preflight failed:")
        for p in problems:
            print(f"  - {p}")
        return 2

    print(f"Ghost Ledger end-to-end ({args.mode}) against {API_BASE_URL}")
    report = run(args.mode, args.amount, args.timeout)
    print("\nResult:", "PASS" if report.passed else "FAIL", f"(recovery {report.recovery_id or '—'})")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "mode": report.mode, "started_at": report.started_at, "recovery_id": report.recovery_id,
            "passed": report.passed,
            "checks": [c.__dict__ for c in report.checks],
        }, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
