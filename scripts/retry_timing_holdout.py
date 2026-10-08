"""
Ghost Ledger v3 — holdout comparison for rule-based retry timing (B3).

Compares two retry policies on the same held-out failures:

* ``immediate``   retry at the time of failure (the v2 behaviour)
* ``rule_based``  retry at the time the playbook timing rule chooses

Causes compared: ``insufficient_funds`` and ``gateway_timeout``, the two causes
whose playbooks have a timing rule. ``card_expired`` and ``mandate_lapsed`` get
no automatic retry under either policy, so they are reported but not compared.

IMPORTANT — what the numbers mean
---------------------------------
``data/sample_output/holdout_set.json`` records *what failed*, not *what
happened on a later retry*. This script therefore uses a **simulated outcome
model**, written out below, so the comparison is reproducible:

* ``insufficient_funds``: a retry succeeds with p=0.55 on days 1-25 of the
  month, and p=0.15 from day 26 (the generator's end-of-cycle cash crunch).
* ``gateway_timeout``: a retry succeeds with p=0.70 outside the generator's peak
  windows (11-14, 19-22) and p=0.40 inside them.

Both policies use the same uniform draw per transaction (hashed from the
transaction id), so the comparison is paired. The rule policy is therefore
rewarded for the same effects the model encodes. The result shows whether the
rule matches the model's assumptions. It is not evidence of real-world uplift.
Real uplift needs a live A/B test (``strategies/ab_test.py``,
``AB_RETRY_TIMING_EXPERIMENT=1``).

Usage::

    python scripts/retry_timing_holdout.py            # full holdout
    python scripts/retry_timing_holdout.py --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.timing import CRUNCH_START_DAY, apply_rule, is_peak_hour  # noqa: E402

HOLDOUT = ROOT / "data" / "sample_output" / "holdout_set.json"
GROUND_TRUTH = ROOT / "data" / "sample_output" / "ground_truth_holdout.json"
REPORT = ROOT / "reports" / "retry_timing_holdout.json"

RULE_FOR_CAUSE: dict[str, str] = {
    "insufficient_funds": "cycle_aware_funds",
    "gateway_timeout": "peak_avoidance",
}
NOT_TIMED = ("card_expired", "mandate_lapsed")
POLICIES = ("immediate", "rule_based")


def simulated_success_probability(cause: str, when: datetime) -> float:
    """The outcome model. Documented in the module docstring; change it there, not here."""
    if cause == "insufficient_funds":
        return 0.15 if when.day >= CRUNCH_START_DAY else 0.55
    if cause == "gateway_timeout":
        return 0.40 if is_peak_hour(when.hour) else 0.70
    raise ValueError(f"no outcome model for {cause!r}")


def paired_uniform(txn_id: str) -> float:
    """Deterministic uniform in [0, 1) per transaction, shared by both policies."""
    digest = hashlib.sha256(f"retry-outcome:{txn_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def retry_time(policy: str, cause: str, failed_at: datetime) -> datetime:
    if policy == "immediate":
        return failed_at
    run_at = apply_rule(RULE_FOR_CAUSE[cause], failed_at)
    return run_at if run_at is not None else failed_at


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar chi-square (no continuity correction) and its two-sided p-value (1 df)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (b - c) ** 2 / (b + c)
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


def run(limit: int | None = None) -> dict[str, Any]:
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    truth: dict[str, str] = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in holdout}

    selected = [txn for txn in sorted(truth) if txn in rows]
    if limit is not None:
        selected = selected[:limit]

    per_cause: dict[str, Counter] = {}
    for txn in selected:
        cause = truth[txn]
        row = rows[txn]
        counts = per_cause.setdefault(cause, Counter())
        counts["n"] += 1
        if cause in NOT_TIMED:
            counts["not_timed"] += 1
            continue
        failed_at = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        u = paired_uniform(txn)
        outcome = {}
        for policy in POLICIES:
            when = retry_time(policy, cause, failed_at)
            outcome[policy] = u < simulated_success_probability(cause, when)
            counts[f"{policy}_ok"] += int(outcome[policy])
            counts[f"{policy}_deferred"] += int(when != failed_at)
        if outcome["rule_based"] and not outcome["immediate"]:
            counts["rule_only"] += 1
        if outcome["immediate"] and not outcome["rule_based"]:
            counts["immediate_only"] += 1

    causes: dict[str, Any] = {}
    for cause in sorted(per_cause):
        c = per_cause[cause]
        n = c["n"]
        entry: dict[str, Any] = {"failures": n}
        if cause in NOT_TIMED:
            entry["note"] = "no automatic retry under either policy; not compared"
            causes[cause] = entry
            continue
        b, cc = c["rule_only"], c["immediate_only"]
        chi2, p = mcnemar(b, cc)
        entry.update({
            "immediate_success_rate": round(c["immediate_ok"] / n, 4),
            "rule_based_success_rate": round(c["rule_based_ok"] / n, 4),
            "uplift_points": round((c["rule_based_ok"] - c["immediate_ok"]) / n * 100, 2),
            "rule_based_deferred": c["rule_based_deferred"],
            "discordant_rule_only": b,
            "discordant_immediate_only": cc,
            "mcnemar_chi2": round(chi2, 3),
            "p_value": float(f"{p:.6g}"),
        })
        causes[cause] = entry

    compared = [c for c in causes.values() if "uplift_points" in c]
    total_n = sum(per_cause[c]["n"] - per_cause[c]["not_timed"] for c in per_cause if c not in NOT_TIMED)
    return {
        "basis": "SIMULATED outcomes (see scripts/retry_timing_holdout.py); holdout has no retry labels",
        "outcome_model": {
            "insufficient_funds": "p=0.55 if day<26 else 0.15",
            "gateway_timeout": "p=0.70 off-peak, 0.40 in peak windows 11-14 and 19-22",
        },
        "rules": RULE_FOR_CAUSE,
        "sample_failures": len(selected),
        "timed_failures": total_n,
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "causes": causes,
        "summary": {
            "compared_causes": len(compared),
            "note": "uplift is a property of the simulated model; compare live with the A/B experiment",
        },
    }


def _print(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("  RETRY TIMING — HOLDOUT COMPARISON (B3)")
    print("=" * 78)
    print(f"  basis: {report['basis']}")
    print(f"  failures in sample: {report['sample_failures']:,}  timed: {report['timed_failures']:,}")
    print("-" * 78)
    print(f"  {'cause':<20}{'n':>7}{'immediate':>12}{'rule':>10}{'uplift pts':>12}{'p-value':>11}")
    for cause, entry in report["causes"].items():
        if "uplift_points" not in entry:
            print(f"  {cause:<20}{entry['failures']:>7}   {entry['note']}")
            continue
        print(
            f"  {cause:<20}{entry['failures']:>7}"
            f"{entry['immediate_success_rate']:>12.2%}{entry['rule_based_success_rate']:>10.2%}"
            f"{entry['uplift_points']:>12.2f}{entry['p_value']:>11.4g}"
        )
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Holdout comparison of retry timing (B3)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N failures")
    parser.add_argument("--out", type=Path, default=REPORT, help="where to write the JSON report")
    args = parser.parse_args(argv)
    report = run(limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)
    print(f"  report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
