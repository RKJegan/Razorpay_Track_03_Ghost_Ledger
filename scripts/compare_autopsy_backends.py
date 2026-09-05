"""
Compare autopsy backends on the same real failures.

Answers one question directly: what does a local LLM actually add over the
deterministic template?

    python scripts/compare_autopsy_backends.py --n 5 --backends template ollama
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.autopsy_reporter import (  # noqa: E402
    build_facts,
    check_hallucinated_numbers,
    generate_autopsy,
)
from database import db_client  # noqa: E402


def pick_failures(n: int, cause: str | None = None):
    """Select real diagnosed failures, optionally filtered to one cause."""
    sql = """
        SELECT f.id AS failure_id, f.transaction_id, f.predicted_cause,
               f.confidence, t.amount, t.txn_type, t.payment_method,
               t.failure_reason_raw, t.timestamp
        FROM failures f
        JOIN transactions t ON t.id = f.transaction_id
    """
    params: list = []
    if cause:
        sql += " WHERE f.predicted_cause = ?"
        params.append(cause)
    sql += " ORDER BY f.id LIMIT ?"
    params.append(n)
    return db_client.query(sql, tuple(params))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="failures to compare")
    ap.add_argument("--cause", default=None, help="filter to one cause bucket")
    ap.add_argument(
        "--backends", nargs="+", default=["template", "ollama"],
        choices=["template", "ollama", "openai"],
    )
    args = ap.parse_args()

    rows = pick_failures(args.n, args.cause)
    if not rows:
        print("No failures found. Run `python main.py` first.")
        return 1

    print("=" * 78)
    print("  AUTOPSY BACKEND COMPARISON")
    print(f"  backends : {', '.join(args.backends)}")
    print(f"  model    : {os.environ.get('LLM_MODEL', 'llama3.2:3b')}")
    print(f"  endpoint : {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"  sample   : {len(rows)} real diagnosed failures"
          + (f" (cause={args.cause})" if args.cause else ""))
    print("=" * 78)

    timings = {b: [] for b in args.backends}
    degraded = {b: 0 for b in args.backends}
    flagged = {b: 0 for b in args.backends}

    for i, r in enumerate(rows, 1):
        cause = r["predicted_cause"]
        code = (r["failure_reason_raw"] or "").split(":")[0]
        facts = build_facts(
            failure_id=r["failure_id"],
            transaction_id=r["transaction_id"],
            cause=cause,
            confidence=r["confidence"],
            amount_inr=r["amount"],
            txn_type=r["txn_type"],
            payment_method=r["payment_method"],
            gateway="(not in ledger)",  # gateway lives in feature context, not the ledger
            error_code=code,
            timestamp=r["timestamp"],
            attempt_count=1,
            recovered_amount=0.0,
        )

        print(f"\n[{i}] {cause} · INR {r['amount']:,.2f} · {code}")
        print("-" * 78)

        for backend in args.backends:
            t0 = time.perf_counter()
            rep = generate_autopsy(r["failure_id"], facts, backend=backend)
            elapsed = time.perf_counter() - t0
            timings[backend].append(elapsed)
            if rep.degraded:
                degraded[backend] += 1
            if rep.hallucination_flags:
                flagged[backend] += 1

            tag = f"{backend}"
            if rep.degraded:
                tag += " [DEGRADED→template]"
            print(f"  {tag:<24} ({elapsed:.2f}s, {rep.model})")
            for line in _wrap(rep.text, 70):
                print(f"      {line}")
            if rep.hallucination_flags:
                print(f"      ⚠ hallucinated numbers: {rep.hallucination_flags}")
        print()

    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'backend':<12} {'avg time':>10} {'degraded':>10} {'flagged':>9}")
    for backend in args.backends:
        ts = timings[backend]
        avg = sum(ts) / len(ts) if ts else 0
        print(f"  {backend:<12} {avg:>9.2f}s {degraded[backend]:>10} "
              f"{flagged[backend]:>9}")
    print()
    print("  Note: the LLM writes explanation text only. It cannot change any")
    print("  rupee, policy decision, or recovery outcome.")
    print("=" * 78)
    return 0


def _wrap(text: str, width: int):
    words, line = text.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            yield line
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        yield line


if __name__ == "__main__":
    raise SystemExit(main())
