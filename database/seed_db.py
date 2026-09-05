"""
Load generated transactions into the SQLite `transactions` table.

Kept separate from the generator so that data production and persistence stay
independently testable, and so `--load-db` remains an explicit opt-in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import db_client  # noqa: E402

# Column order must match the spec's `transactions` DDL exactly.
_COLUMNS = (
    "id",
    "merchant_id",
    "customer_id",
    "amount",
    "status",
    "txn_type",
    "payment_method",
    "failure_reason_raw",
    "timestamp",
    "is_holdout",
)


def load_transactions(
    transactions: list[dict[str, Any]], batch_size: int = 5_000
) -> tuple[int, float]:
    """
    Insert (or replace) transaction rows into the database.

    Inserted in batches so that a large corpus does not build one enormous
    parameter list in memory.

    Parameters
    ----------
    transactions : list[dict[str, Any]]
        Records produced by ``data.synthetic_generator``. Only the ten spec
        columns are persisted; the ``context`` feature block is kept out of
        the ledger and written separately for failures only.
    batch_size : int, optional
        Rows per committed batch.

    Returns
    -------
    tuple[int, float]
        (number of rows written, resulting database size in MB).
    """
    db_client.init_db()
    placeholders = ", ".join("?" * len(_COLUMNS))
    sql = (
        f"INSERT OR REPLACE INTO transactions "
        f"({', '.join(_COLUMNS)}) VALUES ({placeholders})"
    )
    written = 0
    for start in range(0, len(transactions), batch_size):
        chunk = transactions[start : start + batch_size]
        rows = [tuple(t[c] for c in _COLUMNS) for t in chunk]
        written += db_client.executemany(sql, rows)
    db_client.execute("PRAGMA optimize")
    size_mb = db_client.DB_PATH.stat().st_size / (1024 * 1024)
    return written, size_mb


def load_subscriptions(subscriptions: list[dict[str, Any]]) -> int:
    """
    Persist subscription mandates into the audit trail as structured records.

    The spec has no `subscriptions` table, so mandates are stored as audit
    entries rather than by inventing a fifth table. This keeps the schema
    faithful to GHOST_LEDGER_PROJECT_SPEC section 11.

    Parameters
    ----------
    subscriptions : list[dict[str, Any]]
        Subscription mandate records from the generator.

    Returns
    -------
    int
        Number of rows written.
    """
    db_client.init_db()
    import json
    import uuid
    from datetime import datetime

    rows = []
    for s in subscriptions:
        rows.append(
            (
                f"seed_sub_{uuid.uuid4().hex[:12]}",
                datetime.now().isoformat(sep=" ", timespec="seconds"),
                "synthetic_generator",
                "load_subscription",
                json.dumps(s, default=str),
                json.dumps({"stored_as": "audit_trail", "reason": "no subscriptions table in spec"}, default=str),
                "synthetic mandate record for subscription recovery agent",
                1,
            )
        )
    sql = (
        "INSERT OR REPLACE INTO audit_trail "
        "(id, timestamp, component, action, input_data, output_data, decision_reason, success) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    return db_client.executemany(sql, rows)


if __name__ == "__main__":
    import json

    from config import SAMPLE_OUTPUT_DIR

    txns = json.loads((SAMPLE_OUTPUT_DIR / "demo_window.json").read_text(encoding="utf-8"))
    subs = json.loads((SAMPLE_OUTPUT_DIR / "subscriptions.json").read_text(encoding="utf-8"))
    n, mb = load_transactions(txns)
    print(f"[seed] loaded {n:,} transactions into SQLite ({mb:.2f} MB)")
    print(f"[seed] loaded {load_subscriptions(subs):,} subscription records")
