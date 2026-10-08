"""
Ghost Ledger — schema migrations for the unified database.

Run order when the app starts (see ``db_client.init_db``):

1. ``schema.sql``  — the v2 tables, created with ``IF NOT EXISTS``.
2. :func:`run_migrations` — this module:
   * creates the v3 tables from :mod:`database.models`;
   * applies the numbered migrations below exactly once, recording each in
     ``schema_migrations``;
   * every statement is idempotent, so re-running is always safe.

Add a new change as a new numbered entry in ``MIGRATIONS``. Never edit an
entry that has already shipped.

Command line::

    python -m database.migrations          # apply pending migrations
    python -m database.migrations --status # list applied migrations
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from sqlalchemy import text

from config import MERCHANT_MAX_COUNT
from database.engine import get_engine
from database.models import Base

logger = logging.getLogger(__name__)

# (version, name, statements). Statements must be idempotent.
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "v3_core_tables",
        # Tables are created by Base.metadata.create_all before this runs;
        # this entry only records that the v3 table set exists.
        [],
    ),
    (
        2,
        "v3_guards",
        [
            # Merchant cap: at most MERCHANT_MAX_COUNT rows, enforced by the DB.
            f"""
            CREATE TRIGGER IF NOT EXISTS trg_merchants_cap
            BEFORE INSERT ON merchants
            WHEN (SELECT COUNT(*) FROM merchants) >= {int(MERCHANT_MAX_COUNT)}
            BEGIN
                SELECT RAISE(ABORT, 'merchant limit reached: at most {int(MERCHANT_MAX_COUNT)} merchants');
            END
            """,
            # Append-only event log: no updates, no deletes.
            """
            CREATE TRIGGER IF NOT EXISTS trg_recovery_events_no_update
            BEFORE UPDATE ON recovery_events
            BEGIN
                SELECT RAISE(ABORT, 'recovery_events is append-only');
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS trg_recovery_events_no_delete
            BEFORE DELETE ON recovery_events
            BEGIN
                SELECT RAISE(ABORT, 'recovery_events is append-only');
            END
            """,
        ],
    ),
]

LATEST_VERSION: int = MIGRATIONS[-1][0]


def _now() -> str:
    """Return the current local timestamp in the project's canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def run_migrations() -> int:
    """
    Create v3 tables and apply every pending numbered migration.

    Returns
    -------
    int
        The highest migration version now applied.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database cannot be opened or a statement fails. Callers see
        the failure; nothing is swallowed.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY,"
            " name TEXT NOT NULL,"
            " applied_at TEXT NOT NULL)"
        )
        applied = {
            row[0]
            for row in conn.execute(text("SELECT version FROM schema_migrations"))
        }
        for version, name, statements in MIGRATIONS:
            if version in applied:
                continue
            for statement in statements:
                conn.exec_driver_sql(statement)
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, name, applied_at) "
                    "VALUES (:v, :n, :t)"
                ),
                {"v": version, "n": name, "t": _now()},
            )
            logger.info("applied migration %s (%s)", version, name)
    return LATEST_VERSION


def migration_status() -> list[tuple[int, str, str | None]]:
    """
    List every known migration with the time it was applied, if ever.

    Returns
    -------
    list[tuple[int, str, str | None]]
        ``(version, name, applied_at)``; ``applied_at`` is None if pending.
    """
    run_migrations()
    engine = get_engine()
    with engine.connect() as conn:
        applied = {
            row[0]: row[1]
            for row in conn.execute(text("SELECT version, applied_at FROM schema_migrations"))
        }
    return [(v, n, applied.get(v)) for v, n, _ in MIGRATIONS]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Ghost Ledger schema migrations")
    parser.add_argument("--status", action="store_true", help="list migrations and status")
    args = parser.parse_args(argv)
    if args.status:
        from database.db_client import init_db

        init_db()
        for version, name, applied_at in migration_status():
            state = applied_at or "PENDING"
            print(f"  v{version:<3} {name:<20} {state}")
        return 0
    from database.db_client import init_db

    init_db()  # v2 schema + v3 migrations, one call
    print(f"[migrations] database at schema version {LATEST_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
