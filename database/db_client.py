"""
Ghost Ledger v2 — SQLite client.

Zero-setup, portable persistence layer. Responsibilities:
  * create the schema from ``schema.sql`` (idempotent)
  * expose a small, explicit set of write/read helpers used by the pipeline
  * never silently swallow errors — callers must see failures

Deliberately not an ORM: the project has four tables and a hard requirement
that every financial action be traceable, so explicit SQL is easier to audit.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from config import DB_PATH, LOG_LEVEL, PROJECT_ROOT

SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"

_local = threading.local()


def _connect() -> sqlite3.Connection:
    """
    Return a per-thread SQLite connection.

    SQLite connections are not shareable across threads by default, and the
    Streamlit dashboard reads while the pipeline writes, so each thread gets
    its own handle.

    Returns
    -------
    sqlite3.Connection
        Connection with row access by column name enabled.
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """
    Context manager that commits on success and rolls back on exception.

    Yields
    ------
    sqlite3.Connection
        The active connection for the enclosed block.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(verbose: bool = False) -> Path:
    """
    Create every table if it does not already exist.

    Parameters
    ----------
    verbose : bool, optional
        Print the resolved database path when True.

    Returns
    -------
    Path
        Filesystem path of the SQLite database file.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with transaction() as conn:
        conn.executescript(sql)
    if verbose and LOG_LEVEL.upper() in {"DEBUG", "INFO"}:
        print(f"[db] schema ready at {DB_PATH.relative_to(PROJECT_ROOT)}")
    return DB_PATH


def execute(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
    """
    Execute a single statement and commit.

    Parameters
    ----------
    sql : str
        Parameterised SQL text. Values must never be f-string interpolated.
    params : Sequence[Any] | dict[str, Any], optional
        Bound parameters.

    Returns
    -------
    sqlite3.Cursor
        Cursor carrying ``lastrowid`` and ``rowcount``.
    """
    with transaction() as conn:
        return conn.execute(sql, params)


def executemany(sql: str, seq_of_params: Iterable[Sequence[Any]]) -> int:
    """
    Execute a parameterised statement against many rows in one transaction.

    Parameters
    ----------
    sql : str
        Parameterised SQL text.
    seq_of_params : Iterable[Sequence[Any]]
        Iterable of parameter tuples.

    Returns
    -------
    int
        Total number of rows affected.
    """
    with transaction() as conn:
        cur = conn.executemany(sql, seq_of_params)
        return cur.rowcount or 0


def query(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
    """
    Run a SELECT and return all rows.

    Parameters
    ----------
    sql : str
        Parameterised SELECT statement.
    params : Sequence[Any] | dict[str, Any], optional
        Bound parameters.

    Returns
    -------
    list[sqlite3.Row]
        Rows addressable by column name; empty list when nothing matches.
    """
    conn = _connect()
    return list(conn.execute(sql, params))


def query_one(
    sql: str, params: Sequence[Any] | dict[str, Any] = ()
) -> sqlite3.Row | None:
    """
    Run a SELECT and return the first row, or None.

    Parameters
    ----------
    sql : str
        Parameterised SELECT statement.
    params : Sequence[Any] | dict[str, Any], optional
        Bound parameters.

    Returns
    -------
    sqlite3.Row | None
        First matching row, or None if the result set is empty.
    """
    conn = _connect()
    return conn.execute(sql, params).fetchone()


def scalar(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
    """
    Return the first column of the first row, or None.

    Used for COUNT/SUM aggregates, e.g. the dashboard headline figure.

    Parameters
    ----------
    sql : str
        Parameterised aggregate SELECT.
    params : Sequence[Any] | dict[str, Any], optional
        Bound parameters.

    Returns
    -------
    Any
        Scalar result, or None when no row is returned.
    """
    row = query_one(sql, params)
    return None if row is None else row[0]


def reset_db() -> None:
    """
    Drop and recreate every table.

    Used by ``make reset`` and by end-to-end tests that need a clean slate.
    Destructive by design.
    """
    conn = _connect()
    tables = [
        "autopsy_reports",
        "recoveries",
        "failures",
        "transactions",
        "audit_trail",
    ]
    with transaction() as c:
        c.execute("PRAGMA foreign_keys=OFF")
        for t in tables:
            c.execute(f"DROP TABLE IF EXISTS {t}")
        c.execute("PRAGMA foreign_keys=ON")
    init_db()


def to_json(value: Any) -> str | None:
    """
    Serialise a value for storage in a TEXT column.

    Parameters
    ----------
    value : Any
        Anything JSON-serialisable, or None.

    Returns
    -------
    str | None
        Compact JSON string, or None when ``value`` is None.
    """
    if value is None:
        return None
    return json.dumps(value, default=str, separators=(",", ":"))


if __name__ == "__main__":
    path = init_db(verbose=True)
    print(f"[db] tables: {[r[0] for r in query('SELECT name FROM sqlite_master WHERE type=? ORDER BY name', ('table',))]}")
