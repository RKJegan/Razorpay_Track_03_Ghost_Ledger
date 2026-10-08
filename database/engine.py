"""
Ghost Ledger v3 — SQLAlchemy engine on the unified database.

There is ONE database file (``DB_PATH``). The v2 tables are still written by
the explicit SQL helpers in :mod:`database.db_client`; the v3 tables are
mapped with SQLAlchemy models in :mod:`database.models`. Both use this same
file, with the same pragmas:

* ``journal_mode=WAL``  — the dashboard can read while the webhook writes.
* ``foreign_keys=ON``   — every foreign key is enforced, as in v2.
* ``busy_timeout``      — concurrent writers wait instead of failing at once.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB_PATH

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """
    Return the process-wide SQLAlchemy engine, creating it on first use.

    Returns
    -------
    Engine
        Engine bound to ``DB_PATH`` with v2-compatible pragmas applied to
        every new connection.
    """
    global _engine, _session_factory
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )

        @event.listens_for(_engine, "connect")
        def _apply_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        logger.debug("SQLAlchemy engine created for %s", DB_PATH)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional session: commit on success, roll back on error.

    Yields
    ------
    Session
        An open session. Exceptions are re-raised after rollback so callers
        always see failures (nothing is silently swallowed).
    """
    get_engine()
    assert _session_factory is not None  # set by get_engine()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Close pooled connections and forget the engine (used by tests and shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
