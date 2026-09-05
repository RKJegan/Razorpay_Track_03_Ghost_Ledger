"""
Pytest configuration.

The suite runs against a **copy** of the demo database, at
``data/test_ghost_ledger.db``. Two reasons:

1. End-to-end fixtures must insert real transaction/failure rows to satisfy
   foreign keys, and must delete them again. Doing that in the demo database
   would corrupt the figures the dashboard shows.
2. Diagnoser tests need the generated corpus to be present, so the copy is
   seeded from the real database when it exists.

``VACUUM INTO`` is used rather than a file copy because the real database runs
in WAL mode, where recent commits may live only in the ``-wal`` sidecar file.
"""

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before `config` is imported: DB_PATH is resolved once at import.
os.environ["DB_PATH"] = "data/test_ghost_ledger.db"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _seed_test_db():
    """Copy the demo database into the test database, then open it."""
    real = ROOT / "data" / "ghost_ledger.db"
    test = ROOT / "data" / "test_ghost_ledger.db"

    # VACUUM INTO refuses to overwrite, so clear any previous run first.
    for stale in (test, test.with_suffix(".db-wal"), test.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    if real.exists():
        src = sqlite3.connect(str(real))
        try:
            src.execute(f"VACUUM INTO '{test.as_posix()}'")
        finally:
            src.close()

    from database import db_client

    db_client.init_db()
    yield
