"""
Regression test: the API must bootstrap a FRESH database completely.

The bug this guards against: the app ran only the v3 migrations on startup,
so the v2 ``audit_trail`` table was never created and every audit write failed
with "no such table". The shared test database is always pre-built, so the
normal suite could not see that. This test starts the app against an empty
file, in a subprocess, exactly as ``uvicorn api.main:app`` does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROBE = r"""
import sqlite3
from fastapi.testclient import TestClient
from api.main import app
from config import DB_PATH

with TestClient(app) as client:
    assert client.get("/health").status_code == 200
conn = sqlite3.connect(str(DB_PATH))
names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
print("TABLES=" + ",".join(sorted(names)))
print("TRIGGERS=" + ",".join(sorted(triggers)))
"""


def test_api_bootstraps_an_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    tables = set(result.stdout.split("TABLES=")[1].splitlines()[0].split(","))
    # v2 tables (schema.sql) — the one the bug was missing
    for name in ("transactions", "failures", "recoveries", "audit_trail", "autopsy_reports"):
        assert name in tables, f"v2 table {name} missing after API startup"
    # v3 tables (SQLAlchemy models)
    for name in ("merchants", "recovery_cases", "recovery_events", "webhook_events", "schema_migrations"):
        assert name in tables, f"v3 table {name} missing after API startup"

    triggers = set(result.stdout.split("TRIGGERS=")[1].splitlines()[0].split(","))
    assert "trg_merchants_cap" in triggers
    assert "trg_recovery_events_no_update" in triggers
    assert "trg_recovery_events_no_delete" in triggers
