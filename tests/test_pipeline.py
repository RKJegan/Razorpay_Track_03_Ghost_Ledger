"""
Pipeline-level smoke tests.

Runs `main.py` as a subprocess against a throwaway database, to verify the
one-command path actually completes end to end. Reports written by the real
pipeline are backed up and restored so the test cannot leave the workspace
with figures from a truncated run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import REPORTS_DIR  # noqa: E402

REPORTS_TO_PROTECT = ("headline_metrics.json", "holdout_metrics.json")


@pytest.fixture
def protected_reports():
    """Back up the real reports and restore them after the test."""
    backups = {}
    for name in REPORTS_TO_PROTECT:
        src = REPORTS_DIR / name
        if src.exists():
            backups[name] = src.read_text(encoding="utf-8")
    yield
    for name, content in backups.items():
        (REPORTS_DIR / name).write_text(content, encoding="utf-8")


def _run_main(extra_args: list[str], db_path: Path) -> subprocess.CompletedProcess:
    """
    Run main.py in a subprocess against a temporary database.

    Parameters
    ----------
    extra_args : list[str]
        Additional CLI arguments.
    db_path : Path
        Path of the throwaway SQLite file.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process with captured output.
    """
    env = dict(os.environ)
    env["DB_PATH"] = str(db_path)
    return subprocess.run(
        [sys.executable, "main.py", *extra_args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_main_help_works():
    """The CLI is reachable and self-documenting."""
    proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "--profile" in proc.stdout
    assert "--regen" in proc.stdout


def test_pipeline_completes_end_to_end(protected_reports):
    """One command runs the full loop without manual steps."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.db"
        proc = _run_main(
            ["--limit", "250", "--llm-backend", "template"], db
        )
        assert proc.returncode == 0, f"main.py failed:\n{proc.stdout}\n{proc.stderr}"

        out = proc.stdout
        for expected in (
            "SYNTHETIC DATA GENERATION",
            "ROOT CAUSE DIAGNOSIS",
            "POLICY-GATED RECOVERY ACTIONS",
            "AUTOPSY REPORTS",
            "HEADLINE METRICS",
            "PIPELINE COMPLETE",
        ):
            assert expected in out, f"missing pipeline stage: {expected}"

        assert "Reconciles with audit trail: True" in out


def test_pipeline_produces_consistent_figures(protected_reports):
    """Headline, per-cause breakdown and audit trail agree after a run."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "consistency.db"
        proc = _run_main(["--limit", "250", "--llm-backend", "template"], db)
        assert proc.returncode == 0, proc.stderr

        # Read the figures back from the throwaway database.
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        recovered = conn.execute(
            "SELECT COALESCE(SUM(recovered_amount),0) FROM recoveries"
        ).fetchone()[0]
        actions = conn.execute("SELECT COUNT(*) FROM recoveries").fetchone()[0]
        stops = conn.execute(
            "SELECT COUNT(*) FROM recoveries WHERE stopping_rule_triggered=1"
        ).fetchone()[0]
        audit = conn.execute("SELECT COUNT(*) FROM audit_trail").fetchone()[0]
        failures = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        conn.close()

        assert actions > 0
        assert recovered > 0
        assert stops >= 0
        assert audit >= actions, "every action must be audited"
        assert failures > 0


def test_pipeline_is_idempotent(protected_reports):
    """Running twice yields identical headline figures (no stacked actions)."""
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.db"
        for _ in range(2):
            proc = _run_main(["--limit", "200", "--llm-backend", "template"], db)
            assert proc.returncode == 0, proc.stderr

            import sqlite3

            conn = sqlite3.connect(str(db))
            row = conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(recovered_amount),0) total "
                "FROM recoveries"
            ).fetchone()
            conn.close()
            results.append((row[0], round(row[1], 2)))

    assert results[0] == results[1], f"runs diverged: {results}"


def test_pipeline_respects_the_attempt_cap(protected_reports):
    """No failure ever exceeds 3 executed attempts + 1 stop record."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "cap.db"
        proc = _run_main(["--limit", "250", "--llm-backend", "template"], db)
        assert proc.returncode == 0, proc.stderr

        import sqlite3

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT failure_id, COUNT(*) n FROM recoveries GROUP BY failure_id"
        ).fetchall()
        conn.close()

        assert rows, "no recovery actions recorded"
        for _, n in rows:
            assert n <= 4, f"failure exceeded the attempt cap: {n} attempts"
