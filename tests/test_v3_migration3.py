"""
Migration 3 must upgrade a v2-era database that has recovery_cases WITHOUT
customer_id, keep its rows, and be safe to run twice.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

from database import migrations


def _legacy_engine(tmp_path):  # noqa: ANN001
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE recovery_cases (id TEXT PRIMARY KEY, merchant_id TEXT, amount_inr REAL)"
        )
        conn.exec_driver_sql("INSERT INTO recovery_cases VALUES ('rcv_old', 'm1', 499.0)")
    return engine


def _columns(conn, table: str) -> set[str]:  # noqa: ANN001
    return {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}


def test_adds_customer_id_to_a_legacy_table_and_keeps_rows(tmp_path) -> None:  # noqa: ANN001
    engine = _legacy_engine(tmp_path)
    with engine.begin() as conn:
        assert "customer_id" not in _columns(conn, "recovery_cases")
        assert migrations._add_column_if_missing(conn, "recovery_cases", "customer_id", "TEXT") is True
        row = conn.execute(text("SELECT id, amount_inr, customer_id FROM recovery_cases")).one()
    assert row[0] == "rcv_old" and row[1] == 499.0
    assert row[2] is None  # existing rows keep working; the new column is empty


def test_guard_is_safe_to_run_twice(tmp_path) -> None:  # noqa: ANN001
    engine = _legacy_engine(tmp_path)
    with engine.begin() as conn:
        first = migrations._add_column_if_missing(conn, "recovery_cases", "customer_id", "TEXT")
    with engine.begin() as conn:
        second = migrations._add_column_if_missing(conn, "recovery_cases", "customer_id", "TEXT")
        assert _columns(conn, "recovery_cases") >= {"id", "merchant_id", "amount_inr", "customer_id"}
    assert first is True and second is False


def test_migration_three_is_registered_and_recorded() -> None:
    versions = [v for v, _, _ in migrations.MIGRATIONS]
    assert 3 in versions
    assert migrations.LATEST_VERSION >= 3
    applied = {version: applied_at for version, _, applied_at in migrations.migration_status()}
    assert applied[3] is not None
