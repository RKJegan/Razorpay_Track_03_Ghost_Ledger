"""
Single source of truth for every number the dashboard shows.

The build prompt (step 13) requires the dashboard headline to match the audit
trail totals **exactly**. The only reliable way to guarantee that is for both
to call the same functions — so `main.py` and every dashboard component import
from here rather than writing their own SQL.

Every function reads the database at call time, so a stale dashboard is
impossible by construction.
"""

from __future__ import annotations

from typing import Any

from database import db_client
from diagnoser.eval_holdout import load_split_method

# Reused so the dashboard's SQL matches the pipeline's exactly.
_HOLDOUT_JOIN = """
    FROM recoveries r
    JOIN failures f ON f.id = r.failure_id
    JOIN transactions t ON t.id = f.transaction_id
    WHERE t.is_holdout = 1
"""


def compute_headline_metrics() -> dict[str, Any]:
    """
    Compute the headline figures for the held-out batch.

    Returns
    -------
    dict[str, Any]
        Keys: ``n_transactions``, ``amount_at_risk_inr``,
        ``amount_recovered_inr``, ``recovery_rate``, ``stopping_rule_events``,
        ``recovery_actions``, ``audit_records``, ``reconciled``, ``basis``,
        ``split_method``, ``outcomes``, ``n_failures``.
    """
    n_txns = int(
        db_client.scalar("SELECT COUNT(*) FROM transactions WHERE is_holdout=1") or 0
    )
    n_failures = int(
        db_client.scalar(
            "SELECT COUNT(*) FROM failures f JOIN transactions t "
            "ON t.id = f.transaction_id WHERE t.is_holdout=1"
        )
        or 0
    )
    at_risk = float(
        db_client.scalar(
            "SELECT COALESCE(SUM(estimated_value),0) FROM failures f "
            "JOIN transactions t ON t.id = f.transaction_id WHERE t.is_holdout=1"
        )
        or 0.0
    )
    recovered = float(db_client.scalar(
        f"SELECT COALESCE(SUM(r.recovered_amount),0) {_HOLDOUT_JOIN}"
    ) or 0.0)
    stops = int(db_client.scalar(
        f"SELECT COUNT(*) {_HOLDOUT_JOIN} AND r.stopping_rule_triggered=1"
    ) or 0)
    actions = int(db_client.scalar(f"SELECT COUNT(*) {_HOLDOUT_JOIN}") or 0)
    audit_rows = int(db_client.scalar("SELECT COUNT(*) FROM audit_trail") or 0)

    # Reconciliation: whole-DB action total must equal the holdout figure,
    # because the pipeline only ever acts on the held-out batch.
    all_actions_total = float(
        db_client.scalar("SELECT COALESCE(SUM(recovered_amount),0) FROM recoveries") or 0.0
    )

    # Per-ATTEMPT-ROW breakdown (every row in `recoveries`; sums to
    # `recovery_actions`). Useful for auditing retry behaviour, but it is NOT
    # "how many failures ended in each state" — a failure that eventually
    # succeeds can still contribute earlier 'fail' rows here.
    attempt_outcomes: dict[str, int] = {}
    for row in db_client.query(
        f"SELECT r.outcome AS outcome, COUNT(*) AS n {_HOLDOUT_JOIN} GROUP BY r.outcome"
    ):
        attempt_outcomes[row["outcome"]] = int(row["n"])

    # Per-FAILURE final outcome (the outcome of each failure's LAST attempt;
    # sums to `n_failures`, not `recovery_actions`). This is what "how many
    # cases succeeded / were stopped / were blocked" actually means, and it is
    # what the pipeline's own console output and headline_metrics.json report
    # — so compute_headline_metrics() must return the same thing here, or the
    # dashboard and the weekly report (which call this function directly)
    # silently disagree with the CLI/JSON output that overwrote this field.
    outcomes: dict[str, int] = {"success": 0, "fail": 0, "stopped": 0, "blocked": 0}
    for row in db_client.query(
        f"""
        WITH last_attempt AS (
            SELECT r.failure_id, r.outcome,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.failure_id ORDER BY r.attempt_number DESC
                   ) AS rn
            FROM recoveries r
        )
        SELECT la.outcome AS outcome, COUNT(*) AS n
        FROM last_attempt la
        JOIN failures f ON f.id = la.failure_id
        JOIN transactions t ON t.id = f.transaction_id
        WHERE la.rn = 1 AND t.is_holdout = 1
        GROUP BY la.outcome
        """
    ):
        outcomes[row["outcome"]] = int(row["n"])

    return {
        "n_transactions": n_txns,
        "n_failures": n_failures,
        "amount_at_risk_inr": round(at_risk, 2),
        "amount_recovered_inr": round(recovered, 2),
        "recovery_rate": round(recovered / at_risk, 4) if at_risk else 0.0,
        "stopping_rule_events": stops,
        "recovery_actions": actions,
        "audit_records": audit_rows,
        "reconciled": abs(all_actions_total - recovered) < 0.01,
        "outcomes": outcomes,
        "attempt_outcomes": attempt_outcomes,
        "basis": "held-out batch",
        "split_method": load_split_method(),
    }


def recovery_over_time() -> list[dict[str, Any]]:
    """
    Daily recovered vs at-risk amounts on the held-out batch.

    Attempts are aggregated per failure in a CTE *before* joining, otherwise a
    failure with 3 attempts contributes its estimated_value three times and
    the at-risk figure is inflated by the average attempt count.

    Returns
    -------
    list[dict[str, Any]]
        One row per day: ``day``, ``at_risk``, ``recovered``,
        ``recovery_rate``, ``attempts``, ``stops``.
    """
    rows = db_client.query(
        """
        WITH rec AS (
            SELECT failure_id,
                   COALESCE(SUM(recovered_amount),0) AS recovered,
                   COUNT(*)                          AS attempts,
                   SUM(CASE WHEN stopping_rule_triggered=1 THEN 1 ELSE 0 END) AS stops
            FROM recoveries
            GROUP BY failure_id
        )
        SELECT substr(t.timestamp,1,10) AS day,
               COUNT(*)                              AS failures,
               COALESCE(SUM(f.estimated_value),0)     AS at_risk,
               COALESCE(SUM(rec.recovered),0)         AS recovered,
               COALESCE(SUM(rec.attempts),0)          AS attempts,
               COALESCE(SUM(rec.stops),0)             AS stops
        FROM failures f
        JOIN transactions t ON t.id = f.transaction_id
        LEFT JOIN rec ON rec.failure_id = f.id
        WHERE t.is_holdout = 1
        GROUP BY day
        ORDER BY day
        """
    )
    out = []
    for r in rows:
        at_risk = float(r["at_risk"] or 0)
        rec = float(r["recovered"] or 0)
        out.append(
            {
                "day": r["day"],
                "failures": int(r["failures"]),
                "at_risk": at_risk,
                "recovered": rec,
                "recovery_rate": (rec / at_risk) if at_risk else 0.0,
                "attempts": int(r["attempts"] or 0),
                "stops": int(r["stops"] or 0),
            }
        )
    return out


def cause_breakdown() -> list[dict[str, Any]]:
    """
    Recovery performance per diagnosed cause bucket.

    Attempts are aggregated per failure in a CTE before joining, so
    ``at_risk`` counts each failure exactly once no matter how many times it
    was retried.

    Returns
    -------
    list[dict[str, Any]]
        One row per cause: ``cause``, ``failures``, ``at_risk``,
        ``recovered``, ``recovery_rate``, ``stops``, ``attempts``.
    """
    rows = db_client.query(
        """
        WITH rec AS (
            SELECT failure_id,
                   COALESCE(SUM(recovered_amount),0) AS recovered,
                   COUNT(*)                          AS attempts,
                   SUM(CASE WHEN stopping_rule_triggered=1 THEN 1 ELSE 0 END) AS stops
            FROM recoveries
            GROUP BY failure_id
        )
        SELECT f.predicted_cause                  AS cause,
               COUNT(*)                           AS failures,
               COALESCE(SUM(f.estimated_value),0) AS at_risk,
               COALESCE(SUM(rec.recovered),0)     AS recovered,
               COALESCE(SUM(rec.attempts),0)      AS attempts,
               COALESCE(SUM(rec.stops),0)         AS stops
        FROM failures f
        JOIN transactions t ON t.id = f.transaction_id
        LEFT JOIN rec ON rec.failure_id = f.id
        WHERE t.is_holdout = 1
        GROUP BY f.predicted_cause
        ORDER BY at_risk DESC
        """
    )
    out = []
    for r in rows:
        at_risk = float(r["at_risk"] or 0)
        rec = float(r["recovered"] or 0)
        out.append(
            {
                "cause": r["cause"],
                "failures": int(r["failures"]),
                "at_risk": at_risk,
                "recovered": rec,
                "recovery_rate": (rec / at_risk) if at_risk else 0.0,
                "attempts": int(r["attempts"] or 0),
                "stops": int(r["stops"] or 0),
            }
        )
    return out


def stopping_rule_events(limit: int = 200) -> list[dict[str, Any]]:
    """
    Return the explicit stopping-rule events, newest first.

    Parameters
    ----------
    limit : int, optional
        Maximum events to return.

    Returns
    -------
    list[dict[str, Any]]
        One row per stopped case, with customer, cause, amount and reason.
    """
    rows = db_client.query(
        """
        SELECT r.id, r.failure_id, r.agent_name, r.attempt_number,
               r.stopping_reason, r.executed_at, f.predicted_cause,
               f.estimated_value, t.customer_id, t.txn_type, t.amount
        FROM recoveries r
        JOIN failures f ON f.id = r.failure_id
        JOIN transactions t ON t.id = f.transaction_id
        WHERE r.stopping_rule_triggered = 1
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(r) for r in rows]

def count_stopping_rule_events() -> int:
    return int(
        db_client.scalar(
            "SELECT COUNT(*) FROM recoveries WHERE stopping_rule_triggered = 1"
        )
        or 0
    )

def recent_audit(limit: int = 500, component: str | None = None) -> list[dict[str, Any]]:
    """
    Return recent audit-trail rows for the filterable table.

    Parameters
    ----------
    limit : int, optional
        Maximum rows.
    component : str, optional
        Filter to one component.

    Returns
    -------
    list[dict[str, Any]]
        Audit records with JSON fields decoded.
    """
    from database.audit_trail import recent

    return recent(limit=limit, component=component)


def diagnoser_metrics() -> dict[str, Any] | None:
    """
    Load the held-out diagnoser metrics produced by ``diagnoser/train.py``.

    Returns
    -------
    dict[str, Any] | None
        The stored report, or None if training has not run.
    """
    from config import REPORTS_DIR

    path = REPORTS_DIR / "holdout_metrics.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def autopsy_stats() -> dict[str, Any]:
    """
    Count autopsy reports and hallucination flags.

    Returns
    -------
    dict[str, Any]
        ``total``, ``models`` (label -> count), ``sample_text``.
    """
    total = int(db_client.scalar("SELECT COUNT(*) FROM autopsy_reports") or 0)
    rows = db_client.query(
        "SELECT model, COUNT(*) AS n FROM autopsy_reports GROUP BY model ORDER BY n DESC"
    )
    sample = db_client.query_one(
        "SELECT failure_id, report_text, model, basis FROM autopsy_reports LIMIT 1"
    )
    return {
        "total": total,
        "models": {r["model"]: int(r["n"]) for r in rows},
        "sample": dict(sample) if sample else None,
    }
