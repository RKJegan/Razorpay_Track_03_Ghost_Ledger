"""
FR-006 — Audit trail.

Every diagnosis, every policy check and every agent action is written here
**before execution where the action is permitted, and instead-of execution
where it is not**. NFR-004 and Definition of Done require 100% of financial
actions to be traceable, so this module is deliberately hard to bypass: it
never raises on logging failure, and it never silently drops a record.

Schema (spec §11, verbatim):
    audit_trail(id, timestamp, component, action, input_data, output_data,
                decision_reason, success)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from database import db_client

# Components that write to the trail. Kept as constants so a typo cannot
# create a phantom component name that the dashboard never queries.
COMPONENTS = (
    "synthetic_generator",
    "diagnoser",
    "policy_engine",
    "payment_failure_agent",
    "subscription_agent",
    "razorpay_client",
    "autopsy_reporter",
    "pipeline",
)


def _now() -> str:
    """
    Return the current timestamp in the project's canonical format.

    Returns
    -------
    str
        ``YYYY-MM-DD HH:MM:SS``
    """
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def log(
    component: str,
    action: str,
    input_data: Any = None,
    output_data: Any = None,
    decision_reason: str | None = None,
    success: bool = True,
    entity_id: str | None = None,
) -> str:
    """
    Append one record to the audit trail.

    Parameters
    ----------
    component : str
        Originating component; should be one of :data:`COMPONENTS`.
    action : str
        What was done, e.g. ``policy_check``, ``create_payment_link``.
    input_data : Any, optional
        JSON-serialisable inputs.
    output_data : Any, optional
        JSON-serialisable outputs.
    decision_reason : str, optional
        Why this happened. Mandatory in practice for policy checks.
    success : bool, optional
        Whether the action succeeded.
    entity_id : str, optional
        Related failure/transaction id, for traceability.

    Returns
    -------
    str
        The generated audit record id.

    Notes
    -----
    Never raises. A logging failure must not take down the pipeline, but it is
    surfaced on stderr so it is impossible to miss during a demo.
    """
    record_id = f"aud_{uuid.uuid4().hex[:20]}"
    try:
        db_client.execute(
            "INSERT INTO audit_trail "
            "(id, timestamp, component, action, input_data, output_data, "
            " decision_reason, success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id,
                _now(),
                component,
                action,
                db_client.to_json(_with_entity(input_data, entity_id)),
                db_client.to_json(output_data),
                decision_reason,
                int(bool(success)),
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive by design
        print(
            f"[audit] WARNING: failed to write audit record "
            f"({component}/{action}): {exc}",
            flush=True,
        )
    return record_id


def _with_entity(data: Any, entity_id: str | None) -> Any:
    """
    Attach the entity id to the logged input payload.

    Parameters
    ----------
    data : Any
        Original input payload.
    entity_id : str, optional
        Entity id to attach.

    Returns
    -------
    Any
        Payload with ``entity_id`` merged in when applicable.
    """
    if entity_id is None:
        return data
    if isinstance(data, dict):
        return {**data, "entity_id": entity_id}
    return {"value": data, "entity_id": entity_id}


def log_policy_check(
    action: str,
    decision: Any,
    payload: dict[str, Any],
    entity_id: str | None = None,
) -> str:
    """
    Log a policy-engine ruling. **Called before execution, pass or fail.**

    Parameters
    ----------
    action : str
        The action that was evaluated.
    decision : PolicyDecision
        The ruling.
    payload : dict[str, Any]
        The action payload that was evaluated.
    entity_id : str, optional
        Related failure id.

    Returns
    -------
    str
        Audit record id.
    """
    return log(
        component="policy_engine",
        action=f"policy_check:{action}",
        input_data=payload,
        output_data={
            "allowed": decision.allowed,
            "reason_code": decision.reason_code,
            "rule": decision.rule,
            "stopping_rule_triggered": decision.stopping_rule_triggered,
            "stopping_reason": decision.stopping_reason,
            "requires_approval": decision.requires_approval,
        },
        decision_reason=decision.reason,
        success=decision.allowed,
        entity_id=entity_id,
    )


def log_action(
    agent_name: str,
    action: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    decision_reason: str,
    success: bool,
    entity_id: str | None = None,
) -> str:
    """
    Log an executed (or refused) agent action.

    Parameters
    ----------
    agent_name : str
        Acting agent.
    action : str
        Action performed.
    payload : dict[str, Any]
        What was sent.
    result : dict[str, Any]
        What came back.
    decision_reason : str
        Why.
    success : bool
        Outcome.
    entity_id : str, optional
        Related failure id.

    Returns
    -------
    str
        Audit record id.
    """
    return log(
        component=agent_name,
        action=action,
        input_data=payload,
        output_data=result,
        decision_reason=decision_reason,
        success=success,
        entity_id=entity_id,
    )


def recent(limit: int = 50, component: str | None = None) -> list[dict[str, Any]]:
    """
    Read the most recent audit records.

    Parameters
    ----------
    limit : int, optional
        Maximum records to return.
    component : str, optional
        Filter to a single component.

    Returns
    -------
    list[dict[str, Any]]
        Records, newest first, with JSON fields decoded.
    """
    if component:
        rows = db_client.query(
            "SELECT * FROM audit_trail WHERE component = ? "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (component, limit),
        )
    else:
        rows = db_client.query(
            "SELECT * FROM audit_trail ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (limit,),
        )
    return [_decode(r) for r in rows]


def _decode(row: Any) -> dict[str, Any]:
    """
    Parse a row into a plain dict with JSON fields decoded.

    Parameters
    ----------
    row : sqlite3.Row
        Raw row.

    Returns
    -------
    dict[str, Any]
        Decoded record.
    """
    out = dict(row)
    for key in ("input_data", "output_data"):
        raw = out.get(key)
        if raw:
            try:
                out[key] = json.loads(raw)
            except (ValueError, TypeError):
                pass
    return out


def count() -> int:
    """
    Return the total number of audit records.

    Returns
    -------
    int
        Row count.
    """
    return int(db_client.scalar("SELECT COUNT(*) FROM audit_trail") or 0)
