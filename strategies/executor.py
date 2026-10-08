"""
Ghost Ledger v3 — playbook executor (B1, side effects).

:class:`PlaybookExecutor.apply` turns a routed :class:`StrategyPlan` into
recorded facts: events in the recovery log and dunning touches in
``dunning_touches``. It does not create payment links. The runner does that,
through the policy-gated executor, after this step.

Every write is an append or an idempotent insert. Replaying the same plan
does not duplicate dunning touches.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database import recovery_store
from strategies import dunning
from strategies.playbooks import Playbook
from strategies.router import ACTION_SCHEDULE_RETRY, StrategyPlan

EXECUTOR_NAME = "strategy_executor"


class PlaybookExecutor:
    """Records a plan as events and schedules its dunning touches."""

    def apply(
        self,
        recovery_id: str,
        plan: StrategyPlan,
        playbook: Playbook,
        *,
        amount_inr: float,
        txn_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Write the plan's events and dunning touches. Returns a small summary."""
        recovery_store.append_event(
            recovery_id, "playbook_selected", plan.to_dict(), created_by=EXECUTOR_NAME,
        )

        if plan.suggested_methods:
            recovery_store.append_event(
                recovery_id, "method_suggested",
                {"methods": list(plan.suggested_methods)}, created_by=EXECUTOR_NAME,
            )

        if plan.failover_to:
            recovery_store.append_event(
                recovery_id, "gateway_failover",
                {"to": plan.failover_to, "reasons": list(plan.reasons)}, created_by=EXECUTOR_NAME,
            )

        if plan.action == ACTION_SCHEDULE_RETRY and plan.retry_at is not None:
            recovery_store.append_event(
                recovery_id, "retry_scheduled",
                {
                    "run_at": plan.retry_at.isoformat(sep=" ", timespec="seconds"),
                    "rule": plan.timing_rule,
                    "reasons": list(plan.reasons),
                },
                created_by=EXECUTOR_NAME,
            )

        inserted = 0
        if plan.touches:
            next_step = dunning.next_step_for(plan.action, plan.suggested_methods)
            messages = {
                t.touch_no: dunning.render_message(
                    playbook.templates[t.template],
                    amount_inr=amount_inr, txn_id=txn_id, next_step=next_step,
                )
                for t in playbook.touches
            }
            inserted = dunning.schedule_touches(
                recovery_id, plan.touches, base=now, messages=messages,
            )
            if inserted:
                recovery_store.append_event(
                    recovery_id, "dunning_scheduled",
                    {"touches": len(plan.touches), "new": inserted}, created_by=EXECUTOR_NAME,
                )
        return {"touches_inserted": inserted, "action": plan.action}
