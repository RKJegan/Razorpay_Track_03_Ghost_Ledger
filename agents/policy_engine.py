"""
FR-003 — Recovery Policy Engine (guardrails).

Pure Python. 100% deterministic. No LLM, no network, no randomness, no clock
reads inside the decision itself. The same inputs always produce the same
output — that property is what makes it safe to put in front of an agent that
can move money.

Rules (from GHOST_LEDGER_PROJECT_SPEC §6 FR-003)
------------------------------------------------
R1  No action over POLICY_MAX_AUTO_APPROVE_INR (₹10,000) unless the action
    carries an explicit approval flag.
R2  At most POLICY_MAX_ATTEMPTS_PER_FAILURE (3) recovery attempts per customer
    per failure.
R3  On the POLICY_STOP_ON_FAILED_ATTEMPT-th (3rd) FAILED attempt: STOP,
    escalate, and log the reason. The agent halts. It does not retry, does not
    "try once more", and cannot override.
R4  Every action is logged before execution, pass or fail.

Where this sits
---------------
    Diagnoser (AI)  ->  POLICY ENGINE (this file, deterministic)  ->  Agent
The AI proposes a cause; only this module decides whether an action may run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import (
    POLICY_MAX_ATTEMPTS_PER_FAILURE,
    POLICY_MAX_AUTO_APPROVE_INR,
    POLICY_STOP_ON_FAILED_ATTEMPT,
)

# Canonical reason codes, so logs and tests can assert on stable strings
# rather than prose that drifts.
REASON_OK = "OK"
REASON_AMOUNT_LIMIT = "AMOUNT_EXCEEDS_AUTO_APPROVE_LIMIT"
REASON_ATTEMPT_LIMIT = "ATTEMPT_LIMIT_REACHED"
REASON_STOPPING_RULE = "STOPPING_RULE_3_FAILED_ATTEMPTS"
REASON_INVALID_AMOUNT = "INVALID_AMOUNT"
REASON_UNKNOWN_AGENT = "UNKNOWN_AGENT"

STOP_REASON_TEMPLATE = (
    "{n} failed recovery attempts for customer {customer} on failure {failure} "
    "reached the policy stopping rule (max {limit}). No further automated "
    "attempts. Escalated for manual review."
)

KNOWN_AGENTS = ("payment_failure_agent", "subscription_agent")


@dataclass(frozen=True)
class RecoveryAction:
    """
    A proposed recovery action, before the policy engine has ruled on it.

    Attributes
    ----------
    failure_id : str
        Identifier of the failure being acted on.
    customer_id : str
        Customer the action targets (part of the per-customer attempt key).
    agent_name : str
        Name of the agent requesting the action.
    action_type : str
        e.g. ``create_payment_link`` or ``retry_mandate``.
    amount_inr : float
        Amount the action would attempt to recover.
    attempt_number : int
        1-based sequence number of this attempt for this failure.
    approved : bool
        Explicit human approval flag. Required for amounts above the
        auto-approve ceiling.
    metadata : dict[str, Any]
        Optional additional context carried into the audit log.
    """

    failure_id: str
    customer_id: str
    agent_name: str
    action_type: str
    amount_inr: float
    attempt_number: int = 1
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """
    The engine's ruling on a :class:`RecoveryAction`.

    Attributes
    ----------
    allowed : bool
        True only if the action may be executed.
    reason_code : str
        One of the module-level ``REASON_*`` constants.
    reason : str
        Human-readable explanation, stored in the audit trail.
    stopping_rule_triggered : bool
        True when R3 fired. The agent must halt permanently for this failure.
    stopping_reason : str | None
        Escalation text when the stopping rule fires.
    requires_approval : bool
        True when the action would be allowed if a human approved it.
    rule : str
        Identifier of the rule that decided the outcome.
    """

    allowed: bool
    reason_code: str
    reason: str
    stopping_rule_triggered: bool = False
    stopping_reason: str | None = None
    requires_approval: bool = False
    rule: str = "R0"


class PolicyEngine:
    """
    Stateless, deterministic rule evaluator.

    The engine deliberately takes attempt history as explicit inputs rather
    than querying a database. That keeps it pure, unit-testable without
    fixtures, and immune to hidden state. Callers supply the counts.

    Examples
    --------
    >>> engine = PolicyEngine()
    >>> action = RecoveryAction(
    ...     failure_id="f1", customer_id="c1", agent_name="payment_failure_agent",
    ...     action_type="create_payment_link", amount_inr=499.0, attempt_number=1,
    ... )
    >>> engine.evaluate(action, prior_failed_attempts=0).allowed
    True
    """

    def __init__(
        self,
        max_auto_approve_inr: float | None = None,
        max_attempts: int | None = None,
        stop_on_failed_attempt: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_auto_approve_inr : float, optional
            Override for R1 ceiling. Defaults to config.
        max_attempts : int, optional
            Override for R2 limit. Defaults to config.
        stop_on_failed_attempt : int, optional
            Override for R3 threshold. Defaults to config.
        """
        self.max_auto_approve_inr = (
            POLICY_MAX_AUTO_APPROVE_INR if max_auto_approve_inr is None else max_auto_approve_inr
        )
        self.max_attempts = (
            POLICY_MAX_ATTEMPTS_PER_FAILURE if max_attempts is None else max_attempts
        )
        self.stop_on_failed_attempt = (
            POLICY_STOP_ON_FAILED_ATTEMPT
            if stop_on_failed_attempt is None
            else stop_on_failed_attempt
        )

    # -- helpers -----------------------------------------------------------
    def attempt_key(self, action: RecoveryAction) -> tuple[str, str]:
        """
        Return the (customer_id, failure_id) key attempts are counted against.

        Parameters
        ----------
        action : RecoveryAction
            The proposed action.

        Returns
        -------
        tuple[str, str]
            Grouping key for R2 and R3.
        """
        return (action.customer_id, action.failure_id)

    # -- main --------------------------------------------------------------
    def evaluate(
        self,
        action: RecoveryAction,
        prior_failed_attempts: int = 0,
        prior_total_attempts: int = 0,
    ) -> PolicyDecision:
        """
        Rule on a proposed action.

        Rule order is deliberate: the stopping rule (R3) is checked FIRST,
        because a failure that has already burned its attempts must halt even
        if a later rule would also have blocked it. That guarantees the
        audit trail records a STOP, not a generic denial.

        Parameters
        ----------
        action : RecoveryAction
            The proposed action.
        prior_failed_attempts : int, optional
            Number of prior attempts for this (customer, failure) that failed.
        prior_total_attempts : int, optional
            Number of prior attempts of any outcome for this (customer, failure).

        Returns
        -------
        PolicyDecision
            The ruling. ``allowed`` is the only field that gates execution.
        """
        # --- input validation (never silently pass on bad input) -----------
        if action.amount_inr is None or action.amount_inr <= 0:
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_INVALID_AMOUNT,
                reason=(
                    f"Invalid amount {action.amount_inr!r}: recovery actions must "
                    f"carry a positive amount."
                ),
                rule="R0",
            )
        if action.agent_name not in KNOWN_AGENTS:
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_UNKNOWN_AGENT,
                reason=(
                    f"Unknown agent {action.agent_name!r}. Known agents: "
                    f"{', '.join(KNOWN_AGENTS)}."
                ),
                rule="R0",
            )

        # --- R3: stopping rule FIRST --------------------------------------
        if prior_failed_attempts >= self.stop_on_failed_attempt:
            stopping_reason = STOP_REASON_TEMPLATE.format(
                n=prior_failed_attempts,
                customer=action.customer_id,
                failure=action.failure_id,
                limit=self.stop_on_failed_attempt,
            )
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_STOPPING_RULE,
                reason=stopping_reason,
                stopping_rule_triggered=True,
                stopping_reason=stopping_reason,
                rule="R3",
            )

        # --- R1: amount ceiling -------------------------------------------
        if action.amount_inr > self.max_auto_approve_inr and not action.approved:
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_AMOUNT_LIMIT,
                reason=(
                    f"Amount INR {action.amount_inr:,.2f} exceeds the auto-approve "
                    f"ceiling of INR {self.max_auto_approve_inr:,.2f} and the action "
                    f"carries no explicit approval flag. Requires human approval."
                ),
                requires_approval=True,
                rule="R1",
            )

        # --- R2: attempt cap ------------------------------------------------
        if prior_total_attempts >= self.max_attempts:
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_ATTEMPT_LIMIT,
                reason=(
                    f"Attempt {action.attempt_number} would exceed the cap of "
                    f"{self.max_attempts} recovery attempts for customer "
                    f"{action.customer_id} on failure {action.failure_id} "
                    f"({prior_total_attempts} already recorded)."
                ),
                rule="R2",
            )
        if action.attempt_number > self.max_attempts:
            return PolicyDecision(
                allowed=False,
                reason_code=REASON_ATTEMPT_LIMIT,
                reason=(
                    f"Attempt number {action.attempt_number} exceeds the cap of "
                    f"{self.max_attempts} recovery attempts per customer per failure."
                ),
                rule="R2",
            )

        # --- passed ---------------------------------------------------------
        note = REASON_OK
        if action.amount_inr > self.max_auto_approve_inr:
            note = f"{REASON_OK} (above auto-approve ceiling, explicit approval present)"
        return PolicyDecision(
            allowed=True,
            reason_code=REASON_OK,
            reason=(
                f"Action permitted: {action.action_type} for INR "
                f"{action.amount_inr:,.2f} on failure {action.failure_id} "
                f"(attempt {action.attempt_number} of {self.max_attempts}; "
                f"{prior_failed_attempts} prior failures)."
            ),
            rule="R0",
        )

    def describe(self) -> str:
        """
        Return a human-readable summary of the active limits.

        Returns
        -------
        str
            Multi-line description of the three rules.
        """
        return (
            f"R1  no action above INR {self.max_auto_approve_inr:,.2f} "
            f"without an explicit approval flag\n"
            f"R2  at most {self.max_attempts} recovery attempts per customer "
            f"per failure\n"
            f"R3  on the {self.stop_on_failed_attempt}rd failed attempt: STOP, "
            f"escalate, log the reason (checked first, cannot be overridden)"
        )


# ---------------------------------------------------------------------------
# DB-backed attempt counting (thin, optional helper)
# ---------------------------------------------------------------------------
def count_prior_attempts(customer_id: str, failure_id: str) -> tuple[int, int]:
    """
    Count prior recovery attempts for a (customer, failure) pair.

    Kept out of the engine itself so the engine stays pure; this is the
    adapter that feeds it real counts.

    Parameters
    ----------
    customer_id : str
        Customer identifier.
    failure_id : str
        Failure identifier.

    Returns
    -------
    tuple[int, int]
        (number of prior FAILED or STOPPED attempts, total prior attempts).
    """
    from database import db_client

    row_total = db_client.scalar(
        "SELECT COUNT(*) FROM recoveries WHERE failure_id = ?", (failure_id,)
    )
    row_failed = db_client.scalar(
        "SELECT COUNT(*) FROM recoveries WHERE failure_id = ? "
        "AND outcome IN ('fail', 'stopped')",
        (failure_id,),
    )
    # customer_id is part of the attempt key; guard against id collisions by
    # confirming the failure belongs to this customer.
    owner = db_client.scalar(
        "SELECT t.customer_id FROM failures f "
        "JOIN transactions t ON t.id = f.transaction_id "
        "WHERE f.id = ?",
        (failure_id,),
    )
    if owner is not None and owner != customer_id:
        raise ValueError(
            f"failure {failure_id} belongs to customer {owner}, not {customer_id}"
        )
    return int(row_failed or 0), int(row_total or 0)
