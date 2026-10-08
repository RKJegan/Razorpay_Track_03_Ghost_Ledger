"""
Ghost Ledger v3 — alternate payment-method suggestion (B7).

After a customer's card has failed ``after_card_failures`` times on the same
recovery, suggest other methods (UPI, netbanking, ...). The suggestion is a
list of method names chosen by a fixed order in the playbook, filtered by gateway
health. A method that is verified degraded is never suggested. An unknown
method may still be suggested, because it has no bad evidence against it.

Trust boundary: no LLM. The customer-facing wording lives in the playbook
templates, and this module produces only method names.
"""

from __future__ import annotations

from typing import Iterable


def suggest_methods(
    card_failures: int,
    threshold: int | None,
    candidates: Iterable[str],
    degraded: Iterable[str] = (),
) -> tuple[str, ...]:
    """
    Return the methods to suggest, in playbook order. Empty when not triggered.

    Parameters
    ----------
    card_failures : int
        Failed card attempts on this recovery so far.
    threshold : int | None
        ``after_card_failures`` from the playbook. None disables suggestions.
    candidates : iterable of str
        Playbook order of alternative methods.
    degraded : iterable of str
        Methods whose gateway health is currently ``degraded`` (never suggested).
    """
    if threshold is None or card_failures < threshold:
        return ()
    bad = set(degraded)
    return tuple(m for m in candidates if m != "card" and m not in bad)
