"""
FR-005 — Autopsy Report Generator.

Turns a diagnosed failure into a 2–3 sentence plain-English explanation.

Constraints (build prompt, hard constraint 1)
---------------------------------------------
**The LLM is never in the money path.** It produces prose and nothing else.
It does not choose an amount, does not decide to retry, does not decide to
stop. Those are policy-engine decisions and this module cannot influence them.
If the LLM is unreachable, the system falls back to a deterministic template
and keeps working — the prose is cosmetic, the pipeline is not.

Backends (``LLM_BACKEND``)
--------------------------
``ollama``    local Ollama HTTP endpoint (default for development)
``openai``    OpenAI-compatible chat completions
``template``  no model, no network, fully deterministic

Hallucination guard
-------------------
Every generated report is parsed for numeric tokens, and any number that does
not appear in the structured facts passed to the prompt is flagged. Flagged
reports are still stored (hiding them would be worse) but they are counted and
surfaced, per the spec's "spot-check for hallucinated numbers".
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from config import (
    CAUSE_LABELS_HUMAN,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OLLAMA_BASE_URL,
)

NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

# Words that legitimately contain digits but are not claims about the data.
_ALLOWED_BARE_NUMBERS = {"2", "3"}


@dataclass
class AutopsyReport:
    """
    A generated explanation, with full provenance.

    Attributes
    ----------
    failure_id : str
        Failure explained.
    text : str
        The 2–3 sentence explanation.
    model : str
        Which backend/model produced it, e.g. ``ollama:llama3.2:3b``.
    basis : str
        Human-readable description of what the text was derived from.
    hallucination_flags : list[str]
        Numeric tokens in the text that were not present in the input facts.
    degraded : bool
        True when the requested LLM was unavailable and a fallback was used.
    latency_ms : float
        Generation time.
    """

    failure_id: str
    text: str
    model: str
    basis: str
    hallucination_flags: list[str]
    degraded: bool
    latency_ms: float


def build_facts(
    failure_id: str,
    transaction_id: str,
    cause: str,
    confidence: float,
    amount_inr: float,
    txn_type: str,
    payment_method: str,
    gateway: str,
    error_code: str | None,
    timestamp: str,
    attempt_count: int = 0,
    recovered_amount: float = 0.0,
) -> dict[str, Any]:
    """
    Assemble the structured fact block that grounds the explanation.

    Only these values are handed to the model, so any number in the output
    that is not in here is by definition invented.

    Parameters
    ----------
    failure_id : str
        Failure identifier.
    transaction_id : str
        Originating transaction.
    cause : str
        Diagnosed root cause.
    confidence : float
        Diagnoser confidence, 0–1.
    amount_inr : float
        Amount at stake.
    txn_type : str
        ``one_off`` or ``subscription``.
    payment_method : str
        ``upi`` / ``card`` / ``netbanking``.
    gateway : str
        Acquiring gateway.
    error_code : str | None
        Raw decline code.
    timestamp : str
        When the failure occurred.
    attempt_count : int, optional
        Recovery attempts made so far.
    recovered_amount : float, optional
        Rupees recovered so far.

    Returns
    -------
    dict[str, Any]
        The fact block.
    """
    return {
        "failure_id": failure_id,
        "transaction_id": transaction_id,
        "root_cause": cause,
        "root_cause_human": CAUSE_LABELS_HUMAN.get(cause, cause),
        "confidence_pct": round(confidence * 100, 1),
        "amount_inr": round(amount_inr, 2),
        "transaction_type": txn_type,
        "payment_method": payment_method,
        "gateway": gateway,
        "error_code": error_code or "unknown",
        "failure_timestamp": timestamp,
        "recovery_attempts": attempt_count,
        "recovered_amount_inr": round(recovered_amount, 2),
    }


SYSTEM_PROMPT = (
    "You are a payments analyst writing incident notes for a merchant. "
    "Write exactly 2-3 short sentences, plain English, no bullet points, no "
    "headings, no markdown. Use ONLY the numbers supplied in the FACTS block. "
    "Never invent an amount, a date, a count, or a percentage. Do not offer "
    "recommendations or next steps. Explain what failed and why."
)


def build_prompt(facts: dict[str, Any]) -> str:
    """
    Build the structured, data-injected user prompt.

    Parameters
    ----------
    facts : dict[str, Any]
        Output of :func:`build_facts`.

    Returns
    -------
    str
        The user-turn prompt.
    """
    return (
        "FACTS (these are the only values you may reference):\n"
        f"{json.dumps(facts, indent=2)}\n\n"
        "Explain this failed payment in 2-3 sentences for the merchant."
    )


def _template_report(facts: dict[str, Any]) -> str:
    """
    Deterministic fallback explanation, generated with no model.

    Parameters
    ----------
    facts : dict[str, Any]
        The fact block.

    Returns
    -------
    str
        A 2–3 sentence explanation built from templates.
    """
    cause = facts["root_cause_human"]
    amount = f"INR {facts['amount_inr']:,.2f}"
    method = str(facts["payment_method"]).upper()
    gateway = str(facts["gateway"]).upper()
    code = facts["error_code"]
    when = facts["failure_timestamp"]
    kind = (
        "a subscription mandate debit"
        if facts["transaction_type"] == "subscription"
        else "a one-off payment"
    )
    conf = facts["confidence_pct"]

    s1 = (
        f"{kind.capitalize()} of {amount} via {method} on {gateway} failed at "
        f"{when} with decline code {code}."
    )
    s2 = (
        f"The diagnoser attributes this to {cause.lower()} with {conf}% "
        f"confidence."
    )
    if facts["recovery_attempts"] > 0:
        rec = facts["recovered_amount_inr"]
        s3 = (
            f"After {facts['recovery_attempts']} recovery attempt(s), "
            f"INR {rec:,.2f} has been recovered so far."
            if rec > 0
            else f"After {facts['recovery_attempts']} recovery attempt(s), "
            f"no money has been recovered yet."
        )
    else:
        s3 = "No recovery attempt has been made yet."
    return f"{s1} {s2} {s3}"


def _call_ollama(prompt: str) -> tuple[str, str]:
    """
    Call a local Ollama endpoint.

    Parameters
    ----------
    prompt : str
        User-turn prompt.

    Returns
    -------
    tuple[str, str]
        (generated text, model label).

    Raises
    ------
    RuntimeError
        On any transport or HTTP error.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": "qwen3:4b-instruct-2507-q4_K_M",
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": LLM_TEMPERATURE},
        },
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()
    text = (body.get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError("empty completion from Ollama")
    return text, f"ollama:{LLM_MODEL}"


def _call_openai(prompt: str) -> tuple[str, str]:
    """
    Call an OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    prompt : str
        User-turn prompt.

    Returns
    -------
    tuple[str, str]
        (generated text, model label).

    Raises
    ------
    RuntimeError
        When no API key is configured, or on transport/HTTP error.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    resp = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": 200,
        },
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["message"]["content"].strip()
    return text, f"openai:{LLM_MODEL}"


def check_hallucinated_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """
    Flag numeric tokens in the text that are not present in the facts.

    Parameters
    ----------
    text : str
        Generated explanation.
    facts : dict[str, Any]
        The fact block that grounded it.

    Returns
    -------
    list[str]
        Suspicious tokens. Empty means every number traced back to the facts.
    """
    allowed: set[str] = set()
    for value in facts.values():
        if isinstance(value, (int, float)):
            allowed.add(f"{value:,}")
            allowed.add(f"{value:.2f}")
            allowed.add(f"{value:.1f}")
            allowed.add(str(int(value)) if float(value).is_integer() else str(value))
            allowed.add(str(value))
        elif isinstance(value, str):
            allowed.update(NUMBER_RE.findall(value))
    allowed |= _ALLOWED_BARE_NUMBERS

    flags = []
    for token in NUMBER_RE.findall(text):
        if token not in allowed:
            try:
                numeric = float(token.replace(",", ""))
            except ValueError:
                continue
            if not any(
                abs(numeric - float(a.replace(",", ""))) < 0.011
                for a in allowed
                if _is_numeric(a)
            ):
                flags.append(token)
    return flags


def _is_numeric(token: str) -> bool:
    """Return True when a token parses as a number."""
    try:
        float(token.replace(",", ""))
        return True
    except ValueError:
        return False


def generate_autopsy(
    failure_id: str,
    facts: dict[str, Any],
    backend: str | None = None,
) -> AutopsyReport:
    """
    Generate one autopsy explanation.

    Never raises: on any backend failure it falls back to the deterministic
    template and marks the report ``degraded``.

    Parameters
    ----------
    failure_id : str
        Failure being explained.
    facts : dict[str, Any]
        Grounding facts.
    backend : str, optional
        Override ``LLM_BACKEND``.

    Returns
    -------
    AutopsyReport
        The explanation with provenance and hallucination flags.
    """
    backend = (backend or LLM_BACKEND).lower()
    prompt = build_prompt(facts)
    basis = (
        "structured facts injected from the diagnosed failure "
        f"(failure {failure_id}); model writes prose only"
    )

    t0 = time.time()
    text: str
    model_label: str
    degraded = False
    try:
        if backend == "ollama":
            text, model_label = _call_ollama(prompt)
        elif backend == "openai":
            text, model_label = _call_openai(prompt)
        else:
            text, model_label = _template_report(facts), "template-v1"
        # A blank completion is a failure, not a result. Without this check a
        # model that returns an empty string would silently produce an empty
        # autopsy rather than falling back to the template.
        if not text or not text.strip():
            raise RuntimeError("empty completion from backend")
    except Exception as exc:
        # Text-only component: degrade, do not crash.
        text = _template_report(facts)
        model_label = "template-v1 (fallback)"
        degraded = True
        print(
            f"[autopsy] {backend} unavailable ({type(exc).__name__}); "
            f"using deterministic template.",
            flush=True,
        )

    latency = (time.time() - t0) * 1000
    flags = check_hallucinated_numbers(text, facts)
    return AutopsyReport(
        failure_id=failure_id,
        text=text,
        model=model_label,
        basis=basis,
        hallucination_flags=flags,
        degraded=degraded,
        latency_ms=round(latency, 1),
    )


def persist(report: AutopsyReport) -> None:
    """
    Store an autopsy report in the ``autopsy_reports`` table.

    Parameters
    ----------
    report : AutopsyReport
        Report to persist.
    """
    from database import db_client

    db_client.execute(
        "INSERT OR REPLACE INTO autopsy_reports "
        "(failure_id, report_text, model, basis, generated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            report.failure_id,
            report.text,
            report.model,
            report.basis,
            datetime.now().isoformat(sep=" ", timespec="seconds"),
        ),
    )
