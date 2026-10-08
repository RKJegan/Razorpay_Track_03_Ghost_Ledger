"""
Ghost Ledger v2 - Central configuration.

Single source of truth for every tunable constant, path and policy limit.
Values are read from the environment (.env) where they are secrets or
deployment-specific, and are otherwise fixed here so that a batch run is
reproducible from the seed alone.

No secrets are ever hardcoded here. Anything sensitive must come from .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_OUTPUT_DIR: Path = DATA_DIR / "sample_output"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, SAMPLE_OUTPUT_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env if present (never required to exist - defaults are all safe/offline)
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    """Read a string from the environment, falling back to ``default``."""
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment ('1'/'true'/'yes' are truthy)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------
# Environment / identity
# --------------------------------------------------------------------------
MERCHANT_ID: str = _env_str("MERCHANT_ID", "merchant_demo_001")
DATA_SEED: int = _env_int("DATA_SEED", 42)
DB_PATH: Path = PROJECT_ROOT / _env_str("DB_PATH", "data/ghost_ledger.db")
LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")

# --------------------------------------------------------------------------
# Razorpay (test mode only - never live)
# --------------------------------------------------------------------------
RAZORPAY_KEY_ID: str = _env_str("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET: str = _env_str("RAZORPAY_KEY_SECRET", "")

# When True the client never touches the network and returns Razorpay-shaped
# simulated responses. This is the default: the project is built and judged
# without live credentials, and the same code path runs against real
# test-mode endpoints the moment real rzp_test_* keys are placed in .env.
#
#   RAZORPAY_LIVE_TEST_MODE=0  -> simulated client (default, offline safe)
#   RAZORPAY_LIVE_TEST_MODE=1  -> real test-mode API, auto-falls back to the
#                                 simulator on auth/network/timeout failure
RAZORPAY_LIVE_TEST_MODE: bool = _env_bool("RAZORPAY_LIVE_TEST_MODE", False)
RAZORPAY_TIMEOUT_SECONDS: float = _env_float("RAZORPAY_TIMEOUT_SECONDS", 6.0)

# --------------------------------------------------------------------------
# LLM autopsy reporter (text only - never in the money path)
# --------------------------------------------------------------------------
# Backends:
#   "ollama"    -> local Ollama HTTP endpoint (default for development)
#   "openai"    -> OpenAI-compatible chat completions API
#   "template"  -> deterministic, no model, zero network
LLM_BACKEND: str = _env_str("LLM_BACKEND", "ollama").lower()
LLM_MODEL: str = _env_str("LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY: str = _env_str("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_TIMEOUT_SECONDS: float = _env_float("LLM_TIMEOUT_SECONDS", 12.0)
LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.2)

# --------------------------------------------------------------------------
# Policy engine limits (FR-003) - deterministic, pure Python
# --------------------------------------------------------------------------
# Rule 1: no single recovery action above this amount without an explicit
#         approval flag set on the action payload.
POLICY_MAX_AUTO_APPROVE_INR: float = _env_float("POLICY_MAX_AUTO_APPROVE_INR", 10_000.0)
# Rule 2: at most this many recovery attempts per customer per failure.
POLICY_MAX_ATTEMPTS_PER_FAILURE: int = _env_int("POLICY_MAX_ATTEMPTS_PER_FAILURE", 3)
# Rule 3: on the Nth failed attempt -> STOP, escalate, log the reason.
POLICY_STOP_ON_FAILED_ATTEMPT: int = _env_int("POLICY_STOP_ON_FAILED_ATTEMPT", 3)

# --------------------------------------------------------------------------
# Ghost Ledger v3 — real-time platform (single project, single database)
# --------------------------------------------------------------------------
# Master switch for the cause-specific strategy engine (Track B). When 0 the
# system runs the original one-action-per-cause v2 behaviour. Rollback-safe.
ENABLE_ADVANCED_STRATEGIES: bool = _env_bool("ENABLE_ADVANCED_STRATEGIES", False)

# Shared secret configured on the Razorpay dashboard for the webhook. It is
# NOT the key secret. Never printed or logged. Empty = webhooks refused (503).
RAZORPAY_WEBHOOK_SECRET: str = _env_str("RAZORPAY_WEBHOOK_SECRET", "")

# Hard cap on merchants stored in the database (enforced by a SQL trigger too).
MERCHANT_MAX_COUNT: int = 10

# Largest webhook body accepted, in bytes. Larger bodies are rejected with 413.
WEBHOOK_MAX_BODY_BYTES: int = _env_int("WEBHOOK_MAX_BODY_BYTES", 1_000_000)

# Recovery payments above this amount need a human approval (Track A3).
# Reuses the policy ceiling so there is exactly one number for the rule.
APPROVAL_THRESHOLD_INR: float = POLICY_MAX_AUTO_APPROVE_INR

# Operator key guarding the approval routes. Humans acting on approvals send it
# in the X-Operator-Key header. Empty = approval routes refuse all requests.
OPERATOR_API_KEY: str = _env_str("OPERATOR_API_KEY", "")

# Background jobs (A4). Off by default so tests and batch runs never start
# threads; the run guide turns it on for the live demo.
SCHEDULER_ENABLED: bool = _env_bool("SCHEDULER_ENABLED", False)
JOB_SETTLEMENT_POLL_SECONDS: int = _env_int("JOB_SETTLEMENT_POLL_SECONDS", 180)
JOB_RECONCILE_SECONDS: int = _env_int("JOB_RECONCILE_SECONDS", 300)
JOB_RETRY_SECONDS: int = _env_int("JOB_RETRY_SECONDS", 600)
JOB_CLEAN_LINKS_SECONDS: int = _env_int("JOB_CLEAN_LINKS_SECONDS", 900)
# Exponential backoff after a failed job run: base * 2^(streak-1), capped at max.
JOB_BACKOFF_BASE_SECONDS: int = _env_int("JOB_BACKOFF_BASE_SECONDS", 30)
JOB_BACKOFF_MAX_SECONDS: int = _env_int("JOB_BACKOFF_MAX_SECONDS", 1800)
# Hours after which an unpaid payment link is treated as expired.
PAYMENT_LINK_EXPIRY_HOURS: int = _env_int("PAYMENT_LINK_EXPIRY_HOURS", 24)


# --------------------------------------------------------------------------
# Dataset shape (FR-001)
# --------------------------------------------------------------------------
# Generation is sized by PRESET so that "a large training corpus" and "a fast
# smoke-test run" are the same code path. Any preset value can still be
# overridden individually by the DATASET_DAYS / N_CUSTOMERS /
# AVG_DAILY_TRANSACTIONS / HOLDOUT_DAYS env vars, but overrides are applied
# on top of the preset and are echoed into the manifest so a reported metric
# can always be traced back to the corpus that produced it.
DATASET_PRESETS: dict[str, dict[str, float]] = {
    # Fast, small, for unit tests and a 2-second smoke run.
    "demo": {"days": 30, "customers": 700, "daily": 150, "holdout_frac": 0.20, "holdout_days": 7},
    # Default. Large enough to train the diagnoser properly (~9k failure rows)
    # while keeping generation ~13s and peak memory ~200MB.
    "train": {"days": 180, "customers": 2000, "daily": 500, "holdout_frac": 0.20, "holdout_days": 14},
    # Heaviest setting: a full year, ~23k failure rows. ~33s, ~430MB peak.
    "max": {"days": 365, "customers": 2500, "daily": 600, "holdout_frac": 0.20, "holdout_days": 14},
}

DATASET_PROFILE: str = _env_str("DATASET_PROFILE", "train").lower()
if DATASET_PROFILE not in DATASET_PRESETS:
    DATASET_PROFILE = "train"
_PRESET: dict[str, int] = DATASET_PRESETS[DATASET_PROFILE]

DATASET_DAYS: int = _env_int("DATASET_DAYS", _PRESET["days"])
N_CUSTOMERS: int = _env_int("N_CUSTOMERS", _PRESET["customers"])
AVG_DAILY_TRANSACTIONS: int = _env_int(
    "AVG_DAILY_TRANSACTIONS", _PRESET["daily"]
)

# Fraction of the corpus reserved as the held-out batch (random strategies).
HOLDOUT_FRACTION: float = _env_float("HOLDOUT_FRACTION", _PRESET["holdout_frac"])
# Trailing days reserved as the held-out batch (temporal strategy only).
HOLDOUT_DAYS: int = _env_int("HOLDOUT_DAYS", int(_PRESET["holdout_days"]))

# --------------------------------------------------------------------------
# Split strategy
# --------------------------------------------------------------------------
#   "random_grouped" (default) - Random, but assigned at CUSTOMER level.
#       Every transaction belonging to a held-out customer goes to the holdout,
#       so no customer appears on both sides. This is random in the sense that
#       matters (unbiased w.r.t. time and cause) while blocking the leakage
#       described below.
#
#   "random_rows"   - Random at transaction level. INFLATES METRICS: the same
#       customer's failures land in both train and holdout, and customer-level
#       features (mandate_days_overdue, days_to_card_expiry,
#       customer_prior_failure_count) are near-constant per customer, so the
#       model can memorise customers instead of learning causes. Provided for
#       measurement only - see the leakage experiment in the phase report.
#
#   "temporal"      - Trailing HOLDOUT_DAYS are held out, split by timestamp.
#       Strictest w.r.t. time, but conflates cause-learning with distribution
#       drift (mandates age, cards expire), which depresses scores for reasons
#       unrelated to model quality.
SPLIT_STRATEGY: str = _env_str("SPLIT_STRATEGY", "random_grouped").lower()
if SPLIT_STRATEGY not in {"random_grouped", "random_rows", "temporal"}:
    SPLIT_STRATEGY = "random_grouped"

# Cross-validation: stratified by class, grouped by customer, so that CV folds
# inherit the same leakage guarantees as the outer split.
CV_FOLDS: int = _env_int("CV_FOLDS", 5)
CV_RANDOM_STATE: int = _env_int("CV_RANDOM_STATE", 42)

# --------------------------------------------------------------------------
# Root-cause taxonomy (FR-002)
# --------------------------------------------------------------------------
CAUSE_BUCKETS: tuple[str, ...] = (
    "card_expired",
    "insufficient_funds",
    "gateway_timeout",
    "mandate_lapsed",
)

CAUSE_LABELS_HUMAN: dict[str, str] = {
    "card_expired": "Card expired",
    "insufficient_funds": "Insufficient funds",
    "gateway_timeout": "Gateway timeout",
    "mandate_lapsed": "Mandate lapsed",
}

# --------------------------------------------------------------------------
# Recovery economics - deterministic, rule-based (never AI-decided)
# --------------------------------------------------------------------------
# Recovery amount is always the original transaction value. These constants
# only shape the simulated/sandbox settlement behaviour and the backoff
# schedule; they never influence whether an action is permitted.
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (0, 60, 300)  # attempt 1, 2, 3
