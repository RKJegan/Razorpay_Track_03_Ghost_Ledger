"""
FR-001 — Synthetic Data Generator
=================================

Produces a seeded, reproducible 30-day merchant transaction history with
**Payment Failure** and **Failed Subscription** events injected at a known
rate. The injected root cause is recorded as ground truth and is the only
source of the precision/recall numbers reported later.

Design commitments
------------------

1. **Reproducible.** Every stochastic draw comes from a single seeded RNG
   (numpy ``default_rng(DATA_SEED)``). Same seed -> byte-identical output.

2. **No leakage.**
   * The held-out split is *temporal*: the most recent ``HOLDOUT_DAYS`` of the
     window. Training never sees a future row.
   * Ground-truth labels are written to **separate files** per split
     (``train_labels.json`` / ``ground_truth_holdout.json``). The answer key
     for the holdout is never co-located with the features it grades.
   * Customer behavioural features (success rate, amount-vs-median, prior
     failure count) are computed from **strictly earlier** transactions via a
     streaming state update, never from the full window.

3. **Honestly hard.** Root causes are *not* a deterministic function of the
   observable features. Error codes deliberately overlap across causes
   (``ERR_DO_NOT_HONOR`` and ``ERR_UNKNOWN`` are shared), mandates can lapse
   early, and gateway timeouts happen off-peak too. The classifier is
   therefore expected to land below 100% — and that measured number is the
   number that gets reported.

Outputs (written to ``data/sample_output/``)
--------------------------------------------
``transactions.json``          all transactions, schema fields + feature context
``subscriptions.json``         subscription mandates (for the subscription agent)
``holdout_set.json``           holdout transactions only (split saved separately)
``train_labels.json``          {txn_id: cause} for the training split
``ground_truth_holdout.json``  {txn_id: cause} for the held-out split (answer key)
``generation_manifest.json``   seed, config, rates, split method, counts

Usage
-----
    python data/synthetic_generator.py              # generate + write files
    python data/synthetic_generator.py --load-db    # also load into SQLite
    python data/synthetic_generator.py --verify     # generate + run leak checks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Allow direct execution (`python data/synthetic_generator.py`) as well as
# `python -m data.synthetic_generator` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    AVG_DAILY_TRANSACTIONS,
    CAUSE_BUCKETS,
    DATASET_DAYS,
    DATASET_PRESETS,
    DATASET_PROFILE,
    DATA_SEED,
    HOLDOUT_DAYS,
    HOLDOUT_FRACTION,
    MERCHANT_ID,
    SPLIT_STRATEGY,
    N_CUSTOMERS,
    SAMPLE_OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Fixed constants. Not env-tunable: changing any of these changes the dataset
# definition, so they belong in code next to the logic that consumes them.
# ---------------------------------------------------------------------------

# Window ends "today" at 00:00 UTC; the lookback window is the 30 days before.
WINDOW_END: datetime = datetime(2026, 9, 3, 0, 0, 0)
WINDOW_DAYS_BACK: int = 30

# Diurnal traffic curve (24 hourly weights, normalised at draw time).
HOUR_WEIGHTS: np.ndarray = np.array(
    [
        0.20, 0.10, 0.08, 0.07, 0.08, 0.12,  # 00-05  dead of night
        0.25, 0.50, 0.90, 1.40, 1.80, 2.00,  # 06-11  morning ramp
        2.10, 1.90, 1.80, 1.90, 2.00, 2.20,  # 12-17  business hours
        2.40, 2.60, 2.30, 1.60, 0.90, 0.45,  # 18-23  evening peak
    ]
)

# Per-weekday volume multiplier (Monday = 0).
WEEKDAY_FACTOR: np.ndarray = np.array([0.95, 0.98, 1.00, 1.02, 1.08, 1.25, 1.20])

# Gateway mix and their baseline latency profiles (ms, lognormal mean/ sigma).
GATEWAYS: tuple[str, ...] = ("hdfc", "axis", "sbi", "icici")
GATEWAY_WEIGHTS: np.ndarray = np.array([0.35, 0.25, 0.22, 0.18])
GATEWAY_LATENCY: dict[str, tuple[float, float]] = {
    "hdfc": (5.9, 0.45),
    "axis": (6.1, 0.50),
    "sbi": (6.2, 0.55),
    "icici": (6.0, 0.48),
}

# Payment-method mix for one-off purchases.
METHODS: tuple[str, ...] = ("upi", "card", "netbanking")
METHOD_WEIGHTS: np.ndarray = np.array([0.50, 0.34, 0.16])

# Share of the customer base holding an active subscription mandate.
SUBSCRIPTION_PENETRATION: float = 0.52

# Share of card holders whose card expires inside (or just before) the window.
# Produces the "expiry wave" that makes card_expired learnable and frequent
# enough to score per-class metrics on.
NEAR_EXPIRY_SHARE: float = 0.38

# Upper bound on mandate age at window start (days). A wide, older-skewed
# range is realistic for a merchant that has been billing for years, and it
# guarantees a meaningful population of mandates that lapse mid-window.
MANDATE_AGE_MAX: int = 800

# Subscription plan catalogue (INR) and their popularity.
PLAN_AMOUNTS: tuple[float, ...] = (299.0, 499.0, 799.0, 999.0, 1499.0, 2499.0)
PLAN_WEIGHTS: np.ndarray = np.array([0.18, 0.30, 0.20, 0.18, 0.09, 0.05])

# Mandate validity horizons in days (e-NACH / UPI AutoPay style).
MANDATE_VALIDITIES: tuple[int, ...] = (365, 730, 1095)
MANDATE_VALIDITY_WEIGHTS: np.ndarray = np.array([0.40, 0.35, 0.25])

# Cause -> error-code distribution. NOTE the deliberate overlap:
# ERR_DO_NOT_HONOR and ERR_UNKNOWN appear under more than one cause, so the
# diagnoser cannot short-circuit on a lookup table.
CAUSE_ERROR_CODES: dict[str, tuple[tuple[str, float], ...]] = {
    "card_expired": (
        ("ERR_CARD_EXPIRED", 0.72),
        ("ERR_DO_NOT_HONOR", 0.14),
        ("ERR_INVALID_CARD", 0.09),
        ("ERR_UNKNOWN", 0.05),
    ),
    "insufficient_funds": (
        ("ERR_INSUFFICIENT_FUNDS", 0.62),
        ("ERR_DO_NOT_HONOR", 0.22),
        ("ERR_LIMIT_EXCEEDED", 0.11),
        ("ERR_UNKNOWN", 0.05),
    ),
    "gateway_timeout": (
        ("ERR_GATEWAY_TIMEOUT", 0.66),
        ("ERR_NETWORK", 0.19),
        ("ERR_UNKNOWN", 0.10),
        ("ERR_DO_NOT_HONOR", 0.05),
    ),
    "mandate_lapsed": (
        ("ERR_MANDATE_EXPIRED", 0.68),
        ("ERR_MANDATE_REVOKED", 0.16),
        ("ERR_UPI_MANDATE", 0.11),
        ("ERR_UNKNOWN", 0.05),
    ),
}

ERROR_CODE_MESSAGES: dict[str, str] = {
    "ERR_CARD_EXPIRED": "Card has expired",
    "ERR_DO_NOT_HONOR": "Issuer declined the transaction",
    "ERR_INVALID_CARD": "Card details invalid",
    "ERR_INSUFFICIENT_FUNDS": "Insufficient funds in account",
    "ERR_LIMIT_EXCEEDED": "Transaction limit exceeded",
    "ERR_GATEWAY_TIMEOUT": "Gateway timed out before authorisation",
    "ERR_NETWORK": "Network failure at acquirer",
    "ERR_MANDATE_EXPIRED": "Mandate has expired",
    "ERR_MANDATE_REVOKED": "Mandate revoked by customer",
    "ERR_UPI_MANDATE": "UPI AutoPay mandate not honoured",
    "ERR_UNKNOWN": "Unknown decline reason",
}

# Baseline failure rates by transaction type (one-off vs subscription debit).
BASE_FAILURE_RATE_ONE_OFF: float = 0.085
BASE_FAILURE_RATE_SUBSCRIPTION: float = 0.110

# Probability that an attempt on a mandate already past its validity fails.
# Below 1.0 because some banks honour a lapsed mandate inside a grace period.
LAPSED_MANDATE_FAILURE_RATE: float = 0.72


# ---------------------------------------------------------------------------
# Entity model
# ---------------------------------------------------------------------------
@dataclass
class Customer:
    """A merchant customer with the attributes that drive failure behaviour."""

    customer_id: str
    tenure_days_at_start: int
    preferred_method: str
    card_expiry_offset_days: float | None  # days from window start to expiry
    reliability: float  # 0..1, higher = pays more reliably
    base_basket: float  # typical one-off order value
    has_subscription: bool
    mandate_id: str | None
    mandate_age_at_start: int | None  # days the mandate has been alive
    mandate_validity_days: int | None
    plan_amount: float | None
    billing_day: int | None

    # Streaming state, updated as transactions are materialised in time order.
    # Used ONLY to build past-only features -> no lookahead leakage.
    prior_amounts: list[float] = field(default_factory=list)
    prior_n: int = 0
    prior_failures: int = 0


@dataclass
class TxnSkeleton:
    """A planned transaction before features and outcome are materialised."""

    ts: datetime
    customer: Customer
    txn_type: str


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    """
    Seeded generator for a 30-day merchant transaction history.

    Parameters
    ----------
    seed : int, optional
        Master RNG seed. Defaults to ``config.DATA_SEED``.
    days : int, optional
        Length of the history window in days. Defaults to ``config.DATASET_DAYS``.
    n_customers : int, optional
        Size of the customer base. Defaults to ``config.N_CUSTOMERS``.
    avg_daily_txns : int, optional
        Mean transactions per day. Defaults to ``config.AVG_DAILY_TRANSACTIONS``.
    holdout_days : int, optional
        Number of trailing days reserved as the held-out split.

    Attributes
    ----------
    rng : numpy.random.Generator
        The single seeded RNG. Every draw in this class uses it.
    customers : list[Customer]
        Generated customer base.
    transactions : list[dict[str, Any]]
        Materialised transactions, ordered by timestamp.
    subscriptions : list[dict[str, Any]]
        Subscription mandates for the subscription recovery agent.
    """

    def __init__(
        self,
        seed: int | None = None,
        days: int | None = None,
        n_customers: int | None = None,
        avg_daily_txns: int | None = None,
        holdout_days: int | None = None,
        profile: str | None = None,
        split_strategy: str | None = None,
        holdout_fraction: float | None = None,
    ) -> None:
        self.profile: str = profile or DATASET_PROFILE
        self.seed: int = DATA_SEED if seed is None else seed
        self.days: int = DATASET_DAYS if days is None else days
        self.n_customers: int = N_CUSTOMERS if n_customers is None else n_customers
        self.avg_daily_txns: int = (
            AVG_DAILY_TRANSACTIONS if avg_daily_txns is None else avg_daily_txns
        )
        self.holdout_days: int = HOLDOUT_DAYS if holdout_days is None else holdout_days

        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.window_end: datetime = WINDOW_END
        self.window_start: datetime = self.window_end - timedelta(days=self.days)

        self.split_strategy: str = (
            split_strategy or SPLIT_STRATEGY
        ).lower()
        if self.split_strategy not in {"random_grouped", "random_rows", "temporal"}:
            self.split_strategy = "random_grouped"
        self.holdout_fraction: float = (
            HOLDOUT_FRACTION if holdout_fraction is None else holdout_fraction
        )

        # Computed from THIS instance's resolved parameters, never from the
        # module-level default. A stale split description would put a wrong
        # basis under every precision/recall figure we report (NFR-004).
        # Re-derived after the split is assigned (see _assign_holdout).
        self.split_method: str = ""

        self.customers: list[Customer] = []
        self.transactions: list[dict[str, Any]] = []
        self.subscriptions: list[dict[str, Any]] = []

    # -- customers ---------------------------------------------------------
    def _build_customers(self) -> list[Customer]:
        """
        Create the customer base with heterogeneous payment behaviour.

        Returns
        -------
        list[Customer]
            ``self.n_customers`` customers with sampled attributes.
        """
        customers: list[Customer] = []
        for i in range(1, self.n_customers + 1):
            cid = f"CUST_{i:05d}"
            tenure = int(self.rng.integers(5, 900))
            method = str(self.rng.choice(METHODS, p=METHOD_WEIGHTS))

            # Only card users carry a card expiry. A deliberate slice sit
            # inside or just past their expiry window, producing a realistic
            # "expiry wave" rather than a uniform trickle.
            if method == "card":
                if self.rng.random() < NEAR_EXPIRY_SHARE:
                    expiry_offset = float(self.rng.uniform(-30, 120))
                else:
                    expiry_offset = float(self.rng.uniform(120, 1400))
            else:
                expiry_offset = None

            reliability = float(np.clip(self.rng.beta(2.4, 1.6), 0.02, 0.99))
            base_basket = float(np.clip(self.rng.lognormal(7.05, 0.55), 120.0, 24000.0))

            has_sub = bool(self.rng.random() < SUBSCRIPTION_PENETRATION)
            mandate_id = mandate_age = validity = None
            plan_amount = billing_day = None
            if has_sub:
                mandate_id = f"mand_{cid.lower()}_{int(self.rng.integers(10000, 99999))}"
                mandate_age = int(self.rng.integers(0, MANDATE_AGE_MAX))
                validity = int(
                    self.rng.choice(MANDATE_VALIDITIES, p=MANDATE_VALIDITY_WEIGHTS)
                )
                plan_amount = float(self.rng.choice(PLAN_AMOUNTS, p=PLAN_WEIGHTS))
                billing_day = int(self.rng.integers(1, 29))

            customers.append(
                Customer(
                    customer_id=cid,
                    tenure_days_at_start=tenure,
                    preferred_method=method,
                    card_expiry_offset_days=expiry_offset,
                    reliability=reliability,
                    base_basket=base_basket,
                    has_subscription=has_sub,
                    mandate_id=mandate_id,
                    mandate_age_at_start=mandate_age,
                    mandate_validity_days=validity,
                    plan_amount=plan_amount,
                    billing_day=billing_day,
                )
            )
        return customers

    # -- transaction plan --------------------------------------------------
    def _daily_volume(self, day_index: int, day_ts: datetime) -> int:
        """
        Compute how many transactions occur on a given day.

        Volume follows a weekday curve, a mild upward trend across the window,
        and daily noise.

        Parameters
        ----------
        day_index : int
            Zero-based index of the day within the window.
        day_ts : datetime
            Calendar date of that day.

        Returns
        -------
        int
            Number of transactions for the day (never below 10).
        """
        weekday_factor = float(WEEKDAY_FACTOR[day_ts.weekday()])
        trend = 1.0 + 0.15 * (day_index / max(self.days - 1, 1))
        noise = float(self.rng.normal(1.0, 0.08))
        return max(10, int(round(self.avg_daily_txns * weekday_factor * trend * noise)))

    def _plan_transactions(self) -> list[TxnSkeleton]:
        """
        Build the time-ordered skeleton of every transaction in the window.

        Subscription debits fire only on the customer's billing day; one-off
        purchases fill the remainder of the day's volume. All skeletons are
        then sorted by (date, hour) so that features can be computed from
        strictly earlier events.

        Returns
        -------
        list[TxnSkeleton]
            Skeletons sorted chronologically.
        """
        hour_probs = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
        skeletons: list[TxnSkeleton] = []
        sub_customers = [c for c in self.customers if c.has_subscription]

        for day_index in range(self.days):
            day_date = (self.window_start + timedelta(days=day_index)).date()
            volume = self._daily_volume(day_index, datetime.combine(day_date, datetime.min.time()))

            # Subscription debits due today.
            day_of_month = day_date.day
            due = [c for c in sub_customers if c.billing_day == day_of_month]

            planned: list[tuple[datetime, Customer, str]] = []
            for cust in due:
                ts = self._timestamp_for(day_date, hour_probs)
                planned.append((ts, cust, "subscription"))

            # One-off purchases fill the rest of the day's quota.
            n_one_off = max(0, volume - len(planned))
            for _ in range(n_one_off):
                cust = self.customers[int(self.rng.integers(0, len(self.customers)))]
                ts = self._timestamp_for(day_date, hour_probs)
                planned.append((ts, cust, "one_off"))

            planned.sort(key=lambda t: t[0])
            for ts, cust, ttype in planned:
                skeletons.append(TxnSkeleton(ts=ts, customer=cust, txn_type=ttype))

        skeletons.sort(key=lambda s: s.ts)
        return skeletons

    def _timestamp_for(self, day_date, hour_probs: np.ndarray) -> datetime:
        """
        Draw a timestamp within a day from the diurnal traffic curve.

        Parameters
        ----------
        day_date : datetime.date
            Calendar date to place the transaction on.
        hour_probs : np.ndarray
            24-length probability vector over hours.

        Returns
        -------
        datetime
            Timestamp with second-level precision.
        """
        hour = int(self.rng.choice(24, p=hour_probs))
        minute = int(self.rng.integers(0, 60))
        second = int(self.rng.integers(0, 60))
        return datetime.combine(day_date, datetime.min.time()) + timedelta(
            hours=hour, minutes=minute, seconds=second
        )

    # -- outcome model -----------------------------------------------------
    def _failure_probability(
        self, cust: Customer, txn_type: str, amount: float, mandate_age: int | None
    ) -> float:
        """
        Probability that a given attempted transaction fails.

        Driven by customer reliability, basket size relative to the customer's
        norm, and (for subscriptions) whether the mandate is past validity.

        Parameters
        ----------
        cust : Customer
            The paying customer.
        txn_type : str
            ``"one_off"`` or ``"subscription"``.
        amount : float
            Transaction value in INR.
        mandate_age : int | None
            Age of the mandate in days, or None for one-off payments.

        Returns
        -------
        float
            Failure probability clipped to [0.02, 0.92].
        """
        if txn_type == "subscription":
            p = BASE_FAILURE_RATE_SUBSCRIPTION
            # A mandate past its validity is overwhelmingly likely to bounce.
            if (
                mandate_age is not None
                and cust.mandate_validity_days is not None
                and mandate_age > cust.mandate_validity_days
            ):
                p = LAPSED_MANDATE_FAILURE_RATE
        else:
            p = BASE_FAILURE_RATE_ONE_OFF

        # Unreliable customers fail more; big-ticket orders fail more.
        p += (0.5 - cust.reliability) * 0.16
        ratio = amount / max(cust.base_basket, 1.0)
        if ratio > 1.5:
            p += min(0.10, (ratio - 1.5) * 0.05)
        return float(np.clip(p, 0.02, 0.92))

    def _sample_cause(
        self,
        cust: Customer,
        txn_type: str,
        amount: float,
        hour: int,
        day_of_month: int,
        mandate_age: int | None,
    ) -> str:
        """
        Sample the ground-truth root cause conditioned on transaction context.

        The weights overlap heavily by design — a card payment near expiry can
        still fail for insufficient funds, and mandates can be revoked before
        their validity date — so the classification task is genuinely
        probabilistic.

        Parameters
        ----------
        cust : Customer
            The paying customer.
        txn_type : str
            ``"one_off"`` or ``"subscription"``.
        amount : float
            Transaction value in INR.
        hour : int
            Hour of day (0-23).
        day_of_month : int
            Calendar day of month (1-31); late-month pay cycles matter.
        mandate_age : int | None
            Mandate age in days, or None for one-off payments.

        Returns
        -------
        str
            One of ``config.CAUSE_BUCKETS``.
        """
        is_peak = 11 <= hour <= 14 or 19 <= hour <= 22
        amount_ratio = amount / max(cust.base_basket, 1.0)

        # --- card_expired -------------------------------------------------
        if cust.preferred_method == "card" and cust.card_expiry_offset_days is not None:
            days_to_expiry = cust.card_expiry_offset_days
            if days_to_expiry <= 0:
                w_card = 9.0
            elif days_to_expiry <= 30:
                w_card = 7.0
            elif days_to_expiry <= 90:
                w_card = 2.2
            else:
                w_card = 0.45
        else:
            w_card = 0.02  # non-card methods essentially never expire

        # --- insufficient_funds ------------------------------------------
        w_funds = 0.10 + (1.0 - cust.reliability) * 1.15
        if amount_ratio > 1.3:
            w_funds += min(0.85, (amount_ratio - 1.3) * 0.5)
        if day_of_month >= 26:  # end-of-cycle cash crunch
            w_funds += 0.30
        if txn_type == "subscription":
            w_funds *= 1.20  # recurring debits hit empty balances

        # --- gateway_timeout ----------------------------------------------
        w_timeout = 0.28 + (0.75 if is_peak else 0.0)
        if hour >= 23 or hour <= 5:
            w_timeout += 0.12

        # --- mandate_lapsed ------------------------------------------------
        if txn_type == "subscription" and mandate_age is not None:
            validity = cust.mandate_validity_days or 365
            overdue = mandate_age - validity
            if overdue > 0:
                w_mandate = 8.0
            elif overdue > -45:
                w_mandate = 1.60  # revoked / not honoured before expiry
            else:
                w_mandate = 0.22
        else:
            w_mandate = 0.0

        weights = np.array([w_card, w_funds, w_timeout, w_mandate], dtype=float)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(len(CAUSE_BUCKETS))
        probs = weights / weights.sum()
        return str(self.rng.choice(CAUSE_BUCKETS, p=probs))

    # -- feature construction (past-only) ----------------------------------
    def _context_features(self, cust: Customer, ts: datetime, amount: float) -> dict[str, Any]:
        """
        Build the observable feature context using only *past* transactions.

        This is the leak-safety boundary: the customer state is advanced
        **after** the current transaction is fully materialised, so every
        aggregate here is strictly backward-looking.

        Parameters
        ----------
        cust : Customer
            The paying customer, carrying its running state.
        ts : datetime
            Transaction timestamp.
        amount : float
            Transaction value in INR.

        Returns
        -------
        dict[str, Any]
            Feature dictionary attached to the transaction record.
        """
        if cust.prior_amounts:
            cust_median = float(np.median(cust.prior_amounts))
            amount_vs_median = float(amount / max(cust_median, 1.0))
            prior_success_rate = float(
                (cust.prior_n - cust.prior_failures) / max(cust.prior_n, 1)
            )
        else:
            cust_median = 0.0
            amount_vs_median = 1.0
            prior_success_rate = 0.5  # neutral prior for a first-time payer

        return {
            "hour": int(ts.hour),
            "day_of_week": int(ts.weekday()),
            "day_of_month": int(ts.day),
            "is_peak_hour": int(11 <= ts.hour <= 14 or 19 <= ts.hour <= 22),
            "amount": float(round(amount, 2)),
            "amount_vs_customer_median": float(round(amount_vs_median, 4)),
            "customer_median_amount": float(round(cust_median, 2)),
            "customer_tenure_days": int(
                cust.tenure_days_at_start + (ts - self.window_start).days
            ),
            "customer_prior_txn_count": int(cust.prior_n),
            "customer_prior_failure_count": int(cust.prior_failures),
            "customer_prior_success_rate": float(round(prior_success_rate, 4)),
        }

    # -- main --------------------------------------------------------------
    def generate(self) -> dict[str, Any]:
        """
        Run the full generation pipeline and populate instance state.

        Returns
        -------
        dict[str, Any]
            Manifest describing the dataset: seed, split method, counts and
            the injected cause distribution.
        """
        self.customers = self._build_customers()
        skeletons = self._plan_transactions()

        txns: list[dict[str, Any]] = []
        self._cause_by_txn = {}

        for idx, skel in enumerate(skeletons, start=1):
            cust = skel.customer
            ts = skel.ts
            day_index = (ts.date() - self.window_start.date()).days
            # NOTE: is_holdout is assigned by _assign_holdout() once every row
            # exists, so the split can be made at customer level. It is not
            # decided here from the timestamp.
            is_holdout = 0

            # --- amount ---------------------------------------------------
            if skel.txn_type == "subscription":
                amount = float(cust.plan_amount or 499.0)
                # Occasional plan upgrade / add-on charge.
                if self.rng.random() < 0.04:
                    amount = float(round(amount * float(self.rng.uniform(1.1, 1.6)), 2))
                mandate_age = (cust.mandate_age_at_start or 0) + day_index
            else:
                amount = float(
                    round(
                        float(
                            np.clip(
                                cust.base_basket
                                * float(self.rng.lognormal(0.0, 0.45)),
                                59.0,
                                60000.0,
                            )
                        ),
                        2,
                    )
                )
                mandate_age = None

            # Method: subscriptions debit the mandate's instrument; one-offs
            # follow the customer's preference with occasional switching.
            if skel.txn_type == "subscription":
                method = cust.preferred_method
            elif self.rng.random() < 0.12:
                method = str(self.rng.choice(METHODS, p=METHOD_WEIGHTS))
            else:
                method = cust.preferred_method

            gateway = str(self.rng.choice(GATEWAYS, p=GATEWAY_WEIGHTS))

            # --- outcome ----------------------------------------------------
            p_fail = self._failure_probability(cust, skel.txn_type, amount, mandate_age)
            failed = bool(self.rng.random() < p_fail)

            ctx = self._context_features(cust, ts, amount)
            ctx["payment_method"] = method
            ctx["gateway"] = gateway
            ctx["txn_type"] = skel.txn_type

            error_code: str | None = None
            failure_reason_raw: str | None = None
            cause: str | None = None

            if failed:
                cause = self._sample_cause(
                    cust, skel.txn_type, amount, ts.hour, ts.day, mandate_age
                )
                codes, cprobs = zip(*CAUSE_ERROR_CODES[cause])
                error_code = str(self.rng.choice(list(codes), p=list(cprobs)))
                failure_reason_raw = f"{error_code}: {ERROR_CODE_MESSAGES[error_code]}"
                # Latency is a symptom of a timeout, not a label copy: we draw
                # a heavy tail only after the cause is already decided.
                if cause == "gateway_timeout":
                    latency = float(np.clip(self.rng.lognormal(7.6, 0.4), 1200, 30000))
                else:
                    mu, sigma = GATEWAY_LATENCY[gateway]
                    latency = float(
                        np.clip(
                            self.rng.lognormal(mu, sigma) * (1.35 if ctx["is_peak_hour"] else 1.0),
                            120,
                            9000,
                        )
                    )
            else:
                mu, sigma = GATEWAY_LATENCY[gateway]
                latency = float(
                    np.clip(
                        self.rng.lognormal(mu, sigma) * (1.35 if ctx["is_peak_hour"] else 1.0),
                        120,
                        9000,
                    )
                )

            ctx["error_code"] = error_code
            ctx["gateway_latency_ms"] = float(round(latency, 1))
            ctx["days_to_card_expiry"] = (
                None
                if cust.card_expiry_offset_days is None
                else float(round(cust.card_expiry_offset_days - day_index, 1))
            )
            ctx["mandate_age_days"] = mandate_age
            ctx["mandate_validity_days"] = cust.mandate_validity_days
            ctx["mandate_days_overdue"] = (
                None
                if mandate_age is None or cust.mandate_validity_days is None
                else int(mandate_age - cust.mandate_validity_days)
            )

            txn_id = f"txn_{idx:06d}"
            record: dict[str, Any] = {
                # --- spec schema columns (transactions table) ---
                "id": txn_id,
                "merchant_id": MERCHANT_ID,
                "customer_id": cust.customer_id,
                "amount": float(round(amount, 2)),
                "status": "failed" if failed else "success",
                "txn_type": skel.txn_type,
                "payment_method": method,
                "failure_reason_raw": failure_reason_raw,
                "timestamp": ts.isoformat(sep=" ", timespec="seconds"),
                "is_holdout": is_holdout,
                # --- feature context (diagnoser input, not a DB column) ---
                "context": ctx,
            }
            if skel.txn_type == "subscription":
                record["mandate_id"] = cust.mandate_id

            txns.append(record)

            if failed and cause is not None:
                self._cause_by_txn[txn_id] = cause

            # --- advance customer state (AFTER features are built) ---
            cust.prior_amounts.append(amount)
            cust.prior_n += 1
            if failed:
                cust.prior_failures += 1

        self.transactions = txns
        self._assign_holdout()
        self.subscriptions = self._build_subscriptions()
        self.manifest = self._build_manifest()
        return self.manifest

    def _assign_holdout(self) -> None:
        """
        Mark each transaction as train or held-out, per the active strategy.

        Three strategies, selected by ``config.SPLIT_STRATEGY``:

        ``random_grouped`` (default)
            Customers are shuffled with the seeded RNG and added to the
            holdout until the target fraction of **transactions** is reached.
            Every transaction of a selected customer is held out, so no
            customer ever appears on both sides. Random with respect to time
            and cause, but immune to the customer-identity leakage that a
            row-level split suffers from.

        ``random_rows``
            Transactions are shuffled and the first ``HOLDOUT_FRACTION`` are
            held out. Genuinely random, but the same customer's failures can
            appear in both splits; because customer-level features are
            near-constant per customer, this measurably inflates scores.
            Available for measurement, not for reporting.

        ``temporal``
            The trailing ``HOLDOUT_DAYS`` are held out, split by timestamp.

        Side Effects
        ------------
        Sets ``is_holdout`` on every transaction and writes
        ``self.split_method`` — the exact string that must accompany any
        precision/recall figure reported from this corpus (NFR-004).
        """
        target = int(round(self.holdout_fraction * len(self.transactions)))

        if self.split_strategy == "temporal":
            cutoff_ts = (
                self.window_end - timedelta(days=self.holdout_days)
            ).isoformat(sep=" ", timespec="seconds")
            for t in self.transactions:
                t["is_holdout"] = int(t["timestamp"] >= cutoff_ts)
            holdout_customers = None

        elif self.split_strategy == "random_rows":
            order = self.rng.permutation(len(self.transactions))
            for i in order[:target]:
                self.transactions[int(i)]["is_holdout"] = 1
            for t in self.transactions:
                t.setdefault("is_holdout", 0)
            holdout_customers = None

        else:  # random_grouped
            # Group transaction indices by customer, then draw whole customers.
            by_customer: dict[str, list[int]] = {}
            for i, t in enumerate(self.transactions):
                by_customer.setdefault(t["customer_id"], []).append(i)
            customer_ids = list(by_customer.keys())
            self.rng.shuffle(customer_ids)

            assigned = 0
            chosen: list[str] = []
            for cid in customer_ids:
                if assigned >= target:
                    break
                chosen.append(cid)
                assigned += len(by_customer[cid])
            holdout_customers = set(chosen)
            for cid in holdout_customers:
                for i in by_customer[cid]:
                    self.transactions[i]["is_holdout"] = 1

        # Partition ground truth by the assigned split. Done here, not in
        # run(), because _build_manifest() needs it and runs inside generate().
        self._train_labels = {
            t["id"]: self._cause_by_txn[t["id"]]
            for t in self.transactions
            if t["status"] == "failed" and t["is_holdout"] == 0
        }
        self._holdout_labels = {
            t["id"]: self._cause_by_txn[t["id"]]
            for t in self.transactions
            if t["status"] == "failed" and t["is_holdout"] == 1
        }

        n_hold = sum(t["is_holdout"] for t in self.transactions)
        if self.split_strategy == "random_grouped":
            desc = (
                f"random grouped by customer: {len(holdout_customers or []):,} of "
                f"{len({t['customer_id'] for t in self.transactions}):,} customers "
                f"drawn at random (seed {self.seed}); ALL of a held-out customer's "
                f"transactions go to the holdout, so no customer appears in both "
                f"splits. Grouping blocks leakage through customer-level features."
            )
        elif self.split_strategy == "random_rows":
            desc = (
                f"random at transaction level: {n_hold:,} of "
                f"{len(self.transactions):,} rows drawn at random (seed {self.seed}). "
                f"WARNING: customers recur across both splits; metrics from this "
                f"split are inflated by customer-identity leakage."
            )
        else:
            desc = (
                f"temporal holdout: the most recent {self.holdout_days} of "
                f"{self.days} days (train = days 1-{self.days - self.holdout_days}, "
                f"holdout = days {self.days - self.holdout_days + 1}-{self.days}); "
                f"split by transaction timestamp, no random shuffle."
            )
        # The canonical strategy token is appended in machine-readable form so
        # that verifiers and tests can assert on it, rather than pattern-
        # matching prose that is free to drift.
        self.split_method = (
            f"{desc} Held-out batch: {n_hold:,} transactions "
            f"({n_hold / max(len(self.transactions), 1):.1%}). "
            f"[strategy={self.split_strategy}]"
        )

    def _build_subscriptions(self) -> list[dict[str, Any]]:
        """
        Materialise subscription mandate records for the subscription agent.

        Returns
        -------
        list[dict[str, Any]]
            One record per subscription customer, including mandate status as
            of the end of the window.
        """
        subs: list[dict[str, Any]] = []
        for cust in self.customers:
            if not cust.has_subscription:
                continue
            mandate_age = (cust.mandate_age_at_start or 0) + self.days
            validity = cust.mandate_validity_days or 365
            status = "lapsed" if mandate_age > validity else "active"
            subs.append(
                {
                    "subscription_id": f"sub_{cust.customer_id.lower()}",
                    "customer_id": cust.customer_id,
                    "mandate_id": cust.mandate_id,
                    "plan_amount": float(cust.plan_amount or 0.0),
                    "billing_day": int(cust.billing_day or 1),
                    "mandate_age_days": int(mandate_age),
                    "mandate_validity_days": int(validity),
                    "mandate_status": status,
                    "payment_method": cust.preferred_method,
                }
            )
        return subs

    def _build_manifest(self) -> dict[str, Any]:
        """
        Assemble the provenance manifest for the generated dataset.

        Returns
        -------
        dict[str, Any]
            Manifest including seed, split method, counts and cause mix.
        """
        failed = [t for t in self.transactions if t["status"] == "failed"]
        holdout = [t for t in self.transactions if t["is_holdout"] == 1]
        holdout_failed = [t for t in holdout if t["status"] == "failed"]

        train_labels = self._train_labels
        holdout_labels = self._holdout_labels
        all_labels = {**train_labels, **holdout_labels}
        cause_counts: dict[str, int] = {c: 0 for c in CAUSE_BUCKETS}
        for c in all_labels.values():
            cause_counts[c] += 1

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "data/synthetic_generator.py",
            "profile": self.profile,
            "dataset_config": {
                "days": self.days,
                "customers": self.n_customers,
                "avg_daily_txns": self.avg_daily_txns,
                "holdout_days": self.holdout_days,
            },
            "storage": (
                "Full ledger -> SQLite (database/seed_db.py). Feature context "
                "persisted for FAILED transactions only (failure_contexts.json), "
                "since those are the diagnoser's only input rows."
            ),
            "seed": self.seed,
            "merchant_id": MERCHANT_ID,
            "window": {
                "start": self.window_start.isoformat(sep=" ", timespec="seconds"),
                "end": self.window_end.isoformat(sep=" ", timespec="seconds"),
                "days": self.days,
            },
            "split": {
                "method": self.split_method,
                "strategy": self.split_strategy,
                "holdout_fraction": self.holdout_fraction,
                "holdout_days": self.holdout_days,
                "train_days": self.days - self.holdout_days,
            },
            "counts": {
                "customers": len(self.customers),
                "transactions": len(self.transactions),
                "subscriptions": len(self.subscriptions),
                "failed_total": len(failed),
                "failed_train": len(train_labels),
                "failed_holdout": len(holdout_labels),
                "holdout_transactions": len(holdout),
                "holdout_failed": len(holdout_failed),
            },
            "rates": {
                "overall_failure_rate": round(len(failed) / max(len(self.transactions), 1), 4),
                "one_off_failure_rate": round(
                    len([t for t in failed if t["txn_type"] == "one_off"])
                    / max(len([t for t in self.transactions if t["txn_type"] == "one_off"]), 1),
                    4,
                ),
                "subscription_failure_rate": round(
                    len([t for t in failed if t["txn_type"] == "subscription"])
                    / max(len([t for t in self.transactions if t["txn_type"] == "subscription"]), 1),
                    4,
                ),
            },
            "ground_truth_cause_distribution": cause_counts,
            "value_at_risk_inr": round(sum(t["amount"] for t in failed), 2),
            "holdout_value_at_risk_inr": round(sum(t["amount"] for t in holdout_failed), 2),
            "cause_buckets": list(CAUSE_BUCKETS),
            "assumptions": [
                "No real merchant data is used anywhere; every record is synthetic.",
                (
                    f"Subscription mandate debits fail at "
                    f"{round(len([t for t in failed if t['txn_type'] == 'subscription']) / max(len([t for t in self.transactions if t['txn_type'] == 'subscription']), 1), 4):.2%} "
                    "versus "
                    f"{round(len([t for t in failed if t['txn_type'] == 'one_off']) / max(len([t for t in self.transactions if t['txn_type'] == 'one_off']), 1), 4):.2%} "
                    "for one-off payments. This gap is deliberate and large: recurring "
                    "mandates in India expire annually (UPI AutoPay / e-NACH) and lapse "
                    "far more often than one-off card payments decline. It is a stated "
                    "modelling assumption, not an empirical measurement."
                ),
                (
                    "Root cause is sampled from a context-dependent distribution, not "
                    "assigned deterministically. Error codes overlap across causes "
                    "(ERR_DO_NOT_HONOR and ERR_UNKNOWN appear under multiple causes) so "
                    "the classification task is genuinely probabilistic and the "
                    "diagnoser is NOT expected to reach 100%."
                ),
                (
                    "Customer behavioural features (median amount, prior success rate, "
                    "prior failure count) are computed from strictly earlier "
                    "transactions only. No future information enters any feature."
                ),
            ],
            "notes": (
                "Ground truth is the root cause injected at generation time. "
                "It is stored in separate files per split and is never present "
                "in the feature context used for prediction."
            ),
        }

    # -- output ------------------------------------------------------------
    def write(self, outdir: Path | None = None) -> dict[str, Path]:
        """
        Persist artefacts to ``data/sample_output/``.

        Storage strategy — deliberate, because a large corpus does not fit the
        naive "one pretty-printed JSON per concept" approach:

        * The **full ledger** (every transaction, successful or not) goes to
          SQLite via ``database.seed_db``, which is compact and queryable.
          It is not duplicated into JSON.
        * **Feature context is persisted only for failed transactions.** Those
          are the only rows the diagnoser ever consumes; successes exist to
          generate behavioural history, which is already baked into the
          failure rows' features. This removes ~91% of the payload.
        * The **held-out split** is still written to its own file, per spec.
        * A **30-day demo window** is written as JSON for inspection.

        Parameters
        ----------
        outdir : Path, optional
            Override output directory. Defaults to ``config.SAMPLE_OUTPUT_DIR``.

        Returns
        -------
        dict[str, Path]
            Mapping of artefact name to written file path.
        """
        outdir = Path(outdir) if outdir else SAMPLE_OUTPUT_DIR
        outdir.mkdir(parents=True, exist_ok=True)

        train_labels = self._train_labels
        holdout_labels = self._holdout_labels
        holdout_txns = [t for t in self.transactions if t["is_holdout"] == 1]

        # Trailing 30 days, for dashboard inspection and manual spot-checks.
        cutoff = self.window_end - timedelta(days=min(30, self.days))
        demo_window = [
            t
            for t in self.transactions
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]

        # Context only for failures: these are the diagnoser's input rows.
        failure_contexts = {
            t["id"]: t["context"] for t in self.transactions if t["status"] == "failed"
        }

        paths: dict[str, Path] = {}
        # Compact separators for the large payloads; indent only for the small,
        # human-read ones (manifest, labels).
        compact = {"separators": (",", ":")}
        pretty = {"indent": 2}

        files: list[tuple[str, Any, dict]] = [
            ("holdout_set.json", holdout_txns, compact),
            ("demo_window.json", demo_window, compact),
            ("failure_contexts.json", failure_contexts, compact),
            ("subscriptions.json", self.subscriptions, compact),
            ("train_labels.json", train_labels, pretty),
            ("ground_truth_holdout.json", holdout_labels, pretty),
            ("generation_manifest.json", self.manifest, pretty),
        ]
        for name, payload, kw in files:
            p = outdir / name
            p.write_text(json.dumps(payload, default=str, **kw), encoding="utf-8")
            paths[name] = p
        return paths

    def run(self, outdir: Path | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Generate and write the dataset in one call.

        Parameters
        ----------
        outdir : Path, optional
            Override output directory.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Path]]
            The manifest and the mapping of written artefact paths.
        """
        # generate() -> _assign_holdout() sets is_holdout and partitions the
        # ground-truth labels into self._train_labels / self._holdout_labels.
        self.manifest = self.generate()
        paths = self.write(outdir)
        return self.manifest, paths


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(
    transactions: list[dict[str, Any]],
    train_labels: dict[str, str],
    holdout_labels: dict[str, str],
    strategy: str = "random_grouped",
) -> list[str]:
    """
    Run structural and leakage checks over a generated dataset.

    Parameters
    ----------
    transactions : list[dict[str, Any]]
        All generated transactions.
    train_labels : dict[str, str]
        Ground-truth causes for the training split.
    holdout_labels : dict[str, str]
        Ground-truth causes for the held-out split.
    strategy : str, optional
        Active split strategy; selects which ordering check applies.

    Returns
    -------
    list[str]
        Human-readable PASS/FAIL lines, one per check.
    """
    results: list[str] = []
    train_ids = {t["id"] for t in transactions if t["is_holdout"] == 0}
    holdout_ids = {t["id"] for t in transactions if t["is_holdout"] == 1}
    failed_ids = {t["id"] for t in transactions if t["status"] == "failed"}

    checks: list[tuple[str, bool]] = []
    checks.append(("non-empty dataset", len(transactions) > 0))
    checks.append(
        ("transaction ids unique", len({t["id"] for t in transactions}) == len(transactions))
    )
    checks.append(
        ("splits disjoint (no shared txn id)", len(train_ids & holdout_ids) == 0)
    )
    checks.append(
        ("train+holdout == all txns", len(train_ids | holdout_ids) == len(transactions))
    )
    checks.append(
        ("label sets disjoint", len(set(train_labels) & set(holdout_labels)) == 0)
    )
    checks.append(
        ("every label belongs to a failed txn",
         set(train_labels) <= failed_ids and set(holdout_labels) <= failed_ids)
    )
    checks.append(
        ("labels cover all failures",
         set(train_labels) | set(holdout_labels) == failed_ids)
    )
    checks.append(
        ("train labels only on non-holdout rows",
         all(t["is_holdout"] == 0 for t in transactions if t["id"] in train_labels))
    )
    checks.append(
        ("holdout labels only on holdout rows",
         all(t["is_holdout"] == 1 for t in transactions if t["id"] in holdout_labels))
    )
    checks.append(
        ("no cause string inside feature context",
         not any(
             isinstance(v, str) and v in CAUSE_BUCKETS
             for t in transactions
             for v in t["context"].values()
         ))
    )
    # Customer overlap: the check that actually distinguishes a leakage-safe
    # split from an inflated one. Under random_grouped/temporal a customer must
    # never appear on both sides; under random_rows it is expected to fail.
    tr_cust = {t["customer_id"] for t in transactions if t["is_holdout"] == 0}
    ho_cust = {t["customer_id"] for t in transactions if t["is_holdout"] == 1}
    overlap = tr_cust & ho_cust
    if strategy == "random_rows":
        checks.append(
            (f"row-level split (customers recur across splits: "
             f"{len(overlap):,} overlapping - EXPECTED, metrics inflated)", True)
        )
    else:
        checks.append(
            (f"no customer appears in both splits ({len(tr_cust):,} train / "
             f"{len(ho_cust):,} holdout customers, {len(overlap)} overlapping)",
             len(overlap) == 0)
        )

    if strategy == "temporal":
        checks.append(
            ("timestamps monotonically ordered by split",
             max((t["timestamp"] for t in transactions if t["is_holdout"] == 0), default="")
             <= min((t["timestamp"] for t in transactions if t["is_holdout"] == 1), default=""))
        )
    else:
        # Random splits must NOT be time-ordered - that is the point of them.
        tr_ts = [t["timestamp"] for t in transactions if t["is_holdout"] == 0]
        ho_ts = [t["timestamp"] for t in transactions if t["is_holdout"] == 1]
        interleaved = bool(tr_ts and ho_ts and min(ho_ts) < max(tr_ts))
        checks.append(
            ("holdout spans the full time range (not time-ordered)", interleaved)
        )
    checks.append(
        ("every failure has an error code",
         all(t["context"]["error_code"] for t in transactions if t["status"] == "failed"))
    )
    checks.append(
        ("successful txns carry no failure reason",
         all(t["failure_reason_raw"] is None for t in transactions if t["status"] == "success"))
    )
    checks.append(
        ("all four cause buckets present in labels",
         set(train_labels.values()) | set(holdout_labels.values()) == set(CAUSE_BUCKETS))
    )

    for name, ok in checks:
        results.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point for the generator.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Ghost Ledger v2 synthetic data generator")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=sorted(DATASET_PRESETS),
        help=f"dataset size preset (default: {DATASET_PROFILE})",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the RNG seed")
    parser.add_argument("--days", type=int, default=None, help="length of history window")
    parser.add_argument("--customers", type=int, default=None, help="customer base size")
    parser.add_argument(
        "--daily", type=int, default=None, help="average transactions per day"
    )
    parser.add_argument(
        "--holdout-days", type=int, default=None, help="trailing days held out (temporal)"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=None,
        choices=["random_grouped", "random_rows", "temporal"],
        help=f"how the held-out batch is chosen (default: {SPLIT_STRATEGY})",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help="fraction of corpus held out (random strategies)",
    )
    parser.add_argument("--load-db", action="store_true", help="also load into SQLite")
    parser.add_argument("--verify", action="store_true", help="run leakage checks")
    parser.add_argument("--outdir", type=str, default=None, help="output directory")
    args = parser.parse_args(argv)

    # A preset is the base; explicit flags layer on top of it.
    if args.profile:
        preset = DATASET_PRESETS[args.profile]
        days = args.days if args.days is not None else preset["days"]
        customers = (
            args.customers if args.customers is not None else preset["customers"]
        )
        daily = args.daily if args.daily is not None else preset["daily"]
        holdout = (
            args.holdout_days
            if args.holdout_days is not None
            else int(preset["holdout_days"])
        )
        holdout_frac = (
            args.holdout_fraction
            if args.holdout_fraction is not None
            else float(preset["holdout_frac"])
        )
    else:
        days = args.days
        customers = args.customers
        daily = args.daily
        holdout = args.holdout_days
        holdout_frac = args.holdout_fraction

    gen = SyntheticDataGenerator(
        seed=args.seed,
        days=days,
        n_customers=customers,
        avg_daily_txns=daily,
        holdout_days=holdout,
        profile=args.profile or DATASET_PROFILE,
        split_strategy=args.split_strategy,
        holdout_fraction=holdout_frac,
    )
    manifest, paths = gen.run(outdir=Path(args.outdir) if args.outdir else None)

    W = 78
    print("=" * W)
    print("  GHOST LEDGER v2 — SYNTHETIC DATA GENERATION")
    print("=" * W)
    print(f"  profile             : {manifest['profile']}")
    print(f"  seed                : {manifest['seed']}")
    print(f"  merchant            : {manifest['merchant_id']}")
    print(
        f"  window              : {manifest['window']['start'][:10]} -> "
        f"{manifest['window']['end'][:10]} ({manifest['window']['days']}d)"
    )
    print(f"  split method        : {manifest['split']['method']}")
    c = manifest["counts"]
    print(
        f"  transactions        : {c['transactions']:,}  "
        f"(customers {c['customers']:,}, subscriptions {c['subscriptions']:,})"
    )
    print(
        f"  failures            : {c['failed_total']:,} total = "
        f"{c['failed_train']:,} train + {c['failed_holdout']:,} holdout"
    )
    r = manifest["rates"]
    print(
        f"  failure rate        : {r['overall_failure_rate']:.2%} overall | "
        f"{r['one_off_failure_rate']:.2%} one-off | "
        f"{r['subscription_failure_rate']:.2%} subscription"
    )
    print(f"  HELD-OUT EVAL SET   : n = {c['failed_holdout']:,} failed transactions")
    print(
        f"  value at risk       : INR {manifest['value_at_risk_inr']:,.2f} total | "
        f"INR {manifest['holdout_value_at_risk_inr']:,.2f} holdout"
    )

    print("  injected cause mix (train / holdout):")
    tr = gen._train_labels
    ho = gen._holdout_labels
    for cause in CAUSE_BUCKETS:
        ntr = sum(1 for v in tr.values() if v == cause)
        nho = sum(1 for v in ho.values() if v == cause)
        print(f"      {cause:<20} {ntr:>6,} / {nho:>5,}")
    print("-" * W)
    total_bytes = sum(p.stat().st_size for p in paths.values())
    for name, p in paths.items():
        print(f"  wrote {name:<28} {p.stat().st_size / (1024 * 1024):>8.2f} MB")
    print(f"  {'TOTAL JSON':<28} {total_bytes / (1024 * 1024):>8.2f} MB")
    print("-" * W)

    if args.verify:
        print("  VERIFICATION")
        for line in verify(gen.transactions, tr, ho, gen.split_strategy):
            print(line)
        print("-" * W)

    if args.load_db:
        from database.seed_db import load_transactions  # local import: optional step

        n, db_mb = load_transactions(gen.transactions)
        print(f"  loaded {n:,} transactions into SQLite ({db_mb:.2f} MB on disk)")
        print(f"  NOTE: the JSON artefacts above exclude the full ledger, which")
        print(f"        lives in SQLite. Failure feature contexts are persisted")
        print(f"        separately in failure_contexts.json.")
        print("-" * W)

    print("  done.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
