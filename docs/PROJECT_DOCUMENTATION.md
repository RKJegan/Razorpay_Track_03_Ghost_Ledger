# Ghost Ledger v2 — Complete Project Documentation

**Razorpay AI Buildathon · Track 03: AI Revenue Recovery**
Documentation date: 2026-09-03 · Build status: complete · Test status: **127/127 passing**

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Problem and scope](#2-problem-and-scope)
3. [System architecture](#3-system-architecture)
4. [Data generation](#4-data-generation)
5. [Root-cause diagnoser](#5-root-cause-diagnoser)
6. [Policy engine](#6-policy-engine)
7. [Recovery agents](#7-recovery-agents)
8. [Razorpay integration](#8-razorpay-integration)
9. [Autopsy reporter](#9-autopsy-reporter)
10. [Audit trail](#10-audit-trail)
11. [Metrics and dashboard](#11-metrics-and-dashboard)
12. [Results](#12-results)
13. [Testing](#13-testing)
14. [Running the system](#14-running-the-system)
15. [Configuration reference](#15-configuration-reference)
16. [Design decisions](#16-design-decisions)
17. [Honest limitations](#17-honest-limitations)
18. [Failure log](#18-failure-log)
19. [Repository layout](#19-repository-layout)

---

## 1. Executive summary

Ghost Ledger v2 is an autonomous revenue-recovery system for failed Razorpay
payments. It detects failed transactions, classifies *why* each one failed,
attempts a bounded recovery action under hard policy guardrails, explains every
case in plain English, and reports recovered revenue against an audit trail that
reconciles to the rupee.

**The headline number:**

> **₹1,907,719.68 recovered of ₹2,646,060.94 at risk — 72.10%** across 1,815
> held-out failed transactions.

**What makes this more than a classifier demo:**

| Property | How it is delivered |
|---|---|
| Money is actually measured, not claimed | Every rupee traces to a row in `recoveries`, summed by `metrics.py`, reconciled against `audit_trail` |
| The AI is bounded | No action executes without a policy check logged *before* it; a 3-strike rule permanently halts a case |
| The AI never moves money | The LLM writes prose only. It is structurally excluded from the execution path |
| Every metric is attributable | Macro-F1 is decomposed against a no-ML lookup baseline and an ablated model |
| Known failure modes are first-class | 440 cases escalated rather than retried — the stopping rule firing is a *feature*, not a crash |
| It runs in one command | Clean checkout → full pipeline in **78 seconds**, no manual steps |

**Scale:** 7,002 lines of application code, 2,007 lines of tests, 102,262
synthetic transactions, 5 database tables, 4 cause classes, 40 engineered
features.

---

## 2. Problem and scope

### 2.1 The track requirement

Track 03 asks for AI-driven revenue recovery. The stated bar:

> *"Don't just identify the problem. Show measured money recovered across a
> batch, with compliant escalation, stopping rules, and an audit trail."*

Ghost Ledger v2 is built to that sentence literally. Identification is the cheap
part; the system is judged on measured recovery, bounded action, and auditability.

### 2.2 Scope delivered

| Priority | Feature | Status |
|---|---|---|
| **P0** | Payment Failure Recovery (FR-004a) | ✅ Complete |
| **P0** | Failed Subscription Recovery (FR-004b) | ✅ Complete |
| **P0** | Root-cause diagnosis (FR-002) | ✅ Complete |
| **P0** | Policy engine / stopping rules (FR-003) | ✅ Complete |
| **P0** | Audit trail (FR-006) | ✅ Complete |
| **P0** | Autopsy reports (FR-005) | ✅ Complete |
| **P1** | Razorpay integration (FR-008) | ✅ Interface complete, offline simulator |
| **P1** | Weekly report (FR-009) | ✅ Complete |
| P2 | Advanced analytics, UPI AutoPay specifics | ⛔ Out of scope by decision |

P2 was explicitly excluded. The four cause buckets are the four that matter for
the P0 flows.

---

## 3. System architecture

### 3.1 The six-stage pipeline

`main.py` orchestrates a linear, resumable pipeline:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 1/6  SYNTHETIC DATA GENERATION                              │
  │           102,262 transactions → SQLite ledger + feature context  │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 2/6  ROOT CAUSE DIAGNOSIS                                   │
  │           XGBoost → 4 cause buckets + confidence                 │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 3/6  PERSIST DIAGNOSES  (+ reset_action_state)              │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 4/6  POLICY-GATED RECOVERY ACTIONS                          │
  │                                                                  │
  │   for each attempt:                                              │
  │     1. POLICY CHECK ──► audit (always, pass or fail)  ◄── FR-003 │
  │     2. if STOP  → escalate, halt permanently                     │
  │        if DENY  → block this attempt                             │
  │        if ALLOW → execute via Razorpay client                    │
  │     3. record outcome, back off                                  │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 5/6  AUTOPSY REPORTS  (text only — never in money path)     │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ STEP 6/6  HEADLINE METRICS  (single source of truth)             │
  └──────────────────────────────────────────────────────────────────┘
```

### 3.2 Trust boundary — the single most important design property

```
        ┌─────────────────── TRUSTED PATH ───────────────────┐
        │  ledger → diagnoser → policy engine → Razorpay     │
        │  deterministic · audited · bounded · money moves   │
        └────────────────────────────────────────────────────┘

        ┌────────────── UNTRUSTED PATH (prose only) ─────────┐
        │  LLM autopsy reporter → explanation text           │
        │  non-deterministic · never touches execution      │
        └────────────────────────────────────────────────────┘
```

The LLM has **no callable path into the recovery loop**. It reads facts and
writes sentences. If it fails, times out, returns nothing, or invents numbers,
the pipeline continues unaffected.

### 3.3 Module map

| Module | Responsibility |
|---|---|
| `config.py` | All settings, every one with a safe offline default |
| `data/synthetic_generator.py` | Deterministic corpus generation (FR-001) |
| `database/db_client.py` | SQLite access, WAL mode, FK enforcement, parameterised SQL |
| `database/schema.sql` | 5 tables |
| `database/seed_db.py` | Ledger → SQLite loader |
| `database/audit_trail.py` | Append-only compliance log (FR-006) |
| `diagnoser/root_cause_classifier.py` | XGBoost model wrapper |
| `diagnoser/train.py` | Tuning, CV, held-out evaluation, attribution |
| `diagnoser/eval_holdout.py` | Metric computation |
| `diagnoser/split_experiment.py` | Split-strategy comparison |
| `agents/policy_engine.py` | Deterministic guardrails (FR-003) |
| `agents/recovery_base.py` | Shared recovery sequence |
| `agents/payment_failure_agent.py` | One-off payment recovery (FR-004a) |
| `agents/subscription_agent.py` | Mandate retry recovery (FR-004b) |
| `agents/autopsy_reporter.py` | LLM explanations (FR-005) |
| `api/razorpay_client.py` | Simulated + live Razorpay clients (FR-008) |
| `metrics.py` | **Single source of truth** for all dashboard figures |
| `dashboard/app.py` | Streamlit UI |
| `main.py` | Pipeline orchestrator |
| `weekly_report.py` | Markdown report generator (FR-009) |

---

## 4. Data generation

### 4.1 Why synthetic

No real merchant data exists in this environment, and none was invented from
outside sources. Every record is generated by `data/synthetic_generator.py` from
a single seed. This is a stated assumption, not a shortcut: the generator's
parameters are documented in `generation_manifest.json` so a reader can judge
whether the corpus is a fair test bed.

### 4.2 Corpus profile (`train` preset, seed 42)

| Property | Value |
|---|---|
| Transactions | **102,262** |
| Customers | 2,000 |
| Subscriptions | 1,030 |
| Window | 180 days (2026-03-07 → 2026-09-03) |
| Total failed | **9,090** |
| — train | 7,275 |
| — holdout | **1,815** |
| Holdout transactions | 20,479 (20.0%) |
| Overall failure rate | 8.89% |
| Value at risk (corpus) | ₹13,348,370.81 |
| Value at risk (holdout) | **₹2,646,060.94** |

### 4.3 Failure rates by flow

| Flow | Failure rate |
|---|---:|
| One-off payments | 7.55% |
| **Subscription mandates** | **30.30%** |

The 4× gap is deliberate and documented. In India, recurring mandates (UPI
AutoPay, e-NACH) expire annually and lapse far more often than one-off card
payments decline. This is a **modelling assumption**, explicitly labelled as
such in the manifest — not an empirical measurement.

### 4.4 Cause distribution (ground truth, full corpus)

| Cause | Count |
|---|---:|
| insufficient_funds | 3,781 |
| gateway_timeout | 2,905 |
| mandate_lapsed | 1,220 |
| card_expired | 1,184 |

### 4.5 Properties the generator guarantees

1. **Reproducibility** — same seed produces a byte-identical corpus.
2. **Zero customer overlap** between train and holdout.
3. **Ground truth never appears in the feature context.** It is stored in
   separate per-split files and is structurally absent from prediction input.
4. **Error codes overlap across causes** — `ERR_DO_NOT_HONOR` and `ERR_UNKNOWN`
   appear under multiple causes, so the task is genuinely probabilistic and the
   diagnoser is *not* expected to reach 100%.
5. **No future information** in any feature. Customer behavioural features
   (median amount, prior success rate, prior failure count) are computed from
   strictly earlier transactions.
6. **All four buckets present in both splits.**

### 4.6 Split strategy

Default is **`random_grouped`**: 400 of 2,000 customers drawn at random; all of
a held-out customer's transactions go to the holdout. Grouping is essential —
customer-level features would leak across a row-level random split and inflate
every score.

Selectable via `SPLIT_STRATEGY`:

| Strategy | Description |
|---|---|
| `random_grouped` | Random customers, all their transactions (default) |
| `random_rows` | Random individual transactions |
| `temporal` | Last N days held out |

---

## 5. Root-cause diagnoser

### 5.1 Model

XGBoost multiclass classifier, **40 engineered features**:

- **Amount context** — `amount`, `amount_vs_customer_median`, `customer_median_amount`
- **Customer history** — `customer_tenure_days`, `customer_prior_txn_count`, `customer_prior_failure_count`, `customer_prior_success_rate`
- **Temporal** — `hour`, `day_of_week`, `day_of_month`, `is_peak_hour`
- **Infrastructure** — `gateway_latency_ms`, gateway one-hot (`axis`/`hdfc`/`icici`/`sbi`)
- **Card/mandate lifecycle** — `days_to_card_expiry`, `mandate_age_days`, `mandate_validity_days`, `mandate_days_overdue`
- **Categorical** — payment method, transaction type
- **Error code** — 12 one-hot columns

### 5.2 Tuning

40 configurations × 5-fold **`StratifiedGroupKFold(groups=customer_id)`**,
scored on `f1_macro`, run on the **train split only** (7,275 failures).

Best parameters:

```json
{
  "max_depth": 3, "n_estimators": 600, "learning_rate": 0.03,
  "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 1,
  "reg_lambda": 2.0, "gamma": 0.1
}
```

Cross-validation: **macro-F1 0.9726 ± 0.0019**
Fold scores: 0.9706 · 0.9738 · 0.9740 · 0.9745 · 0.9701

The tight spread (±0.0019) is the point of grouped CV — it shows the score is
not an artifact of one lucky split.

### 5.3 Held-out performance (n = 1,815)

| Cause | Precision | Recall | F1 | n (truth) |
|---|---:|---:|---:|---:|
| gateway_timeout | 0.9805 | 1.0000 | **0.9901** | 602 |
| mandate_lapsed | 0.9862 | 0.9907 | **0.9885** | 216 |
| insufficient_funds | 0.9753 | 0.9715 | **0.9734** | 772 |
| card_expired | 0.9535 | 0.9111 | **0.9318** | 225 |
| **macro avg** | 0.9739 | 0.9683 | **0.9709** | 1,815 |
| **weighted avg** | 0.9756 | 0.9758 | 0.9756 | 1,815 |

**Accuracy 0.9758 · Macro-F1 0.9709**

`card_expired` is the weakest bucket (F1 0.9318), and it is also the bucket with
the lowest settlement probability — the two facts compound into the lowest
recovery rate. This is visible honestly in the dashboard rather than hidden.

### 5.4 Attribution — read this before quoting 0.9709

The error code Razorpay returns is already highly informative. Anyone could
write a lookup table. So the score is decomposed:

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| `error_code` → majority cause **lookup, no ML** | 0.9289 | 0.9291 |
| **Full model (40 features)** | 0.9758 | **0.9709** |
| Model with `error_code` removed (28 features) | 0.8738 | 0.8264 |

**The honest framing:**

- **+0.0418 macro-F1** over a lookup table anyone could write.
- **0.8264** from transaction context alone, when the error code is withheld.

Both statements are true and both matter. The generator's error-code
distributions were deliberately **not** weakened to make the first number look
more modest — depressing a metric on purpose is the same dishonesty as inflating
one.

---

## 6. Policy engine

`agents/policy_engine.py` — pure Python, stateless, deterministic, no I/O.
23 dedicated tests.

### 6.1 The rules

| Rule | Constraint | On violation |
|---|---|---|
| **R0a** | Amount must be positive | `INVALID_AMOUNT` |
| **R0b** | Agent must be a known agent | `UNKNOWN_AGENT` |
| **R1** | No action above **₹10,000** auto-approved | `AMOUNT_EXCEEDS_AUTO_APPROVE_LIMIT` |
| **R2** | At most **3 attempts** per failure | `ATTEMPT_LIMIT_REACHED` |
| **R3** | **Stopping rule**: after 3 failed attempts, halt permanently | `STOPPING_RULE_3_FAILED_ATTEMPTS` |

### 6.2 Rule ordering is deliberate

**R3 is evaluated first.** This matters: a case that has burned through its
attempts must log a `STOP` with a reason, not a generic denial. Checking the
amount ceiling first would produce a misleading audit record — the ledger would
say "too expensive" when the truth is "we already tried three times and stopped."

The STOP also survives an `approved=True` flag and any amount. Once the stopping
rule fires, nothing in the system will retry that case.

### 6.3 Known agents

```python
KNOWN_AGENTS = ("payment_failure_agent", "subscription_agent")
```

An unknown agent identifier is rejected, so a typo or a new agent cannot
silently bypass policy.

### 6.4 The check is logged before execution

Every policy evaluation — pass **or** fail — writes an audit record *before* any
money moves. The audit trail therefore records intent, not just outcome: a
denied action is as much a part of the compliance story as an executed one.

---

## 7. Recovery agents

### 7.1 Shared sequence

`agents/recovery_base.py` implements one sequence used by both agents, so the
two flows cannot drift apart:

```
for attempt in 1 .. max_attempts + 1:
    count prior attempts from `recoveries`
    ──► POLICY CHECK          (logged first, always)
    ──► STOP   → escalate, record reason, break
    ──► DENY   → block this attempt, continue
    ──► ALLOW  → execute
              ├─ success → record recovered amount, break
              └─ fail    → record, back off, next attempt
```

The `+1` in the loop bound is what makes the stopping rule **observable**: the
loop runs one extra iteration specifically so the STOP decision is recorded in
the ledger rather than the loop simply ending.

Backoff is **recorded, not slept** — the pipeline runs in seconds, not hours.

### 7.2 The two agents

| Agent | Flow | Action |
|---|---|---|
| `payment_failure_agent` | One-off payments | Create a Razorpay Payment Link |
| `subscription_agent` | Recurring mandates | Retry the subscription mandate |

Both are thin wrappers supplying their own `execute` callable to the shared
sequence.

### 7.3 Settlement outcomes

Success is cause-dependent and attempt-dependent (see §8.2). This produces
genuinely different recovery profiles per cause — the system is not equally good
at everything, and the dashboard shows that.

---

## 8. Razorpay integration

### 8.1 Two clients, one interface

`api/razorpay_client.py` exposes `get_razorpay_client()`, returning either:

| Client | When | Behaviour |
|---|---|---|
| `SimulatedRazorpayClient` | Default | Offline, seeded, Razorpay-shaped responses |
| `LiveRazorpayClient` | `RAZORPAY_LIVE_TEST_MODE=1` + keys | Real test-mode API, auto-degrades on error |

Swapping between them is a **one-environment-variable change with no calling-code
modification**. This is how the system honours FR-008 while running in an
environment with no credentials.

### 8.2 Simulator design

The simulator is not random noise. Settlement probability is conditioned on both
**cause** and **attempt number**:

| Cause | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---:|---:|---:|
| gateway_timeout | 0.78 | 0.72 | 0.65 |
| mandate_lapsed | 0.55 | 0.45 | 0.35 |
| insufficient_funds | 0.35 | 0.42 | 0.30 |
| card_expired | 0.12 | 0.10 | 0.08 |

This encodes the real-world intuition that retrying a transient gateway timeout
usually works, while retrying an expired card usually does not.

Injected real-world failures:
- **4% 504 Gateway Timeout**
- **2% 429 Rate Limited**

Neither raises. Both are returned as structured failure responses, so the
recovery loop handles them like any other failed attempt.

### 8.3 Response fidelity

Responses match real Razorpay shapes:
- Amounts in **paise**
- Payment link IDs prefixed `plink_`
- Short URLs on `rzp.io`
- Real error codes (`ERR_CARD_EXPIRED`, `ERR_INSUFFICIENT_FUNDS`, …)

`response_is_recovered()` accepts only `paid` / `captured` / `completed` — a
`created` or `pending` link is **not** counted as recovered money.

### 8.4 Degradation

A live client that hits an auth, network, or timeout error falls back to the
simulator and marks the response `degraded=True` (NFR-002). The pipeline
continues; the degradation is visible rather than silent.

---

## 9. Autopsy reporter

`agents/autopsy_reporter.py` — FR-005, text only.

### 9.1 Pipeline

```
build_facts()  →  build_prompt()  →  [backend]  →  check_hallucinated_numbers()
```

Facts are injected as structured JSON, not prose. The model writes sentences
around supplied values; it never computes anything.

### 9.2 Backends

| Backend | Use |
|---|---|
| `ollama` | Local model, default for development |
| `openai` | Alternative, pluggable via the same adapter |
| `template` | Deterministic, zero-network fallback |

The adapter is pluggable by design: the submission can swap to a stronger final
model without touching the pipeline.

### 9.3 Guarantees

1. **Never raises.** Any backend failure — unreachable host, timeout, empty
   completion, misconfigured key — falls back to the template with
   `degraded=True`.
2. **Never invents numbers.** `check_hallucinated_numbers()` flags any numeric
   token in the output that is not present in the facts. Flagged text is stored
   and flagged, not silently hidden.
3. **Provenance on every report** — model, basis, latency.

In the current run: **1,815 reports generated, 0 hallucination flags, 0 degraded.**

---

## 10. Audit trail

`database/audit_trail.py` — FR-006, append-only.

### 10.1 Design commitments

- **Write-ahead**: the policy check is logged *before* execution.
- **Never raises**: `log()` catches and prints a warning rather than crashing a
  recovery run. An audit system that can take down the pipeline it audits is a
  liability.
- **Complete**: pass and fail both recorded.

### 10.2 Components

```
synthetic_generator · diagnoser · policy_engine · payment_failure_agent
subscription_agent · razorpay_client · autopsy_reporter · pipeline
```

### 10.3 Volume in the current run

**7,633 audit records** across 3,816 recovery actions — roughly two records per
action (the policy check plus the execution result), plus generation and
diagnosis records.

The headline reconciliation check compares the sum of `recoveries` against the
audit-trail totals and reports **PASS**.

---

## 11. Metrics and dashboard

### 11.1 Single source of truth

`metrics.py` is the **only** place dashboard figures are computed. The Streamlit
app and `weekly_report.py` both call it, which is why the dashboard, the weekly
report, and the JSON artifacts cannot disagree.

### 11.2 Dashboard sections

| Section | Content |
|---|---|
| Headline | Recovered / At Risk / N / Rate + basis + reconciliation check |
| Recovery over time | 180-day trend |
| Cause breakdown | Per-cause at-risk, recovered, rate, stops |
| Stopping-rule panel | Escalated cases with reasons |
| Diagnoser panel | Precision/recall with **n** and split method, plus attribution |
| Autopsy sample | Generated explanations |
| Audit trail | Filterable, denials included |

A persistent banner states that settlement is simulated whenever
`RAZORPAY_LIVE_TEST_MODE=0`.

### 11.3 The double-counting bug and its fix

**This is the most important bug in the build's history.**

Joining `failures` directly to `recoveries` multiplies each failure's
`estimated_value` by its number of attempts. A failure with 3 attempts was
counted 3 times, inflating per-cause and per-day "at risk" to roughly **2.1×**
(₹5.58M reported vs ₹2.65M actual).

**The fix:** every query that sums `estimated_value` alongside `recoveries` now
aggregates recoveries to one row per failure in a `WITH rec AS (...)` CTE
*before* joining.

**The guard:** `test_per_cause_totals_match_the_headline` and
`test_daily_totals_match_the_headline` assert the breakdowns sum to the headline
exactly (₹0.01 tolerance). The bug cannot silently return.

---

## 12. Results

### 12.1 Headline — held-out batch, n = 1,815 failures

| | |
|---|---|
| **₹ Recovered** | **₹1,907,719.68** |
| ₹ At Risk | ₹2,646,060.94 |
| **Recovery rate** | **72.10%** |
| Transactions in batch | 20,479 |
| Failures | 1,815 |
| Recovery actions | 3,816 |
| Audit records | 7,633 |
| **Reconciliation** | ✅ **PASS** |

### 12.2 Final outcome per failure (1,815)

| Outcome | Count | Share |
|---|---:|---:|
| Recovered | 1,368 | 75.4% |
| Escalated (stopping rule) | 440 | 24.2% |
| Blocked by policy | 7 | 0.4% |
| Abandoned without stop | 0 | 0.0% |

Every single failure reached a terminal state. **None was silently dropped.**

### 12.3 Outcome per action (3,816)

| Outcome | Count |
|---|---:|
| Success | 1,368 |
| Failed attempt | 2,001 |
| Stopped | 440 |
| Blocked | 7 |

### 12.4 Per-cause breakdown

| Cause | Failures | At risk | Recovered | Rate | Stopped |
|---|---:|---:|---:|---:|---:|
| insufficient_funds | 769 | ₹1,285,999.87 | ₹857,974.98 | 66.7% | 237 |
| gateway_timeout | 614 | ₹890,265.80 | ₹835,005.68 | **93.8%** | 10 |
| mandate_lapsed | 217 | ₹162,742.40 | ₹124,950.01 | 76.8% | 43 |
| card_expired | 215 | ₹307,052.87 | ₹89,789.01 | **29.2%** | 150 |

The spread tells a coherent story: `gateway_timeout` is transient and retries
well (93.8%). `card_expired` requires the customer to act, so automated retry
rarely works (29.2%) and most cases correctly escalate (150 of 215 stopped).

The system's behaviour matches the domain — evidence the recovery logic is
grounded in how these failures actually behave, not applied uniformly.

### 12.5 Diagnoser

| Metric | Value |
|---|---|
| Macro-F1 (held-out) | **0.9709** |
| Accuracy (held-out) | 0.9758 |
| CV macro-F1 | 0.9726 ± 0.0019 |
| n | 1,815 |
| Lift over lookup baseline | **+0.0418** |
| Context-only (error code ablated) | 0.8264 |

### 12.6 Timings

| Operation | Duration |
|---|---:|
| Clean checkout, one command | **78s** |
| Warm pipeline | 6–20s |
| Full 40-config tuning | 174s |
| Full test suite (127 tests) | 85s |

Tuning is excluded from the demo path — the trained model loads from
`models/root_cause_classifier.json`.

### 12.7 Determinism verified

Three consecutive `python main.py` runs produced **byte-identical** figures:

| Run | Recovered | At risk | Actions | Stops |
|---|---:|---:|---:|---:|
| 1 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |
| 2 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |
| 3 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |

---

## 13. Testing

**127 tests, 7 files, all passing.**

| Test file | Tests | Module |
|---|---:|---|
| `test_policy_engine.py` | 23 | `agents/policy_engine.py` |
| `test_data_generator.py` | 21 | `data/synthetic_generator.py` |
| `test_razorpay_client.py` | 20 | `api/razorpay_client.py` |
| `test_autopsy.py` | 17 | `agents/autopsy_reporter.py` |
| `test_metrics.py` | 15 | `metrics.py` |
| `test_diagnoser.py` | 14 | `diagnoser/*` |
| `test_end_to_end.py` | 12 | agents + audit trail |
| `test_pipeline.py` | 5 | `main.py` |

### 13.1 What the sweep found

Writing the tests surfaced **five real defects**, four in code already believed
working:

1. **Pipeline not idempotent** — a second run stacked actions on the first,
   hitting the stopping rule immediately. Fixed with `reset_action_state()`.
2. **Foreign-key violation** in that fix (deleted `failures` before
   `autopsy_reports`). Reordered children before parents.
3. **Empty LLM completion** produced a blank autopsy marked healthy. Blank
   completions now route to fallback.
4. **Attribution silently dropped** — `main.py` overwrote
   `holdout_metrics.json` without it. Moved into shared `evaluate_model()`.
5. **Brittle prose assertion** — replaced with a machine-readable
   `[strategy=random_grouped]` marker.

### 13.2 Not covered

1. **Streamlit UI rendering** — data layer tested via `metrics.py`; HTTP
   endpoints return 200; widget rendering verified by eye.
2. **Live Razorpay** — no credentials exist. `LiveRazorpayClient` is tested for
   *failure behaviour*, not against the real API.
3. **Real LLM output quality** — synthetic completions injected; the Ollama path
   is tested for degradation, not prose quality.
4. **Concurrency** — the pipeline is single-threaded.

---

## 14. Running the system

### 14.1 One command (clean checkout)

```bash
python3 -m pip install -r requirements.txt
python main.py
```

Generates the corpus, trains the diagnoser, runs recovery, writes autopsies, and
prints headline metrics. **78 seconds, zero manual steps.**

### 14.2 Other entry points

```bash
python main.py --help                  # all flags
python main.py --regen                 # rebuild the corpus from seed
python main.py --retrain               # force hyperparameter tuning
python main.py --reset-db              # rebuild the database
python main.py --limit 250             # smaller batch
python main.py --no-autopsy            # skip LLM explanations
python main.py --llm-backend template  # deterministic autopsies, no network

python diagnoser/train.py              # train + evaluate + attribute
python diagnoser/train.py --no-tune    # fast path, skip tuning
python weekly_report.py                # FR-009 markdown report
streamlit run dashboard/app.py         # dashboard on :8501
```

### 14.3 Testing

```bash
python3 -m pytest tests/ -q                          # 127 tests, ~85s
python3 -m pytest tests/ --ignore=tests/test_pipeline.py   # fast, ~7s
```

### 14.4 Environment

Python 3.13.14 · numpy 2.3.5 · pandas 2.2.3 · scikit-learn 1.6.1 ·
xgboost 3.4.1 · streamlit 1.63.0 · altair 6.2.2 · pytest 9.0.3 · razorpay 2.0.1

Tested under a 2 GB RAM constraint.

---

## 15. Configuration reference

All settings live in `.env` (see `.env.example`). Every value has a safe offline
default — with the file untouched, the system runs fully offline.

### Data

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SEED` | `42` | Generation seed |
| `DATASET_PROFILE` | `train` | Size preset |
| `SPLIT_STRATEGY` | `random_grouped` | `random_grouped` / `random_rows` / `temporal` |
| `HOLDOUT_FRACTION` | `0.2` | Share of customers held out |
| `CV_FOLDS` | `5` | Cross-validation folds |

### Policy

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_MAX_AUTO_APPROVE_INR` | `10000` | Auto-approval ceiling (R1) |
| `POLICY_MAX_ATTEMPTS_PER_FAILURE` | `3` | Attempt cap (R2) |
| `POLICY_STOP_ON_FAILED_ATTEMPT` | `3` | Failures before permanent stop (R3) |

### Razorpay

| Variable | Default | Meaning |
|---|---|---|
| `RAZORPAY_KEY_ID` | *(empty)* | Test-mode key |
| `RAZORPAY_KEY_SECRET` | *(empty)* | Test-mode secret |
| `RAZORPAY_LIVE_TEST_MODE` | `0` | `1` = use real test-mode API |
| `RAZORPAY_TIMEOUT_SECONDS` | `6.0` | Request timeout |

### LLM

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` / `openai` / `template` |
| `LLM_MODEL` | `llama3.2:3b` | Model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LLM_TIMEOUT_SECONDS` | `12.0` | Generation timeout |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |

---

## 16. Design decisions

### 16.1 Seven hard constraints

1. **No LLM in the money path.** Autopsy only.
2. **Razorpay test-mode only.** Never live keys.
3. **Every financial action is policy-checked and logged before execution**,
   pass or fail. A STOP halts permanently.
4. **No hardcoded secrets.**
5. **No metric without `n` and split method beside it.**
6. **No P2 until P0 is complete** and one clean end-to-end run passes.
7. **One command starts the system.**

### 16.2 Trade-offs accepted

| Decision | Alternative rejected | Why |
|---|---|---|
| Offline Razorpay simulator | Fake arbitrary success rates | Shape fidelity + one-env-var swap to live |
| Random grouped split | Temporal split | Matches the instruction; grouping prevents leakage |
| Grouped CV (`StratifiedGroupKFold`) | Plain KFold | Row-level folds leak customer features |
| SQLite | Postgres | Zero-install, one command, adequate at this scale |
| XGBoost | Deep learning | Tabular, 40 features, needs calibration and speed |
| Recorded backoff | Real `sleep` | Demo must finish in seconds |
| Template LLM fallback | Crash on LLM failure | Never let prose break the money path |

### 16.3 Attribution as an anti-hype measure

Publishing a bare "97% accurate" figure would be misleading, because a lookup
table on the error code already reaches 92.9%. The attribution table is rendered
in the dashboard and the weekly report **next to** the headline score so the
claim is never read without its context.

---

## 17. Honest limitations

1. **Settlement is simulated.** No Razorpay credentials exist in this
   environment. The ₹ figures measure the pipeline's behaviour against a
   documented, cause-conditioned settlement model — not real money. Swapping in
   test-mode keys changes this without code changes, but the numbers would move.

2. **The corpus is synthetic.** Generated from stated assumptions, not merchant
   data. Failure rates are modelling choices, not measurements.

3. **Recovery rates are model-dependent.** `card_expired`'s 29.2% follows from a
   simulated 8–12% settlement probability. Real expired-card recovery depends on
   customer responsiveness, which the simulator cannot capture.

4. **No customer contact channel.** The system retries payments; it does not
   send dunning emails or SMS. Causes needing customer action necessarily
   underperform.

5. **Single-threaded.** No concurrency in the recovery loop or audit writer.

6. **The dashboard is not browser-tested.** Data layer tested; rendering
   verified manually.

7. **One DoD item is unmet:** a backup demo video cannot be produced in this
   environment. Every other definition-of-done item passes.

### Open questions awaiting sign-off

- **Additive `autopsy_reports` table** vs strict spec purity — the spec's schema
  did not include it; it was added to make explanations auditable.
- **180-day corpus** vs the spec's literal 30-day "history" — the larger window
  was generated to satisfy the request for a large training dataset.

---

## 18. Failure log

`FAILURES.md` records **17 dated entries** — every real blocker, written the day
it happened rather than reconstructed later. Format: what broke → why it
matters → resolution → status.

Significant entries:

| Entry | Resolution |
|---|---|
| No Razorpay test-mode credentials | Offline simulator behind the live interface |
| No SHAP on Python 3.13 | Use `pred_contribs`, handle 3-D multiclass output |
| Attempt-multiplication double counting | CTE aggregation + two permanent regression tests |
| Clean checkout crashed (manifest present, ledger empty) | Regenerate when `ledger_rows == 0` |
| Stale diagnoser metrics after auto-training | Shared `evaluate_model()`, refreshed every run |
| Pipeline not idempotent | `reset_action_state()` at run start |
| FK violation in that reset | Delete children before parents |
| Empty LLM completion → blank autopsy | Treat blank as failure, route to fallback |
| Attribution dropped by `main.py` | Moved into shared `evaluate_model()` |

---

## 19. Repository layout

```
ghost-ledger/
├── main.py                      Pipeline orchestrator
├── metrics.py                   Single source of truth for all figures
├── weekly_report.py             FR-009 markdown report
├── config.py                    All settings, safe offline defaults
├── requirements.txt
├── README.md                    Quick start
├── FAILURES.md                  17 dated failure entries
├── .env.example
│
├── agents/
│   ├── policy_engine.py         FR-003 guardrails
│   ├── recovery_base.py         Shared recovery sequence
│   ├── payment_failure_agent.py FR-004a
│   ├── subscription_agent.py    FR-004b
│   └── autopsy_reporter.py      FR-005, text only
│
├── api/
│   └── razorpay_client.py       FR-008, simulator + live
│
├── data/
│   ├── synthetic_generator.py   FR-001
│   └── sample_output/           Manifest, labels, contexts
│
├── database/
│   ├── schema.sql               5 tables
│   ├── db_client.py
│   ├── seed_db.py
│   └── audit_trail.py           FR-006
│
├── diagnoser/
│   ├── root_cause_classifier.py
│   ├── train.py                 Tuning + CV + attribution
│   ├── eval_holdout.py
│   └── split_experiment.py
│
├── dashboard/
│   ├── app.py
│   └── components/              4 reusable panels
│
├── models/                      Trained classifier + metadata
├── reports/                     Metrics, weekly report, test report
├── tests/                       7 files, 127 tests
└── docs/                        This document
```

### Database schema (5 tables)

| Table | Purpose |
|---|---|
| `transactions` | Full ledger, 102,262 rows |
| `failures` | Diagnosed failures with predicted cause + confidence |
| `recoveries` | Every attempt, outcome, recovered amount, stop flag |
| `audit_trail` | Append-only compliance log |
| `autopsy_reports` | Generated explanations with provenance |

Foreign keys: `recoveries.failure_id` → `failures.id` → `transactions.id`.

---

## Appendix A — Quick reference card

```
RECOVERED      ₹1,907,719.68
AT RISK        ₹2,646,060.94
RATE           72.10%
FAILURES       1,815  (held out)
ACTIONS        3,816
STOPPED        440
AUDIT RECORDS  7,633
RECONCILED     PASS

MACRO-F1       0.9709  (n=1,815, held out)
CV             0.9726 ± 0.0019  (5-fold grouped, train only)
LIFT           +0.0418 over error-code lookup
ABLATED        0.8264  (error code withheld)

CORPUS         102,262 txns · 2,000 customers · 180 days
TESTS          127 passing
CLEAN RUN      78 seconds
```

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| **At risk** | Sum of `estimated_value` for all held-out failures |
| **Recovered** | Sum of `recovered_amount` on successful recovery actions |
| **Stopping rule (R3)** | After 3 failed attempts, halt permanently and escalate |
| **GROUPED split** | Hold out customers, not rows — prevents customer-level leakage |
| **Attribution** | Decomposing model score vs lookup baseline and ablated model |
| **Autopsy** | Plain-English explanation of one failure |
| **Degraded** | A component fell back and flagged that it did |
| **Paise** | Razorpay's minor currency unit; ₹1 = 100 paise |

---

*All figures computed from `metrics.py`, the same source of truth as the
dashboard. Recovery settlement in this environment is simulated unless Razorpay
test credentials are configured.*
