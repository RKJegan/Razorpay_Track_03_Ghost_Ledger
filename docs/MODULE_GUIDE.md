# Module guide

This file explains every source file in the project. Each per-file section
summarises that file's own docstring and lists its public classes and functions,
so the entries match the code as it was when this guide was written. Files are
grouped by folder below.

## How the folders fit together

| Folder | Role | Track |
|---|---|---|
| `data/` | Synthetic transaction generator and the seeded dataset (deterministic, seed 42). | v2 |
| `diagnoser/` | Root-cause classifier (XGBoost): labels each failed payment with one of four causes. | v2 |
| `agents/` | Deterministic policy engine (R1–R4), recovery sequences, approvals, jobs, autopsy text. | v2 + v3 A3/A4 |
| `api/` | FastAPI app: Razorpay client adapter, webhooks, scheduler, approval and merchant APIs. | v3 A1/A4/A3/A6 |
| `database/` | SQLite access: v2 tables, event log, merchants, approvals, migrations, models. | v2 + v3 A2 |
| `strategies/` | Track B engine: playbooks, router, timing, gateway health, dunning, A/B tests. | v3 B1–B8 |
| `dashboard/` | Streamlit pages: v2 recovery dashboard and v3 live progress page. | v2 + v3 A5 |
| `web/merchant/` | Static HTML/CSS/JS merchant dashboard served by the API at `/merchant`. | v3 A6 |
| `scripts/` | Command-line tools: end-to-end Test Mode check, retry holdout comparison, autopsy backend comparison. | v3 A7, B3 |
| `playbooks/` | YAML playbooks, one per root cause. Read by `strategies/playbooks.py`. | v3 B1 |
| `tests/` | pytest suite (292 tests). | v2 + v3 |
| `reports/`, `models/`, `data/sample_output/` | Generated outputs and the holdout files the pipeline writes or reads. | v2 |

## The request path in one picture

```
Razorpay (or the simulator) ──► api/webhooks.py  (HMAC check, idempotency, store raw event)
                                    │
                                    ▼
                         database/recovery_store.py  (append-only recovery_events)
                                    ▲
diagnoser ─► main.py pipeline ─► agents/recovery_executor.submit_recovery
                                    │
              ENABLE_ADVANCED_STRATEGIES = 0 ──► submit_direct (v2 path)
              ENABLE_ADVANCED_STRATEGIES = 1 ──► strategies/runner.handle_failure
                                    │
                 policy ruling first (R1–R4, always) ─► router (only if allowed)
                                    │
                 create link │ schedule retry │ dunning only
                                    ▼
                    api/razorpay_client.py (simulator or Test Mode)
```

## Where the policy lives

Every amount, approval, stop, and attempt limit is decided in
`agents/policy_engine.py` (rules R1–R4). Nothing else can skip it. The Track B
router runs only after that ruling says "allowed". No module in `strategies/`
imports an LLM client, and a test enforces this.

---

### `agents/__init__.py` (1 lines)
Ghost Ledger agents: deterministic policy gate + two recovery agents.

### `agents/approval_queue.py` (135 lines)
Ghost Ledger v3 — human approval workflow (A3).
Definitions:
- `def approve_recovery` — Approve a pending recovery and execute it through the policy gate.
- `def reject_recovery` — Reject a pending recovery and escalate it

### `agents/autopsy_reporter.py` (468 lines)
FR-005 — Autopsy Report Generator.
Definitions:
- `class AutopsyReport` — A generated explanation, with full provenance.
- `def build_facts` — Assemble the structured fact block that grounds the explanation.
- `def build_prompt` — Build the structured, data-injected user prompt.
- `def check_hallucinated_numbers` — Flag numeric tokens in the text that are not present in the facts.
- `def generate_autopsy` — Generate one autopsy explanation.
- `def persist` — Store an autopsy report in the ``autopsy_reports`` table.

### `agents/payment_failure_agent.py` (104 lines)
FR-004a — Payment Failure Agent.
Definitions:
- `def recover_payment_failure` — Run the Payment Failure recovery sequence for one failure.

### `agents/policy_engine.py` (372 lines)
FR-003 — Recovery Policy Engine (guardrails).
Definitions:
- `class RecoveryAction` — A proposed recovery action, before the policy engine has ruled on it.
- `class PolicyDecision` — The engine's ruling on a :class:`RecoveryAction`.
- `class PolicyEngine` — Stateless, deterministic rule evaluator.
- `def count_prior_attempts` — Count prior recovery attempts for a (customer, failure) pair.

### `agents/recovery_base.py` (353 lines)
Shared recovery sequence for both Resurrector agents.
Definitions:
- `class RecoveryOutcome` — Result of running the recovery sequence for one failure.
- `def run_recovery_sequence` — Run the bounded, policy-gated retry sequence for one failure.

### `agents/recovery_executor.py` (329 lines)
Ghost Ledger v3 — recovery executor (A3, shared by approvals, jobs and the demo).
Definitions:
- `class SubmitResult` — What happened when a recovery was submitted for execution.
- `class Ruling` — The policy ruling for one recovery
- `def rule_on_recovery` — Rule on the next attempt for one recovery
- `def execute_ruling` — Act on a ruling made by :func:`rule_on_recovery`
- `def submit_direct` — The v2-compatible path: rule, then execute
- `def submit_recovery` — Single entry point for v3 recovery actions (approvals, jobs, demo).

### `agents/scheduled_jobs.py` (461 lines)
Ghost Ledger v3 — background jobs (A4).
Definitions:
- `class TransientJobError` — A recoverable failure, such as a degraded or failed Razorpay response.
- `def backoff_seconds` — Return the wait after ``streak`` consecutive failures: base * 2^(streak-1), capped.
- `def run_guarded` — Run a job, honouring its backoff window, and record the outcome.
- `def job_health` — Return the health record of every job that has run at least once.
- `def check_payment_settlement` — Poll one recovery's payment link
- `def poll_open_links` — Run :func:`check_payment_settlement` on every open link
- `def reconcile_recovery` — Compare a webhook-confirmed capture with the recovery amount, and record the result.
- `def reconcile_all` — Reconcile every settled recovery that is not yet reconciled.
- `def retry_recovery` — Create the next attempt if the last link is dead (expired or the payment failed).
- `def retry_due_recoveries` — Run the retry pass
- `def clean_expired_links` — Handle expired payment links.

### `agents/subscription_agent.py` (110 lines)
FR-004b — Subscription Recovery Agent.
Definitions:
- `def recover_subscription_failure` — Run the Subscription Recovery sequence for one failed mandate debit.

### `api/__init__.py` (0 lines)
(no module docstring)

### `api/approvals_api.py` (125 lines)
Ghost Ledger v3 — approval routes (A3).
Definitions:
- `def require_operator` — Check the operator key
- `class ApproveBody` — Who is approving.
- `class RejectBody` — Who is rejecting, and why
- `def list_all` — List approvals, newest first.
- `def get_one` — One approval with the full timeline of its recovery.
- `def approve` — Approve and execute
- `def reject` — Reject and escalate

### `api/main.py` (82 lines)
Ghost Ledger v3 — FastAPI application.
Definitions:
- `def lifespan` — Bring the unified database up to date on startup.
- `def health` — Liveness check

### `api/merchant_api.py` (214 lines)
Ghost Ledger v3 — merchant-scoped API and web view (A6).
Definitions:
- `def require_merchant` — Resolve the merchant from the bearer key
- `def me` — Identify the merchant behind the key.
- `def summary` — This merchant's live state: counts, active and failing recoveries, pending approvals.
- `def list_recoveries` — This merchant's recoveries, newest first.
- `def get_recovery` — One recovery with its sanitised timeline
- `def merchant_page` — The merchant dashboard page.
- `def merchant_js` — Page script.
- `def merchant_css` — Page styles.

### `api/operator_api.py` (93 lines)
Ghost Ledger v3 — operator summary for the live dashboard (A5).
Definitions:
- `def build_summary` — Build the dashboard payload
- `def summary` — Live state for the operator dashboard.

### `api/razorpay_client.py` (501 lines)
FR-008 — Razorpay test-mode integration.
Definitions:
- `class RazorpayResponse` — Normalised response from any Razorpay-shaped operation.
- `class SimulatedRazorpayClient` — Offline, seeded, Razorpay-shaped client.
- `class LiveRazorpayClient` — Real Razorpay **test-mode** client, with graceful degradation.
- `def get_razorpay_client` — Return the configured Razorpay client.
- `def response_is_recovered` — Return True when a response represents money actually recovered.

### `api/scheduler.py` (99 lines)
Ghost Ledger v3 — background scheduler wiring (A4).
Definitions:
- `def job_table` — Return (job name, function, interval seconds) for every A4 job (+ dunning when B is on).
- `def run_job` — Run one A4 job by name, through the backoff guard
- `def build_scheduler` — Create (but do not start) a scheduler with every A4 job registered.
- `def start_scheduler` — Start the scheduler if ``SCHEDULER_ENABLED`` is set
- `def stop_scheduler` — Stop the scheduler if it is running
- `def scheduler_running` — True if the background scheduler is currently running.

### `api/webhooks.py` (421 lines)
Ghost Ledger v3 — Razorpay webhook listener (A1).
Definitions:
- `def verify_signature` — Check a Razorpay webhook signature in constant time.
- `class ParsedWebhook` — The fields of a Razorpay event that Ghost Ledger uses.
- `def parse_razorpay_event` — Extract the fields Ghost Ledger needs from a Razorpay event body.
- `def store_webhook_event` — Persist a verified webhook
- `def process_webhook_event` — Apply one stored webhook to its recovery
- `def razorpay_webhook` — Receive a Razorpay webhook delivery.

### `config.py` (265 lines)
Ghost Ledger v2 - Central configuration.

### `dashboard/__init__.py` (0 lines)
(no module docstring)

### `dashboard/app.py` (146 lines)
FR-007 — Recovery Dashboard.
Definitions:
- `def main` — Render the dashboard.

### `dashboard/components/__init__.py` (0 lines)
(no module docstring)

### `dashboard/components/audit_trail_table.py` (77 lines)
Filterable audit-trail table.
Definitions:
- `def render` — Render the audit trail with component and outcome filters.

### `dashboard/components/headline_metric.py` (74 lines)
Headline figures: Recovered / At Risk / N transactions, on the held-out batch.
Definitions:
- `def render` — Render the headline metric row.

### `dashboard/components/precision_recall_panel.py` (83 lines)
Diagnoser precision / recall panel.
Definitions:
- `def render` — Render the diagnoser evaluation and attribution panels.

### `dashboard/components/stopping_rule_panel.py` (79 lines)
Stopping-rule panel.
Definitions:
- `def render` — Render the stopping-rule events panel.

### `dashboard/live_progress.py` (173 lines)
Ghost Ledger v3 — live progress dashboard (A5).
Definitions:
- `def api_get` — GET from the API
- `def api_post` — POST to the API
- `def live_panel` — The auto-refreshing part of the page.
- `def main` — Page entry point.

### `data/__init__.py` (0 lines)
(no module docstring)

### `data/synthetic_generator.py` (1414 lines)
FR-001 — Synthetic Data Generator =================================
Definitions:
- `class Customer` — A merchant customer with the attributes that drive failure behaviour.
- `class TxnSkeleton` — A planned transaction before features and outcome are materialised.
- `class SyntheticDataGenerator` — Seeded generator for a 30-day merchant transaction history.
- `def verify` — Run structural and leakage checks over a generated dataset.
- `def main` — Command-line entry point for the generator.

### `database/__init__.py` (0 lines)
(no module docstring)

### `database/approvals.py` (222 lines)
Ghost Ledger v3 — approval records (A3, data layer).
Definitions:
- `class ApprovalNotFoundError` — Raised when no approval has the requested id.
- `class ApprovalAlreadyDecidedError` — Raised when an approval has already been approved or rejected.
- `def create_approval` — Open a pending approval for a recovery.
- `def get_approval` — Return one approval as a dict, or None.
- `def get_pending_approval_for` — Return the open approval for a recovery, if one exists (prevents duplicates).
- `def list_approvals` — Return one page of approvals, newest first, plus the total match count.
- `def claim_decision` — Move a pending approval to ``approved`` or ``rejected``, exactly once.

### `database/audit_trail.py` (295 lines)
FR-006 — Audit trail.
Definitions:
- `def log` — Append one record to the audit trail.
- `def log_policy_check` — Log a policy-engine ruling
- `def log_action` — Log an executed (or refused) agent action.
- `def recent` — Read the most recent audit records.
- `def count` — Return the total number of audit records.

### `database/db_client.py` (273 lines)
Ghost Ledger v2 — SQLite client.
Definitions:
- `def transaction` — Context manager that commits on success and rolls back on exception.
- `def init_db` — Create every table if it does not already exist.
- `def init_db_v2` — Alias for :func:`init_db`, kept because the v3 run instructions call it.
- `def execute` — Execute a single statement and commit.
- `def executemany` — Execute a parameterised statement against many rows in one transaction.
- `def query` — Run a SELECT and return all rows.
- `def query_one` — Run a SELECT and return the first row, or None.
- `def scalar` — Return the first column of the first row, or None.
- `def reset_db` — Drop and recreate every table.
- `def to_json` — Serialise a value for storage in a TEXT column.

### `database/engine.py` (93 lines)
Ghost Ledger v3 — SQLAlchemy engine on the unified database.
Definitions:
- `def get_engine` — Return the process-wide SQLAlchemy engine, creating it on first use.
- `def session_scope` — Provide a transactional session: commit on success, roll back on error.
- `def dispose_engine` — Close pooled connections and forget the engine (used by tests and shutdown).

### `database/merchants.py` (226 lines)
Ghost Ledger v3 — merchant registry (up to 10 merchants).
Definitions:
- `class MerchantLimitError` — Raised when creating a merchant would exceed :data:`MERCHANT_MAX_COUNT`.
- `class DuplicateMerchantError` — Raised when a merchant id is already registered.
- `def hash_api_key` — Return the SHA-256 hex digest of an API key.
- `def generate_api_key` — Return a new random API key (43 URL-safe characters, 256 bits).
- `def count_merchants` — Return how many merchants are registered.
- `def create_merchant` — Register a merchant and return its new plain-text API key.
- `def rotate_api_key` — Issue a new API key for an existing merchant, invalidating the old one.
- `def verify_api_key` — Return True when ``api_key`` belongs to an active merchant.
- `def merchant_for_api_key` — Return the id of the active merchant that owns ``api_key``, or None.
- `def list_merchants` — Return every merchant (never the key hash).
- `def main` — Command-line entry point for merchant administration.

### `database/migrations.py` (215 lines)
Ghost Ledger — schema migrations for the unified database.
Definitions:
- `def run_migrations` — Create v3 tables and apply every pending numbered migration.
- `def migration_status` — List every known migration with the time it was applied, if ever.
- `def main` — Command-line entry point.

### `database/models.py` (221 lines)
Ghost Ledger v3 — SQLAlchemy models (new tables only).
Definitions:
- `class Base` — Declarative base shared by every v3 model.
- `class Merchant` — A merchant that uses Ghost Ledger
- `class RecoveryCase` — Cached current state of one recovery
- `class RecoveryEvent` — One immutable stage transition of a recovery
- `class WebhookEvent` — A verified inbound webhook
- `class Approval` — A human decision request for a recovery above the auto-approve ceiling (A3).
- `class JobState` — Per-job health for the scheduler: failure streak and the next allowed run (A4).
- `class BatchRun` — Progress of a long-running batch, read live by the progress dashboard (A5).
- `class DunningTouch` — One scheduled customer reminder in a dunning sequence (B5)
- `class GatewayObservation` — One payment outcome per payment method, used for gateway health (B4).
- `class AbAssignment` — Stable A/B assignment of one unit to one variant, with its first outcome (B6).

### `database/recovery_store.py` (466 lines)
Ghost Ledger v3 — event-sourced recovery tracking (A2).
Definitions:
- `class UnknownRecoveryError` — Raised when an event or query names a recovery that does not exist.
- `class UnknownStageError` — Raised when an event names a stage outside :data:`STAGES`.
- `def new_recovery_id` — Generate a new recovery identifier.
- `def create_case` — Open a new recovery case in status ``pending``.
- `def append_event` — Append one immutable stage event and update the cached case status.
- `def has_payment_recorded` — Return True if this payment is already recorded as a capture or confirmation.
- `def has_provisional_settlement` — Return True when the latest settlement was seen by a status poll and not yet confirmed by a webhook
- `def case_exists` — Return True when a recovery with this id exists.
- `def has_event_for_payment` — Return True if this recovery already has ``stage`` recorded for ``payment_id``.
- `def get_recovery_status` — Return the cached current state of one recovery.
- `def get_recovery_timeline` — Return every event for a recovery, in the order they were appended.
- `def get_active_recoveries` — Return recoveries still in status ``pending``, newest first.
- `def get_settled_today` — Return recoveries that settled on ``day`` (default: today, local time).
- `def count_events` — Return how many events of the given stages a recovery has (for attempt counting).
- `def latest_event` — Return the most recent event of the given stages, or None.
- `def list_cases` — Return one page of cases, filtered, newest first, plus the total match count.
- `def merchant_exists` — Return True when the merchant row exists.

### `database/seed_db.py` (127 lines)
Load generated transactions into the SQLite `transactions` table.
Definitions:
- `def load_transactions` — Insert (or replace) transaction rows into the database.
- `def load_subscriptions` — Persist subscription mandates into the audit trail as structured records.

### `diagnoser/__init__.py` (0 lines)
(no module docstring)

### `diagnoser/eval_holdout.py` (260 lines)
FR-002 evaluation — held-out precision / recall.
Definitions:
- `def load_split_method` — Read the split description written by the generator for this corpus.
- `def compute_metrics` — Per-class and aggregate precision / recall / F1.
- `def evaluate_holdout` — Build the complete, self-describing held-out evaluation record.
- `def print_report` — Render an evaluation report, with n and split method always visible.
- `def print_leakage_comparison` — Print a side-by-side comparison of the same model across split strategies.

### `diagnoser/root_cause_classifier.py` (562 lines)
FR-002 — Root Cause Diagnoser =============================
Definitions:
- `def load_failure_dataset` — Assemble the modelling table: features + ground-truth cause + split flag.
- `def build_feature_matrix` — Turn the modelling table into a numeric design matrix.
- `class RootCauseClassifier` — XGBoost root-cause classifier with grouped-CV tuning and SHAP reasons.

### `diagnoser/split_experiment.py` (171 lines)
Split-strategy leakage experiment.
Definitions:
- `def assign_split` — Return a boolean "is holdout" mask for a given strategy.
- `def main` — Run the three-way split comparison and print the result.

### `diagnoser/train.py` (383 lines)
Train the root-cause diagnoser end to end.
Definitions:
- `def evaluate_model` — Evaluate a fitted model on the held-out split and build the report.
- `def main` — Command-line entry point for diagnoser training.
- `def error_code_lookup_baseline` — Baseline with no model at all: look up the most likely cause per error code.
- `def run_attribution_checks` — Quantify how much of the score comes from the error code vs
- `def print_ablation_table` — Print the attribution comparison.

### `main.py` (520 lines)
Ghost Ledger v2 — one-command pipeline.
Definitions:
- `def step_strategies` — Track B startup check: validate the playbooks and print one line per cause
- `def step_generate` — Ensure a dataset exists, regenerating from the seed when asked.
- `def step_diagnose` — Load or train the diagnoser and score every failed transaction.
- `def step_persist_failures` — Write one ``failures`` row per failed transaction.
- `def step_recover` — Run the policy-gated recovery agents over the batch.
- `def reset_action_state` — Clear the results of any previous pipeline run.
- `def step_autopsy` — Generate autopsy explanations.
- `def step_metrics` — Compute the headline figures straight from the database.
- `def main` — Run the whole pipeline.

### `metrics.py` (337 lines)
Single source of truth for every number the dashboard shows.
Definitions:
- `def compute_headline_metrics` — Compute the headline figures for the held-out batch.
- `def recovery_over_time` — Daily recovered vs at-risk amounts on the held-out batch.
- `def cause_breakdown` — Recovery performance per diagnosed cause bucket.
- `def stopping_rule_events` — Return the explicit stopping-rule events, newest first.
- `def count_stopping_rule_events`
- `def recent_audit` — Return recent audit-trail rows for the filterable table.
- `def diagnoser_metrics` — Load the held-out diagnoser metrics produced by ``diagnoser/train.py``.
- `def autopsy_stats` — Count autopsy reports and hallucination flags.

### `playbooks/card_expired.yaml` (23 lines)
(no module docstring)

### `playbooks/gateway_timeout.yaml` (22 lines)
(no module docstring)

### `playbooks/insufficient_funds.yaml` (23 lines)
(no module docstring)

### `playbooks/mandate_lapsed.yaml` (18 lines)
(no module docstring)

### `scripts/compare_autopsy_backends.py` (144 lines)
Compare autopsy backends on the same real failures.
Definitions:
- `def pick_failures` — Select real diagnosed failures, optionally filtered to one cause.
- `def main`

### `scripts/e2e_test_mode.py` (325 lines)
Ghost Ledger v3 — end-to-end check (A7). Plain Python, no test framework.
Definitions:
- `class Check`
- `class Report`
- `def preflight` — Return a list of problems
- `def ensure_merchant` — Create the e2e merchant if needed, rotate its key, and return the key in memory only.
- `def new_case`
- `def open_link` — Ask the executor for a link, retrying transient call failures
- `def stages`
- `def wait_until`
- `def post_signed_webhook` — Send a webhook exactly as Razorpay would, signed with the configured secret.
- `def run`
- `def main`

### `scripts/retry_timing_holdout.py` (205 lines)
Ghost Ledger v3 — holdout comparison for rule-based retry timing (B3).
Definitions:
- `def simulated_success_probability` — The outcome model
- `def paired_uniform` — Deterministic uniform in [0, 1) per transaction, shared by both policies.
- `def retry_time`
- `def mcnemar` — McNemar chi-square (no continuity correction) and its two-sided p-value (1 df).
- `def run`
- `def main`

### `strategies/__init__.py` (18 lines)
Ghost Ledger v3 — multi-strategy recovery engine (Track B).

### `strategies/ab_test.py` (196 lines)
Ghost Ledger v3 — A/B tests for strategies (B6).
Definitions:
- `def bucket_of` — Stable bucket in 0..9999 for a (experiment, unit) pair.
- `def variant_for_bucket` — Control gets the first ``split_percent`` of buckets, treatment the rest.
- `class ZTestResult` — Output of :func:`two_proportion_z`.
- `def two_proportion_z` — Two-sided two-proportion z-test
- `class ExperimentReport` — Result of evaluating one experiment
- `class ABTestManager` — Assigns units to variants and evaluates outcomes.

### `strategies/dunning.py` (242 lines)
Ghost Ledger v3 — dunning sequencer (B5).
Definitions:
- `def send_via_channel` — MOCK sender
- `def render_message` — Fill the three allowed template fields
- `def next_step_for` — Return the fixed customer-facing sentence for an action, plus any method suggestion.
- `def schedule_touches` — Insert the touches that do not exist yet
- `def cancel_for_recovery` — Cancel every scheduled touch for a recovery (for example, on settlement).
- `def run_due` — Send every touch that is due
- `def list_touches` — Return every touch for a recovery, in sequence order.

### `strategies/executor.py` (85 lines)
Ghost Ledger v3 — playbook executor (B1, side effects).
Definitions:
- `class PlaybookExecutor` — Records a plan as events and schedules its dunning touches.

### `strategies/gateway.py` (180 lines)
Ghost Ledger v3 — gateway health and failover (B4).
Definitions:
- `class HealthSnapshot` — The health of one payment method at one moment.
- `class GatewayHealthMonitor` — Records payment outcomes and reports per-method health over a rolling window.
- `class FailoverDecision` — Whether to move a payment to another route, and why.
- `def decide_failover` — Pure failover rule over health snapshots
- `class GatewayFailoverEngine` — Reads live health from the monitor, then applies :func:`decide_failover`.

### `strategies/methods.py` (42 lines)
Ghost Ledger v3 — alternate payment-method suggestion (B7).
Definitions:
- `def suggest_methods` — Return the methods to suggest, in playbook order

### `strategies/playbooks.py` (334 lines)
Ghost Ledger v3 — YAML playbooks (B1).
Definitions:
- `class PlaybookError` — Raised when a playbook file or the playbook set is invalid.
- `class Touch` — One dunning reminder: when (minutes after the failure), how, and which template.
- `class Playbook` — A validated playbook for one root cause.
- `def load_playbook_set` — Load and validate every playbook in ``directory``.
- `class PlaybookLoader` — Holds the validated playbook set and reloads it when files change.
- `def default_loader` — Return the process-wide loader (created on first use).

### `strategies/router.py` (172 lines)
Ghost Ledger v3 — deterministic strategy router (B2).
Definitions:
- `class RouteContext` — Everything the router may look at
- `class StrategyPlan` — The routed decision for one attempt, with the reason for each step.
- `def route` — Choose the action for one attempt

### `strategies/runner.py` (232 lines)
Ghost Ledger v3 — strategy runner (B8 glue).
Definitions:
- `def build_context` — Collect the facts the router may use
- `def handle_failure` — Run one recovery attempt through the policy gate, then the strategy layer.
- `def retry_one` — Decide whether one open recovery needs an attempt now
- `def retry_pass` — Run :func:`retry_one` over every pending recovery
- `def observe_payment` — Webhook hook, called after a payment event is recorded
- `def dry_run_summary` — Return one readable line per cause, from the validated playbooks

### `strategies/timing.py` (96 lines)
Ghost Ledger v3 — rule-based retry timing (B3).
Definitions:
- `def is_peak_hour` — True when ``hour`` falls in a generator peak window.
- `def cycle_aware_funds` — Defer funds retries inside the end-of-cycle window
- `def peak_avoidance` — Step a gateway retry forward out of peak hours
- `def no_timing` — Retry immediately.
- `def apply_rule` — Run a named rule

### `tests/__init__.py` (0 lines)
(no module docstring)

### `tests/conftest.py` (52 lines)
Pytest configuration.

### `tests/test_autopsy.py` (260 lines)
Tests for the autopsy reporter (FR-005).

### `tests/test_data_generator.py` (260 lines)
Tests for the synthetic data generator (FR-001).

### `tests/test_diagnoser.py` (193 lines)
Tests for the root-cause diagnoser (FR-002).

### `tests/test_end_to_end.py` (401 lines)
Integration and edge-case tests.

### `tests/test_metrics.py` (192 lines)
Tests for the metrics module — the single source of truth for the dashboard.

### `tests/test_pipeline.py` (174 lines)
Pipeline-level smoke tests.

### `tests/test_policy_engine.py` (231 lines)
Tests for the recovery policy engine (FR-003).

### `tests/test_razorpay_client.py` (244 lines)
Tests for the Razorpay client (FR-008).

### `tests/test_v3_approvals.py` (303 lines)
Tests for A3 — the approval queue and the policy-gated recovery executor.

### `tests/test_v3_bootstrap.py` (60 lines)
Regression test: the API must bootstrap a FRESH database completely.

### `tests/test_v3_dashboard.py` (85 lines)
Smoke tests for the A5 live dashboard using Streamlit's AppTest.

### `tests/test_v3_e2e_script.py` (143 lines)
Tests for the A7 end-to-end script.

### `tests/test_v3_jobs.py` (313 lines)
Tests for A4 — scheduled jobs: settlement polling, reconciliation, retries, expired-link cleanup, backoff, and the scheduler wiring.

### `tests/test_v3_merchant_api.py` (239 lines)
Tests for A6 — the merchant-scoped API, the sanitised timeline, and the web page.

### `tests/test_v3_merchants.py` (108 lines)
Tests for the merchant registry: the 10-merchant cap, hashed keys, and constant-time key verification (database/merchants.py).

### `tests/test_v3_migration3.py` (52 lines)
Migration 3 must upgrade a v2-era database that has recovery_cases WITHOUT customer_id, keep its rows, and be safe to run twice.

### `tests/test_v3_operator.py` (46 lines)
Operator summary (A5 data source): shape, failure filtering, and auth.

### `tests/test_v3_recovery_store.py` (147 lines)
Tests for A2 — event-sourced recovery tracking (database/recovery_store.py), and for the append-only guarantee on recovery_events.

### `tests/test_v3_strategies.py` (468 lines)
Unit tests for Track B (B1-B7): playbooks, router, timing, gateway health, dunning, A/B tests, method suggestion, and migration re-runs.

### `tests/test_v3_strategies_e2e.py` (374 lines)
B9 — end-to-end tests for the strategy engine with ENABLE_ADVANCED_STRATEGIES on.

### `tests/test_v3_webhooks.py` (283 lines)
Tests for A1 — the Razorpay webhook listener (api/webhooks.py, api/main.py).

### `web/merchant/app.css` (97 lines)
(no module docstring)

### `web/merchant/app.js` (270 lines)
(no module docstring)

### `web/merchant/index.html` (91 lines)
(no module docstring)

### `weekly_report.py` (199 lines)
FR-009 — Weekly Recovery Report (P1).
Definitions:
- `def build_report` — Assemble the weekly report as markdown.
- `def main` — Write the weekly report to ``reports/weekly_recovery_report.md``.

