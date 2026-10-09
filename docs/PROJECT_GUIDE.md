# Ghost Ledger — complete project guide

Ghost Ledger is a payment-recovery system. When a customer's payment fails, it
works out *why* (card expired, insufficient funds, gateway timeout, or lapsed
mandate), decides *what to do* under strict rules, acts through Razorpay (or a
simulator), and records every step in an audit trail. The language model, where
used, only writes explanations. It never decides money, approvals, retries,
gateways, timing, or stops.

This guide covers the whole project (v2 base plus the v3 upgrade). For each
source file, see [MODULE_GUIDE.md](MODULE_GUIDE.md). For a VS Code walkthrough,
see [VSCODE_TEST_GUIDE.md](VSCODE_TEST_GUIDE.md). For the strategy engine in
depth, see [STRATEGIES.md](STRATEGIES.md).

---

## 1. What is in the project

| Part | What it does | Where |
|---|---|---|
| **v2 pipeline** | Generates synthetic payment data, diagnoses each failure, runs policy-gated recovery, writes reports and the audit trail. | `main.py` |
| **Diagnoser** | XGBoost classifier that labels each failed payment with one of four causes. | `diagnoser/` |
| **Policy engine** | Four fixed rules (R1–R4). Every recovery action passes through them. | `agents/policy_engine.py` |
| **v2 dashboard** | Streamlit page with headline figures, the audit trail, and the stopping-rule panel. | `dashboard/app.py` |
| **Track A (real time)** | Webhooks with signature checks, an event log, human approvals above ₹10,000, background jobs with backoff, a live progress page, a merchant API and web page, and an end-to-end Test Mode script. | `api/`, `database/`, `agents/`, `dashboard/live_progress.py`, `web/merchant/`, `scripts/e2e_test_mode.py` |
| **Track B (strategies)** | Cause-specific playbooks, a deterministic router, retry timing, gateway health and failover, dunning, A/B tests, and method suggestions. Off by default. | `strategies/`, `playbooks/` |

---

## 2. The four causes and the policy rules

**Causes** (`config.CAUSE_BUCKETS`): `card_expired`, `insufficient_funds`,
`gateway_timeout`, `mandate_lapsed`.

**Policy rules** (`agents/policy_engine.py`; the rules are unchanged in v3):

| Rule | Meaning |
|---|---|
| **R1** | No automatic action above ₹10,000 (`POLICY_MAX_AUTO_APPROVE_INR`). Above that, a human must approve. An approval lifts R1 only. |
| **R2** | At most 3 attempts per failure (`POLICY_MAX_ATTEMPTS_PER_FAILURE`). |
| **R3** | On the 3rd failed attempt, stop permanently and escalate. |
| **R4** | Every ruling is written to the audit trail before anything is executed. |

Denial takes precedence: an approval cannot override R2 or R3.

**Trust boundary.** The LLM (optional, for autopsy text only) never sets an
amount, approval, retry, gateway, time, or stop. Razorpay responses are recorded
as facts. A link response that says "paid" from the simulator does not settle a
recovery. Only a verified webhook or a confirmed status read does.

---

## 3. Quick start (v2 base)

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --no-autopsy            # generates data and the model, runs the pipeline
streamlit run dashboard/app.py         # http://localhost:8501
```

First run takes about a minute or two (training). Later runs are faster. The
pipeline is deterministic (seed 42), so the figures are reproducible.

Headline figures on the sample dataset (`train` profile, 102,262 ledger rows):

| Figure | Value |
|---|---|
| Held-out transactions | 20,479 |
| Failures in the holdout | 1,815 |
| Diagnoser held-out macro-F1 | 0.9704 |
| Recovery rate | **72.10%** (published, committed `reports/headline_metrics.json`) |
| Stopping-rule stops | **440** (published) |
| Recovery actions | **3,816** (published) |
| Audit records | **7,633** (published; reconciles with the recovery actions) |

A fresh rebuild in the sandbox (Python 3.11, xgboost 3.2.0, not the pinned 3.4.1)
gave 72.23%, 438 stops, 3,814 actions, and 7,629 audit records. The difference is
not yet explained. Treat the committed `reports/` files as the reference, and
re-run on Python 3.12 with the pinned requirements to compare.

---

## 4. Running the v3 pieces

Open a separate terminal for each. All commands run from the project folder.

| Piece | Command | Address | Needs |
|---|---|---|---|
| v2 dashboard | `streamlit run dashboard/app.py` | http://localhost:8501 | the pipeline run once |
| API (webhooks, approvals, merchant API, merchant page) | `uvicorn api.main:app --host 0.0.0.0 --port 8000` | http://localhost:8000/health | nothing required |
| Live progress page (Track A5) | `streamlit run dashboard/live_progress.py --server.port 8502` | http://localhost:8502 | the API running, and `OPERATOR_API_KEY` set |
| Merchant dashboard (Track A6) | open http://localhost:8000/merchant | — | a merchant API key (below) |

**Live progress page.** Set `OPERATOR_API_KEY` in `.env` (any long random string)
before starting the API and the page. Without it, the operator routes refuse
every request by design.

**Merchant dashboard.** Create a merchant (prints its API key once):

```bash
python -m database.merchants create --id merchant_demo --name "Demo merchant"
python -m database.merchants list          # no keys shown
python -m database.merchants rotate-key --id merchant_demo   # issues a new key; the old one stops working
```

Paste the key into the merchant page. Each merchant sees only its own
recoveries. The database holds only a hash of each key. Up to 10 merchants are
allowed.

**Turning on the strategy engine (Track B).** Set `ENABLE_ADVANCED_STRATEGIES=1`
in `.env`, then run the pipeline. It prints the playbook summary first. Set it
back to `0` to return to v2 behaviour.

**Scheduler (Track A4).** Set `SCHEDULER_ENABLED=1` to run background jobs with
the API: settlement poll, reconcile, retry, expired-link cleanup, and (with
Track B on) dunning. Off by default so tests and batch runs never start threads.

---

### Real-time transaction and recovery demo

The end-to-end script runs one full recovery: a payment fails, the system
creates a second payment link, the second payment succeeds, the recovery
settles, and reconciliation matches the amounts. It also checks that the
merchant API shows only that merchant's own timeline.

Two modes:

* `--mode offline` (works in any sandbox, no Razorpay account). The script posts correctly signed webhooks to the local API. Run it with the API up and the same `RAZORPAY_WEBHOOK_SECRET`:

  ```bash
  python scripts/e2e_test_mode.py --mode offline
  ```

  Expect 15 `[PASS]` lines and `Result: PASS`.
* `--mode live` (real Razorpay Test Mode, on your machine). The script prints
  payment links. You pay the first one with a method that fails (for example
  `failure@razorpay`) and the second with a test card that succeeds. Setup steps
  are in section 7.

Where to watch it: the **live progress page** (port 8502) shows recoveries as
they move through their stages. The **merchant page** (`/merchant`) shows the
merchant's own timeline once a merchant key is entered.

## 5. HTTP endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness, plus the feature flags in effect. |
| `POST /webhooks/razorpay` | `X-Razorpay-Signature` (HMAC-SHA256 over the raw body) | Receives Razorpay events. Refused (503) if no webhook secret is set. Duplicate event ids are processed once. |
| `GET /api/approvals`, `GET /api/approvals/{id}` | `X-Operator-Key` | List and read approval requests. |
| `POST /api/approvals/{id}/approve`, `.../reject` | `X-Operator-Key` | Decide an approval. Approval runs through the policy gate. |
| `GET /api/operator/summary` | `X-Operator-Key` | Figures for the live progress page. |
| `GET /api/merchant/me`, `/summary`, `/recoveries`, `/recoveries/{id}` | `Authorization: Bearer <merchant key>` | Merchant-scoped data only. |
| `GET /merchant` (and `/merchant/app.js`, `/merchant/app.css`) | none (the page asks for the key) | The merchant web dashboard. |

Interactive API docs are at http://localhost:8000/docs when the API runs.

---

## 6. Configuration

Everything is set in `.env` (copy `.env.example`). Every value has a safe
default. Never commit `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_PROFILE` | `train` | `demo` (5k transactions), `train` (102k), `max` (251k). |
| `DATA_SEED` | `42` | Seed. The output is byte-reproducible. |
| `SPLIT_STRATEGY` | `random_grouped` | Holdout split: `random_grouped`, `random_rows`, or `temporal`. |
| `DB_PATH` | `data/ghost_ledger.db` | One SQLite file for v2 and v3. |
| `LLM_BACKEND` | `ollama` | Autopsy text: `ollama`, `openai`, or `template` (offline). |
| `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET` | empty | Test-mode keys (must start with `rzp_test_`). |
| `RAZORPAY_LIVE_TEST_MODE` | `0` | `1` = real Razorpay Test Mode API. `0` = simulator. |
| `RAZORPAY_WEBHOOK_SECRET` | empty | Secret from the Razorpay dashboard (not the key secret). Empty = webhooks refused. |
| `OPERATOR_API_KEY` | empty | Key for approval and operator routes. Empty = refused. |
| `POLICY_MAX_AUTO_APPROVE_INR` | `10000` | R1 ceiling. |
| `POLICY_MAX_ATTEMPTS_PER_FAILURE` | `3` | R2 cap. |
| `SCHEDULER_ENABLED` | `0` | Start background jobs with the API. |
| `JOB_*_SECONDS`, `JOB_BACKOFF_*` | see file | Job intervals and backoff. |
| `ENABLE_ADVANCED_STRATEGIES` | `0` | Track B switch. `0` = exactly v2 behaviour. |
| `JOB_DUNNING_SECONDS` | `900` | Dunning job interval (Track B). |
| `GATEWAY_HEALTH_*`, `GATEWAY_DEGRADED_BELOW` | see file | Gateway health window, sample size, and threshold. |
| `AB_ALPHA`, `AB_MIN_SAMPLE_PER_ARM`, `AB_RETRY_TIMING_EXPERIMENT` | see file | A/B test settings. The experiment is opt-in. |

---

## 7. Razorpay: simulator or Test Mode

`api/razorpay_client.py` is an adapter. The rest of the code calls one
interface, and `get_razorpay_client()` picks the implementation:

* **Simulator** (default). No network. Deterministic. Used by tests and the batch pipeline.
* **Live Test Mode** (`RAZORPAY_LIVE_TEST_MODE=1` plus `rzp_test_` keys). Real Test Mode API calls.

To test against Razorpay Test Mode:

1. Set the test keys and `RAZORPAY_LIVE_TEST_MODE=1` in `.env`.
2. Expose the API to the internet (Razorpay must reach the webhook). For example, `ngrok http 8000`.
3. In the Razorpay Test Dashboard, add a webhook pointing to `https://<your-tunnel>/webhooks/razorpay`, with the events `payment.captured` and `payment.failed`. Copy its secret into `RAZORPAY_WEBHOOK_SECRET`.
4. Run the end-to-end script: `python scripts/e2e_test_mode.py --mode live`. It prints payment links. You pay them in the browser using Razorpay test cards or UPI test handles.

Razorpay documentation on test payments is the reference for test cards and
UPI handles (`success@razorpay`, `failure@razorpay`). The sandbox used to build
this project could not reach Razorpay, so **live Test Mode has not been run
from that sandbox**. Run step 4 on your own machine.

---

## 8. Tests

```bash
pytest tests -q                 # 292 tests, about 70 seconds
pytest tests/test_v3_strategies_e2e.py -q    # just the strategy end-to-end tests
```

The suite copies `data/ghost_ledger.db` to `data/test_ghost_ledger.db` at the
start, so tests never change the demo figures. Tests use a fake Razorpay client
and fixed clock values, so they are deterministic and make no network calls.

What is covered: the stopping rule, denial precedence, the diagnoser's output
shape, webhook signatures and idempotency, approvals, job backoff, merchant
isolation, the dashboard, the strategy engine (flag on and off), and migrations
re-running safely.

---

## 9. Data and reports

| Path | Content | In git? |
|---|---|---|
| `data/ghost_ledger.db` | The single SQLite database (v2 and v3 tables). | No (regenerated) |
| `data/sample_output/` | Generated holdout, ground truth, and manifest files. | Yes |
| `models/` | Trained classifier (JSON). | No (regenerated) |
| `reports/headline_metrics.json`, `holdout_metrics.json` | Figures the dashboard shows. | Yes |
| `reports/retry_timing_holdout.json` | Track B3 holdout comparison (simulated outcomes). | Yes |
| `playbooks/*.yaml` | Track B playbooks. Edit and the engine reloads them. | Yes |

Migrations run at startup (`database/migrations.py`) and are safe to re-run.
Track B added three tables: `dunning_touches`, `gateway_observations`, and
`ab_assignments`.

---

## 10. Security notes

* **Secrets stay out of the repo.** `.env` is git-ignored. Only `.env.example` (blank values) is committed.
* **Webhooks** are checked with HMAC-SHA256 over the raw request body, using a constant-time comparison (`hmac.compare_digest`). A bad signature gets 401. A missing secret gets 503. Malformed JSON gets 400.
* **API keys** (merchant and operator) are stored as hashes. Merchant keys are shown once.
* **Merchant isolation.** Every merchant route filters by the key's merchant. A merchant cannot read another merchant's data.
* **The event log is append-only.** Database triggers block updates and deletes on `recovery_events`.
* **Dunning recipients** are customer references, not contact details. Ghost Ledger does not store phone numbers or emails.

---

## 11. Limitations (read before quoting numbers)

* **The holdout comparison for retry timing uses simulated outcomes.** The dataset has no retry results. See `docs/STRATEGIES.md`.
* **Timing rules copy the generator's assumptions** (the day-26 cash crunch and peak hours). They are not measured bank behaviour.
* **Gateway failover is record-only** (decision confirmed by the project owner). When a route is degraded and a healthier one exists, the audit trail records the switch that would happen. The customer's payment link is not changed. Enforcing would change what customers see, so it needs a separate decision.
* **Dunning sends nothing real.** `send_via_channel` is a mock. Connect a provider in that one function.
* **Settlement is simulated** in the batch pipeline. Only Track A's verified webhook path settles a real Test Mode payment.
* **Python 3.12** is the version `requirements.txt` targets (it pins xgboost 3.4.1). The build sandbox used Python 3.11 with xgboost 3.2.0. Run the same checks on 3.12 on your machine.
* **The live Test Mode run** has not been executed from the sandbox (no route to Razorpay).

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named 'xxx'` | Activate the virtual environment, then `pip install -r requirements.txt`. |
| `xgboost` install fails | Use Python 3.12 (the pin needs it). |
| The dashboard says "no data" | Run `python main.py --no-autopsy` once. |
| Live progress says it cannot reach the API | Start `uvicorn api.main:app --port 8000` first. Set `API_BASE_URL` if the API is elsewhere. |
| Live progress shows 503 or "not configured" | Set `OPERATOR_API_KEY` in `.env` and restart the API and the page. |
| Webhook returns 503 | `RAZORPAY_WEBHOOK_SECRET` is empty. Set it. |
| Webhook returns 401 | The secret in `.env` does not match the Razorpay dashboard. |
| Merchant page shows `invalid merchant key` (401) | The key is wrong, or it was replaced with `rotate-key`. Use the newest key, or rotate again (the old key stops working). |
| `merchant limit reached: at most 10 merchants` | At most 10 merchants. Use an existing one. |
| Strategy engine does nothing | `ENABLE_ADVANCED_STRATEGIES` is `0`. |
| Playbook warning in the log | A YAML file is invalid. The last good set keeps running. See `PlaybookLoader.last_error`. |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Recovery | One failed payment being worked on, from failure to settlement or escalation. |
| Attempt | One payment link created for a recovery. |
| Ruling | The policy engine's decision on one attempt. |
| Stopping rule | R3: after the third failed attempt, the recovery is escalated and no more automatic attempts are made. |
| Holdout | Failures set aside from training to test the diagnoser and the recovery figures. |
| Playbook | A YAML file that says what the router may do for one cause. |
| Dunning | A fixed sequence of customer reminders. |
| Failover | Moving a customer to another payment method when the current one is degraded. |
| A/B test | Splitting units into control and treatment to compare outcomes. |
| Holdout simulation | The Track B timing comparison, which uses a stated model of retry outcomes. |
