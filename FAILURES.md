# FAILURES.md — Ghost Ledger v2

> Every real blocker, written **the day it happened**, not reconstructed later.
> Format: dated entry → what broke → why → what was done about it → status.

---

## 2026-09-03 — Blocker: no Razorpay test-mode credentials in the build environment

**What:** The build environment has no `rzp_test_*` key pair, and the project
spec (FR-004/FR-008) requires recovery actions to execute against Razorpay
test-mode endpoints (Payment Links, Subscription mandate retry, refund lookup).

**Why it matters:** Without a decision here the demo has no execution path, and
"measured money recovered" — the track's core success bar — is unmeasurable.

**Resolution:** Build a **simulated Razorpay client** behind the exact same
interface the real SDK client will use (`api/razorpay_client.py`). The
simulator:

* returns Razorpay-shaped response objects (field names, id prefixes
  `plink_`/`sub_`/`rfnd_`, paise-denominated amounts, ISO timestamps),
* is **seeded** from the same `DATA_SEED`, so outcomes are reproducible,
* injects realistic failure modes — timeouts, rate limiting (429), and
  auth errors — so the graceful-degrade path (NFR-002) is exercised,
* settles outcomes from the **root cause**, so recovery rates differ per cause
  bucket the way they do in production (an expired card stays broken until the
  customer updates it; a gateway timeout usually succeeds on retry).

Swapping to live test mode is a single env change: set `RAZORPAY_KEY_ID`,
`RAZORPAY_KEY_SECRET`, and `RAZORPAY_LIVE_TEST_MODE=1`. No calling code changes.

**Status:** RESOLVED (design) — simulator pending implementation in Phase 3.
**Honesty note for the demo:** any "₹ recovered" figure produced while running
on the simulator must be labelled as simulated settlement. This labelling is
enforced in the dashboard, not left to the presenter.

---

## 2026-09-03 — Data defect: generator v1 produced an unusable class balance

**What:** First run of `data/synthetic_generator.py` (seed 42, 400 customers,
180 txns/day) produced:

| Cause | Count | Share |
|---|---|---|
| insufficient_funds | 327 | 66.3% |
| gateway_timeout | 120 | 24.3% |
| card_expired | 31 | 6.3% |
| mandate_lapsed | **15** | **3.0%** |

with a **53.6% failure rate on subscription debits**.

**Why it matters:** Three separate problems.
1. `mandate_lapsed` at n=15 total (≈4 in the holdout split) makes per-class
   precision/recall meaningless — and it is one of the four buckets the spec
   requires the diagnoser to distinguish.
2. The Subscription Recovery Agent (FR-004) had almost no work to do, gutting
   half the demo.
3. A 53.6% mandate failure rate is not credible; a judge would rightly
   challenge it, and the burden of proof is on us.

**Root cause:** Two independent mis-specifications.
* Mandate ages were drawn from `uniform(0, 520)` days while 60% of mandates
  had a 365-day validity, so only ~30% could ever lapse — and subscription
  debits are already rare (one per customer per month, ~2.3% of all txns).
* The `insufficient_funds` weight (`0.25 + (1-reliability)*2.2`, plus amount
  and end-of-month bonuses) dominated the softmax by roughly 5x over every
  other cause.

**Fix (iterated empirically, not guessed):**
* Customer base 400 → 700; subscription penetration 35% → 52%; daily volume
  180 → 150 (raises subscription debits from 2.3% → 7.3% of all txns).
* Mandate age `uniform(0, 800)` with validity mix reweighted to
  (365d 40% / 730d 35% / 1095d 25%) → ~25% lapse exposure.
* Lapsed-mandate failure probability 0.88 → 0.72 (banks honour some lapsed
  mandates inside a grace period).
* Base subscription failure rate 0.160 → 0.110.
* `insufficient_funds` weight reduced ~2x; `card_expired` near-expiry weights
  raised and the near-expiry cohort widened 22% → 38% of card holders.

**Result after fix (seed 42):**

| Cause | Train | Holdout | Share |
|---|---|---|---|
| insufficient_funds | 141 | 39 | 37.9% |
| gateway_timeout | 106 | 38 | 30.3% |
| mandate_lapsed | 67 | 10 | 16.2% |
| card_expired | 53 | 21 | 15.6% |

Overall failure rate 9.27%; subscription debits 32.97%. Holdout n=108.

**Residual risk (accepted, must be stated in the pitch):** `mandate_lapsed`
has only n=10 in the holdout split. Per-class metrics for that bucket carry a
wide confidence interval and will be reported **with their n**, per NFR-004.
A 33% subscription failure rate is high but defensible for a merchant with an
aging mandate book; it is recorded as a stated modelling assumption in
`generation_manifest.json` rather than buried.

**Status:** RESOLVED — verified reproducible, all 14 leak checks pass.

---

## 2026-09-03 — Environment constraint: `shap` not installable on Python 3.13

**What:** The spec's technology table cites XGBoost as "explainable via SHAP".
The build environment is Python 3.13.14 and the `shap` package has no reliable
wheel for it.

**Why it matters:** NFR-003 requires every AI output to carry a human-readable
reason. Without per-prediction attributions the diagnoser is a black box, which
undercuts the whole "explainable" claim.

**Resolution:** Use **XGBoost's built-in tree-SHAP** via
`booster.predict(dmatrix, pred_contribs=True)`, which returns exact SHAP
values for tree models directly from the booster. This is the same algorithm
the `shap` package delegates to for `TreeExplainer`, so the output is
equivalent — without the extra dependency. Top-3 contributing features per
prediction are surfaced as the human-readable reason.

**Status:** RESOLVED (approach) — implementation in Phase 2.

---

## 2026-09-03 — Documentation conflict: v1 and v2 specs disagree

**What:** `project_requirement_razorpay.md` is **v1.0** (open track): 6 ghost
types, Prophet, pgmpy, Isolation Forest, LangGraph, a `ghosts`/`resurrections`
schema, and a ">80% accuracy" target. `GHOST_LEDGER_PROJECT_SPEC.txt` and
`ghost-ledger-prebuild-doc.md` are **v2** (narrowed): 2 loss types, no
Prophet/pgmpy/IsolationForest/LangGraph, a `transactions`/`failures`/
`recoveries`/`audit_trail` schema, and "report honestly, never bare accuracy".

**Why it matters:** Building against the wrong document silently doubles the
scope and violates the build prompt's P2 ban.

**Resolution:** Per `GHOST_LEDGER_AGENTIC_BUILD_PROMPT` ("If anything here
conflicts with that file, the spec file wins"), **v2 governs**. v1's extra
ghost types (cascade, decay, suppression, pre-fraud) are explicitly P2 and are
**not** being built. Confirmed with the project owner.

**Status:** RESOLVED — v2 confirmed as the build target.

---

## 2026-09-03 — Scaling: the 5k-transaction corpus was too small to train on

**What:** The Phase 1 corpus (30 days, 700 customers, 150 txns/day) yielded
**5,126 transactions but only 367 training failures** — and just **10
`mandate_lapsed` examples in the holdout split**. Too small to train a
4-class classifier or to report any per-class metric with a straight face.

**Why it matters:** `mandate_lapsed` at n=10 gives a recall estimate with a
confidence interval so wide it is decoration, not measurement.

**Constraint discovered while measuring (this was the real blocker):** naive
scaling does not work, because **storage blows up before compute does.**

| Corpus | Txns | Failures | Gen time | Peak RAM | Naive JSON |
|---|---|---|---|---|---|
| 30d / 700c / 150 | 5,126 | 475 | 0.7 s | 46 MB | 4.9 MB |
| 90d / 1200c / 300 | 30,577 | 2,702 | 3.9 s | 85 MB | 29 MB |
| 180d / 2000c / 500 | 102,262 | 9,090 | 13.4 s | 196 MB | **98 MB** |
| 365d / 2500c / 600 | 251,251 | 23,026 | 33.4 s | 428 MB | **240 MB** |

The per-transaction `context` block is ~950 bytes, so a pretty-printed JSON
dump runs ~1 KB/row. At the 180-day setting that is 98 MB — which alone would
exceed the workspace snapshot cap, before SQLite is counted. The sandbox also
has only **2 GB RAM** (1.5 GB available), so the 365-day setting at 428 MB
peak is the practical ceiling.

**Fix — two changes:**

1. **Stop persisting what is never read.** The diagnoser only ever consumes
   **failed** transactions. Successful rows exist to generate customer
   behavioural history, and that history is *already baked into* each failure
   row's features at generation time. So feature context is persisted for
   failures only — removing ~91% of the payload.
   * Full ledger (all 102,262 rows, 10 spec columns) -> SQLite, 14.8 MB.
   * Failure contexts (9,090 rows) -> `failure_contexts.json`, 4.5 MB.
   * **Total 45 MB instead of 113 MB.**

2. **Size by preset, not by editing constants.** `config.DATASET_PRESETS`
   defines `demo` / `train` / `max`; `--profile` selects one and individual
   `--days` / `--customers` / `--daily` / `--holdout-days` flags layer on top.

**Result (profile `train`, seed 42):**

```
102,262 transactions | 2,000 customers | 1,030 subscriptions | 180 days
9,090 failures (8.89%) | 8,294 train | 796 holdout
Holdout per class: card_expired 90 | insufficient_funds 364
                   gateway_timeout 238 | mandate_lapsed 104
Generation 14.8 s | 196 MB peak | 24 MB JSON + 14.8 MB SQLite
```

`mandate_lapsed` holdout went **n=10 -> n=104**. Reproducibility re-verified
at scale: two seed-42 runs produced byte-identical `failure_contexts.json`.

**Status:** RESOLVED.

---

## 2026-09-03 — Bug: stale split description would have mislabelled every metric

**What:** After adding presets, running `--profile demo` printed the correct
*counts* (30 days, 475 failures) but the **wrong split basis**:

```
profile      : train                                    <- should be "demo"
split method : temporal holdout: the most recent 14 of 180 days   <- 30-day run
```

**Why it matters:** `SPLIT_METHOD` was a module-level string built from config
defaults at import time, so it described the *default* preset rather than the
run actually performed. Every precision/recall figure is required by NFR-004
to carry its evaluation set and split method. A correct number under a wrong
label is still a misreported number — and this would have shipped straight
into the dashboard and the pitch.

**Root cause:** Module-level constant computed once at import, then read by a
per-instance code path. Classic default-vs-instance confusion, introduced by
the preset change.

**Fix:** `SyntheticDataGenerator.__init__` now derives `self.split_method`
from its own resolved `self.days` / `self.holdout_days`, and accepts a
`profile` argument threaded through from the CLI. The manifest records
`self.profile` and the resolved `dataset_config` alongside it. The now-unused
module-level `SPLIT_METHOD` import was removed so this cannot silently regress.

**Verified:** `--profile demo` -> "most recent 7 of 30 days";
`--profile demo --days 60 --holdout-days 10` -> "most recent 10 of 60 days".

**Status:** RESOLVED — caught by inspection before any metric was reported.

---

## 2026-09-03 — Finding: 97.6% accuracy is real, but the error code does most of it

**What:** With the random grouped split the diagnoser reaches **accuracy
0.9758, macro-F1 0.9709** on n=1,815 held-out failures. That is high enough to
look wrong, so it was not accepted until it was decomposed.

**Why it matters:** A headline accuracy that is really just a lookup table
would misrepresent the system to a judge, and the track rewards *judgment* in
how AI is applied. High numbers that no one can explain are a liability.

**Investigation — three-way attribution, identical model and features:**

| Condition | Accuracy | Macro-F1 |
|---|---|---|
| `error_code` -> majority cause **lookup, no ML** | 0.9289 | 0.9291 |
| Full model (40 features) | 0.9758 | **0.9709** |
| Model with all `error_code` features removed (28 features) | 0.8738 | 0.8264 |

**Conclusion:** the error code carries most of the signal — as it does in
production, where Razorpay returns a decline code. The model nonetheless adds
a real **+0.042 macro-F1 over the lookup floor**, and still reaches 0.8264 from
transaction context alone when the error code is withheld entirely. Both
numbers are reported side by side rather than quoting 97.6% alone.

**Decision:** the generator's error-code distributions were **not** altered to
make the score look more "believable". Weakening a signal purely to depress a
metric is the same dishonesty as strengthening one to inflate it. The honest
move is to publish the decomposition, which is now emitted by
`diagnoser/train.py` on every run and stored in `reports/holdout_metrics.json`
under `attribution`.

**Status:** RESOLVED — reported with decomposition, not hidden.

---

## 2026-09-03 — Hypothesis tested and DISPROVEN: row-level random does not inflate here

**What:** When the split strategy was changed to random, I predicted that a
**row-level** random split would inflate scores versus a **customer-grouped**
one, because customer-level features (`mandate_days_overdue`,
`days_to_card_expiry`, `customer_prior_failure_count`) are near-constant per
customer and would let the model memorise customers.

I ran the experiment anyway. **The prediction was wrong.**

| Strategy | n | Accuracy | Macro-F1 | Customers in both splits |
|---|---|---|---|---|
| `random_grouped` | 1,820 | 0.9824 | **0.9769** | **0** |
| `random_rows` | 1,818 | 0.9758 | 0.9698 | 1,050 |
| `temporal` | 801 | 0.9813 | 0.9794 | 619 |

Row-level random came out **lower**, not higher. Identical corpus, features and
hyperparameters in every arm; only the split differs, so the delta is
attributable to the split alone.

**Why the reasoning failed:** the hypothesis assumed meaningful
customer-specific signal to memorise. There is very little. The model is
dominated by `error_code` and `gateway_latency_ms`, which are **per-transaction**
rather than per-customer, so seeing the same customer twice buys the model
almost nothing. The ablation supports this: strip the error code and
performance falls to 0.8264, meaning the customer-level cues were never doing
much work.

**Consequences:**
* The owner's requested random split is **safe** — the grouped variant is kept
  as the default because it is the more defensible choice, not because the
  alternative measured worse.
* All three strategies land within ~0.01 macro-F1 of each other, i.e. within
  noise. The choice of split is not a material lever on the reported number
  for this dataset.

**Lesson recorded:** the leakage guard was built on an untested assumption.
It is still the right default — guarding costs nothing and protects against a
corpus where customer signal *is* strong — but it should have been presented
as a hypothesis to test rather than a fact. The experiment now lives in
`diagnoser/split_experiment.py` so the claim can be re-checked whenever the
feature set changes.

**Status:** RESOLVED — hypothesis disproven, guard retained, claim now measured.

---

## 2026-09-03 — Bug: per-cause breakdown inflated by attempt multiplication

**What:** The weekly report's per-cause table summed to **₹5,579,616** at
risk while the headline said **₹2,646,060.94** — a 2.1× discrepancy, which is
exactly the average number of recovery attempts per failure (3,816 / 1,815).

**Why it matters:** Two different numbers for the same quantity, on screen at
the same time, is the fastest way to lose a judge's trust. The build prompt
requires the headline to reconcile with the audit trail; a breakdown that
contradicts the headline is the same defect one level down.

**Root cause:** `cause_breakdown()` and `recovery_over_time()` joined
`failures` directly to `recoveries`. A failure with 3 attempts produces 3
joined rows, so `SUM(f.estimated_value)` counted its value 3 times. The
headline metric was correct only because it never joined the two tables.

**Fix:** Aggregate recoveries per failure in a CTE *before* joining:

```sql
WITH rec AS (
    SELECT failure_id,
           SUM(recovered_amount) AS recovered,
           COUNT(*) AS attempts,
           SUM(CASE WHEN stopping_rule_triggered=1 THEN 1 ELSE 0 END) AS stops
    FROM recoveries GROUP BY failure_id
)
SELECT ... FROM failures f
JOIN transactions t ON t.id = f.transaction_id
LEFT JOIN rec ON rec.failure_id = f.id
```

**Verified after fix:** per-cause sums equal the headline exactly —
at risk ₹2,646,060.94, recovered ₹1,907,719.68, failures 1,815.

**Status:** RESOLVED.

---

## 2026-09-03 — Test isolation: fixtures were polluting the demo database

**What:** End-to-end tests insert real `transactions` / `failures` rows so
that `recoveries` can satisfy its foreign keys. Those rows were written into
the *demo* database, so the dashboard's "N transactions" drifted from 20,479
to 20,485, and the split description printed a stale count beside it.

**Why it matters:** Test data silently contaminating the figures shown to a
judge. The drift is small enough to miss by eye and large enough to be wrong.

**Fix:** `tests/conftest.py` now points `DB_PATH` at
`data/test_ghost_ledger.db` and seeds it from the real database with
`VACUUM INTO` at session start. `VACUUM INTO` is used rather than a file copy
because the demo database runs in WAL mode, where recent commits may live only
in the `-wal` sidecar file.

**Discovered along the way:** the foreign keys that caused the original test
failures were working correctly — the tests were wrong, not the schema. The
constraint was kept and the tests were fixed to create proper parents.

**Status:** RESOLVED — 49 tests pass, demo database regenerated clean.

---

## 2026-09-03 — Gap: `main.py` exited on a clean checkout (violated constraint 7)

**What:** `main.py` raised `SystemExit` when no trained model was found,
telling the user to run `diagnoser/train.py` first. That is two commands on a
clean checkout, which the build prompt's hard constraint 7 forbids.

**Fix:** `main.py` now trains the diagnoser automatically when the model is
absent, using a reduced 15-config CV search (~1 min) to stay inside the
5-minute demo budget. `python diagnoser/train.py` remains the full 40-config
path for the reported numbers.

**Status:** RESOLVED.

---

## 2026-09-03 — Schema deviation: added one table beyond spec §11

**What:** `GHOST_LEDGER_PROJECT_SPEC` §11 defines exactly four tables, none of
which has a column suited to storing the FR-005 autopsy text.

**Resolution:** Added `autopsy_reports(failure_id, report_text, model, basis,
generated_at)`. **Additive only** — no spec table or column was modified,
renamed, or removed. Subscription mandates are likewise *not* given a new
table; they are persisted as `audit_trail` entries to keep the schema faithful.

**Status:** OPEN — flagged for owner approval at the Phase 1 checkpoint.

---

## 2026-09-03 — Bug: pipeline not idempotent (second run stacked a second set of actions)

**What:** Running `python main.py` twice in a row produced a different, worse
answer than the first run. A second pass over the same holdout failures saw the
prior run's attempts, counted them as `prior_total_attempts`, and immediately
hit the R2 attempt cap and the R3 stopping rule — so the second run's figures
described the *union of two runs*, not one.

**Why it mattered:** The dashboard headline is the number the submission is
judged on. A pipeline whose output depends on how many times you happened to
run it is not reproducible, and the "one command starts the system" DoD item is
worthless if that command only works once.

**Detection:** `tests/test_pipeline.py::test_pipeline_is_idempotent` — runs the
pipeline twice against a throwaway database and asserts `(action_count,
recovered_total)` are identical both times.

**Resolution:** Added `reset_action_state()` to `main.py`, called at the start of
every run. It clears `recoveries`, `autopsy_reports`, `failures`, and pipeline
audit rows. The transaction ledger is deliberately left alone (it is regenerated
only with `--regen`) and generator audit records are preserved.

**Verified:** Three consecutive runs returned identical figures —
`at risk 2,646,060.94 / recovered 1,907,719.68 / 3,816 actions / 440 stops`.

**Status:** RESOLVED.

---

## 2026-09-03 — Bug: `reset_action_state()` failed with a foreign-key violation

**What:** The first version of the reset function deleted `failures` before
`autopsy_reports`. Both `recoveries` and `autopsy_reports` carry foreign keys to
`failures.id`, and foreign key enforcement is on, so the reset raised
`sqlite3.IntegrityError: FOREIGN KEY constraint failed` and aborted the pipeline
at step 3.

**Why it mattered:** A bug found while fixing the previous bug. It would have
stopped a clean run dead.

**Resolution:** Reordered to delete children before parents — `recoveries`,
`autopsy_reports`, then `failures`, then the audit rows (which have no FK).

**Status:** RESOLVED. Caught by running the pipeline immediately after the fix,
before writing the test.

---

## 2026-09-03 — Bug: an empty LLM completion produced a blank autopsy instead of falling back

**What:** `generate_autopsy()` only fell back to the template when the backend
*raised* an exception. A model that returned an empty string (a real failure
mode for local Ollama with a cold or overloaded model) produced
`AutopsyReport(text="", degraded=False)` — a silently empty report recorded as
healthy.

**Why it mattered:** An empty autopsy shown in the dashboard looks like a bug in
the product, and `degraded=False` would have hidden the cause.

**Detection:** `tests/test_autopsy.py::test_empty_llm_completion_falls_back`
monkeypatches the backend to return `("", ...)`.

**Resolution:** Treat a blank completion as a failure. `generate_autopsy()` now
raises `RuntimeError("empty completion from backend")` when the returned text is
empty or whitespace-only, which routes it into the existing fallback path.

**Status:** RESOLVED.

---

## 2026-09-03 — Bug: `main.py` overwrote the diagnoser report without the attribution study

**What:** Attribution (the error-code lookup floor, the full model, and the
error-code-ablated model) was computed only in `diagnoser/train.py`'s `main()`.
When `main.py` later ran its own evaluation, it rewrote
`reports/holdout_metrics.json` with a report that had no `attribution` key,
silently dropping the decomposition from the dashboard.

**Why it mattered:** The attribution is what makes the ML claim honest — it
shows how much of the score is just the error-code lookup that anyone could
write. Losing it after every pipeline run left the dashboard showing an
unexplained precision/recall panel.

**Detection:** `tests/test_metrics.py::test_attribution_is_present_and_ordered`.

**Resolution:** Moved `run_attribution_checks()` inside `evaluate_model()`, which
both entry points share. The stored report is now identical whichever path
produced it.

**Verified:** After a full `main.py` run, `attribution` is present with all three
rows — lookup 0.9291, full 0.9709, ablated 0.8264, lift +0.0418.

**Status:** RESOLVED.

---

## 2026-09-03 — Test debt: 49 tests covered 3 of 8 modules

**What:** The suite tested the policy engine, the diagnoser and one end-to-end
path. The data generator, Razorpay client, autopsy reporter, metrics module and
pipeline orchestrator had no tests at all.

**Why it mattered:** The five bugs above all lived in the untested modules. Notably
the attempt-multiplication bug (per-cause sums 2.1x the headline) had been fixed
but never had a regression test, so nothing prevented it recurring.

**Resolution:** Grew the suite from 49 to **127 tests** across 7 files:
`test_policy_engine.py` (23), `test_diagnoser.py` (14), `test_end_to_end.py` (12),
`test_data_generator.py` (21), `test_razorpay_client.py` (20),
`test_autopsy.py` (17), `test_metrics.py` (15), `test_pipeline.py` (5).

Added explicit regression tests for the double-counting bug
(`test_per_cause_totals_match_the_headline`, `test_daily_totals_match_the_headline`)
that assert per-cause and per-day sums equal the headline exactly.

**Status:** RESOLVED — 127 passing.
