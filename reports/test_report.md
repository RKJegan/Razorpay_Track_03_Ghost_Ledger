# Test Report — Ghost Ledger v2

**Date:** 2026-09-03
**Command:** `python3 -m pytest tests/ -q`
**Result:** **127 passed** in 85s — 0 failed, 0 skipped, 0 warnings-as-errors.
**Environment:** Python 3.13.14, 2 GB RAM, pytest 9.0.3.

---

## 1. Coverage by module

The suite went from 49 tests covering 3 of 8 modules to **127 tests covering
every module in the project**.

| Test file | Tests | Module under test | What it proves |
|---|---:|---|---|
| `test_policy_engine.py` | 23 | `agents/policy_engine.py` | FR-003 rules, ₹10k ceiling, 3-attempt cap, stopping rule precedence |
| `test_diagnoser.py` | 14 | `diagnoser/*` | Training, held-out evaluation, grouped CV, no leakage |
| `test_end_to_end.py` | 12 | agents + audit trail | Full loop with stub clients (always succeeds / always fails / always times out) |
| `test_data_generator.py` | 21 | `data/synthetic_generator.py` | Reproducibility, schema, split integrity, **zero customer overlap** |
| `test_razorpay_client.py` | 20 | `api/razorpay_client.py` | Simulator determinism, settlement rates, Razorpay-shaped payloads, degradation |
| `test_autopsy.py` | 17 | `agents/autopsy_reporter.py` | Never raises, falls back on outage, flags invented numbers |
| `test_metrics.py` | 15 | `metrics.py` | Headline reconciliation, **regression guards for double counting** |
| `test_pipeline.py` | 5 | `main.py` | One-command completion, idempotency, attempt cap |

**Total: 127**

---

## 2. Bugs the sweep found and fixed

Writing the tests surfaced **five real defects**, four of them in code that was
already believed to be working. All are recorded in `FAILURES.md` with dates.

### 2.1 Pipeline was not idempotent — `main.py`

Running the pipeline twice stacked a second set of recovery actions on top of
the first. The second pass counted the prior run's attempts and immediately hit
the attempt cap and stopping rule, so the reported figures described the union
of two runs.

*Fix:* added `reset_action_state()`, called at the start of every run. It clears
`recoveries`, `autopsy_reports`, `failures` and pipeline audit rows; the
transaction ledger is preserved.

*Verified:* three consecutive runs returned identical figures.

### 2.2 Foreign-key violation in that reset — `main.py`

The first version deleted `failures` before `autopsy_reports`. Both children
carry foreign keys to `failures.id`, so the reset raised `IntegrityError` and
aborted the run. Reordered to delete children before parents.

### 2.3 Empty LLM completion produced a blank autopsy — `agents/autopsy_reporter.py`

Fallback only triggered when the backend *raised*. A model returning an empty
string yielded `AutopsyReport(text="", degraded=False)` — an empty report
recorded as healthy. A blank completion is now treated as a failure and routed
into the fallback path.

### 2.4 Attribution silently dropped from the diagnoser report — `main.py`

Attribution was computed only in `train.py::main`, but `main.py` rewrote
`reports/holdout_metrics.json` with a report lacking it — so the dashboard lost
the score decomposition after every pipeline run. `run_attribution_checks()` was
moved into the shared `evaluate_model()`.

*Verified:* all three rows survive a full `main.py` run — lookup 0.9291, full
0.9709, ablated 0.8264, lift **+0.0418**.

### 2.5 Brittle assertion on prose — `data/synthetic_generator.py`

A test asserted the split strategy token appeared in a human-readable
description. Rather than pattern-match prose, the generator now appends a
machine-readable `[strategy=random_grouped]` marker to `split_method`.

---

## 3. Regression guards

Two bugs had been fixed earlier **without** a test to keep them fixed. Both now
have explicit guards:

- **Attempt multiplication.** Joining `failures` straight to `recoveries`
  multiplied each failure's value by its attempt count, inflating per-cause and
  per-day "at risk" ~2.1× (₹5.58M vs ₹2.65M).
  `test_per_cause_totals_match_the_headline` and
  `test_daily_totals_match_the_headline` assert the breakdowns sum to the
  headline **exactly** (tolerance ₹0.01).

- **Clean-checkout crash.** Checking only for `generation_manifest.json` missed
  the case where the JSON artefacts survive while the SQLite ledger is gone.
  `main.py` now also regenerates when `ledger_rows == 0`; a full clean-checkout
  run is verified below.

---

## 4. Verification beyond the unit tests

### 4.1 Clean checkout (the real acceptance test)

Deleted `data/ghost_ledger.db`, all models and all reports, then ran one command:

```
$ python main.py
...
PIPELINE COMPLETE
total elapsed : 78.0s
```

**78 seconds, zero manual steps.** Within the 5-minute demo budget.

### 4.2 Determinism

Three consecutive `python main.py` runs produced byte-identical headline
figures:

| Run | Recovered | At risk | Actions | Stops |
|---|---:|---:|---:|---:|
| 1 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |
| 2 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |
| 3 | 1,907,719.68 | 2,646,060.94 | 3,816 | 440 |

### 4.3 Dashboard reconciliation

Every dashboard figure was cross-checked against the audit trail:

```
recovered        INR     1,907,719.68
at risk          INR     2,646,060.94
rate                         72.10%
reconciled                     True
stops                            440
macro-F1                   0.9709  (n=1,815)
autopsies                      1,815

per-cause at-risk   2,646,060.94 vs 2,646,060.94 -> MATCH
per-cause recover   1,907,719.68 vs 1,907,719.68 -> MATCH
per-cause failures         1,815 vs 1,815        -> MATCH
per-day   at-risk   2,646,060.94 vs 2,646,060.94 -> MATCH
```

### 4.4 Full tuning + pipeline (canonical numbers)

- Tuning: 40 configs × 5 `StratifiedGroupKFold` on train only — **174.4s**, best
  CV macro-F1 **0.9726 ± 0.0019**.
- Pipeline: warm run **6.2s**.
- Diagnoser (held-out, n=1,815): accuracy **0.9758**, macro-F1 **0.9709**.

---

## 5. Timings

| Operation | Duration |
|---|---:|
| Full unit suite (127 tests) | 85s |
| Clean checkout, one command | 78s |
| Warm pipeline (data + model present) | 6–20s |
| Full 40-config hyperparameter tuning | 174s |

Tuning is excluded from the demo path: the trained model is loaded from
`models/root_cause_classifier.json`.

---

## 6. What the suite does *not* cover

Stated honestly, since a test report that claims total coverage is not credible:

1. **Streamlit UI rendering.** No browser-level test. The dashboard's data layer
   is tested through `metrics.py`, and the HTTP endpoints return 200, but widget
   rendering is verified by eye only.
2. **Live Razorpay.** No test-mode credentials exist in this environment
   (recorded in `FAILURES.md`). `LiveRazorpayClient` is tested for its failure
   behaviour — raising `RuntimeError` without keys, and degrading to the
   simulator — but never against the real API.
3. **Real LLM output.** Autopsy tests inject synthetic completions. The Ollama
   path is tested for *degradation*, not for the quality of real generated prose.
4. **Concurrency.** The pipeline is single-threaded; no test exercises
   simultaneous writers to the audit trail.

---

## 7. Reproducing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q        # 127 passed
python main.py                     # full pipeline, clean checkout, ~78s
python weekly_report.py            # FR-009 report
streamlit run dashboard/app.py     # dashboard on :8501
```
