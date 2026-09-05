# Connecting Ollama & Testing the Whole Project

**Date:** 2026-09-04 · **Audience:** you, on your own machine
**Read §1 first** — it answers "what is the LLM actually for here".

---

## 1. What the LLM does and does NOT do

This is the most important thing to understand before you connect anything.

```
┌───────────────── TRUSTED PATH (money moves here) ─────────────────┐
│  ledger → XGBoost diagnoser → policy engine → Razorpay client     │
│  deterministic · audited · bounded · no LLM anywhere              │
└──────────────────────────────────────────────────────────────────┘

┌──────────── UNTRUSTED PATH (prose only) ────────────┐
│  Ollama → autopsy sentence → dashboard "why" column │
│  cannot change a rupee, a policy call, or an outcome│
└─────────────────────────────────────────────────────┘
```

**The LLM's entire job is FR-005:** turn a diagnosis into one or two sentences a
human can read. That is it.

This is a hard constraint from the spec, not a limitation of our design:

> *"No LLM for financial decision-making (only explanations)."*

And from the build prompt:

> *"……output 2–3 sentences, text only, no decision-making."*

**So the honest answer to "what's the use of the LLM" is: it explains. It does
not decide, act, or recover.** The ₹19 lakh is recovered by the deterministic
agent loop with the XGBoost cause classifier. The LLM watches and writes
sentences about it.

### 1.1 What I measured when I actually ran it

I installed Ollama in the sandbox and compared backends on real failures.
Results (model `qwen2.5:0.5b`, 2 CPUs, 1.3 GB RAM — a weak model because the
sandbox only has 2 GB):

| | Template | Ollama |
|---|---|---|
| Time per autopsy | **0.00s** | 3.1–4.9s |
| Rupees changed | 0 | 0 |
| Numbers correct | ✅ | ✅ (copied from facts) |
| **Semantically correct** | ✅ | ❌ **several errors** |

Actual errors produced by the model:

| Model output | Problem |
|---|---|
| *"an amount of 2141.02 in **RON**"* | Romanian Leu — wrong currency |
| *"an amount of 999.0 in **RIN**"* | currency typo |
| *"the **merchant** had insufficient funds"* | wrong entity — it's the customer |
| *"the **merchant** experienced 1 recovery attempt"* | wrong entity |
| *"was **recovered** without any recovery attempts"* | **fabricated a recovery that did not happen** |
| *"resulting in an amount of…"* (5 of 8 cases) | causally wrong — the amount is an input, not a result |

**The most serious one:** it claimed money was recovered when nothing was. In a
revenue-recovery product, that is the worst possible error.

**And critically — our hallucination guard flagged ZERO of these.** It catches
invented *numbers*. These were invented *facts*. The model reused real numbers
in false statements.

### 1.2 What that means for you

**Keep the template as the default for the full batch.** It is instant,
deterministic, and factually correct because every value is interpolated from
the database rather than generated.

**Use Ollama to demonstrate the adapter works**, on a small subset, with a
decent model. Do not run all 1,815 autopsies through a local model before your
video — at ~5s each that is over 2 hours, for prose that is worse.

**This is a strength for your submission, not a weakness.** Razorpay grades
*"AI judgment — the right tool in the right place, **and where you chose not to
use one**."* You can now say: *we tested an LLM on this task and chose rules
instead, here is the evidence.* Almost no other submission will have that.

---

## 2. Installing Ollama

### macOS / Windows

Download from **https://ollama.com** and run the installer. The app runs the
server for you on `http://localhost:11434`.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve          # or: sudo systemctl start ollama
```

### Verify it is running

```bash
curl http://localhost:11434/api/version
# {"version":"0.33.3"}
```

---

## 3. Pulling a model

```bash
ollama pull llama3.2:3b      # ~2 GB, needs ~4 GB RAM free — the configured default
```

Memory guidance:

| Model | Download | RAM needed | Quality |
|---|---|---|---|
| `qwen2.5:0.5b` | 400 MB | ~1 GB | poor — made factual errors |
| `llama3.2:1b` | 1.3 GB | ~2 GB | weak |
| `llama3.2:3b` | 2.0 GB | ~4 GB | acceptable — **use this** |
| `llama3.1:8b` | 4.7 GB | ~8 GB | good, if your laptop can take it |

Pick based on your free RAM. On any laptop with 8 GB+ you can use 3b or 8b.

---

## 4. Connecting it to Ghost Ledger

The project **already defaults to Ollama**. No code change needed.

`config.py`:

```python
LLM_BACKEND     = "ollama"                       # default
LLM_MODEL       = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TIMEOUT_SECONDS = 12.0
LLM_TEMPERATURE     = 0.2
```

### Basic run

```bash
ollama serve                              # terminal 1 (skip if the app is running)
ollama pull llama3.2:3b                   # once
python main.py --llm-backend ollama       # terminal 2
```

### Recommended: cap the autopsy count

```bash
python main.py --llm-backend ollama --autopsy-limit 50
```

Full batch = 1,815 autopsies. At ~3–5s each on a laptop that is 1.5–2.5 hours.
Capping gives you real LLM output in about 2 minutes and leaves the rest on the
template.

### Override the model / endpoint without editing files

```bash
LLM_MODEL=llama3.1:8b python main.py --llm-backend ollama --autopsy-limit 20
OLLAMA_BASE_URL=http://192.168.1.50:11434 python main.py --llm-backend ollama
```

### Side-by-side comparison before you commit

```bash
LLM_MODEL=llama3.2:3b python scripts/compare_autopsy_backends.py --n 5
```

Shows template vs Ollama on the same real failures, with timing, degradation
and hallucination flags. **Run this first** — it tells you whether your model
is good enough to put in the video.

---

## 5. Safety: it cannot break your demo

| Failure | Behaviour |
|---|---|
| Ollama not running | falls back to template, `degraded=True` |
| Request timeout (>12s) | falls back to template |
| Model returns empty string | falls back to template |
| Model unreachable / HTTP error | falls back to template |
| Model invents a number | text kept, flagged in `hallucination_flags` |

All five are covered by `tests/test_autopsy.py`. The pipeline never raises.

Every run I did while building used `--llm-backend template` precisely because
this guarantee holds — the numbers are identical either way.

---

## 6. Testing the entire project

### 6.1 Setup

```bash
cd ghost-ledger
python3 -m pip install -r requirements.txt
```

If pip complains about permissions: `python3 -m pip install --user -r requirements.txt`

### 6.2 Full test suite — 127 tests

```bash
python3 -m pytest tests/ -q
# 127 passed in ~85s
```

Fast version (skips the slow subprocess pipeline tests, ~7s):

```bash
python3 -m pytest tests/ --ignore=tests/test_pipeline.py -q
```

| File | Tests | Covers |
|---|---:|---|
| `test_policy_engine.py` | 23 | FR-003 rules, ₹10k ceiling, attempt cap, stopping rule |
| `test_data_generator.py` | 21 | Reproducibility, schema, split integrity |
| `test_razorpay_client.py` | 20 | Simulator determinism, payload shapes, degradation |
| `test_autopsy.py` | 17 | Never raises, falls back, flags invented numbers |
| `test_metrics.py` | 15 | Headline reconciliation, double-count regression |
| `test_diagnoser.py` | 14 | Training, held-out eval, grouped CV |
| `test_end_to_end.py` | 12 | Full loop with stub clients |
| `test_pipeline.py` | 5 | One-command completion, idempotency, attempt cap |

### 6.3 Full pipeline — the one-command test

```bash
python main.py
```

Expected (about 78s on a clean checkout):

```
Amount at risk      : INR 2,646,060.94
Amount recovered    : INR 1,907,719.68
Recovery rate       : 72.10%
Recovery actions    : 3,816
Stopping-rule stops : 440
Audit records       : 7,633
Reconciles with audit trail: True
```

If you see those numbers, everything works.

### 6.4 Other entry points

```bash
python main.py --help                  # all flags
python main.py --regen                 # rebuild corpus from seed
python main.py --retrain               # force hyperparameter tuning (~3 min)
python main.py --reset-db              # rebuild the database
python main.py --limit 250             # smaller batch
python main.py --no-autopsy            # skip explanations entirely

python diagnoser/train.py              # train + evaluate + attribution
python weekly_report.py                # FR-009 report
streamlit run dashboard/app.py         # dashboard on :8501
```

### 6.5 Verify determinism (optional but impressive)

```bash
python main.py | grep "Amount recovered"
python main.py | grep "Amount recovered"
python main.py | grep "Amount recovered"
```

All three lines must be identical: `INR 1,907,719.68`.

### 6.6 Verify the dashboard reconciles

```bash
python -c "
from metrics import compute_headline_metrics, cause_breakdown
m = compute_headline_metrics()
c = cause_breakdown()
print('headline  ', round(m['amount_at_risk_inr'], 2))
print('per-cause ', round(sum(x['at_risk'] for x in c), 2))
print('MATCH' if abs(sum(x['at_risk'] for x in c) - m['amount_at_risk_inr']) < 0.01 else 'MISMATCH')
"
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: dotenv` | `pip install -r requirements.txt` |
| `could not connect to ollama server` | run `ollama serve` in another terminal |
| `model 'llama3.2:3b' not found` | `ollama pull llama3.2:3b` |
| Ollama very slow / `signal: killed` | not enough RAM — use a smaller model |
| All autopsies say `template-v1` | expected — Ollama was unreachable, fell back |
| DB errors after editing schema | `python main.py --reset-db` |
| Tests fail after a fresh checkout | reinstall deps, then `python main.py` once |

---

## 8. Recommended order for your demo

```bash
# 1. install
python3 -m pip install -r requirements.txt

# 2. verify the system works without any LLM (78s)
python main.py --llm-backend template

# 3. run the tests
python3 -m pytest tests/ -q

# 4. install Ollama and check it responds
ollama serve &
curl http://localhost:11434/api/version

# 5. pull a model
ollama pull llama3.2:3b

# 6. compare before committing to it
LLM_MODEL=llama3.2:3b python scripts/compare_autopsy_backends.py --n 5

# 7. if the output looks good, use it on a subset
python main.py --llm-backend ollama --autopsy-limit 50

# 8. dashboard
streamlit run dashboard/app.py
```

---

## 9. If you only do one thing

Run **step 6**. It prints template and Ollama output side by side for the same
five failures. Read them. If Ollama is better, use it. If it is not — and on a
small model it will not be — use the template and say in your submission that
you tested the LLM and deliberately chose not to rely on it.

That decision, backed by printed evidence, scores better on Razorpay's *"AI
judgment"* criterion than using an LLM because the track has "AI" in the name.
