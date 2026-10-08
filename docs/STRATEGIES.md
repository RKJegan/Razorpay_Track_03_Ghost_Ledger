# Multi-strategy recovery engine (Track B)

Ghost Ledger v3 adds a cause-specific strategy layer on top of the v2 policy
engine. It decides *how* a recovery is attempted (link now, retry later,
dunning only, or suggest another method) for each root cause. It never decides
*whether* an attempt is allowed. That stays with the policy engine (R1 amount
ceiling, R2 attempt cap, R3 stopping rule, R4 audit).

## Switch

| Setting | Default | Meaning |
|---|---|---|
| `ENABLE_ADVANCED_STRATEGIES` | `0` | `0` = exactly the v2 / A-track behaviour. `1` = strategy layer on. This is the rollback switch. |

With the switch at `0`, no strategy code runs on the live path. The retry job,
the scheduler, and the webhook handler behave as before. The only change
visible with the switch off is that webhook events store the payment method
in their event detail. That field is additive.

## Order of operations for one attempt

1. **Policy first.** `agents/recovery_executor.rule_on_recovery` runs R1–R4 and
   writes the R4 audit entry before anything else.
2. **Blocked is final.** A stop, a denial, or an approval request is returned
   exactly as v2 returns it. The router does not run.
3. **Router second (B2).** Only if the policy allows does
   `strategies/router.route` build a plan. It is a pure, deterministic function
   of the playbook, the time, and the gateway health snapshots.
4. **Executor records (B1).** `strategies/executor.PlaybookExecutor.apply`
   writes `playbook_selected`, and optionally `retry_scheduled`,
   `method_suggested`, `gateway_failover`, and `dunning_scheduled`.
5. **Action.** `create_link` goes through `execute_ruling` (the same
   policy-gated code path as v2). `schedule_retry` and `dunning_only` create no
   link.

If a playbook is missing or invalid, the attempt falls back to the v2 path
(immediate link, still policy-gated) and the event records why.

## Playbooks (B1)

One YAML file per root cause in `playbooks/`. Shipped: `card_expired`,
`insufficient_funds`, `gateway_timeout`, `mandate_lapsed`.

```yaml
cause: insufficient_funds
version: 1
description: "..."
route: create_link            # create_link | dunning_only
retry:
  timing_rule: cycle_aware_funds   # none | cycle_aware_funds | peak_avoidance
failover:
  enabled: false
  candidates: [upi, netbanking]
method_suggestion:
  after_card_failures: 2
  candidates: [upi, netbanking]
dunning:
  touches:
    - {offset_minutes: 0, channel: sms, template: funds_first}
templates:
  funds_first: "Hello, your payment of INR {amount_inr} for {txn_id} did not go through. {next_step}"
```

The loader (`PlaybookLoader`) rejects a set if any of these hold: unknown
keys, an unknown cause, a cause with no playbook or with two, a file name that
does not match its cause, an unknown route, timing rule, channel, or method,
decreasing touch offsets, more than 6 touches, a template that names a missing
field, or a template field other than `amount_inr`, `txn_id`, `next_step`.

**Hot reload.** Edit a file while the app runs. The next lookup reloads the set.
If the new set is invalid, the last good set keeps running and the error is kept
in `PlaybookLoader.last_error`.

## Timing rules (B3)

| Rule | Cause | What it does |
|---|---|---|
| `cycle_aware_funds` | insufficient_funds | From day 26 of the month, defer to the 2nd of next month at 10:00. |
| `peak_avoidance` | gateway_timeout | Step a retry forward in 30-minute steps out of the peak windows 11–14 and 19–22. |
| `none` | card_expired, mandate_lapsed | Immediate. |

The peak windows and the day-26 cash crunch are the ones the synthetic
generator uses (`data/synthetic_generator.py`). They are assumptions of the
dataset, not measured bank behaviour.

**Holdout comparison.** `python scripts/retry_timing_holdout.py` compares
immediate retry with the rule on the held-out failures. It writes
`reports/retry_timing_holdout.json`. The holdout records what failed, not what
happened on a retry, so the outcomes are **simulated** from a model written at
the top of the script. The uplift it reports shows whether the rule matches
that model. It is not evidence of real-world uplift. Use the A/B experiment
(below) for that.

## Gateway health and failover (B4)

Every verified `payment.captured` or `payment.failed` webhook records one
observation for its payment method (`gateway_observations`). Over a 60-minute
window, a method is:

* `unknown` below `GATEWAY_HEALTH_MIN_SAMPLE` (20) observations. Never acted on.
* `healthy` at or above `GATEWAY_DEGRADED_BELOW` (80%) captured.
* `degraded` below that.

Failover moves to a candidate only when the current route is `degraded` and the
candidate is verified `healthy`. Unknown data never triggers a switch. The first
route is assumed to be `card`.

## Dunning (B5)

`dunning_touches` holds one row per (recovery, touch). Scheduling is idempotent.
A job (`dunning`, every `JOB_DUNNING_SECONDS`, only with the switch on) sends
due touches. Before each send it checks the recovery. If the recovery is no
longer pending (for example, the customer paid), the touch is cancelled and
nothing is sent. A webhook capture also cancels the rest of the sequence.

`strategies/dunning.send_via_channel` is a **mock**. It makes no network call
and returns a simulated receipt. To connect a real provider, change that one
function. Recipients are customer ids, never phone numbers or addresses, because
Ghost Ledger does not store contact details.

## A/B tests (B6)

`strategies/ab_test.py` assigns each unit to `control` or `treatment` by a
SHA-256 hash of `experiment:unit`. The assignment is stored, so a later change
to the split does not move an existing unit. Evaluation is a two-sided
two-proportion z-test on the first outcome per unit. There is no verdict until
both arms have `AB_MIN_SAMPLE_PER_ARM` (100) units.

Set `AB_RETRY_TIMING_EXPERIMENT=1` (with the switch on) to run the live timing
experiment. Control is immediate retry (v2 timing) and treatment is the playbook
rule. It is off by default because it changes customer timing.

## Alternate payment method (B7)

After `after_card_failures` card failures on one recovery, the playbook's
candidate methods (in order) are suggested, skipping any that are verified
degraded. The suggestion is recorded as `method_suggested` and appears in the
dunning text.

## Tables

| Table | Purpose |
|---|---|
| `recovery_cases`, `recovery_events` | v2/v3 recovery log (unchanged). New stages: `retry_scheduled`, `method_suggested`, `gateway_failover`, `dunning_scheduled`, `dunning_cancelled`. |
| `dunning_touches` | Scheduled and sent reminders. Unique on (recovery, touch). |
| `gateway_observations` | Payment outcome per method, for health. |
| `ab_assignments` | Stable experiment assignment and first outcome. |

All of these live in the same SQLite file as v2. Migration 4 (`v3_strategies`)
is idempotent and safe to re-run.

## Trust boundary

No module in `strategies/` imports an LLM client (a test enforces this). Every
amount, approval, stop, timing decision, and method choice is deterministic
Python. Customer-facing text comes from playbook templates with three fields
filled in.

## Tests

* `tests/test_v3_strategies.py`: B1–B7 units (validation, hot reload, router,
  timing, gateway health, dunning, A/B z-test, method suggestion, migrations).
* `tests/test_v3_strategies_e2e.py`: B8/B9 end to end through `submit_recovery`,
  the retry job, and the webhook endpoint, with the switch on and off.

```bash
pytest tests -q
python scripts/retry_timing_holdout.py
ENABLE_ADVANCED_STRATEGIES=1 python main.py --no-autopsy   # dry-run summary, then the v2 pipeline
```
