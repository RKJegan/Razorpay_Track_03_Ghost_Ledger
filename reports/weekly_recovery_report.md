# Ghost Ledger v2 — Weekly Recovery Report

_Generated 2026-09-03T09:04:49_

## Headline

- **Recovered:** ₹1,907,719.68
- **At risk:** ₹2,646,060.94
- **Recovery rate:** 72.10%
- **Transactions in batch:** 20,479
- **Failures in batch:** 1,815
- **Recovery actions taken:** 3,816
- **Audit records written:** 7,633

**Basis:** held-out batch · N = 20,479 transactions

**Split method:** random grouped by customer: 400 of 2,000 customers drawn at random (seed 42); ALL of a held-out customer's transactions go to the holdout, so no customer appears in both splits. Grouping blocks leakage through customer-level features. Held-out batch: 20,479 transactions (20.0%). [strategy=random_grouped]

**Reconciliation:** PASS — headline figures match the audit-trail totals.

## Action outcomes

| Outcome | Count |
|---|---:|
| fail | 2,001 |
| success | 1,368 |
| stopped | 440 |
| blocked | 7 |

## Stopping-rule events (cases escalated, not retried)

**440** cases hit the 3-failed-attempt rule.

Most recent:

- `fail_txn_009860` — customer `CUST_01527`, cause `mandate_lapsed`, ₹799.00 at risk
- `fail_txn_079566` — customer `CUST_00556`, cause `insufficient_funds`, ₹2,065.15 at risk
- `fail_txn_042586` — customer `CUST_00175`, cause `card_expired`, ₹372.44 at risk
- `fail_txn_049626` — customer `CUST_01126`, cause `insufficient_funds`, ₹1,691.68 at risk
- `fail_txn_023950` — customer `CUST_00253`, cause `insufficient_funds`, ₹498.69 at risk

_Reason recorded:_ 3 failed recovery attempts for customer CUST_01527 on failure fail_txn_009860 reached the policy stopping rule (max 3). No further automated attempts. Escalated for manual review.

## Top cause buckets

| Cause | Failures | At risk | Recovered | Rate | Stopped |
|---|---:|---:|---:|---:|---:|
| insufficient_funds | 769 | ₹1,285,999.87 | ₹857,974.98 | 66.7% | 237 |
| gateway_timeout | 614 | ₹890,265.80 | ₹835,005.68 | 93.8% | 10 |
| card_expired | 215 | ₹307,052.87 | ₹89,789.01 | 29.2% | 150 |
| mandate_lapsed | 217 | ₹162,742.40 | ₹124,950.01 | 76.8% | 43 |

## Diagnoser performance

Held-out **n = 1,815** failed transactions · macro-F1 **0.9709** · accuracy **0.9758**

| Cause | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| card_expired | 0.9535 | 0.9111 | 0.9318 | 225 |
| insufficient_funds | 0.9753 | 0.9715 | 0.9734 | 772 |
| gateway_timeout | 0.9805 | 1.0000 | 0.9901 | 602 |
| mandate_lapsed | 0.9862 | 0.9907 | 0.9885 | 216 |

_Attribution:_ model lift over a no-ML error-code lookup is **+0.0418** macro-F1; with the error code withheld the model still reaches **0.8264**.

## Activity window

180 days covered (2026-03-07 → 2026-09-02).

## Autopsy reports

1,815 explanations generated (template-v1: 1,815).

---

_All figures are computed from the same source of truth as the dashboard (`metrics.py`). Recovery settlement in this environment is simulated unless Razorpay test credentials are configured._
