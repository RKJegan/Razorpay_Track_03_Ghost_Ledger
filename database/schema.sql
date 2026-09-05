-- ============================================================================
-- Ghost Ledger v2 — SQLite schema
-- Track: Razorpay AI Buildathon — 03: AI Revenue Recovery
--
-- The four tables below are transcribed VERBATIM from GHOST_LEDGER_PROJECT_SPEC
-- section 11 (DATABASE SCHEMA). They are the system of record for the demo.
--
-- One additive table (`autopsy_reports`) follows, clearly marked. It exists
-- because FR-005 requires the LLM autopsy text to be persisted and queryable,
-- and none of the four spec tables has a text column that fits. It is additive
-- only: no spec table or column is modified, renamed or removed.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- SPEC TABLE 1/4 — transactions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    id                TEXT PRIMARY KEY,
    merchant_id       TEXT NOT NULL,
    customer_id       TEXT NOT NULL,
    amount            REAL NOT NULL,
    status            TEXT NOT NULL,      -- success/failed/retrying
    txn_type          TEXT NOT NULL,      -- one_off/subscription
    payment_method    TEXT,
    failure_reason_raw TEXT,
    timestamp         TEXT NOT NULL,
    is_holdout        INTEGER DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- SPEC TABLE 2/4 — failures
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS failures (
    id                 TEXT PRIMARY KEY,
    transaction_id     TEXT REFERENCES transactions(id),
    predicted_cause    TEXT NOT NULL,
    confidence         REAL NOT NULL,
    ground_truth_cause TEXT,              -- null unless is_holdout
    detected_at        TEXT NOT NULL,
    estimated_value    REAL NOT NULL
);

-- ---------------------------------------------------------------------------
-- SPEC TABLE 3/4 — recoveries
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recoveries (
    id                      TEXT PRIMARY KEY,
    failure_id              TEXT REFERENCES failures(id),
    agent_name              TEXT NOT NULL,
    action_type             TEXT NOT NULL,
    attempt_number          INTEGER NOT NULL,
    policy_check_passed     INTEGER NOT NULL,
    policy_check_reason     TEXT,
    stopping_rule_triggered INTEGER DEFAULT 0,
    stopping_reason         TEXT,
    executed_at             TEXT,
    outcome                 TEXT,         -- success/fail/stopped
    recovered_amount        REAL,
    razorpay_response       TEXT
);

-- ---------------------------------------------------------------------------
-- SPEC TABLE 4/4 — audit_trail
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_trail (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    component       TEXT NOT NULL,
    action          TEXT NOT NULL,
    input_data      TEXT,
    output_data     TEXT,
    decision_reason TEXT,
    success         INTEGER NOT NULL
);

-- ===========================================================================
-- ADDITIVE TABLE (not in spec section 11) — see header note for rationale
-- ===========================================================================
CREATE TABLE IF NOT EXISTS autopsy_reports (
    failure_id   TEXT NOT NULL REFERENCES failures(id),
    report_text  TEXT NOT NULL,
    model        TEXT NOT NULL,   -- e.g. "ollama:llama3.2:3b" / "template-v1"
    basis        TEXT NOT NULL,   -- human-readable provenance of the text
    generated_at TEXT NOT NULL,
    PRIMARY KEY (failure_id)
);

-- ---------------------------------------------------------------------------
-- Indexes. Purely additive; no schema change.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_txn_holdout     ON transactions(is_holdout);
CREATE INDEX IF NOT EXISTS idx_txn_status      ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_failures_txn    ON failures(transaction_id);
CREATE INDEX IF NOT EXISTS idx_recoveries_fail ON recoveries(failure_id);
CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_trail(component);
CREATE INDEX IF NOT EXISTS idx_audit_ts        ON audit_trail(timestamp);
