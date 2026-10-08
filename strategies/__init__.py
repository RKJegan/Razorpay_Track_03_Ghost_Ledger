"""
Ghost Ledger v3 — multi-strategy recovery engine (Track B).

Modules
-------
playbooks   B1  YAML playbooks: PlaybookLoader (validate, hot reload)
executor    B1  PlaybookExecutor (records a plan as events and dunning rows)
router      B2  deterministic StrategyRouter (pure; runs only after policy allows)
timing      B3  rule-based retry timing (pure functions)
gateway     B4  GatewayHealthMonitor and GatewayFailoverEngine
dunning     B5  DunningSequencer (dunning_touches; send_via_channel is mocked)
ab_test     B6  stable-hash A/B assignment and two-proportion z-test
methods     B7  alternate payment-method suggestion
runner      B8  glue: handle_failure, retry_pass, observe_payment

Everything here is behind ``config.ENABLE_ADVANCED_STRATEGIES`` (default off).
No module in this package imports an LLM client.
"""
