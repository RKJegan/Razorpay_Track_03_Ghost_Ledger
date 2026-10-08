"""
Ghost Ledger v3 — FastAPI application.

Run the webhook listener (Terminal 1 in the run guide)::

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints are added per component. Currently:

* ``POST /webhooks/razorpay``  — signed Razorpay webhook (A1)
* ``/api/approvals``           — human approval queue (A3, operator key)
* ``/api/operator/summary``    — live state for the Streamlit dashboard (A5)
* ``/api/merchant/*``          — merchant-scoped API, bearer key (A6)
* ``/merchant``                — merchant HTML/JS dashboard (A6)
* ``GET  /health``             — liveness and feature-flag status

The database schema is migrated on startup (idempotent). Background jobs
(settlement poll, reconcile, retry, expired-link cleanup) run only when
``SCHEDULER_ENABLED=1``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

import config
from api.approvals_api import router as approvals_router
from api.merchant_api import page_router as merchant_page_router
from api.merchant_api import router as merchant_router
from api.operator_api import router as operator_router
from api.scheduler import start_scheduler, stop_scheduler
from api.webhooks import router as webhook_router
from database.db_client import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Bring the unified database up to date on startup.

    ``init_db`` creates the v2 tables (schema.sql) AND runs the v3 migrations.
    Calling only the v3 migrations leaves the v2 tables missing (for example
    ``audit_trail``), which silently breaks every audit write.
    """
    init_db()
    logger.info("database ready at %s", config.DB_PATH)
    start_scheduler()  # no-op unless SCHEDULER_ENABLED=1
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(
    title="Ghost Ledger v3 API",
    version="3.0.0",
    description="Real-time revenue recovery: webhooks, approvals, merchant views.",
    lifespan=lifespan,
)
app.include_router(webhook_router)
app.include_router(approvals_router)
app.include_router(operator_router)
app.include_router(merchant_router)
app.include_router(merchant_page_router)


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness check. Reports feature flags so an operator can see the mode."""
    return {
        "status": "ok",
        "advanced_strategies": config.ENABLE_ADVANCED_STRATEGIES,
        "live_test_mode": config.RAZORPAY_LIVE_TEST_MODE,
        "webhook_secret_configured": bool(config.RAZORPAY_WEBHOOK_SECRET),
        "scheduler_enabled": config.SCHEDULER_ENABLED,
    }
