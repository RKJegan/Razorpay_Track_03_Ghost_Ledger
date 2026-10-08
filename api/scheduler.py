"""
Ghost Ledger v3 — background scheduler wiring (A4).

Started from the FastAPI lifespan hook when ``SCHEDULER_ENABLED=1``. Off by
default, so the webhook server and the tests never start background threads
unless asked to.

Each job runs through :func:`agents.scheduled_jobs.run_guarded`, so a failure
backs off and never stops the scheduler. ``max_instances=1`` and
``coalesce=True`` stop one slow run from piling up behind the next.
"""

from __future__ import annotations

import logging
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from agents import scheduled_jobs as jobs

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def job_table() -> list[tuple[str, Callable[[], object], int]]:
    """Return (job name, function, interval seconds) for every A4 job."""
    return [
        ("settlement_poll", jobs.poll_open_links, config.JOB_SETTLEMENT_POLL_SECONDS),
        ("reconcile", jobs.reconcile_all, config.JOB_RECONCILE_SECONDS),
        ("retry", jobs.retry_due_recoveries, config.JOB_RETRY_SECONDS),
        ("expired_link_cleanup", jobs.clean_expired_links, config.JOB_CLEAN_LINKS_SECONDS),
    ]


def run_job(name: str) -> None:
    """
    Run one A4 job by name, through the backoff guard. Never raises.

    Module-level on purpose: APScheduler serialises jobs by a module:function
    reference, so a closure would not be accepted.
    """
    functions = {job_name: fn for job_name, fn, _ in job_table()}
    outcome = jobs.run_guarded(name, functions[name])
    if outcome["status"] == "failed":
        logger.warning("job %s failed: %s", name, outcome)


def build_scheduler() -> BackgroundScheduler:
    """Create (but do not start) a scheduler with every A4 job registered."""
    scheduler = BackgroundScheduler(
        job_defaults={"max_instances": 1, "coalesce": True, "misfire_grace_time": 60},
        timezone="Asia/Kolkata",
    )
    for name, fn, seconds in job_table():
        scheduler.add_job(
            run_job,
            IntervalTrigger(seconds=seconds),
            args=[name],
            id=name,
            name=name,
            replace_existing=True,
        )
    return scheduler


def start_scheduler() -> BackgroundScheduler | None:
    """Start the scheduler if ``SCHEDULER_ENABLED`` is set. Returns None when off."""
    global _scheduler
    if not config.SCHEDULER_ENABLED:
        logger.info("scheduler disabled (SCHEDULER_ENABLED=0)")
        return None
    if _scheduler is None:
        _scheduler = build_scheduler()
        _scheduler.start()
        logger.info("scheduler started with jobs: %s", [j[0] for j in job_table()])
    return _scheduler


def stop_scheduler() -> None:
    """Stop the scheduler if it is running. Safe to call when it is not."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def scheduler_running() -> bool:
    """True if the background scheduler is currently running."""
    return _scheduler is not None and _scheduler.running
