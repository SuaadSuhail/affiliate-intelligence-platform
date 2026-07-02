"""
Scheduled background jobs for the Affiliate Intelligence Platform.

Jobs
----
run_scheduled_leakage_scan  — daily promo-code leakage scan at 03:00 UTC
start_scheduler             — create and start the APScheduler BackgroundScheduler
"""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.logging_config import get_logger

logger = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def run_scheduled_leakage_scan() -> None:
    """Run a full promo-code leakage scan and log the result."""
    from src.storage.database import SessionLocal
    from src.scraping.leakage_scraper import check_leakage

    db = SessionLocal()
    try:
        result = check_leakage(db, scan_type="scheduled")
        logger.info(
            "Scheduled leakage scan complete",
            extra={
                "sites_checked": result["sites_checked"],
                "sites_failed": len(result["sites_failed"]),
                "new_leaks": len(result["new_leaks"]),
            },
        )
    except Exception as exc:
        logger.error("Scheduled leakage scan failed", extra={"error": str(exc)})
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler:
    """
    Create and start the APScheduler BackgroundScheduler.

    Idempotent: if the scheduler is already running, return it immediately
    rather than registering a second instance.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler already running — skipping reinitialisation")
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        run_scheduled_leakage_scan,
        trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="leakage_daily_scan",
        replace_existing=True,
    )
    _scheduler.start()

    job_ids = [job.id for job in _scheduler.get_jobs()]
    logger.info("Scheduler started", extra={"jobs": job_ids})

    return _scheduler