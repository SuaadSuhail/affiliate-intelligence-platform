"""
Scheduled background jobs for the Affiliate Intelligence Platform.

Jobs
----
run_scheduled_leakage_scan  — daily promo-code leakage scan at 03:00 UTC
run_scheduled_seo_scan      — weekly SEO rank check, Monday 04:00 UTC
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


def run_scheduled_seo_scan() -> None:
    """
    Run a full SEO rank check and log the result.

    Weekly cadence (Monday 04:00 UTC), not daily like the leak scan: search
    engine rankings move on the order of days-to-weeks as crawlers re-index
    and ranking signals accumulate, unlike a promo-code leak which can
    appear or disappear within hours. Real rank-tracking tools (SEMrush,
    Ahrefs) default to weekly position tracking for the same reason — daily
    checks would mostly measure noise against a metric that doesn't
    materially shift day to day. Offset from the leak scan's 03:00 UTC slot
    to avoid both jobs contending for resources at the same instant.
    """
    from src.storage.database import SessionLocal
    from src.seo.checker import check_seo

    db = SessionLocal()
    try:
        result = check_seo(db, scan_type="scheduled")
        logger.info(
            "Scheduled SEO scan complete",
            extra={
                "keywords_checked": result["keywords_checked"],
                "not_found": len(result["not_found"]),
            },
        )
    except Exception as exc:
        logger.error("Scheduled SEO scan failed", extra={"error": str(exc)})
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
    _scheduler.add_job(
        run_scheduled_seo_scan,
        trigger=CronTrigger(day_of_week="mon", hour=4, minute=0, timezone="UTC"),
        id="seo_weekly_scan",
        replace_existing=True,
    )
    _scheduler.start()

    job_ids = [job.id for job in _scheduler.get_jobs()]
    logger.info("Scheduler started", extra={"jobs": job_ids})

    return _scheduler