"""
Leakage Router
==============
Endpoints for triggering and querying promo-code leakage scans.

POST /leakage/scan                   — run a full leakage scan in background; returns task_id
GET  /leakage/results                — all recorded LeakedCode rows (newest first)
GET  /leakage/results/{affiliate_id} — LeakedCode rows for one affiliate ([] if none)
"""

from __future__ import annotations

import uuid as _uuid
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.api.task_store import set_task
from src.core.logging_config import get_logger
from src.storage.database import SessionLocal, get_db

logger = get_logger(__name__)
router = APIRouter()


# ─── Background task ──────────────────────────────────────────────────────────

def _run_leakage_scan_task(task_id: str) -> None:
    """Run a full leakage scan across all affiliates in a background thread."""
    set_task(task_id, "running")
    logger.info("Leakage scan task started", extra={"task_id": task_id})
    db = SessionLocal()
    try:
        from src.scraping.leakage_scraper import check_leakage
        result = check_leakage(db, scan_type="on_demand")
        set_task(task_id, "complete", result=result)
        logger.info(
            "Leakage scan task complete",
            extra={
                "task_id": task_id,
                "sites_checked": result["sites_checked"],
                "new_leaks": len(result["new_leaks"]),
            },
        )
    except Exception as exc:
        logger.error("Leakage scan task failed", extra={"task_id": task_id, "error": str(exc)})
        set_task(task_id, "failed", error=str(exc))
    finally:
        db.close()


# ─── POST /scan ───────────────────────────────────────────────────────────────

@router.post("/scan", dependencies=[Depends(get_api_key)])
async def trigger_leakage_scan(background_tasks: BackgroundTasks) -> dict:
    """
    Run a full promo-code leakage scan across all affiliates in the background.
    Returns immediately with a task_id. Poll GET /task/{task_id} for status.
    """
    task_id = str(uuid4())
    set_task(task_id, "pending")
    background_tasks.add_task(_run_leakage_scan_task, task_id)
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": f"Leakage scan started in background. Poll GET /task/{task_id} for status.",
    }


# ─── GET /results ─────────────────────────────────────────────────────────────

@router.get("/results", response_model=list[dict])
def get_leakage_results(db: Session = Depends(get_db)) -> list[dict]:
    """List all recorded promo-code leak events, newest first."""
    from src.storage.models import LeakedCode
    rows = (
        db.query(LeakedCode)
        .order_by(LeakedCode.found_at.desc())
        .all()
    )
    return [
        {
            "id": str(r.id),
            "affiliate_id": str(r.affiliate_id),
            "code": r.code,
            "site": r.site,
            "source_url": r.source_url,
            "scan_type": r.scan_type,
            "found_at": r.found_at.isoformat() if r.found_at else None,
        }
        for r in rows
    ]


# ─── GET /results/{affiliate_id} ──────────────────────────────────────────────

@router.get("/results/{affiliate_id}", response_model=list[dict])
def get_leakage_results_for_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    List recorded leak events for one affiliate, newest first.
    Returns an empty list (not 404) when no leaks have been recorded — a clean
    result is a valid, normal outcome.
    """
    try:
        uid = _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid affiliate_id: must be a UUID")

    from src.storage.models import LeakedCode
    rows = (
        db.query(LeakedCode)
        .filter(LeakedCode.affiliate_id == uid)
        .order_by(LeakedCode.found_at.desc())
        .all()
    )
    return [
        {
            "id": str(r.id),
            "affiliate_id": str(r.affiliate_id),
            "code": r.code,
            "site": r.site,
            "source_url": r.source_url,
            "scan_type": r.scan_type,
            "found_at": r.found_at.isoformat() if r.found_at else None,
        }
        for r in rows
    ]
