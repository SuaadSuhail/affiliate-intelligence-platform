"""
SEO Router
==========
Endpoints for triggering and querying SEO rank-tracking signals.

POST /seo/scan                   — run a full SEO rank check in background; returns task_id
GET  /seo/results                — all recorded SeoSignal rows (newest first)
GET  /seo/results/{affiliate_id} — SeoSignal rows for one affiliate ([] if none)
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

def _run_seo_scan_task(task_id: str) -> None:
    """Run a full SEO rank check across all tracked affiliates in a background thread."""
    set_task(task_id, "running")
    logger.info("SEO scan task started", extra={"task_id": task_id})
    db = SessionLocal()
    try:
        from src.seo.checker import check_seo
        result = check_seo(db, scan_type="on_demand")
        set_task(task_id, "complete", result=result)
        logger.info(
            "SEO scan task complete",
            extra={
                "task_id": task_id,
                "keywords_checked": result["keywords_checked"],
            },
        )
    except Exception as exc:
        logger.error("SEO scan task failed", extra={"task_id": task_id, "error": str(exc)})
        set_task(task_id, "failed", error=str(exc))
    finally:
        db.close()


# ─── POST /scan ───────────────────────────────────────────────────────────────

@router.post("/scan", dependencies=[Depends(get_api_key)])
async def trigger_seo_scan(background_tasks: BackgroundTasks) -> dict:
    """
    Run a full SEO rank check across all tracked affiliates in the background.
    Returns immediately with a task_id. Poll GET /task/{task_id} for status.
    """
    task_id = str(uuid4())
    set_task(task_id, "pending")
    background_tasks.add_task(_run_seo_scan_task, task_id)
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": f"SEO scan started in background. Poll GET /task/{task_id} for status.",
    }


# ─── GET /results ─────────────────────────────────────────────────────────────

@router.get("/results", response_model=list[dict])
def get_seo_results(db: Session = Depends(get_db)) -> list[dict]:
    """List all recorded SEO rank-check signals, newest first."""
    from src.storage.models import SeoSignal
    rows = (
        db.query(SeoSignal)
        .order_by(SeoSignal.checked_at.desc())
        .all()
    )
    return [
        {
            "id": str(r.id),
            "affiliate_id": str(r.affiliate_id),
            "keyword": r.keyword,
            "rank": r.rank,
            "rank_change": r.rank_change,
            "search_volume": r.search_volume,
            "checked_at": r.checked_at.isoformat() if r.checked_at else None,
        }
        for r in rows
    ]


# ─── GET /results/{affiliate_id} ──────────────────────────────────────────────

@router.get("/results/{affiliate_id}", response_model=list[dict])
def get_seo_results_for_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    List recorded SEO rank-check signals for one affiliate, newest first.
    Returns an empty list (not 404) when none have been recorded — a clean
    result is a valid, normal outcome.
    """
    try:
        uid = _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid affiliate_id: must be a UUID")

    from src.storage.models import SeoSignal
    rows = (
        db.query(SeoSignal)
        .filter(SeoSignal.affiliate_id == uid)
        .order_by(SeoSignal.checked_at.desc())
        .all()
    )
    return [
        {
            "id": str(r.id),
            "affiliate_id": str(r.affiliate_id),
            "keyword": r.keyword,
            "rank": r.rank,
            "rank_change": r.rank_change,
            "search_volume": r.search_volume,
            "checked_at": r.checked_at.isoformat() if r.checked_at else None,
        }
        for r in rows
    ]
