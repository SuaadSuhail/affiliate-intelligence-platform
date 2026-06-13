"""
Processing Router
=================
Endpoints that run NLP tagging and embedding generation over communications.

POST /process/nlp        — tag all untagged communications (synchronous)
POST /process/embeddings — embed all unembedded communications (synchronous)
POST /process/full       — NLP + embeddings end-to-end in background
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.api.task_store import set_task
from src.core.logging_config import get_logger
from src.storage.database import SessionLocal, get_db

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(get_api_key)])


# ─── NLP ──────────────────────────────────────────────────────────────────────

@router.post("/nlp")
def run_nlp(db: Session = Depends(get_db)) -> dict:
    """
    Run NLP tagging on all untagged communications.

    Delegates to process_all_communications() which queries every Communication
    where tags = [], runs the spaCy pipeline, and writes tags + sentiment back
    to PostgreSQL.
    """
    from src.ingestion.nlp_processor import process_all_communications

    result = process_all_communications(db)
    db.commit()
    return result


# ─── Embeddings ───────────────────────────────────────────────────────────────

@router.post("/embeddings")
def run_embeddings(db: Session = Depends(get_db)) -> dict:
    """
    Generate embeddings for all communications that have no embedding_id yet.

    Delegates to embed_all_communications() which chunks each communication's
    raw_text, encodes with all-MiniLM-L6-v2, stores in ChromaDB, and writes
    the first chunk doc_id to communications.embedding_id.
    """
    from src.ingestion.embedding_generator import embed_all_communications, vector_store

    result = embed_all_communications(db, vector_store)
    db.commit()
    return result


# ─── Full pipeline (background) ───────────────────────────────────────────────

def _run_process_full_task(task_id: str) -> None:
    """Run ETL → NLP → Embeddings pipeline in a background thread."""
    set_task(task_id, "running")
    logger.info("Process full task started", extra={"task_id": task_id})
    db = SessionLocal()
    try:
        from src.ingestion.nlp_processor import process_all_communications
        from src.ingestion.embedding_generator import embed_all_communications, vector_store

        etl_result: dict = {"status": "skipped"}
        try:
            from src.ingestion.etl_pipeline import run_full_pipeline
            run_full_pipeline()
            db.expire_all()
            etl_result = {"status": "complete"}
        except Exception as exc:
            etl_result = {"status": "error", "detail": str(exc)}

        nlp_result = process_all_communications(db)
        db.commit()

        emb_result = embed_all_communications(db, vector_store)
        db.commit()

        result = {"etl": etl_result, "nlp": nlp_result, "embeddings": emb_result}
        set_task(task_id, "complete", result=result)
        logger.info("Process full task complete", extra={"task_id": task_id})
    except Exception as exc:
        logger.error("Process full task failed", extra={"task_id": task_id, "error": str(exc)})
        set_task(task_id, "failed", error=str(exc))
    finally:
        db.close()


@router.post("/full")
async def run_full(background_tasks: BackgroundTasks) -> dict:
    """
    Run the complete processing pipeline in the background.

    Order: ETL (optional) → NLP tagging → Embeddings.
    Returns immediately with a task_id. Poll GET /task/{task_id} for status.
    """
    task_id = str(uuid4())
    set_task(task_id, "pending")
    background_tasks.add_task(_run_process_full_task, task_id)
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": f"Processing started in background. Poll GET /task/{task_id} for status.",
    }