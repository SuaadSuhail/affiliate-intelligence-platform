"""
Processing Router
=================
Endpoints that run NLP tagging and embedding generation over communications.

POST /process/nlp        — tag all untagged communications
POST /process/embeddings — embed all unembedded communications
POST /process/full       — NLP + embeddings end-to-end (idempotent)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.storage.database import get_db
from src.storage.models import Communication

router = APIRouter()


# ─── NLP ──────────────────────────────────────────────────────────────────────

@router.post("/nlp")
def run_nlp(db: Session = Depends(get_db)) -> dict:
    """
    Run NLP tagging on all untagged communications.

    Queries every Communication where tags = [], applies keyword/sentiment
    detection via process_text(), and writes tags + sentiment back to PostgreSQL.
    """
    from src.ingestion.nlp_processor import process_text

    untagged = (
        db.query(Communication)
        .filter(func.jsonb_array_length(Communication.tags) == 0)
        .all()
    )

    processed = 0
    tag_summary: dict[str, int] = {}

    for comm in untagged:
        result = process_text(comm.content or "")
        comm.tags = result.tags
        comm.sentiment_score = result.sentiment_score
        comm.sentiment_label = result.sentiment_label
        processed += 1
        for tag in result.tags:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1

    db.commit()
    return {
        "total_processed": processed,
        "total_tagged": sum(1 for c in untagged if c.tags),
        "tag_summary": tag_summary,
    }


# ─── Embeddings ───────────────────────────────────────────────────────────────

@router.post("/embeddings")
def run_embeddings(db: Session = Depends(get_db)) -> dict:
    """
    Generate embeddings for all communications that have no embedding_id yet.

    Chunks each communication's content, encodes with all-MiniLM-L6-v2,
    stores in ChromaDB, and writes the doc_id to communications.embedding_id.
    """
    from src.ingestion.embedding_generator import get_generator

    already_embedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.isnot(None))
        .count()
    )
    unembedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.is_(None))
        .all()
    )

    gen = get_generator()
    total_embedded = 0

    for comm in unembedded:
        try:
            doc_id = gen.index_communication(
                comm_id=str(comm.id),
                content=comm.content or "",
                affiliate_id=str(comm.affiliate_id),
                channel=comm.channel,
                direction=comm.direction,
                sentiment_label=comm.sentiment_label or "neutral",
                tags=comm.tags or [],
                occurred_at=comm.occurred_at.isoformat() if comm.occurred_at else "",
            )
            comm.embedding_id = doc_id
            total_embedded += 1
        except Exception as exc:
            print(f"[process/embeddings] Skipping comm {comm.id}: {exc}")

    db.commit()
    return {
        "total_processed": total_embedded,
        "already_embedded": already_embedded,
        "errors": len(unembedded) - total_embedded,
    }


# ─── Full pipeline ────────────────────────────────────────────────────────────

@router.post("/full")
def run_full(db: Session = Depends(get_db)) -> dict:
    """
    Run the complete processing pipeline end-to-end.

    Order: ETL (optional) → NLP tagging → Embeddings.
    Each step is idempotent — only unprocessed records are touched.
    """
    from src.ingestion.nlp_processor import process_text
    from src.ingestion.embedding_generator import get_generator

    # Step 1 — ETL (optional, catches import/runtime errors gracefully)
    etl_result: dict = {"status": "skipped"}
    try:
        from src.ingestion.etl_pipeline import run_full_pipeline
        run_full_pipeline()
        db.expire_all()  # refresh session after ETL's own commits
        etl_result = {"status": "complete"}
    except Exception as exc:
        etl_result = {"status": "error", "detail": str(exc)}

    # Step 2 — NLP
    untagged = (
        db.query(Communication)
        .filter(func.jsonb_array_length(Communication.tags) == 0)
        .all()
    )
    tag_summary: dict[str, int] = {}
    for comm in untagged:
        result = process_text(comm.content or "")
        comm.tags = result.tags
        comm.sentiment_score = result.sentiment_score
        comm.sentiment_label = result.sentiment_label
        for tag in result.tags:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1
    db.commit()

    nlp_result = {
        "total_processed": len(untagged),
        "total_tagged": sum(1 for c in untagged if c.tags),
        "tag_summary": tag_summary,
    }

    # Step 3 — Embeddings
    already_embedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.isnot(None))
        .count()
    )
    unembedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.is_(None))
        .all()
    )
    gen = get_generator()
    total_embedded = 0
    for comm in unembedded:
        try:
            doc_id = gen.index_communication(
                comm_id=str(comm.id),
                content=comm.content or "",
                affiliate_id=str(comm.affiliate_id),
                channel=comm.channel,
                direction=comm.direction,
                sentiment_label=comm.sentiment_label or "neutral",
                tags=comm.tags or [],
                occurred_at=comm.occurred_at.isoformat() if comm.occurred_at else "",
            )
            comm.embedding_id = doc_id
            total_embedded += 1
        except Exception as exc:
            print(f"[process/full] Embedding error for {comm.id}: {exc}")
    db.commit()

    return {
        "etl": etl_result,
        "nlp": nlp_result,
        "embeddings": {
            "total_processed": total_embedded,
            "already_embedded": already_embedded,
        },
    }