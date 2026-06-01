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
from sqlalchemy.orm import Session

from src.storage.database import get_db

router = APIRouter()


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


# ─── Full pipeline ────────────────────────────────────────────────────────────

@router.post("/full")
def run_full(db: Session = Depends(get_db)) -> dict:
    """
    Run the complete processing pipeline end-to-end.

    Order: ETL (optional) → NLP tagging → Embeddings.
    Each step is idempotent — only unprocessed records are touched.
    """
    from src.ingestion.nlp_processor import process_all_communications
    from src.ingestion.embedding_generator import embed_all_communications, vector_store

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
    nlp_result = process_all_communications(db)
    db.commit()

    # Step 3 — Embeddings
    emb_result = embed_all_communications(db, vector_store)
    db.commit()

    return {
        "etl": etl_result,
        "nlp": nlp_result,
        "embeddings": emb_result,
    }