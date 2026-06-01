"""
Ingestion Router
================
Endpoints that trigger data ingestion from mock files or CSV uploads.

POST /ingest/full           — run full ETL (affiliates + comms + ChromaDB profiles)
POST /ingest/affiliates     — re-ingest affiliates CSV only
POST /ingest/communications — re-ingest emails.txt + transcripts.txt
POST /ingest/csv            — upload a CSV file of affiliates
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.storage.database import get_db

router = APIRouter()


@router.post("/full")
def ingest_full() -> dict:
    """
    Run the complete ETL pipeline from mock data files.
    Steps: affiliates.csv → emails.txt → transcripts.txt → ChromaDB profiles.
    """
    try:
        from src.ingestion.etl_pipeline import run_full_pipeline
        run_full_pipeline()
        return {"status": "complete", "message": "Full ETL pipeline finished"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ETL error: {exc}")


@router.post("/affiliates")
def ingest_affiliates() -> dict:
    """Re-ingest affiliates from data/mock/affiliates.csv only."""
    try:
        from src.ingestion.etl_pipeline import ingest_affiliates_csv, DATA_DIR
        ids = ingest_affiliates_csv(DATA_DIR / "affiliates.csv")
        return {"status": "complete", "affiliates_processed": len(ids)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}")


@router.post("/communications")
def ingest_communications() -> dict:
    """Re-ingest communications from data/mock/emails.txt and transcripts.txt."""
    try:
        from src.ingestion.etl_pipeline import ingest_communications_file, DATA_DIR
        email_ids = ingest_communications_file(DATA_DIR / "emails.txt")
        trans_ids = ingest_communications_file(DATA_DIR / "transcripts.txt")
        return {
            "status": "complete",
            "emails_processed": len(email_ids),
            "transcripts_processed": len(trans_ids),
            "total": len(email_ids) + len(trans_ids),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}")


@router.post("/csv")
async def ingest_csv(file: UploadFile = File(...)) -> dict:
    """
    Upload a CSV file of affiliates to ingest into the platform.
    Expected columns: name, email, company, tier, join_date, country,
                      niche, traffic_source, monthly_revenue
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")
    content = await file.read()
    csv_text = content.decode("utf-8")
    try:
        from src.ingestion.etl_pipeline import ingest_csv_content
        result = ingest_csv_content(csv_text)
        return {"status": "success", "filename": file.filename, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV ingest error: {exc}")