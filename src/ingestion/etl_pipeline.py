"""
ETL Pipeline
============
Loads mock data into PostgreSQL via SQLAlchemy.

Two jobs
--------
Job 1 — run_affiliate_etl(db)
    Reads data/mock/affiliates.csv with Pandas, parses last_contact_at as a
    proper datetime, upserts Affiliate rows by name.

Job 2 — run_communications_etl(db)
    Reads data/mock/emails.txt (source=email) and
    data/mock/transcripts.txt (source=call).
    Each communication block begins with [AFFILIATE: Name] and [DATE: ...].
    Upserts Communication rows by (affiliate_id, occurred_at).
    Tags and embeddings are left empty — filled by the NLP and embedding
    pipeline in a later stage.

Orchestrator
-----------
run_full_etl(db) — runs Job 1 then Job 2, commits on success, rolls back
    the entire transaction if anything fails.

Direct run (seed mock data)
---------------------------
    python src/ingestion/etl_pipeline.py

FastAPI usage
-------------
    from src.ingestion.etl_pipeline import run_full_etl, run_affiliate_etl, run_communications_etl
    # pass a Session from Depends(get_db); the function commits / rolls back internally
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.orm import Session

load_dotenv()

from src.storage.database import db_session, init_db
from src.storage.models import Affiliate, Communication

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"

_VALID_STATUSES = {"active", "at_risk", "churned", "high_growth"}
_VALID_SOURCES  = {"email", "call", "api_event"}


# ─── Private helpers ──────────────────────────────────────────────────────────

def _derive_status(churn: float, growth: float, health: float) -> str:
    """Derive affiliate status from score values."""
    if churn >= 0.8:
        return "churned"
    if churn >= 0.5:
        return "at_risk"
    if health >= 75 and growth >= 0.7:
        return "high_growth"
    return "active"


def _parse_comm_blocks(text: str, default_source: str) -> list[dict]:
    """
    Split a communications text file into individual records.

    Expected block format (repeated for each communication):

        [AFFILIATE: Affiliate Name]
        [DATE: 2026-05-19T09:14:00]

        <full raw text of the email or transcript>

    The [DATE: ...] line is extracted and removed from raw_text.
    Blocks are delimited by the next [AFFILIATE: ...] header.

    Parameters
    ----------
    text           : full file contents
    default_source : 'email' or 'call'

    Returns a list of dicts with keys:
        affiliate_name, occurred_at (datetime, UTC-aware), raw_text, source
    """
    # Split on [AFFILIATE: ...] markers; alternating name / body pairs
    parts = re.split(r"\[AFFILIATE:\s*(.+?)\]", text)
    # parts[0] = any pre-header text (ignore)
    # parts[1], parts[2] = first name, first body
    # parts[3], parts[4] = second name, second body …

    records: list[dict] = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract [DATE: ...] from the body
        date_match = re.search(r"\[DATE:\s*(.+?)\]", body)
        occurred_at: datetime
        if date_match:
            raw_dt = date_match.group(1).strip()
            try:
                dt = datetime.fromisoformat(raw_dt)
                occurred_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                occurred_at = datetime.now(timezone.utc)
        else:
            occurred_at = datetime.now(timezone.utc)

        # Remove the [DATE: ...] line from raw_text and trim whitespace
        raw_text = re.sub(r"\[DATE:\s*.+?\]\n?", "", body).strip()

        if name and raw_text:
            records.append(
                {
                    "affiliate_name": name,
                    "occurred_at": occurred_at,
                    "raw_text": raw_text,
                    "source": default_source,
                }
            )

    return records


# ─── Job 1: Affiliates ────────────────────────────────────────────────────────

def run_affiliate_etl(db: Session) -> dict:
    """
    Job 1 — read data/mock/affiliates.csv with Pandas and upsert Affiliate rows.

    Upsert key : name (case-sensitive)
    Columns used from CSV:
        name, status, revenue_30d, ctr_trend_pct, last_contact_at,
        churn_risk_score, growth_potential_score, health_score
    Ignored columns (kept in CSV for human reference):
        id, email, company, tier, join_date, country, niche,
        traffic_source, days_since_contact (auto-computed by SQLAlchemy)

    Returns
    -------
    {
        "affiliates_processed": int,
        "affiliates_created": int,
        "affiliates_updated": int,
    }

    Raises
    ------
    FileNotFoundError : if affiliates.csv is missing
    """
    csv_path = DATA_DIR / "affiliates.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Affiliates CSV not found at: {csv_path}\n"
            f"Expected location: {csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)

    # Parse last_contact_at into UTC-aware timestamps; NaT becomes None below
    if "last_contact_at" in df.columns:
        df["last_contact_at"] = pd.to_datetime(
            df["last_contact_at"], utc=True, errors="coerce"
        )

    created = 0
    updated = 0

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name == "nan":
            continue

        churn  = float(row.get("churn_risk_score",      0.5) or 0.5)
        growth = float(row.get("growth_potential_score", 0.5) or 0.5)
        health = float(row.get("health_score",           50.0) or 50.0)
        revenue = float(row.get("revenue_30d", 0.0) or 0.0)
        ctr    = float(row.get("ctr_trend_pct", 0.0) or 0.0)

        status_raw = str(row.get("status", "")).strip()
        status = (
            status_raw if status_raw in _VALID_STATUSES
            else _derive_status(churn, growth, health)
        )

        # last_contact_at: convert Pandas Timestamp → Python datetime (UTC)
        lc_val = row.get("last_contact_at")
        last_contact: Optional[datetime] = None
        if lc_val is not None and not pd.isna(lc_val):
            ts = pd.Timestamp(lc_val)
            last_contact = ts.to_pydatetime()
            if last_contact.tzinfo is None:
                last_contact = last_contact.replace(tzinfo=timezone.utc)

        existing = db.query(Affiliate).filter_by(name=name).first()
        if existing:
            affiliate = existing
            updated += 1
        else:
            affiliate = Affiliate()
            db.add(affiliate)
            created += 1

        affiliate.name                   = name
        affiliate.status                 = status
        affiliate.churn_risk_score       = churn
        affiliate.growth_potential_score = growth
        affiliate.health_score           = health
        affiliate.revenue_30d            = revenue
        affiliate.ctr_trend_pct          = ctr
        if last_contact is not None:
            affiliate.last_contact_at = last_contact

        db.flush()  # materialise the generated UUID before moving on
        logger.info("  ✓ Affiliate: %s (%s)", name, status)

    total = created + updated
    logger.info("[etl] Affiliates — processed=%d created=%d updated=%d", total, created, updated)

    return {
        "affiliates_processed": total,
        "affiliates_created": created,
        "affiliates_updated": updated,
    }


# ─── Job 2: Communications ────────────────────────────────────────────────────

def run_communications_etl(db: Session) -> dict:
    """
    Job 2 — read emails.txt and transcripts.txt and upsert Communication rows.

    Block format in each file:

        [AFFILIATE: Affiliate Name]
        [DATE: 2026-05-19T09:14:00]

        <raw text of the email or transcript>

    Upsert key      : (affiliate_id, occurred_at)
    Tags            : left as [] — filled by the NLP pipeline later
    Embedding       : left as None — filled by the embedding pipeline later
    Unknown affiliates : logged as a warning; block is skipped (no crash)

    Returns
    -------
    {
        "communications_processed": int,
        "created": int,
        "updated": int,
    }
    """
    all_blocks: list[dict] = []

    emails_path = DATA_DIR / "emails.txt"
    if emails_path.exists():
        text = emails_path.read_text(encoding="utf-8")
        all_blocks.extend(_parse_comm_blocks(text, default_source="email"))
    else:
        logger.warning("[etl] emails.txt not found at %s — skipping", emails_path)

    transcripts_path = DATA_DIR / "transcripts.txt"
    if transcripts_path.exists():
        text = transcripts_path.read_text(encoding="utf-8")
        all_blocks.extend(_parse_comm_blocks(text, default_source="call"))
    else:
        logger.warning("[etl] transcripts.txt not found at %s — skipping", transcripts_path)

    created = 0
    updated = 0

    for block in all_blocks:
        affiliate_name = block["affiliate_name"]
        occurred_at    = block["occurred_at"]
        raw_text       = block["raw_text"]
        source         = block["source"]

        # Look up the affiliate by name
        affiliate = db.query(Affiliate).filter_by(name=affiliate_name).first()
        if not affiliate:
            logger.warning(
                "[etl] Affiliate %r not found in DB — skipping communication (occurred_at=%s)",
                affiliate_name,
                occurred_at,
            )
            continue

        # Upsert on (affiliate_id, occurred_at)
        existing_comm = (
            db.query(Communication)
            .filter_by(affiliate_id=affiliate.id, occurred_at=occurred_at)
            .first()
        )

        if existing_comm:
            existing_comm.raw_text = raw_text
            existing_comm.source   = source
            updated += 1
            logger.info(
                "  ↺ Updated comm [%s] for %s @ %s", source, affiliate_name, occurred_at
            )
        else:
            comm = Communication(
                affiliate_id    = affiliate.id,
                source          = source,
                raw_text        = raw_text,
                tags            = [],    # NLP pipeline fills this later
                sentiment_score = 0.0,   # NLP pipeline fills this later
                embedding_id    = None,  # embedding pipeline fills this later
                occurred_at     = occurred_at,
            )
            db.add(comm)
            db.flush()
            created += 1
            logger.info(
                "  ✓ Created comm [%s] for %s @ %s", source, affiliate_name, occurred_at
            )

    total = created + updated
    logger.info("[etl] Communications — processed=%d created=%d updated=%d", total, created, updated)

    return {
        "communications_processed": total,
        "created": created,
        "updated": updated,
    }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_full_etl(db: Session) -> dict:
    """
    Run Job 1 (affiliates) then Job 2 (communications) atomically.

    Affiliates are loaded first so the FK constraint on Communications is
    satisfied when the second job runs.

    On success  : commits the transaction and returns combined result dict.
    On any error: rolls back the entire transaction so no partial data is
                  written, then re-raises the exception.

    Returns
    -------
    {
        "affiliates_processed": int,
        "affiliates_created": int,
        "affiliates_updated": int,
        "communications_processed": int,
        "created": int,
        "updated": int,
    }
    """
    try:
        logger.info("[etl] Starting full ETL …")

        aff_result  = run_affiliate_etl(db)
        comm_result = run_communications_etl(db)

        db.commit()
        logger.info("[etl] Full ETL committed successfully.")

        return {**aff_result, **comm_result}

    except Exception:
        db.rollback()
        logger.exception("[etl] Full ETL failed — transaction rolled back.")
        raise


# ─── API-facing upload endpoint (POST /ingest/csv) ────────────────────────────

def ingest_csv_content(csv_content: str) -> dict:
    """
    Accept raw CSV string from the POST /ingest/csv file-upload endpoint and
    upsert Affiliate rows.

    Expected columns : name, status, revenue_30d, ctr_trend_pct
    Optional columns : churn_risk_score, growth_potential_score, health_score

    Returns {created, updated, total}.
    """
    df = pd.read_csv(io.StringIO(csv_content))

    created = 0
    updated = 0

    with db_session() as db:
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            if not name or name == "nan":
                continue

            churn  = float(row.get("churn_risk_score",      0.5) or 0.5)
            growth = float(row.get("growth_potential_score", 0.5) or 0.5)
            health = float(row.get("health_score",           50.0) or 50.0)
            revenue = float(row.get("revenue_30d", 0.0) or 0.0)
            ctr    = float(row.get("ctr_trend_pct", 0.0) or 0.0)

            status_raw = str(row.get("status", "")).strip()
            status = (
                status_raw if status_raw in _VALID_STATUSES
                else _derive_status(churn, growth, health)
            )

            existing = db.query(Affiliate).filter_by(name=name).first()
            if existing:
                existing.status                 = status
                existing.revenue_30d            = revenue
                existing.ctr_trend_pct          = ctr
                existing.churn_risk_score       = churn
                existing.growth_potential_score = growth
                existing.health_score           = health
                updated += 1
            else:
                db.add(
                    Affiliate(
                        name                   = name,
                        status                 = status,
                        revenue_30d            = revenue,
                        ctr_trend_pct          = ctr,
                        churn_risk_score       = churn,
                        growth_potential_score = growth,
                        health_score           = health,
                    )
                )
                created += 1

    return {"created": created, "updated": updated, "total": created + updated}


# ─── Direct script entry point ────────────────────────────────────────────────

def _run_as_script() -> None:
    """Seed the database from the mock data files when run as __main__."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    print("\n═══ Affiliate Intelligence Platform — ETL Pipeline ═══\n")
    print("Initialising database schema …")
    init_db()

    from src.storage.database import SessionLocal

    db = SessionLocal()
    try:
        result = run_full_etl(db)
    finally:
        db.close()

    print("\n✅  ETL pipeline complete.\n")
    print(f"  Affiliates  : {result['affiliates_processed']} processed "
          f"({result['affiliates_created']} created, {result['affiliates_updated']} updated)")
    print(f"  Communications: {result['communications_processed']} processed "
          f"({result['created']} created, {result['updated']} updated)\n")


if __name__ == "__main__":
    _run_as_script()