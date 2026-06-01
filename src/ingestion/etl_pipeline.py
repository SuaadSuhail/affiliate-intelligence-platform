"""
ETL Pipeline
============
Loads mock data into PostgreSQL. NLP tagging and embedding generation
are separate steps — see POST /process/nlp and POST /process/embeddings.

Responsibilities
----------------
1. Read affiliates.csv → upsert Affiliate rows
2. Parse emails.txt + transcripts.txt → insert Communication rows (raw text only)

Run directly to seed mock data:
    python src/ingestion/etl_pipeline.py
"""

import csv
import io
import re
import uuid
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.storage.database import init_db, db_session
from src.storage.models import Affiliate, Communication

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"

# ─── Schema helpers ───────────────────────────────────────────────────────────

_SOURCE_MAP = {"email": "email", "call": "call", "api_event": "api_event"}


def _derive_status(churn: float, growth: float) -> str:
    if churn > 0.6:
        return "at_risk"
    if growth > 0.6:
        return "high_growth"
    return "active"


def _compute_days_since(last_contact_at: Optional[datetime]) -> int:
    if last_contact_at is None:
        return 0
    now = datetime.now(timezone.utc)
    lc = last_contact_at if last_contact_at.tzinfo else last_contact_at.replace(tzinfo=timezone.utc)
    return max(0, (now - lc).days)


# ─── Step 1: Ingest affiliates CSV ────────────────────────────────────────────

def ingest_affiliates_csv(path: Path) -> list[str]:
    """
    Read affiliates.csv and upsert into PostgreSQL.
    Upserts by name (new schema has no email column).
    Returns list of affiliate UUIDs processed.
    """
    ids: list[str] = []
    with db_session() as db:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing = db.query(Affiliate).filter(Affiliate.name == row["name"]).first()
                if existing:
                    aff = existing
                else:
                    aff = Affiliate(id=uuid.uuid4())
                    db.add(aff)

                churn = float(row.get("churn_risk_score", 0.5))
                growth = float(row.get("growth_potential_score", 0.5))
                health = float(row.get("health_score", 50.0))

                aff.name = row["name"]
                aff.status = _derive_status(churn, growth)
                aff.churn_risk_score = churn
                aff.growth_potential_score = growth
                aff.health_score = health
                aff.revenue_30d = float(row.get("monthly_revenue", 0))
                aff.ctr_trend_pct = 0.0

                raw_lc = row.get("last_contact_date") or row.get("last_contact_at")
                if raw_lc:
                    try:
                        lc_dt = datetime.fromisoformat(raw_lc)
                        if lc_dt.tzinfo is None:
                            lc_dt = lc_dt.replace(tzinfo=timezone.utc)
                        aff.last_contact_at = lc_dt
                        aff.days_since_contact = _compute_days_since(lc_dt)
                    except ValueError:
                        pass

                ids.append(str(aff.id))
                print(f"  ✓ Affiliate: {aff.name}")

    print(f"[etl] Affiliates ingested: {len(ids)}")
    return ids


# ─── Step 2: Parse flat text files (emails + transcripts) ─────────────────────

def _parse_blocks(text: str) -> list[dict]:
    """
    Parse ===RECORD_NNN=== delimited blocks from emails.txt / transcripts.txt.
    Returns list of raw field dicts.
    """
    blocks = re.split(r"===\w+===", text)
    records = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        record: dict = {}
        lines = block.split("\n")
        content_lines: list[str] = []
        in_content = False
        for line in lines:
            if not in_content and re.match(r"^\w+:", line):
                key, _, value = line.partition(":")
                record[key.strip().lower()] = value.strip()
            else:
                in_content = True
                content_lines.append(line)
        raw = "\n".join(content_lines).strip()
        # Strip EXPECTED TAGS comment
        raw = re.sub(r"\n---\nEXPECTED TAGS:.*$", "", raw, flags=re.DOTALL).strip()
        record["raw_text"] = raw
        if record.get("affiliate_id"):
            records.append(record)
    return records


def ingest_communications_file(path: Path) -> list[str]:
    """
    Parse a flat text file (emails.txt or transcripts.txt) and insert
    Communication rows into PostgreSQL.

    Inserts raw_text only — tags and embeddings are populated later by
    POST /process/nlp and POST /process/embeddings.
    Returns list of communication UUIDs created.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _parse_blocks(text)
    comm_ids: list[str] = []

    with db_session() as db:
        for block in blocks:
            affiliate_id_str = block.get("affiliate_id", "").strip()
            affiliate = _find_affiliate_by_mock_id(db, affiliate_id_str)
            if not affiliate:
                print(f"  ✗ Affiliate not found: {affiliate_id_str} — skipping")
                continue

            occurred_at_str = block.get("occurred_at", "")
            try:
                occurred_at = datetime.fromisoformat(occurred_at_str)
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            except ValueError:
                occurred_at = datetime.now(timezone.utc)

            raw_text = block.get("raw_text", "")
            # Map block channel → Communication.source enum
            channel_raw = block.get("channel", "email").lower()
            source = _SOURCE_MAP.get(channel_raw, "email")

            comm = Communication(
                affiliate_id=affiliate.id,
                source=source,
                raw_text=raw_text,
                tags=[],
                sentiment_score=0.0,
                occurred_at=occurred_at,
            )
            db.add(comm)
            db.flush()
            comm_ids.append(str(comm.id))

            # Update affiliate last_contact_at and days_since_contact
            if (
                affiliate.last_contact_at is None
                or occurred_at > (
                    affiliate.last_contact_at
                    if affiliate.last_contact_at.tzinfo
                    else affiliate.last_contact_at.replace(tzinfo=timezone.utc)
                )
            ):
                affiliate.last_contact_at = occurred_at
                affiliate.days_since_contact = _compute_days_since(occurred_at)

            print(f"  ✓ Comm [{source}] for {affiliate.name}")

    print(f"[etl] Communications ingested: {len(comm_ids)}")
    return comm_ids


def _find_affiliate_by_mock_id(db, mock_id: str) -> Optional[Affiliate]:
    """
    Map mock IDs (aff-001 etc.) to Affiliate rows by name.
    New schema has no email column — name is the stable lookup key.
    """
    mock_map = {
        "aff-001": "Sarah Chen",
        "aff-002": "Marcus Williams",
        "aff-003": "Priya Sharma",
        "aff-004": "James O'Brien",
        "aff-005": "Aiko Tanaka",
        "aff-006": "Carlos Mendez",
        "aff-007": "Fatima Al-Hassan",
        "aff-008": "Tom Bauer",
        "aff-009": "Nkechi Okonkwo",
        "aff-010": "Rachel Torres",
    }
    name = mock_map.get(mock_id)
    if name:
        return db.query(Affiliate).filter(Affiliate.name == name).first()
    return None


# ─── API-facing ingestion (CSV upload) ────────────────────────────────────────

def ingest_csv_content(csv_content: str) -> dict:
    """
    Accept raw CSV string (from API upload) and upsert affiliates.
    Upserts by name.  Returns summary dict.
    """
    with io.StringIO(csv_content) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with db_session() as db:
        created = 0
        updated = 0
        for row in rows:
            name = row.get("name", "").strip()
            if not name:
                continue
            existing = db.query(Affiliate).filter(Affiliate.name == name).first()
            if existing:
                existing.revenue_30d = float(row.get("monthly_revenue", existing.revenue_30d or 0))
                updated += 1
            else:
                aff = Affiliate(
                    name=name,
                    status="active",
                    revenue_30d=float(row.get("monthly_revenue", 0)),
                )
                db.add(aff)
                created += 1

    return {"created": created, "updated": updated, "total": created + updated}


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_full_pipeline() -> None:
    print("\n═══ Affiliate Intelligence Platform — ETL Pipeline ═══\n")

    print("1/3  Initialising database schema …")
    init_db()

    print("\n2/3  Ingesting affiliates …")
    ingest_affiliates_csv(DATA_DIR / "affiliates.csv")

    print("\n3/3  Ingesting communications (raw text only) …")
    ingest_communications_file(DATA_DIR / "emails.txt")
    ingest_communications_file(DATA_DIR / "transcripts.txt")

    print("\n✅  ETL complete. Run /process/nlp then /process/embeddings to finish.\n")


if __name__ == "__main__":
    run_full_pipeline()