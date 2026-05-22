"""
ETL Pipeline
============
Orchestrates full data ingestion:
  1. Read affiliates.csv → upsert Affiliate rows
  2. Parse emails.txt + transcripts.txt → create Communication rows
  3. Run NLP tagging + sentiment on every communication
  4. Generate embeddings + upsert into ChromaDB
  5. Upsert affiliate profile embeddings

Run directly to seed mock data:
    python src/ingestion/etl_pipeline.py
"""

import csv
import io
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# We need the DB to be up — import lazily to avoid startup errors during testing
from src.storage.database import init_db, db_session
from src.storage.models import Affiliate, Communication
from src.ingestion.nlp_processor import process_text
from src.ingestion.embedding_generator import get_generator

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"


# ─── Step 1: Ingest affiliates CSV ────────────────────────────────────────────

def ingest_affiliates_csv(path: Path) -> list[str]:
    """
    Read affiliates.csv and upsert into PostgreSQL.
    Returns list of affiliate IDs processed.
    """
    ids = []
    with db_session() as db:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing = (
                    db.query(Affiliate).filter_by(email=row["email"]).first()
                )
                if existing:
                    affiliate = existing
                else:
                    affiliate = Affiliate(id=uuid.UUID(row["id"]) if row.get("id") else uuid.uuid4())

                affiliate.name = row["name"]
                affiliate.email = row["email"]
                affiliate.company = row.get("company")
                affiliate.tier = row.get("tier", "bronze")
                affiliate.join_date = datetime.strptime(row["join_date"], "%Y-%m-%d").date()
                affiliate.country = row.get("country")
                affiliate.niche = row.get("niche")
                affiliate.traffic_source = row.get("traffic_source")
                affiliate.monthly_revenue = float(row.get("monthly_revenue", 0))
                affiliate.churn_risk_score = float(row.get("churn_risk_score", 0.5))
                affiliate.growth_potential_score = float(row.get("growth_potential_score", 0.5))
                affiliate.health_score = float(row.get("health_score", 50.0))

                if row.get("last_contact_date"):
                    affiliate.last_contact_date = datetime.fromisoformat(row["last_contact_date"])

                if not existing:
                    db.add(affiliate)

                ids.append(str(affiliate.id))
                print(f"  ✓ Affiliate: {affiliate.name} ({affiliate.email})")

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
        # Extract key: value header lines
        lines = block.split("\n")
        content_lines = []
        in_content = False
        for line in lines:
            if not in_content and re.match(r"^\w+:", line):
                key, _, value = line.partition(":")
                record[key.strip().lower()] = value.strip()
            else:
                in_content = True
                content_lines.append(line)
        record["content"] = "\n".join(content_lines).strip()
        # Strip EXPECTED TAGS comment from content
        record["content"] = re.sub(
            r"\n---\nEXPECTED TAGS:.*$", "", record["content"], flags=re.DOTALL
        ).strip()
        if record.get("affiliate_id"):
            records.append(record)
    return records


def ingest_communications_file(path: Path) -> list[str]:
    """
    Parse a flat text file (emails.txt or transcripts.txt) and insert
    Communication rows with NLP tags + sentiment.
    Returns list of communication IDs created.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _parse_blocks(text)
    comm_ids = []
    gen = get_generator()

    with db_session() as db:
        for block in blocks:
            affiliate_id_str = block.get("affiliate_id", "").strip()
            # Look up the affiliate by the mock ID stored in the CSV
            # In mock data IDs are short strings like "aff-001" stored as emails/names
            # We resolve by scanning affiliates (mock IDs won't be real UUIDs)
            # Strategy: match on name pattern or just create with a stable UUID from the string
            affiliate = _find_affiliate_by_mock_id(db, affiliate_id_str)
            if not affiliate:
                print(f"  ✗ Affiliate not found: {affiliate_id_str} — skipping block")
                continue

            # Parse occurred_at
            occurred_at_str = block.get("occurred_at", "")
            try:
                occurred_at = datetime.fromisoformat(occurred_at_str)
            except ValueError:
                occurred_at = datetime.utcnow()

            content = block.get("content", "")
            subject = block.get("subject", "")
            channel = block.get("channel", "email")
            direction = block.get("direction", "inbound")

            # Run NLP
            nlp_result = process_text(content)

            comm = Communication(
                affiliate_id=affiliate.id,
                channel=channel,
                direction=direction,
                subject=subject,
                content=content,
                sentiment_score=nlp_result.sentiment_score,
                sentiment_label=nlp_result.sentiment_label,
                tags=nlp_result.tags,
                occurred_at=occurred_at,
            )
            db.add(comm)
            db.flush()  # get the generated UUID

            # Embed + index in ChromaDB
            doc_id = gen.index_communication(
                comm_id=str(comm.id),
                content=content,
                affiliate_id=str(affiliate.id),
                channel=channel,
                direction=direction,
                sentiment_label=nlp_result.sentiment_label,
                tags=nlp_result.tags,
                occurred_at=occurred_at.isoformat(),
            )
            comm.embedding_id = doc_id
            comm_ids.append(str(comm.id))

            # Update affiliate last_contact_date
            if (
                affiliate.last_contact_date is None
                or occurred_at > affiliate.last_contact_date
            ):
                affiliate.last_contact_date = occurred_at

            print(
                f"  ✓ Comm [{channel}] for {affiliate.name} | "
                f"sent={nlp_result.sentiment_label} tags={nlp_result.tags}"
            )

    print(f"[etl] Communications ingested: {len(comm_ids)}")
    return comm_ids


def _find_affiliate_by_mock_id(db, mock_id: str) -> Optional[Affiliate]:
    """
    In mock data, affiliate_id values are short strings like 'aff-001'.
    Map them to Affiliate rows by position in the CSV (seeded order).
    """
    mock_map = {
        "aff-001": "sarah.chen@brightleadsmedia.com",
        "aff-002": "m.williams@performanceplus.io",
        "aff-003": "priya@clickhubnetwork.com",
        "aff-004": "jobs@dublindigital.ie",
        "aff-005": "aiko.tanaka@tanaka-affiliates.jp",
        "aff-006": "carlos@trafficbridgeco.mx",
        "aff-007": "fatima@crescentclicks.ae",
        "aff-008": "tbauer@bauer-digital.de",
        "aff-009": "nkechi@lagosgrowthlab.ng",
        "aff-010": "rachel.t@sunsetaffiliatesco.com",
    }
    email = mock_map.get(mock_id)
    if email:
        return db.query(Affiliate).filter_by(email=email).first()
    return None


# ─── Step 3: Index affiliate profiles in ChromaDB ─────────────────────────────

def index_affiliate_profiles() -> None:
    """Embed and upsert all affiliate profile documents into ChromaDB."""
    gen = get_generator()
    with db_session() as db:
        affiliates = db.query(Affiliate).all()
        for aff in affiliates:
            gen.index_affiliate_profile(
                affiliate_id=str(aff.id),
                name=aff.name,
                company=aff.company,
                niche=aff.niche,
                traffic_source=aff.traffic_source,
                tier=aff.tier,
                monthly_revenue=aff.monthly_revenue,
                churn_risk_score=aff.churn_risk_score,
                growth_potential_score=aff.growth_potential_score,
                health_score=aff.health_score,
            )
            print(f"  ✓ Indexed profile: {aff.name}")
    print("[etl] Affiliate profiles indexed in ChromaDB.")


# ─── API-facing ingestion (CSV upload) ────────────────────────────────────────

def ingest_csv_content(csv_content: str) -> dict:
    """
    Accept raw CSV string (from API upload) and ingest affiliates.
    Returns summary dict.
    """
    with io.StringIO(csv_content) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with db_session() as db:
        created = 0
        updated = 0
        for row in rows:
            existing = db.query(Affiliate).filter_by(email=row.get("email", "")).first()
            if existing:
                existing.name = row.get("name", existing.name)
                existing.company = row.get("company", existing.company)
                existing.tier = row.get("tier", existing.tier)
                existing.monthly_revenue = float(row.get("monthly_revenue", existing.monthly_revenue))
                updated += 1
            else:
                aff = Affiliate(
                    name=row["name"],
                    email=row["email"],
                    company=row.get("company"),
                    tier=row.get("tier", "bronze"),
                    join_date=datetime.strptime(row.get("join_date", "2024-01-01"), "%Y-%m-%d").date(),
                    country=row.get("country"),
                    niche=row.get("niche"),
                    traffic_source=row.get("traffic_source"),
                    monthly_revenue=float(row.get("monthly_revenue", 0)),
                )
                db.add(aff)
                created += 1

    return {"created": created, "updated": updated, "total": created + updated}


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_full_pipeline() -> None:
    print("\n═══ Affiliate Intelligence Platform — ETL Pipeline ═══\n")

    print("1/4  Initialising database schema …")
    init_db()

    print("\n2/4  Ingesting affiliates …")
    ingest_affiliates_csv(DATA_DIR / "affiliates.csv")

    print("\n3/4  Ingesting communications …")
    ingest_communications_file(DATA_DIR / "emails.txt")
    ingest_communications_file(DATA_DIR / "transcripts.txt")

    print("\n4/4  Indexing affiliate profiles in ChromaDB …")
    index_affiliate_profiles()

    print("\n✅  ETL pipeline complete.\n")


if __name__ == "__main__":
    run_full_pipeline()
