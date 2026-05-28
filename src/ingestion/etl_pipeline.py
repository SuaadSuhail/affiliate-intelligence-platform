"""
ETL Pipeline
============
Orchestrates full data ingestion:
  1. Read affiliates.csv → upsert Affiliate rows
  2. Parse emails.txt + transcripts.txt → create Communication rows
  3. Run NLP tagging + sentiment on every communication
  4. Generate embeddings + upsert into ChromaDB (affiliate_comms collection)

Run directly to seed mock data:
    python src/ingestion/etl_pipeline.py

Schema reference (src/storage/models.py)
-----------------------------------------
  Affiliate
    id, name, status (active|at_risk|churned|high_growth),
    churn_risk_score, growth_potential_score, health_score,
    revenue_30d, ctr_trend_pct,
    last_contact_at, days_since_contact (auto-computed), updated_at

  Communication
    id, affiliate_id, source (email|call|api_event),
    raw_text, tags (Postgres TEXT[]), sentiment_score,
    embedding_id, occurred_at

CSV column aliases (for backward compatibility with the original mock data)
---------------------------------------------------------------------------
    monthly_revenue  → revenue_30d
    last_contact_date → last_contact_at
"""

import csv
import io
import re
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.storage.database import init_db, db_session
from src.storage.models import Affiliate, Communication
from src.ingestion.nlp_processor import process_text
from src.ingestion.embedding_generator import get_generator

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"

# Valid values for the Communication.source enum
_VALID_SOURCES = {"email", "call", "api_event"}

# Valid values for the Affiliate.status enum
_VALID_STATUSES = {"active", "at_risk", "churned", "high_growth"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _map_source(channel: str) -> str:
    """Map legacy channel/source strings to the Communication.source enum."""
    ch = (channel or "").lower().strip()
    return ch if ch in _VALID_SOURCES else "api_event"


def _derive_status(
    churn_risk: float,
    growth_potential: float,
    health_score: float,
) -> str:
    """
    Derive Affiliate.status from score values when the CSV column is absent or
    does not contain a valid enum value.
    """
    if churn_risk >= 0.8:
        return "churned"
    if churn_risk >= 0.5:
        return "at_risk"
    if health_score >= 75 and growth_potential >= 0.7:
        return "high_growth"
    return "active"


# ─── Step 1: Ingest affiliates CSV ────────────────────────────────────────────

def ingest_affiliates_csv(path: Path) -> list[str]:
    """
    Read affiliates.csv and upsert into PostgreSQL.

    Required CSV columns : name
    Used if present       : status, churn_risk_score, growth_potential_score,
                            health_score, revenue_30d (or monthly_revenue),
                            ctr_trend_pct, last_contact_at (or last_contact_date)
    Ignored               : id, email, company, tier, join_date, country, niche,
                            traffic_source (legacy; not in current schema)

    Returns list of affiliate UUID strings processed.
    """
    ids: list[str] = []

    with db_session() as db:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                if not name:
                    continue

                existing = db.query(Affiliate).filter_by(name=name).first()
                affiliate: Affiliate = existing or Affiliate()

                churn  = float(row.get("churn_risk_score", 0.5))
                growth = float(row.get("growth_potential_score", 0.5))
                health = float(row.get("health_score", 50.0))

                # Accept both new column name and legacy alias
                revenue = float(
                    row.get("revenue_30d") or row.get("monthly_revenue") or 0.0
                )
                ctr = float(row.get("ctr_trend_pct") or 0.0)

                # Accept explicit status or derive from scores
                status_raw = (row.get("status") or "").strip()
                status = (
                    status_raw if status_raw in _VALID_STATUSES
                    else _derive_status(churn, growth, health)
                )

                # Accept both column name variants for last contact timestamp
                lc_raw = row.get("last_contact_at") or row.get("last_contact_date") or ""
                last_contact: Optional[datetime] = None
                if lc_raw:
                    try:
                        lc = datetime.fromisoformat(lc_raw)
                        last_contact = lc if lc.tzinfo else lc.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

                affiliate.name                  = name
                affiliate.status                = status
                affiliate.churn_risk_score      = churn
                affiliate.growth_potential_score = growth
                affiliate.health_score          = health
                affiliate.revenue_30d           = revenue
                affiliate.ctr_trend_pct         = ctr
                if last_contact is not None:
                    affiliate.last_contact_at = last_contact

                if not existing:
                    db.add(affiliate)

                db.flush()  # materialise the generated UUID
                ids.append(str(affiliate.id))
                print(f"  ✓ Affiliate: {affiliate.name} ({affiliate.status})")

    print(f"[etl] Affiliates ingested: {len(ids)}")
    return ids


# ─── Step 2: Parse flat text files (emails + transcripts) ─────────────────────

def _parse_blocks(text: str) -> list[dict]:
    """
    Parse ===RECORD_NNN=== delimited blocks from emails.txt / transcripts.txt.
    Returns a list of raw field dicts (one dict per block).
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
            if not in_content and re.match(r"^\w[\w_]*:", line):
                key, _, value = line.partition(":")
                record[key.strip().lower()] = value.strip()
            else:
                in_content = True
                content_lines.append(line)
        # Join content; strip trailing EXPECTED TAGS annotation
        raw_content = "\n".join(content_lines).strip()
        record["content"] = re.sub(
            r"\n---\nEXPECTED TAGS:.*$", "", raw_content, flags=re.DOTALL
        ).strip()
        if record.get("affiliate_id"):
            records.append(record)
    return records


def ingest_communications_file(path: Path) -> list[str]:
    """
    Parse a flat text file (emails.txt or transcripts.txt) and insert
    Communication rows with NLP tags + sentiment.
    Returns list of communication UUID strings created.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _parse_blocks(text)
    comm_ids: list[str] = []
    gen = get_generator()

    with db_session() as db:
        for block in blocks:
            mock_id = block.get("affiliate_id", "").strip()
            affiliate = _find_affiliate_by_mock_id(db, mock_id)
            if not affiliate:
                print(f"  ✗ Affiliate not found: {mock_id!r} — skipping block")
                continue

            # Parse occurred_at; default to now if missing / unparseable
            occurred_at_raw = block.get("occurred_at", "")
            try:
                occurred_at = datetime.fromisoformat(occurred_at_raw)
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            except ValueError:
                occurred_at = datetime.now(timezone.utc)

            raw_text = block.get("content", "")
            source   = _map_source(block.get("channel", "email"))

            # Run NLP tagging + sentiment
            nlp_result = process_text(raw_text)

            comm = Communication(
                affiliate_id    = affiliate.id,
                source          = source,
                raw_text        = raw_text,
                sentiment_score = nlp_result.sentiment_score,
                tags            = nlp_result.tags,
                occurred_at     = occurred_at,
            )
            db.add(comm)
            db.flush()  # materialise the generated UUID

            # Embed the text and upsert into ChromaDB
            doc_id = gen.index_communication(
                comm_id        = str(comm.id),
                text           = raw_text,
                affiliate_id   = str(affiliate.id),
                affiliate_name = affiliate.name,
                source         = source,
                tags           = nlp_result.tags,
                occurred_at    = occurred_at.isoformat(),
            )
            comm.embedding_id = doc_id
            comm_ids.append(str(comm.id))

            # Keep affiliate.last_contact_at up to date
            if (
                affiliate.last_contact_at is None
                or occurred_at > affiliate.last_contact_at
            ):
                affiliate.last_contact_at = occurred_at

            print(
                f"  ✓ Comm [{source}] for {affiliate.name} | "
                f"sent={nlp_result.sentiment_score:.3f} tags={nlp_result.tags}"
            )

    print(f"[etl] Communications ingested: {len(comm_ids)}")
    return comm_ids


def _find_affiliate_by_mock_id(db, mock_id: str) -> Optional[Affiliate]:
    """
    Map mock-data IDs (e.g. 'aff-001') to Affiliate rows via name lookup.
    The Affiliate model has no email column, so we match on name — which is
    unique across the 10 mock affiliates.
    """
    _mock_name_map: dict[str, str] = {
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
    name = _mock_name_map.get(mock_id)
    if name:
        return db.query(Affiliate).filter_by(name=name).first()
    return None


# ─── API-facing ingestion (CSV upload) ────────────────────────────────────────

def ingest_csv_content(csv_content: str) -> dict:
    """
    Accept raw CSV string (from the POST /ingest/csv endpoint) and upsert
    affiliates into PostgreSQL.

    Expected columns : name, status, revenue_30d, ctr_trend_pct
    Optional columns : churn_risk_score, growth_potential_score, health_score

    Returns a dict: {created: int, updated: int, total: int}.
    """
    with io.StringIO(csv_content) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    created = 0
    updated = 0

    with db_session() as db:
        for row in rows:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            churn  = float(row.get("churn_risk_score", 0.5))
            growth = float(row.get("growth_potential_score", 0.5))
            health = float(row.get("health_score", 50.0))
            status_raw = (row.get("status") or "").strip()
            status = (
                status_raw if status_raw in _VALID_STATUSES
                else _derive_status(churn, growth, health)
            )
            revenue = float(row.get("revenue_30d", 0.0) or 0.0)
            ctr     = float(row.get("ctr_trend_pct", 0.0) or 0.0)

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
                aff = Affiliate(
                    name                   = name,
                    status                 = status,
                    revenue_30d            = revenue,
                    ctr_trend_pct          = ctr,
                    churn_risk_score       = churn,
                    growth_potential_score = growth,
                    health_score           = health,
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

    print("\n3/3  Ingesting communications …")
    ingest_communications_file(DATA_DIR / "emails.txt")
    ingest_communications_file(DATA_DIR / "transcripts.txt")

    print("\n✅  ETL pipeline complete.\n")


if __name__ == "__main__":
    run_full_pipeline()