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
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.core.logging_config import get_logger
from src.storage.database import init_db, db_session
from src.storage.models import Affiliate, Communication

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"

# ─── Schema helpers ───────────────────────────────────────────────────────────

_SOURCE_MAP = {"email": "email", "call": "call", "api_event": "api_event"}


def _derive_status(churn: float, growth: float) -> str:
    if churn > 0.8:
        return "churned"
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

    New CSV format: name, revenue_30d, ctr_trend_pct, days_since_contact, status
    last_contact_at is computed as now() - days_since_contact so it is always
    accurate regardless of when the pipeline runs.

    Legacy CSV format (with churn_risk_score / last_contact_date columns) is
    still supported for backward compatibility.

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

                aff.name = row["name"]

                # Revenue — support both column names
                aff.revenue_30d = float(
                    row.get("revenue_30d") or row.get("monthly_revenue") or 0
                )

                # CTR trend
                aff.ctr_trend_pct = float(row.get("ctr_trend_pct", 0.0))

                # Status — use CSV value if present, otherwise derive from scores
                if row.get("status"):
                    aff.status = row["status"]
                else:
                    churn = float(row.get("churn_risk_score", 0.5))
                    growth = float(row.get("growth_potential_score", 0.5))
                    aff.status = _derive_status(churn, growth)

                # Initial ML scores — kept at defaults so the ML pipeline
                # can overwrite them cleanly on first scoring run
                aff.churn_risk_score = float(row.get("churn_risk_score", 0.5))
                aff.growth_potential_score = float(row.get("growth_potential_score", 0.5))
                aff.health_score = float(row.get("health_score", 50.0))

                # Compute last_contact_at from days_since_contact (new format)
                if row.get("days_since_contact"):
                    days = int(row["days_since_contact"])
                    aff.days_since_contact = days
                    aff.last_contact_at = datetime.now(timezone.utc) - timedelta(days=days)
                else:
                    # Legacy format: parse explicit last_contact_date / last_contact_at
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
                logger.debug("Affiliate upserted", extra={"name": aff.name})

    logger.info("Affiliates ingested", extra={"count": len(ids)})
    return ids


# ─── Step 2: Parse flat text files (emails + transcripts) ─────────────────────

def _parse_blocks(text: str) -> list[dict]:
    """
    Parse communication blocks from emails.txt / transcripts.txt.

    Auto-detects format:
    - New format:    [AFFILIATE: Name] / [DATE: N days ago] / [SOURCE: type]
                     blocks separated by --- lines
    - Legacy format: ===RECORD_NNN=== delimiter with key: value headers
    """
    text = text.strip()
    if text.startswith("["):
        return _parse_bracket_blocks(text)
    return _parse_legacy_blocks(text)


def _parse_bracket_blocks(text: str) -> list[dict]:
    """Parse [KEY: VALUE] header blocks separated by --- lines."""
    records = []
    raw_blocks = re.split(r"\n\s*---+\s*\n", text)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        record: dict = {}
        lines = block.split("\n")
        content_lines: list[str] = []
        header_done = False

        for line in lines:
            if not header_done:
                m = re.match(r"^\[([A-Za-z]+):\s*(.+?)\]\s*$", line.strip())
                if m:
                    record[m.group(1).upper()] = m.group(2).strip()
                    continue
                elif not line.strip():
                    if record:
                        header_done = True
                    continue
                else:
                    header_done = True
            content_lines.append(line)

        record["raw_text"] = "\n".join(content_lines).strip()

        # Resolve "N days ago" → ISO timestamp
        date_str = record.get("DATE", "")
        m = re.match(r"(\d+)\s+days?\s+ago", date_str, re.IGNORECASE)
        if m:
            days = int(m.group(1))
            record["occurred_at"] = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat()

        if record.get("AFFILIATE"):
            records.append(record)

    return records


def _parse_legacy_blocks(text: str) -> list[dict]:
    """Parse ===RECORD_NNN=== delimited blocks (original format)."""
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
            # New format uses AFFILIATE (name); legacy format uses affiliate_id (mock ID)
            if block.get("AFFILIATE"):
                affiliate_name = block["AFFILIATE"].strip()
                affiliate = (
                    db.query(Affiliate).filter(Affiliate.name == affiliate_name).first()
                )
                channel_raw = block.get("SOURCE", "email").lower()
                mock_ref = affiliate_name
            else:
                affiliate_id_str = block.get("affiliate_id", "").strip()
                affiliate = _find_affiliate_by_mock_id(db, affiliate_id_str)
                channel_raw = block.get("channel", "email").lower()
                mock_ref = affiliate_id_str

            if not affiliate:
                logger.warning(
                    "Affiliate not found — skipping communication",
                    extra={"mock_id": mock_ref},
                )
                continue

            occurred_at_str = block.get("occurred_at", "")
            try:
                occurred_at = datetime.fromisoformat(occurred_at_str)
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            except ValueError:
                occurred_at = datetime.now(timezone.utc)

            raw_text = block.get("raw_text", "")
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

            logger.debug(
                "Communication inserted",
                extra={"source": source, "affiliate": affiliate.name},
            )

    logger.info("Communications ingested", extra={"count": len(comm_ids)})
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
    logger.info("ETL pipeline starting")

    logger.info("Step 1/3 — initialising database schema")
    init_db()

    logger.info("Step 2/3 — ingesting affiliates")
    ingest_affiliates_csv(DATA_DIR / "affiliates.csv")

    logger.info("Step 3/3 — ingesting communications (raw text only)")
    ingest_communications_file(DATA_DIR / "emails.txt")
    ingest_communications_file(DATA_DIR / "transcripts.txt")

    logger.info("ETL complete — run /process/nlp then /process/embeddings to finish")


if __name__ == "__main__":
    run_full_pipeline()