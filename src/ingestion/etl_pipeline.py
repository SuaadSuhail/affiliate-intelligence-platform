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
from src.rulebook.recommend import (
    CHURN_AT_RISK_THRESHOLD,
    CHURN_CRITICAL_THRESHOLD,
    GROWTH_HIGH_THRESHOLD,
)
from src.storage.database import init_db, db_session
from src.storage.models import Affiliate, Communication

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mock"

# ─── Schema helpers ───────────────────────────────────────────────────────────

_SOURCE_MAP = {"email": "email", "call": "call", "api_event": "api_event"}


def _derive_status(churn: float, growth: float) -> str:
    """
    Ingest-time status assignment. Stays a separate function from
    src.rulebook.recommend.categorize() — it runs before any communications
    or features exist for a freshly-ingested affiliate, so it can't call the
    full rulebook — but imports the same canonical thresholds so the two
    never disagree.
    """
    if churn > CHURN_CRITICAL_THRESHOLD:
        return "churned"
    if churn > CHURN_AT_RISK_THRESHOLD:
        return "at_risk"
    if growth > GROWTH_HIGH_THRESHOLD:
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

    New CSV format: name, revenue_30d, ctr_trend_pct, days_since_contact, status,
    active_promo_code (optional — blank/absent means no code assigned),
    tracked_keyword (optional — blank/absent means no SEO keyword tracked)
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
                is_new = existing is None
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

                # ML scores — only ever set here for a brand-new affiliate
                # (defaults, so update_all_scores() has something to overwrite
                # on its first run) or when the CSV explicitly provides a
                # value (legacy re-import format). A re-ingest of an existing
                # affiliate with the current CSV format (no score columns)
                # must NOT reset scores that a previous POST /ml/score run
                # already computed — ingest and scoring are separate steps,
                # and re-running the former should never silently wipe the
                # latter's output back to 0.5/0.5/50.0.
                if row.get("churn_risk_score"):
                    aff.churn_risk_score = float(row["churn_risk_score"])
                elif is_new:
                    aff.churn_risk_score = 0.5

                if row.get("growth_potential_score"):
                    aff.growth_potential_score = float(row["growth_potential_score"])
                elif is_new:
                    aff.growth_potential_score = 0.5

                if row.get("health_score"):
                    aff.health_score = float(row["health_score"])
                elif is_new:
                    aff.health_score = 50.0

                # Promo code this affiliate is authorised to share — matched
                # against monitored sites by src.scraping.leakage_scraper.
                # Blank/absent means no code assigned (skipped silently by
                # every leakage scan).
                promo_code = (row.get("active_promo_code") or "").strip()
                aff.active_promo_code = promo_code or None

                # Keyword this affiliate is tracked against for SEO rank
                # checks — matched against src.seo.api_client's fetched rows
                # by src.seo.checker. Blank/absent means no keyword tracked
                # (skipped silently by every SEO check).
                tracked_keyword = (row.get("tracked_keyword") or "").strip()
                aff.tracked_keyword = tracked_keyword or None

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
    Returns list of communication UUIDs actually created this call (does not
    include names already present from a prior run — see idempotency note
    below).

    Idempotent by (affiliate_id, source, raw_text): a communication already
    present from an earlier call is skipped, not re-inserted. The mock files
    represent a fixed historical backstory, not a live moving window — their
    "N days ago" markers are resolved to an absolute occurred_at once, the
    first time a given block is ever ingested, and are then frozen. Without
    this, every re-run (e.g. every POST /ingest/full) would insert a second,
    untagged copy of every affiliate's communications with a freshly
    recomputed occurred_at that is always more recent than any prior copy —
    which is exactly what silently buried real historical signal under
    "Insufficient data" in get_affiliate_summary's driver section.
    occurred_at itself can't be part of the identity key (it's derived at
    parse time, not stable across runs) — (affiliate_id, source, raw_text)
    is the stable proxy for "the same mock communication".
    """
    text = path.read_text(encoding="utf-8")
    blocks = _parse_blocks(text)
    comm_ids: list[str] = []
    skipped = 0

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

            existing = (
                db.query(Communication)
                .filter(
                    Communication.affiliate_id == affiliate.id,
                    Communication.source == source,
                    Communication.raw_text == raw_text,
                )
                .first()
            )
            if existing:
                skipped += 1
                logger.debug(
                    "Communication already ingested — skipping duplicate",
                    extra={"source": source, "affiliate": affiliate.name},
                )
                continue

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

            # Update affiliate last_contact_at and days_since_contact — only
            # for a genuinely new communication. An already-ingested one was
            # already accounted for the first time it was seen, and its
            # occurred_at is now frozen (see idempotency note above), so
            # there is nothing new to recompute against on a repeat run.
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

    logger.info(
        "Communications ingested",
        extra={"count": len(comm_ids), "skipped_duplicates": skipped},
    )
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


# ─── Demo leak seed ───────────────────────────────────────────────────────────

def _all_sites_are_mock() -> bool:
    """
    True only if every configured site is a local file:// fixture — the only
    signal actually safe to trust here.

    SiteConfig.kind is NOT a reliable indicator of "mock vs. real": the
    csr-shell-mock fixture is declared kind="live" for a technical reason
    (it needs the Playwright browser-render fetch path, since its JS shell
    must execute), even though its url is a local fixture file, not a real
    external site — see site_config.py's own comment on that entry. Checking
    the URL scheme directly is the only check that can't be fooled by that.
    """
    from src.scraping.site_config import SITES
    return all(site.url.startswith("file://") for site in SITES)


def seed_demo_leak_scan() -> Optional[dict]:
    """
    Run check_leakage once, covering whichever affiliates currently have an
    active_promo_code set (data/mock/affiliates.csv seeds two: Rachel Torres
    and Marcus Williams — see their active_promo_code values, which match
    codes already present in the scraping fixtures under src/scraping/fixtures/).

    This is the exact same check_leakage() used by POST /leakage/scan and the
    nightly APScheduler job — no separate demo-only code path. Kept as its
    own named function (not inlined into run_full_pipeline()) so it's easy
    to find, read, and remove independently; run_full_pipeline() calls it as
    a clearly separate final step for demo convenience — see that function's
    docstring for why this is a deliberate demo-only call, not something a
    production ingest path should do unconditionally.

    Guarded by _all_sites_are_mock(): if any configured site is not a local
    file:// fixture (i.e. a real site has been enabled in site_config.SITES),
    this logs a warning and returns None instead of scanning — enforcing the
    tradeoff documented above rather than leaving it as something a future
    reader could miss by not reading this docstring. Folding a live external
    scan into routine ingestion would otherwise become a silent side effect
    of POST /ingest/full the moment a real site is turned on.
    """
    if not _all_sites_are_mock():
        logger.warning(
            "seed_demo_leak_scan skipped — a non-fixture (non file://) site "
            "is configured in src.scraping.site_config.SITES. Refusing to "
            "run a live leakage scan as a side effect of routine ingestion; "
            "trigger POST /leakage/scan explicitly instead."
        )
        return None

    from src.storage.database import db_session
    from src.scraping.leakage_scraper import check_leakage

    with db_session() as db:
        result = check_leakage(db, scan_type="on_demand")

    logger.info(
        "Demo leak seed scan complete",
        extra={
            "sites_checked": result["sites_checked"],
            "new_leaks": len(result["new_leaks"]),
        },
    )
    return result


# ─── Demo SEO seed ────────────────────────────────────────────────────────────

def _seo_source_is_mock() -> bool:
    """
    True only if no live SEO API has been wired up yet. Mirrors
    _all_sites_are_mock()'s purpose for the leak checker — an explicit,
    enforced check rather than relying on a docstring being read — adapted
    to SEO's simpler reality: there's no list of sources to inspect, just a
    single global flag for whether a real fetch_seo_data(kind="live") branch
    exists yet.
    """
    from src.seo.api_client import LIVE_API_CONFIGURED
    return not LIVE_API_CONFIGURED


def seed_demo_seo_scan() -> Optional[dict]:
    """
    Run check_seo once, covering whichever affiliates currently have a
    tracked_keyword set (data/mock/affiliates.csv seeds four: Rachel Torres,
    Priya Sharma, Sarah Chen, and Marcus Williams — see their
    tracked_keyword values, which match keywords already present in
    data/mock/seo/rank_tracking_mock.json). Two of those four — Sarah Chen
    and Marcus Williams — are engineered to show a genuinely declining trend.

    This is the exact same check_seo() used by POST /seo/scan and the weekly
    APScheduler job — no separate demo-only code path. Kept as its own named
    function for the same reason as seed_demo_leak_scan(): easy to find,
    read, and remove independently.

    Guarded by _seo_source_is_mock(): if a live SEO API has been configured
    (src.seo.api_client.LIVE_API_CONFIGURED is True), this logs a warning
    and returns None instead of checking — same enforced tradeoff as
    seed_demo_leak_scan(), so folding a live SEO check into routine ingestion
    can't become a silent side effect the moment a real API is wired up.
    """
    if not _seo_source_is_mock():
        logger.warning(
            "seed_demo_seo_scan skipped — a live SEO API is configured "
            "(src.seo.api_client.LIVE_API_CONFIGURED is True). Refusing to "
            "run a live SEO check as a side effect of routine ingestion; "
            "trigger POST /seo/scan explicitly instead."
        )
        return None

    from src.storage.database import db_session
    from src.seo.checker import check_seo

    with db_session() as db:
        result = check_seo(db, scan_type="on_demand")

    logger.info(
        "Demo SEO seed scan complete",
        extra={"keywords_checked": result["keywords_checked"]},
    )
    return result


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_full_pipeline() -> None:
    logger.info("ETL pipeline starting")

    logger.info("Step 1/5 — initialising database schema")
    init_db()

    logger.info("Step 2/5 — ingesting affiliates")
    ingest_affiliates_csv(DATA_DIR / "affiliates.csv")

    logger.info("Step 3/5 — ingesting communications (raw text only)")
    ingest_communications_file(DATA_DIR / "emails.txt")
    ingest_communications_file(DATA_DIR / "transcripts.txt")

    # Steps 4-5 are demo convenience, not a general ETL responsibility: each
    # runs the exact same real check used by its POST /*/scan endpoint, so
    # the affiliates seeded with an active_promo_code / tracked_keyword above
    # already show has_active_leak / search_trend right after ingest, with no
    # separate manual scan step. Safe today because both data sources are
    # fixture/mock-only — each is guarded and will refuse to run and log a
    # warning instead the moment a real site/API is configured, rather than
    # silently starting to fire live external calls as a side effect of
    # routine ingestion.
    logger.info("Step 4/5 — seeding demo leak scan")
    seed_demo_leak_scan()

    logger.info("Step 5/5 — seeding demo SEO scan")
    seed_demo_seo_scan()

    logger.info("ETL complete — run /process/nlp then /process/embeddings to finish")


if __name__ == "__main__":
    run_full_pipeline()