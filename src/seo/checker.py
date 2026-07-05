"""
SEO signal check orchestrator.

check_seo() is the main entry point: it loads tracked keywords from the
database (affiliates.tracked_keyword), fetches rank-tracking data (mock
fixture today — see src.seo.api_client), matches each affiliate's tracked
keyword against the fetched rows, computes rank_change, and persists a new
SeoSignal row per affiliate whose keyword was found.

affiliates.search_trend IS updated here — it's a storage-layer visibility
flag (signal kept separate from and alongside the scores, not folded into
them), not a scoring input. Recomputed via src.seo.analyze.derive_search_trend
from the affiliate's full seo_signals history, not just this run's result,
so the flag stays accurate for an affiliate whose keyword momentarily wasn't
found in a given check.

No time-window dedup (unlike check_leakage): every call is a fresh, legitimate
rank measurement — even an unchanged rank is a real data point worth
recording in the time series, not noise to suppress. There IS an exact-
measurement guard, though: if a signal with the identical
(affiliate_id, keyword, checked_at) already exists, it is not re-inserted.
This is not a time-window suppression — a genuinely new measurement (a real
API returning its own current timestamp) always has a distinct checked_at
and is never affected by this. It only catches literally replaying the same
data point, which matters today because the fixture source
(data/mock/seo/rank_tracking_mock.json) has a fixed checked_at — without
this guard, every repeat call to seed_demo_seo_scan() (e.g. every
POST /ingest/full) would insert a fresh set of byte-for-byte duplicate rows.

Also writes one audit_log entry per affiliate in scan scope (src.audit.log)
— including affiliates whose tracked keyword wasn't found in this run's
data, so "checked, no data available" is on record too, not just "found
a change".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.audit.log import write_audit_entry
from src.core.logging_config import get_logger
from src.seo.analyze import derive_search_trend
from src.seo.api_client import fetch_seo_data

logger = get_logger(__name__)


def _load_tracked_keywords(db: Session) -> dict[str, str]:
    """
    Return {str(affiliate_id): tracked_keyword} for all affiliates that
    have a non-null tracked_keyword.
    """
    from src.storage.models import Affiliate  # local import — avoids hard import-time dep
    rows = (
        db.query(Affiliate.id, Affiliate.tracked_keyword)
        .filter(Affiliate.tracked_keyword.isnot(None))
        .all()
    )
    return {str(row.id): row.tracked_keyword for row in rows}


def _parse_checked_at(raw) -> datetime:
    if raw:
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def check_seo(db: Session, scan_type: str = "scheduled") -> dict:
    """
    Run a full SEO rank check across all tracked keywords.

    Parameters
    ----------
    db        : Active SQLAlchemy session. This function commits once at the
                end; the caller should not commit before returning.
    scan_type : "scheduled" | "on_demand" — kept for parity with
                check_leakage's parameter shape; not currently persisted as
                a SeoSignal column (that table has no scan_type field —
                every check is an equally legitimate data point regardless
                of trigger source).

    Returns
    -------
    {
        "keywords_checked": int,
        "signals": [{"affiliate_id", "keyword", "rank", "rank_change",
                     "search_volume"}, ...],
        "not_found": [affiliate_id, ...],
    }
    """
    from src.storage.models import Affiliate, SeoSignal

    tracked = _load_tracked_keywords(db)
    if not tracked:
        logger.info("no affiliates have a tracked_keyword — SEO scan skipped")
        return {"keywords_checked": 0, "signals": [], "not_found": []}

    logger.info(
        "SEO scan starting",
        extra={"affiliates_tracked": len(tracked), "scan_type": scan_type},
    )

    rows = fetch_seo_data(kind="fixture")
    by_keyword = {row["keyword"].strip().lower(): row for row in rows}

    signals: list[dict] = []
    not_found: list[str] = []

    for aff_id_str, keyword in tracked.items():
        row = by_keyword.get(keyword.strip().lower())
        if row is None:
            logger.warning(
                "tracked keyword not found in SEO data source — no signal this run",
                extra={"affiliate_id": aff_id_str, "keyword": keyword},
            )
            not_found.append(aff_id_str)
            continue

        current_rank = row["position"]
        previous_rank = row.get("previous_position")
        rank_change = (previous_rank - current_rank) if previous_rank is not None else None
        checked_at = _parse_checked_at(row.get("checked_at"))
        aff_uuid = uuid.UUID(aff_id_str)

        # Exact-measurement guard (see module docstring) — not a time-window
        # dedup. Skip only if this precise data point was already recorded.
        existing_signal = (
            db.query(SeoSignal)
            .filter(
                SeoSignal.affiliate_id == aff_uuid,
                SeoSignal.keyword == row["keyword"],
                SeoSignal.checked_at == checked_at,
            )
            .first()
        )
        if existing_signal is not None:
            logger.debug(
                "Identical SEO measurement already recorded — skipping duplicate",
                extra={"affiliate_id": aff_id_str, "keyword": keyword, "checked_at": checked_at.isoformat()},
            )
            signals.append({
                "affiliate_id": aff_id_str,
                "keyword": existing_signal.keyword,
                "rank": existing_signal.rank,
                "rank_change": existing_signal.rank_change,
                "search_volume": existing_signal.search_volume,
            })
            continue

        signal = SeoSignal(
            affiliate_id=aff_uuid,
            keyword=row["keyword"],
            rank=current_rank,
            rank_change=rank_change,
            search_volume=row.get("search_volume"),
            checked_at=checked_at,
        )
        db.add(signal)
        db.flush()

        signals.append({
            "affiliate_id": aff_id_str,
            "keyword": row["keyword"],
            "rank": current_rank,
            "rank_change": rank_change,
            "search_volume": row.get("search_volume"),
        })

    # Recompute search_trend + write one audit entry per affiliate that was
    # in this scan's scope — including ones whose keyword wasn't found this
    # run, so "checked, no data available" is on record too.
    signals_by_affiliate = {s["affiliate_id"]: s for s in signals}
    for aff_id_str, keyword in tracked.items():
        aff_uuid = uuid.UUID(aff_id_str)
        all_signals = db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff_uuid).all()
        trend = derive_search_trend(all_signals)

        db.query(Affiliate).filter(Affiliate.id == aff_uuid).update(
            {"search_trend": trend}, synchronize_session=False
        )

        own_signal = signals_by_affiliate.get(aff_id_str)
        output_snapshot = (
            {
                "rank": own_signal["rank"],
                "rank_change": own_signal["rank_change"],
                "search_trend": trend,
            }
            if own_signal is not None
            else {"rank": None, "rank_change": None, "search_trend": trend, "note": "keyword not found in source"}
        )
        write_audit_entry(
            db,
            stage="signals",
            record_type="affiliate",
            record_id=aff_uuid,
            rule_or_tool="check_seo",
            input_snapshot={"keywords_checked": [keyword]},
            output_snapshot=output_snapshot,
        )

    db.commit()

    logger.info(
        "SEO scan complete",
        extra={
            "scan_type": scan_type,
            "keywords_checked": len(signals),
            "not_found": len(not_found),
        },
    )

    return {"keywords_checked": len(signals), "signals": signals, "not_found": not_found}
