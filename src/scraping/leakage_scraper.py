"""
Leakage scan orchestrator.

check_leakage() is the main entry point: it loads affiliate promo codes from
the database, fetches each configured site via the Playwright-based fetcher,
extracts code candidates, matches them against affiliate codes, deduplicates
against recent scan history, and persists new LeakedCode rows.

Only leaked_codes rows are written here — no churn_risk_score,
growth_potential_score, health_score, or any other Affiliate scoring field is
touched. Flag now, numeric integration later: the scoring pipeline can consume
LeakedCode rows in a separate step once the detection layer is stable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.core.logging_config import get_logger
from src.scraping.extractor import extract_candidate_codes
from src.scraping.fetcher import RobotsDisallowedError, fetch_html
from src.scraping.matcher import match_candidates_to_affiliates
from src.scraping.site_config import SITES

logger = get_logger(__name__)

_DEDUP_WINDOW_HOURS = 20


def _load_affiliate_codes(db: Session) -> dict[str, str]:
    """
    Return {str(affiliate_id): active_promo_code} for all affiliates that
    have a non-null active_promo_code.
    """
    from src.storage.models import Affiliate  # local import — avoids hard import-time dep
    rows = (
        db.query(Affiliate.id, Affiliate.active_promo_code)
        .filter(Affiliate.active_promo_code.isnot(None))
        .all()
    )
    return {str(row.id): row.active_promo_code for row in rows}


def _already_recorded(
    db: Session,
    affiliate_id: str,
    code: str,
    site_name: str,
) -> bool:
    """
    Return True if (affiliate_id, code, site_name) already exists in
    leaked_codes with found_at within the last _DEDUP_WINDOW_HOURS.
    """
    from src.storage.models import LeakedCode  # local import
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_DEDUP_WINDOW_HOURS)
    return (
        db.query(LeakedCode)
        .filter(
            LeakedCode.affiliate_id == uuid.UUID(affiliate_id),
            LeakedCode.code == code,
            LeakedCode.site == site_name,
            LeakedCode.found_at >= cutoff,
        )
        .count()
        > 0
    )


def check_leakage(
    db: Session,
    scan_type: str = "scheduled",
    affiliate_id: str | None = None,
) -> dict:
    """
    Run a full leakage scan across all configured sites.

    For each site: fetch HTML → extract candidates → match against affiliate
    promo codes → deduplicate → persist new LeakedCode rows.

    Only leaked_codes rows are written — no churn/growth/health scores are
    touched. Flag now, numeric integration later.

    Parameters
    ----------
    db           : Active SQLAlchemy session. This function commits once at the
                   end; the caller should not commit before returning.
    scan_type    : "scheduled" | "on_demand" — stored on each LeakedCode row.
    affiliate_id : If given, filter results to this one affiliate after matching.
                   Every site is still fetched — there is no cheaper path when
                   we cannot know in advance which site lists which affiliate's code.

    Returns
    -------
    {
        "sites_checked": int,
        "sites_failed":  [{"site": str, "reason": str}, ...],
        "new_leaks":     [{"affiliate_id", "code", "site", "source_url",
                           "found_at"}, ...],
        "scan_type":     str,
    }
    """
    from src.storage.models import LeakedCode  # local import

    affiliate_codes = _load_affiliate_codes(db)

    if not affiliate_codes:
        logger.info("no affiliates have an active_promo_code — scan skipped")
        return {"sites_checked": 0, "sites_failed": [], "new_leaks": [], "scan_type": scan_type}

    if affiliate_id:
        affiliate_codes = {k: v for k, v in affiliate_codes.items() if k == affiliate_id}
        if not affiliate_codes:
            logger.warning(
                "requested affiliate has no active_promo_code — scan skipped",
                extra={"affiliate_id": affiliate_id},
            )
            return {"sites_checked": 0, "sites_failed": [], "new_leaks": [], "scan_type": scan_type}

    logger.info(
        "leakage scan starting",
        extra={
            "sites": len(SITES),
            "affiliates_with_codes": len(affiliate_codes),
            "scan_type": scan_type,
        },
    )

    sites_checked = 0
    sites_failed: list[dict] = []
    new_leaks: list[dict] = []

    for site in SITES:
        # Wrap each site fetch individually — one broken/disallowed site must
        # not abort the scan for every other site in the loop.
        try:
            html = fetch_html(site.url, kind=site.kind)
        except RobotsDisallowedError:
            logger.warning(
                "site disallowed by robots.txt — skipping",
                extra={"site": site.name},
            )
            sites_failed.append({"site": site.name, "reason": "robots_disallowed"})
            continue
        except Exception as exc:
            logger.warning(
                "site fetch failed — skipping",
                extra={"site": site.name, "reason": str(exc)[:200]},
            )
            sites_failed.append({"site": site.name, "reason": str(exc)[:200]})
            continue

        candidates = extract_candidate_codes(html, site)
        matches = match_candidates_to_affiliates(
            candidates, affiliate_codes, site.name, site.url
        )

        for match in matches:
            if _already_recorded(db, match.affiliate_id, match.affiliate_code, match.site_name):
                logger.debug(
                    "duplicate leak within dedup window — skipping",
                    extra={
                        "site": site.name,
                        "code": match.affiliate_code,
                        "affiliate_id": match.affiliate_id,
                    },
                )
                continue

            now = datetime.now(timezone.utc)
            leaked = LeakedCode(
                affiliate_id=uuid.UUID(match.affiliate_id),
                code=match.affiliate_code,
                site=match.site_name,
                source_url=match.source_url,
                raw_snippet=match.raw_snippet,
                scan_type=scan_type,
                found_at=now,
            )
            db.add(leaked)
            db.flush()  # persist row so _already_recorded sees it within the same run

            new_leaks.append({
                "affiliate_id": match.affiliate_id,
                "code": match.affiliate_code,
                "site": match.site_name,
                "source_url": match.source_url,
                "found_at": leaked.found_at.isoformat(),
            })

        sites_checked += 1

    db.commit()

    logger.info(
        "leakage scan complete",
        extra={
            "scan_type": scan_type,
            "sites_checked": sites_checked,
            "sites_failed": len(sites_failed),
            "new_leaks": len(new_leaks),
        },
    )

    return {
        "sites_checked": sites_checked,
        "sites_failed": sites_failed,
        "new_leaks": new_leaks,
        "scan_type": scan_type,
    }