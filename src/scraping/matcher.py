"""
Matcher for the promo leakage detector.

Compares extracted CandidateCode objects against known affiliate promo codes.
Matching is exact and case-normalised only — no fuzzy matching. A promo code
is an identifier; approximate matching would mean guessing at leaks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.logging_config import get_logger
from src.scraping.extractor import CandidateCode

logger = get_logger(__name__)


@dataclass(frozen=True)
class LeakMatch:
    affiliate_id: str
    affiliate_code: str
    site_name: str
    source_url: str
    raw_snippet: str


def match_candidates_to_affiliates(
    candidates: list[CandidateCode],
    affiliate_codes: dict[str, str],
    site_name: str,
    source_url: str,
) -> list[LeakMatch]:
    """
    Match extracted code candidates against known affiliate promo codes.

    Matching is exact and case-normalised (both sides uppercased/stripped).

    Parameters
    ----------
    candidates      : CandidateCodes from extractor.extract_candidate_codes().
    affiliate_codes : {affiliate_id: active_promo_code} for active affiliates.
    site_name       : name of the site being scanned (stored on LeakMatch).
    source_url      : URL of the scanned page (stored on LeakMatch for audit).

    Returns
    -------
    One LeakMatch per matched candidate. Empty list if no matches.
    """
    # Reverse lookup: normalised code -> affiliate_id
    normalised_lookup: dict[str, str] = {
        code.strip().upper(): aff_id
        for aff_id, code in affiliate_codes.items()
        if code
    }

    matches: list[LeakMatch] = []
    for candidate in candidates:
        normalised = candidate.code.strip().upper()
        aff_id = normalised_lookup.get(normalised)
        if aff_id is not None:
            matches.append(LeakMatch(
                affiliate_id=aff_id,
                affiliate_code=normalised,
                site_name=site_name,
                source_url=source_url,
                raw_snippet=candidate.snippet,
            ))

    if matches:
        logger.info(
            "leak matches found",
            extra={
                "site": site_name,
                "count": len(matches),
                "codes": [m.affiliate_code for m in matches],
            },
        )

    return matches