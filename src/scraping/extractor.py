"""
Code extractor for the promo leakage detector.

Two extraction paths run in sequence for every page:

1. Selector path  — uses the CSS selectors declared in SiteConfig to find
   code elements precisely.  Yields high-confidence hits with merchant context.

2. Regex fallback — scans all visible text for code-shaped tokens regardless
   of page structure.  Catches codes the selectors missed; deduplicates
   against selector hits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from src.core.logging_config import get_logger
from src.scraping.site_config import SiteConfig

logger = get_logger(__name__)

# Tokens that look like promo codes: uppercase letters/digits/hyphens, 4–15 chars,
# containing at least one digit or hyphen OR being 8+ chars long.
# Rejects short all-letter words like "SALE", "FREE", "SAVE" while keeping
# "SAVE15", "TESTLEAK20", "SHIPFREE" (8 chars), "CSRLEAK99", "TOMB-EXCL20", etc.
#
# Uses negative lookarounds instead of \b because \b fires at every hyphen boundary
# (hyphens are \W), which causes two failure modes with hyphenated codes:
#   - "xTOMB-EXCL20": \b fires at the interior -→E boundary, returning only "EXCL20"
#   - "TOMB-EXCL20x": \b fires after the interior -, returning truncated "TOMB-"
# The lookaround defines boundaries using the token's own character class [A-Z0-9\-],
# so lowercase letters (outside the class) correctly delimit the token edges.
_CODE_PATTERN = re.compile(
    r'(?<![A-Z0-9\-])([A-Z0-9][A-Z0-9\-]{3,14})(?![A-Z0-9\-])'
)

_SNIPPET_MAX = 300   # chars of surrounding HTML kept per hit


def _looks_like_code(token: str) -> bool:
    """Return True if token passes the promo-code shape heuristic."""
    if len(token) < 4 or len(token) > 15:
        return False
    has_digit_or_hyphen = any(c.isdigit() or c == "-" for c in token)
    return has_digit_or_hyphen or len(token) >= 8


def _nearest_merchant(element: Tag, merchant_selectors: list[str]) -> str:
    """
    Walk up the DOM from element looking for a sibling or ancestor that
    matches any merchant selector.  Returns stripped text or "" if not found.
    """
    for selector in merchant_selectors:
        # Search within the element's parent container first (sibling proximity)
        parent = element.parent
        while parent:
            hit = parent.select_one(selector)
            if hit and hit is not element:
                return hit.get_text(strip=True)
            parent = parent.parent
    return ""


def _selector_path(
    soup: BeautifulSoup,
    site: SiteConfig,
) -> list[tuple[str, str, str]]:
    """
    Apply site.code_selectors to soup.

    Returns list of (code, merchant_context, snippet) tuples.
    """
    results: list[tuple[str, str, str]] = []
    for selector in site.code_selectors:
        for el in soup.select(selector):
            raw_code = el.get_text(strip=True).upper()
            if not raw_code:
                continue
            merchant = _nearest_merchant(el, site.merchant_selectors)
            snippet = str(el)[:_SNIPPET_MAX]
            results.append((raw_code, merchant, snippet))
    return results


def _regex_fallback(
    soup: BeautifulSoup,
    already_found: set[str],
) -> list[tuple[str, str, str]]:
    """
    Scan all visible text for code-shaped tokens not already found by selectors.

    Returns list of (code, merchant_context="", snippet) tuples.
    """
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set(already_found)

    # Scan body only — page <title> and other <head> elements contain prose that
    # can produce false positives (e.g. "JS-Rendered" in a title yields "JS-R").
    body = soup.body or soup
    visible_text = body.get_text(separator=" ")

    for match in _CODE_PATTERN.finditer(visible_text):
        token = match.group(1).upper()
        if token in seen:
            continue
        if not _looks_like_code(token):
            continue
        seen.add(token)
        start = max(0, match.start() - 60)
        end = min(len(visible_text), match.end() + 60)
        snippet = visible_text[start:end].strip()
        results.append((token, "", snippet))

    return results


@dataclass(frozen=True)
class CandidateCode:
    code: str
    merchant_context: str   # nearby merchant/store name if found, else ""
    snippet: str            # raw HTML element or surrounding text for audit trail


def extract_candidate_codes(html: str, site: SiteConfig) -> list[CandidateCode]:
    """
    Extract promo code candidates from rendered HTML using a two-pass approach.

    Pass 1 — CSS selector path: precise, site-specific, includes merchant context.
    Pass 2 — Regex fallback: broad sweep of visible text for anything missed.

    Deduplication is applied across both passes (by uppercased code value).

    Parameters
    ----------
    html : Fully rendered HTML string (from fetch_html).
    site : SiteConfig for the page being scanned (provides selectors).

    Returns
    -------
    Deduplicated list of CandidateCode, selector hits first.
    """
    soup = BeautifulSoup(html, "lxml")

    selector_hits = _selector_path(soup, site)
    selector_codes = {code for code, _, _ in selector_hits}

    fallback_hits = _regex_fallback(soup, already_found=selector_codes)

    logger.info(
        "extraction complete",
        extra={
            "site": site.name,
            "selector_hits": len(selector_hits),
            "fallback_hits": len(fallback_hits),
            "total": len(selector_hits) + len(fallback_hits),
        },
    )

    candidates = [
        CandidateCode(code=code, merchant_context=merchant, snippet=snippet)
        for code, merchant, snippet in selector_hits + fallback_hits
    ]
    return candidates
