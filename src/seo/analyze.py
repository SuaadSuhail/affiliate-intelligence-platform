"""
SEO signal analysis — pure, deterministic derivation of a search_trend flag
from an affiliate's rank-check history.

No I/O, no DB session — mirrors src.rulebook.recommend's purity. Kept
deliberately separate from src.rulebook.recommend: this is a
signal-detection function (like src.scraping.leakage_scraper.check_leakage),
not a decision function. Do not import this from recommend.py, and do not
let search_trend feed into categorize()/recommend()'s tier calculation —
same principle as the leak signal (see recommend.py's own docstring).
"""

from __future__ import annotations

# rank_change convention: previous_rank - current_rank. Positive means the
# keyword moved to a better (lower-number) position; negative means it
# dropped. These thresholds are a starting judgment call, same spirit as
# src.rulebook.recommend's canonical thresholds — a single source of truth
# for "what counts as declining/improving" for this signal.
DECLINING_THRESHOLD = -3
IMPROVING_THRESHOLD = 3


def _field(signal, name: str):
    """Read a field from either an ORM SeoSignal row or a plain dict —
    same dual-mode pattern as src.rulebook.recommend._leak_code."""
    if isinstance(signal, dict):
        return signal.get(name)
    return getattr(signal, name, None)


def derive_search_trend(signals: list) -> str:
    """
    Classify an affiliate's search_trend as 'declining' | 'stable' |
    'improving' from their seo_signals history.

    Uses only the most recently checked signal's rank_change — rank_change
    already encodes movement since the prior check, so looking further back
    would double-count the same delta. No signals, or a most-recent signal
    with rank_change=None (no prior rank was on record at that check), both
    resolve to 'stable' — there's no evidence of movement yet either way.
    """
    if not signals:
        return "stable"

    latest = max(signals, key=lambda s: _field(s, "checked_at"))
    rank_change = _field(latest, "rank_change")

    if rank_change is None:
        return "stable"
    if rank_change <= DECLINING_THRESHOLD:
        return "declining"
    if rank_change >= IMPROVING_THRESHOLD:
        return "improving"
    return "stable"
