"""
Rulebook — deterministic recommendation layer
===============================================
Single source of truth for "what should we recommend for this affiliate,
and why." Plain code, not an LLM call — every function here is pure
(no DB session, no I/O, no network) so it can be unit tested exhaustively
against boundary values.

This module replaces THREE previously-inconsistent inline implementations
of the same "is this affiliate at risk / high growth" judgment:
  - src/agent/tools.py  get_affiliate_summary()   (churn >= 0.65 / 0.45, growth >= 0.70)
  - src/agent/tools.py  get_portfolio_health()     (churn > 0.5, growth > 0.5, churn > 0.8)
  - src/ml/score_updater.py update_all_scores()    (churn > 0.5, growth > 0.5)

Canonical thresholds
---------------------
CHURN_CRITICAL_THRESHOLD = 0.80
CHURN_AT_RISK_THRESHOLD  = 0.50
GROWTH_HIGH_THRESHOLD    = 0.50

Why these specific numbers: 0.80 already had unanimous agreement across
every call site that used a "churned/critical" cut (get_portfolio_health's
`churned`, score_updater's — indirectly, via the affiliate_status enum —
and the agent's old SYSTEM_PROMPT text), so it carried over unchanged.
For the second cut, get_portfolio_health and score_updater.py — 2 of the
3 conflicting call sites — already agreed on 0.50 for both "at risk" and
"high growth"; get_affiliate_summary was the outlier (0.65/0.45/0.70).
0.50 wins as the majority value and also reads naturally on a 0-1
probability scale ("more likely than not"). get_affiliate_summary's
four-tier action text is preserved (urgent / growth opportunity / monitor
/ healthy) but now keyed off these two shared cutoffs instead of its own
three private ones — nothing needed a fourth tier, "monitor" and "at risk"
were always the same concept wearing different names.

This is now the ONLY place in the codebase that should contain a
churn/growth risk threshold. If you're about to write `> 0.5` or `> 0.8`
against churn_risk_score / growth_potential_score anywhere else, import
`categorize()` or `recommend()` from here instead.

src/ingestion/etl_pipeline.py `_derive_status()` imports these same three
constants for its own ingest-time status assignment, rather than keeping a
private copy — see that file for why it stays a separate function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ─── Canonical thresholds — the single source of truth ───────────────────────

CHURN_CRITICAL_THRESHOLD = 0.80
CHURN_AT_RISK_THRESHOLD = 0.50
GROWTH_HIGH_THRESHOLD = 0.50

_TIER_RECOMMENDATION_TEXT = {
    "churned": "URGENT: Schedule a retention call within 48 hours.",
    "at_risk": "Monitor: Follow up within 7 days.",
    "high_growth": "Growth opportunity: Propose a scale-up plan.",
    "active": "Healthy: Maintain regular check-in cadence.",
}

# Feature keys worth surfacing as evidence when present and non-zero.
_EVIDENCE_FEATURE_KEYS = (
    "days_since_contact",
    "churn_signal_count",
    "competitor_mention_count",
    "escalation_count",
    "comm_count_30d",
)


@dataclass(frozen=True)
class Recommendation:
    """The output of the rulebook: an action plus the facts that produced it."""

    recommendation: str
    reason_code: str
    evidence: list[str] = field(default_factory=list)


# ─── Shared tier classification ───────────────────────────────────────────────

def categorize(churn_risk_score: float, growth_potential_score: float) -> str:
    """
    Classify an affiliate into exactly one mutually-exclusive tier:
    'churned' | 'at_risk' | 'high_growth' | 'active'.

    Mirrors the affiliate_status enum values (src/storage/models.py) and the
    precedence order used by _derive_status() in etl_pipeline.py — churn
    risk is checked before growth potential, so a struggling-but-growing
    affiliate is still flagged as at-risk first.
    """
    churn = 0.0 if churn_risk_score is None else float(churn_risk_score)
    growth = 0.0 if growth_potential_score is None else float(growth_potential_score)

    if churn > CHURN_CRITICAL_THRESHOLD:
        return "churned"
    if churn > CHURN_AT_RISK_THRESHOLD:
        return "at_risk"
    if growth > GROWTH_HIGH_THRESHOLD:
        return "high_growth"
    return "active"


def _leak_code(leak) -> str:
    """Extract a display code from a LeakedCode ORM row or a plain dict."""
    if isinstance(leak, dict):
        return str(leak.get("code", "?"))
    return str(getattr(leak, "code", "?"))


# ─── Full recommendation with evidence bundle ────────────────────────────────

def recommend(affiliate, features: dict | None, leaks: list | None) -> Recommendation:
    """
    Pure function: given an affiliate's current scores, its feature vector,
    and any recorded promo-code leaks, return a Recommendation.

    Parameters
    ----------
    affiliate : an object with .churn_risk_score and .growth_potential_score
                attributes (e.g. src.storage.models.Affiliate). Only those
                two attributes are read — no DB access happens here.
    features  : dict from src.ml.feature_engineering.build_feature_vector(),
                or None if unavailable. Used only to enrich the evidence
                list; scoring/categorization does not depend on it.
    leaks     : list of LeakedCode rows (or dicts with a 'code' key), or
                None/[] if there are none on record. A leak is a separate,
                visible signal, not an input to the score — see below.

    Returns
    -------
    Recommendation(recommendation, reason_code, evidence)
    """
    churn = 0.5 if getattr(affiliate, "churn_risk_score", None) is None else float(affiliate.churn_risk_score)
    growth = 0.5 if getattr(affiliate, "growth_potential_score", None) is None else float(affiliate.growth_potential_score)

    # Tier is a pure function of churn/growth only — leaks (and any other
    # signal) are kept separate from and visible alongside the score, per
    # the target architecture, rather than folded into it.
    tier = categorize(churn, growth)
    reason_code = tier

    evidence: list[str] = [
        f"churn_risk_score={churn:.2f} (at-risk threshold {CHURN_AT_RISK_THRESHOLD:.2f}, "
        f"critical threshold {CHURN_CRITICAL_THRESHOLD:.2f})",
        f"growth_potential_score={growth:.2f} (high-growth threshold {GROWTH_HIGH_THRESHOLD:.2f})",
    ]

    if features:
        for key in _EVIDENCE_FEATURE_KEYS:
            value = features.get(key)
            if value:
                evidence.append(f"{key}={value}")

    if leaks:
        codes = sorted({_leak_code(leak) for leak in leaks})
        evidence.append(
            f"{len(leaks)} unauthorized promo-code leak(s) on record: {', '.join(codes)}"
        )
        reason_code = f"{reason_code}_leak_detected"

    recommendation_text = _TIER_RECOMMENDATION_TEXT[tier]

    return Recommendation(
        recommendation=recommendation_text,
        reason_code=reason_code,
        evidence=evidence,
    )
