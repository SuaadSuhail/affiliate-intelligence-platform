"""
Rulebook Tests
==============
Boundary tests for src/rulebook/recommend.py — the single deterministic
source of truth for churn/growth risk tiers and recommendations.

Run:
    pytest tests/test_rulebook.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rulebook.recommend import (
    CHURN_AT_RISK_THRESHOLD,
    CHURN_CRITICAL_THRESHOLD,
    GROWTH_HIGH_THRESHOLD,
    Recommendation,
    categorize,
    recommend,
)


def _aff(churn: float, growth: float):
    return SimpleNamespace(churn_risk_score=churn, growth_potential_score=growth)


# ─── categorize() — churn boundaries ──────────────────────────────────────────

def test_categorize_churn_just_below_at_risk_threshold_is_active():
    """0.49 is below the 0.50 at-risk cut and below the 0.50 growth cut → active."""
    assert categorize(CHURN_AT_RISK_THRESHOLD - 0.01, 0.0) == "active"


def test_categorize_churn_at_at_risk_threshold_is_active():
    """Threshold comparisons are strict '>' — exactly 0.50 does not qualify."""
    assert categorize(CHURN_AT_RISK_THRESHOLD, 0.0) == "active"


def test_categorize_churn_just_above_at_risk_threshold_is_at_risk():
    assert categorize(CHURN_AT_RISK_THRESHOLD + 0.01, 0.0) == "at_risk"


def test_categorize_churn_just_below_critical_threshold_is_at_risk():
    assert categorize(CHURN_CRITICAL_THRESHOLD - 0.01, 0.0) == "at_risk"


def test_categorize_churn_at_critical_threshold_is_at_risk():
    """Exactly 0.80 does not qualify as churned — strict '>' semantics."""
    assert categorize(CHURN_CRITICAL_THRESHOLD, 0.0) == "at_risk"


def test_categorize_churn_just_above_critical_threshold_is_churned():
    assert categorize(CHURN_CRITICAL_THRESHOLD + 0.01, 0.0) == "churned"


# ─── categorize() — growth boundaries (churn held low so growth tier can win) ─

def test_categorize_growth_just_below_high_threshold_is_active():
    assert categorize(0.0, GROWTH_HIGH_THRESHOLD - 0.01) == "active"


def test_categorize_growth_at_high_threshold_is_active():
    assert categorize(0.0, GROWTH_HIGH_THRESHOLD) == "active"


def test_categorize_growth_just_above_high_threshold_is_high_growth():
    assert categorize(0.0, GROWTH_HIGH_THRESHOLD + 0.01) == "high_growth"


def test_categorize_churn_at_risk_takes_precedence_over_high_growth():
    """A struggling-but-growing affiliate is flagged at_risk first, not high_growth."""
    assert categorize(CHURN_AT_RISK_THRESHOLD + 0.01, GROWTH_HIGH_THRESHOLD + 0.01) == "at_risk"


def test_categorize_handles_none_as_zero():
    assert categorize(None, None) == "active"


# ─── recommend() — end-to-end per tier ────────────────────────────────────────

def test_recommend_healthy_tier():
    aff = _aff(0.2, 0.2)
    rec = recommend(aff, features=None, leaks=None)
    assert isinstance(rec, Recommendation)
    assert rec.reason_code == "active"
    assert "Healthy" in rec.recommendation
    assert any("churn_risk_score=0.20" in e for e in rec.evidence)


def test_recommend_at_risk_tier():
    aff = _aff(0.6, 0.1)
    rec = recommend(aff, features=None, leaks=None)
    assert rec.reason_code == "at_risk"
    assert "Monitor" in rec.recommendation


def test_recommend_churned_tier_is_urgent():
    aff = _aff(0.95, 0.0)
    rec = recommend(aff, features=None, leaks=None)
    assert rec.reason_code == "churned"
    assert "URGENT" in rec.recommendation


def test_recommend_high_growth_tier():
    aff = _aff(0.1, 0.75)
    rec = recommend(aff, features=None, leaks=None)
    assert rec.reason_code == "high_growth"
    assert "Growth opportunity" in rec.recommendation


def test_recommend_missing_scores_default_to_midpoint():
    """None scores must not crash — they fall back to 0.5, same as the rest of the codebase."""
    aff = _aff(None, None)
    rec = recommend(aff, features=None, leaks=None)
    assert rec.reason_code == "active"


# ─── recommend() — feature evidence enrichment ────────────────────────────────

def test_recommend_includes_nonzero_feature_evidence():
    aff = _aff(0.6, 0.1)
    features = {
        "days_since_contact": 35,
        "churn_signal_count": 2,
        "competitor_mention_count": 0,  # zero — should NOT appear in evidence
        "escalation_count": 1,
        "comm_count_30d": 0,  # zero — should NOT appear in evidence
    }
    rec = recommend(aff, features, leaks=None)
    joined = " | ".join(rec.evidence)
    assert "days_since_contact=35" in joined
    assert "churn_signal_count=2" in joined
    assert "escalation_count=1" in joined
    assert "competitor_mention_count" not in joined
    assert "comm_count_30d" not in joined


# ─── recommend() — active leak present ────────────────────────────────────────

def test_recommend_leak_does_not_change_tier_of_healthy_affiliate():
    """Tier selection is a pure function of churn/growth scores only — a leak
    must not move a healthy affiliate into a different tier. It still shows up
    in reason_code (suffix) and evidence, kept separate from and visible
    alongside the score rather than folded into it."""
    aff = _aff(0.1, 0.1)
    leaks = [SimpleNamespace(code="SAVE20"), SimpleNamespace(code="SAVE20")]

    rec_no_leak = recommend(aff, features=None, leaks=None)
    rec_with_leak = recommend(aff, features=None, leaks=leaks)

    assert rec_no_leak.reason_code == "active"
    assert rec_with_leak.reason_code == "active_leak_detected"
    # Same underlying tier — recommendation text (tier-derived) is unchanged.
    assert rec_with_leak.recommendation == rec_no_leak.recommendation
    assert "Healthy" in rec_with_leak.recommendation
    assert any("unauthorized promo-code leak" in e for e in rec_with_leak.evidence)
    assert any("SAVE20" in e for e in rec_with_leak.evidence)
    assert not any("leak" in e for e in rec_no_leak.evidence)


def test_recommend_leak_evidence_deduplicates_codes():
    aff = _aff(0.1, 0.1)
    leaks = [SimpleNamespace(code="DUP10"), SimpleNamespace(code="DUP10")]
    rec = recommend(aff, features=None, leaks=leaks)
    leak_evidence = [e for e in rec.evidence if "leak" in e]
    assert len(leak_evidence) == 1
    assert leak_evidence[0].count("DUP10") == 1


def test_recommend_leak_on_already_urgent_affiliate_appends_suffix_not_override():
    """A leak on a churned affiliate stays urgent — reason_code gains a suffix,
    not a different tier — since 'churned' already implies human attention."""
    aff = _aff(0.95, 0.0)
    leaks = [{"code": "LEAK1"}]
    rec = recommend(aff, features=None, leaks=leaks)
    assert rec.reason_code == "churned_leak_detected"
    assert "URGENT" in rec.recommendation


def test_recommend_accepts_leak_dicts_not_just_orm_objects():
    aff = _aff(0.1, 0.1)
    leaks = [{"code": "DICTCODE"}]
    rec = recommend(aff, features=None, leaks=leaks)
    assert any("DICTCODE" in e for e in rec.evidence)


def test_recommend_empty_leak_list_behaves_like_none():
    aff = _aff(0.1, 0.1)
    rec_none = recommend(aff, features=None, leaks=None)
    rec_empty = recommend(aff, features=None, leaks=[])
    assert rec_none.reason_code == rec_empty.reason_code == "active"
