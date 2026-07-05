"""
ML Module Tests
===============
Tests for feature engineering, rule-based scorers, score updater,
and SHAP explainability.

Run:
    pytest tests/test_ml.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ─── Test 1: build_feature_vector returns dict with all 12 features ───────────

def test_build_feature_vector_has_all_features():
    """build_feature_vector must return a dict with all 12 FEATURE_NAMES keys."""
    from src.ml.feature_engineering import build_feature_vector, FEATURE_NAMES
    from src.storage.models import Affiliate, Communication

    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.name = "Test Affiliate"
    aff.email = "test@example.com"
    aff.tier = "gold"
    aff.monthly_revenue = 12000.0
    aff.churn_risk_score = 0.4
    aff.growth_potential_score = 0.6
    aff.health_score = 60.0
    aff.last_contact_date = datetime.now(timezone.utc)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = aff
    mock_db.query.return_value.filter.return_value.all.return_value = []

    result = build_feature_vector(str(aff.id), mock_db)

    assert "affiliate_id" in result
    assert "affiliate_name" in result
    assert "status" in result

    for feature in FEATURE_NAMES:
        assert feature in result, f"Missing feature: {feature}"

    assert isinstance(result["days_since_contact"], (int, float))
    assert isinstance(result["avg_sentiment_30d"], float)
    assert isinstance(result["comm_count_30d"], (int, float))


# ─── Test 2: calculate_churn_risk_rules — high score for risk signals ─────────

def test_calculate_churn_risk_rules_high_risk():
    """Rule-based churn scorer must return a high score when many risk signals fire."""
    from src.ml.churn_model import calculate_churn_risk_rules

    high_risk_features = {
        "days_since_contact": 35,       # > 30 → +0.35
        "ctr_trend_pct": -3.0,          # < -2.0 → +0.25
        "churn_signal_count": 2,        # >= 2 → +0.25
        "competitor_mention_count": 1,  # >= 1 → +0.20
        "escalation_count": 1,          # >= 1 → +0.15
        "avg_sentiment_30d": -0.5,      # < -0.4 → +0.20
        "comm_count_30d": 0,            # == 0 → +0.15
        "positive_signal_count": 0,
        "revenue_30d": 500.0,
        "sentiment_trend": -0.2,
        "response_rate": 0.1,
        "days_since_positive": 40,
    }
    score = calculate_churn_risk_rules(high_risk_features)
    assert score >= 0.7, f"Expected high churn risk (≥0.7), got {score}"
    assert 0.0 <= score <= 1.0


# ─── Test 3: calculate_growth_potential_rules — high score for growth signals ─

def test_calculate_growth_potential_rules_high_growth():
    """Rule-based growth scorer must return a high score when growth signals fire."""
    from src.ml.growth_model import calculate_growth_potential_rules

    high_growth_features = {
        "positive_signal_count": 4,  # >= 3 → +0.30
        "avg_sentiment_30d": 0.6,    # > 0.4 → +0.25
        "comm_count_30d": 5,         # >= 3 → +0.15
        "revenue_30d": 25000.0,      # > 20000 → +0.20
        "ctr_trend_pct": 3.0,        # > 2.0 → +0.20
        "sentiment_trend": 0.3,      # > 0.1 → +0.10
        "days_since_contact": 2,
        "churn_signal_count": 0,
        "escalation_count": 0,
        "competitor_mention_count": 0,
        "response_rate": 0.8,
        "days_since_positive": 1,
    }
    score = calculate_growth_potential_rules(high_growth_features)
    assert score >= 0.7, f"Expected high growth potential (≥0.7), got {score}"
    assert 0.0 <= score <= 1.0


# ─── Test 4: update_all_scores updates all affiliates ─────────────────────────

def _mock_score_updater_db(affiliates):
    """Shared mock for update_all_scores(): dispatches db.query() by model.
    No affiliate has any recorded leaks — that path is covered independently
    by tests/test_rulebook.py's leak-specific cases."""
    from src.storage.models import Affiliate, LeakedCode, ScoreHistory

    mock_db = MagicMock()

    def query_side_effect(model):
        q = MagicMock()
        if model is Affiliate:
            q.all.return_value = affiliates
        elif model is ScoreHistory:
            # No existing score today → all affiliates need scoring
            filt = MagicMock()
            filt.first.return_value = None
            q.filter.return_value = filt
        elif model is LeakedCode:
            # db.query(LeakedCode).filter(...).order_by(...).limit(...).all()
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            chain.limit.return_value = chain
            chain.all.return_value = []
            q.filter.return_value = chain
        return q

    mock_db.query.side_effect = query_side_effect
    mock_db.add = MagicMock()
    return mock_db


def test_update_all_scores_updates_affiliates():
    """update_all_scores must score every affiliate and return correct counts."""
    from src.ml.score_updater import update_all_scores
    from src.storage.models import Affiliate

    def _make_aff(churn=0.4, growth=0.6):
        a = Affiliate()
        a.id = uuid.uuid4()
        a.name = "Test"
        a.status = "active"
        a.churn_risk_score = churn
        a.growth_potential_score = growth
        a.health_score = 60.0
        a.revenue_30d = 5000.0
        a.ctr_trend_pct = 0.0
        a.last_contact_at = datetime.now(timezone.utc)
        a.days_since_contact = 0
        return a

    aff1 = _make_aff(churn=0.3, growth=0.7)
    aff2 = _make_aff(churn=0.8, growth=0.2)

    mock_db = _mock_score_updater_db([aff1, aff2])

    with patch("src.ml.score_updater.build_feature_vector") as mock_fv, \
         patch("src.ml.score_updater.calculate_churn_risk_rules") as mock_churn, \
         patch("src.ml.score_updater.calculate_growth_potential_rules") as mock_growth:

        mock_fv.return_value = {f: 0.0 for f in [
            "days_since_contact", "revenue_30d", "ctr_trend_pct",
            "avg_sentiment_30d", "comm_count_30d", "churn_signal_count",
            "positive_signal_count", "escalation_count", "competitor_mention_count",
            "sentiment_trend", "response_rate", "days_since_positive",
        ]}
        mock_churn.side_effect = [0.3, 0.8]
        mock_growth.side_effect = [0.7, 0.2]

        result = update_all_scores(mock_db)

    assert result["affiliates_scored"] == 2
    assert "avg_health_score" in result
    assert "at_risk_count" in result
    assert "high_growth_count" in result
    # 2 affiliates x 2 add() calls each (audit_log entry + score_history row).
    assert mock_db.add.call_count == 4


# ─── Test 4b: update_all_scores persists the rulebook's evidence bundle ───────

def test_update_all_scores_persists_evidence_bundle():
    """Each score_history row added must carry a non-null evidence_bundle
    equal to what src.rulebook.recommend.recommend() produces for that
    affiliate's churn/growth scores and feature vector."""
    from src.ml.score_updater import update_all_scores
    from src.storage.models import Affiliate
    from src.rulebook.recommend import recommend

    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.name = "Evidence Test"
    aff.status = "active"
    aff.churn_risk_score = 0.4
    aff.growth_potential_score = 0.6
    aff.health_score = 60.0
    aff.revenue_30d = 5000.0
    aff.ctr_trend_pct = 0.0
    aff.last_contact_at = datetime.now(timezone.utc)
    aff.days_since_contact = 35  # non-zero so it shows up in evidence

    mock_db = _mock_score_updater_db([aff])

    features = {
        "days_since_contact": 35, "revenue_30d": 5000.0, "ctr_trend_pct": 0.0,
        "avg_sentiment_30d": -0.5, "comm_count_30d": 0, "churn_signal_count": 2,
        "positive_signal_count": 0, "escalation_count": 1,
        "competitor_mention_count": 1, "sentiment_trend": 0.0,
        "response_rate": 0.1, "days_since_positive": 40,
    }

    with patch("src.ml.score_updater.build_feature_vector") as mock_fv, \
         patch("src.ml.score_updater.calculate_churn_risk_rules") as mock_churn, \
         patch("src.ml.score_updater.calculate_growth_potential_rules") as mock_growth:

        mock_fv.return_value = features
        mock_churn.return_value = 0.9
        mock_growth.return_value = 0.1

        update_all_scores(mock_db)

    # Two db.add() calls per scored affiliate now: the audit_log entry
    # (written first) and the score_history row (written last) — see
    # test_update_all_scores_writes_audit_entry below for the audit half.
    assert mock_db.add.call_count == 2
    entry = mock_db.add.call_args[0][0]
    assert entry.evidence_bundle is not None

    # aff now carries the scores update_all_scores just assigned to it —
    # recompute recommend() independently with the same inputs and compare.
    expected = recommend(aff, features, leaks=None)
    assert entry.evidence_bundle == expected.evidence


# ─── Test 4c2: update_all_scores writes an audit_log entry per affiliate ──────

def test_update_all_scores_writes_audit_entry():
    """Each scored affiliate must get one audit_log entry (stage='rulebook',
    rule_or_tool='recommend') whose input_snapshot is the exact feature dict
    passed to recommend() and output_snapshot carries the tier + evidence —
    not just an empty/placeholder row."""
    from src.ml.score_updater import update_all_scores
    from src.storage.models import Affiliate, AuditLog
    from src.rulebook.recommend import categorize

    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.name = "Audit Test"
    aff.status = "active"
    aff.churn_risk_score = 0.4
    aff.growth_potential_score = 0.6
    aff.health_score = 60.0
    aff.revenue_30d = 5000.0
    aff.ctr_trend_pct = 0.0
    aff.last_contact_at = datetime.now(timezone.utc)
    aff.days_since_contact = 10

    mock_db = _mock_score_updater_db([aff])

    features = {
        "days_since_contact": 10, "revenue_30d": 5000.0, "ctr_trend_pct": 0.0,
        "avg_sentiment_30d": 0.1, "comm_count_30d": 2, "churn_signal_count": 0,
        "positive_signal_count": 1, "escalation_count": 0,
        "competitor_mention_count": 0, "sentiment_trend": 0.0,
        "response_rate": 0.5, "days_since_positive": 5,
    }

    with patch("src.ml.score_updater.build_feature_vector") as mock_fv, \
         patch("src.ml.score_updater.calculate_churn_risk_rules") as mock_churn, \
         patch("src.ml.score_updater.calculate_growth_potential_rules") as mock_growth:

        mock_fv.return_value = features
        mock_churn.return_value = 0.4
        mock_growth.return_value = 0.6

        update_all_scores(mock_db)

    audit_entries = [
        call.args[0] for call in mock_db.add.call_args_list if isinstance(call.args[0], AuditLog)
    ]
    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry.stage == "rulebook"
    assert entry.record_type == "affiliate"
    assert entry.record_id == aff.id
    assert entry.rule_or_tool == "recommend"
    assert entry.input_snapshot == features
    assert entry.output_snapshot["tier"] == categorize(0.4, 0.6)
    assert isinstance(entry.output_snapshot["evidence"], list)
    assert len(entry.output_snapshot["evidence"]) > 0


# ─── Test 4c: get_scores does not substitute 0.5 for a real 0.0 score ─────────

def test_get_scores_does_not_substitute_0_5_for_real_zero_score():
    """AffiliateScore must report a real 0.0 churn/growth score as 0.0 — the
    old `a.growth_potential_score or 0.5` pattern wrongly substitutes 0.5
    whenever the real score is 0.0, since 0.0 is falsy in Python."""
    from src.api.routers.ml import get_scores
    from src.storage.models import Affiliate, ScoreHistory

    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.name = "Zero Score Affiliate"
    aff.churn_risk_score = 0.0
    aff.growth_potential_score = 0.0
    aff.health_score = 60.0

    mock_db = MagicMock()

    def query_side_effect(model):
        q = MagicMock()
        if model is Affiliate:
            q.order_by.return_value = q
            q.limit.return_value = q
            q.all.return_value = [aff]
        elif model is ScoreHistory:
            # _latest_evidence_by_affiliate: .filter().order_by().all()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
        return q

    mock_db.query.side_effect = query_side_effect

    result = get_scores(limit=50, db=mock_db)

    assert len(result) == 1
    assert result[0].churn_risk_score == 0.0
    assert result[0].growth_potential_score == 0.0


# ─── Test 4c2: get_scores exposes has_active_leak for both states ────────────

def test_get_scores_exposes_has_active_leak():
    """AffiliateScore must surface has_active_leak with the correct value for
    both a leaked and a non-leaked affiliate — the leak signal must be
    visible wherever the affiliate list is rendered, not just in AffiliateOut."""
    from src.api.routers.ml import get_scores
    from src.storage.models import Affiliate, ScoreHistory

    leaked_aff = Affiliate()
    leaked_aff.id = uuid.uuid4()
    leaked_aff.name = "Leaked Affiliate"
    leaked_aff.churn_risk_score = 0.3
    leaked_aff.growth_potential_score = 0.5
    leaked_aff.health_score = 60.0
    leaked_aff.has_active_leak = True

    clean_aff = Affiliate()
    clean_aff.id = uuid.uuid4()
    clean_aff.name = "Clean Affiliate"
    clean_aff.churn_risk_score = 0.2
    clean_aff.growth_potential_score = 0.6
    clean_aff.health_score = 70.0
    clean_aff.has_active_leak = False

    mock_db = MagicMock()

    def query_side_effect(model):
        q = MagicMock()
        if model is Affiliate:
            q.order_by.return_value = q
            q.limit.return_value = q
            q.all.return_value = [leaked_aff, clean_aff]
        elif model is ScoreHistory:
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
        return q

    mock_db.query.side_effect = query_side_effect

    result = get_scores(limit=50, db=mock_db)

    by_name = {r.name: r for r in result}
    assert by_name["Leaked Affiliate"].has_active_leak is True
    assert by_name["Clean Affiliate"].has_active_leak is False


# ─── Test 4d: dashboard() tier counts agree with categorize() at boundaries ───

def test_dashboard_counts_agree_with_categorize_at_boundaries():
    """dashboard()'s at_risk/high_growth/churned counts must exactly match
    src.rulebook.recommend.categorize() for every affiliate, including at the
    exact threshold boundaries — this is precisely the kind of silent
    disagreement Phase 1 closed at the other call sites; dashboard() had its
    own private `> 0.5` / `> 0.8` copy that was missed then."""
    from src.api.routers.ml import dashboard
    from src.storage.models import Affiliate, ScoreHistory
    from src.rulebook.recommend import (
        CHURN_AT_RISK_THRESHOLD,
        CHURN_CRITICAL_THRESHOLD,
        GROWTH_HIGH_THRESHOLD,
        categorize,
    )

    def _aff(churn, growth):
        a = Affiliate()
        a.id = uuid.uuid4()
        a.name = f"aff-{churn}-{growth}"
        a.churn_risk_score = churn
        a.growth_potential_score = growth
        a.health_score = 50.0
        return a

    boundary_affiliates = [
        _aff(CHURN_AT_RISK_THRESHOLD, 0.0),          # exactly 0.50 churn -> active
        _aff(CHURN_AT_RISK_THRESHOLD + 0.01, 0.0),   # just above -> at_risk
        _aff(CHURN_CRITICAL_THRESHOLD, 0.0),         # exactly 0.80 churn -> at_risk
        _aff(CHURN_CRITICAL_THRESHOLD + 0.01, 0.0),  # just above -> churned
        _aff(0.0, GROWTH_HIGH_THRESHOLD),            # exactly 0.50 growth -> active
        _aff(0.0, GROWTH_HIGH_THRESHOLD + 0.01),     # just above -> high_growth
    ]

    mock_db = MagicMock()

    def query_side_effect(model):
        q = MagicMock()
        if model is Affiliate:
            q.order_by.return_value = q
            q.all.return_value = boundary_affiliates
        elif model is ScoreHistory:
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
        return q

    mock_db.query.side_effect = query_side_effect

    result = dashboard(db=mock_db)

    expected_tiers = [
        categorize(a.churn_risk_score, a.growth_potential_score) for a in boundary_affiliates
    ]
    # Sanity check: lock in the actual tier each boundary case lands in, so
    # this test can't pass just by "dashboard agreeing with itself".
    assert expected_tiers == ["active", "at_risk", "at_risk", "churned", "active", "high_growth"]

    assert result.at_risk_count == sum(1 for t in expected_tiers if t == "at_risk")
    assert result.high_growth_count == sum(1 for t in expected_tiers if t == "high_growth")
    assert result.churned_count == sum(1 for t in expected_tiers if t == "churned")


# ─── Test 5: get_shap_explanation returns top_factors list ────────────────────

def test_get_shap_explanation_structure():
    """get_shap_explanation must return a dict with top_factors list."""
    from src.ml.explainability import get_shap_explanation
    from src.ml.feature_engineering import FEATURE_NAMES

    features = {f: 0.1 for f in FEATURE_NAMES}
    aff_id = str(uuid.uuid4())

    # Without a trained model, should return rule-based placeholder
    result = get_shap_explanation(aff_id, features, "churn")

    assert "affiliate_id" in result
    assert result["affiliate_id"] == aff_id
    assert "model_type" in result
    assert result["model_type"] == "churn"
    assert "prediction" in result
    assert isinstance(result["prediction"], float)
    assert 0.0 <= result["prediction"] <= 1.0
    assert "top_factors" in result
    assert isinstance(result["top_factors"], list)


# ─── Test 6: SHAP failure returns explanation_unavailable, not fake zeros ──────

def test_get_shap_explanation_returns_unavailable_not_zeros_on_shap_failure():
    """
    Root-cause regression for the SHAP/XGBoost base_score incompatibility
    (shap.TreeExplainer raising ValueError: could not convert string to
    float: '[3E-1]' against xgboost>=3.0.0-trained models). A failure
    inside SHAP computation must return an explicit
    explanation_unavailable=True shape with an empty top_factors list and
    base_value=None — never fabricated np.zeros() values indistinguishable
    from "every feature genuinely contributes nothing", which is exactly
    how this bug went unnoticed. prediction must still be real: it comes
    from predict_proba(), which does not depend on SHAP succeeding.
    """
    from src.ml.explainability import get_shap_explanation, _load_model
    from src.ml.feature_engineering import FEATURE_NAMES

    if _load_model("churn") is None:
        pytest.skip("No trained churn model on disk — run POST /ml/train first")

    features = {f: 0.1 for f in FEATURE_NAMES}
    aff_id = str(uuid.uuid4())

    with patch(
        "src.ml.explainability.shap.TreeExplainer",
        side_effect=ValueError("could not convert string to float: '[3E-1]'"),
    ):
        result = get_shap_explanation(aff_id, features, "churn")

    assert result["explanation_unavailable"] is True
    assert result["top_factors"] == []
    assert result["base_value"] is None
    assert isinstance(result["prediction"], float)
    assert 0.0 <= result["prediction"] <= 1.0
    assert result.get("note") and "unavailable" in result["note"].lower()


def test_get_shap_explanation_model_not_trained_also_sets_unavailable_flag():
    """
    The pre-existing "model not trained" fallback also sets
    explanation_unavailable=True (added alongside the SHAP-failure fix) so
    callers — and the frontend — have one single flag to check regardless
    of *why* no real explanation exists.
    """
    from src.ml.explainability import get_shap_explanation
    from src.ml.feature_engineering import FEATURE_NAMES

    features = {f: 0.1 for f in FEATURE_NAMES}
    aff_id = str(uuid.uuid4())

    with patch("src.ml.explainability._load_model", return_value=None):
        result = get_shap_explanation(aff_id, features, "churn")

    assert result["explanation_unavailable"] is True
    assert result["top_factors"] == []
    assert result["base_value"] is None
    assert isinstance(result["prediction"], float)


def test_get_shap_explanation_success_path_returns_real_values_and_no_note():
    """
    A successful SHAP computation (real trained model, no injected failure)
    must return explanation_unavailable=False, a real base_value, and 5
    top_factors with real per-feature shap_value/feature_value/direction —
    confirming the narrower exception handling in Part 2 doesn't affect the
    normal success path.
    """
    from src.ml.explainability import get_shap_explanation, _load_model
    from src.ml.feature_engineering import FEATURE_NAMES

    if _load_model("churn") is None:
        pytest.skip("No trained churn model on disk — run POST /ml/train first")

    features = {f: 0.1 for f in FEATURE_NAMES}
    aff_id = str(uuid.uuid4())

    result = get_shap_explanation(aff_id, features, "churn")

    assert result["explanation_unavailable"] is False
    assert result.get("note") is None
    assert result["base_value"] is not None
    assert len(result["top_factors"]) == 5
    for factor in result["top_factors"]:
        assert factor["feature"] in FEATURE_NAMES
        assert isinstance(factor["shap_value"], float)
        assert factor["direction"] in ("increases_risk", "decreases_risk")