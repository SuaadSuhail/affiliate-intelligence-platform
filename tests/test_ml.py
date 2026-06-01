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

def test_update_all_scores_updates_affiliates():
    """update_all_scores must score every affiliate and return correct counts."""
    from src.ml.score_updater import update_all_scores
    from src.storage.models import Affiliate, ScoreHistory

    def _make_aff(churn=0.4, growth=0.6):
        a = Affiliate()
        a.id = uuid.uuid4()
        a.name = "Test"
        a.email = f"{uuid.uuid4()}@test.com"
        a.tier = "silver"
        a.monthly_revenue = 5000.0
        a.churn_risk_score = churn
        a.growth_potential_score = growth
        a.health_score = 60.0
        a.last_contact_date = datetime.now(timezone.utc)
        return a

    aff1 = _make_aff(churn=0.3, growth=0.7)
    aff2 = _make_aff(churn=0.8, growth=0.2)

    mock_db = MagicMock()

    def query_side_effect(model):
        q = MagicMock()
        if model is Affiliate:
            q.all.return_value = [aff1, aff2]
        elif model is ScoreHistory:
            # No existing score today → both need scoring
            filt = MagicMock()
            filt.first.return_value = None
            q.filter.return_value = filt
        return q

    mock_db.query.side_effect = query_side_effect
    mock_db.add = MagicMock()

    with patch("src.ml.score_updater.build_feature_vector") as mock_fv, \
         patch("src.ml.score_updater.predict_churn_risk") as mock_churn, \
         patch("src.ml.score_updater.predict_growth_potential") as mock_growth:

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
    assert mock_db.add.call_count == 2


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