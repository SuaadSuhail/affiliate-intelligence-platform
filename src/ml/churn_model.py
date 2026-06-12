"""
Churn Risk Model
================
Predicts churn_risk_score (0.0–1.0) for each affiliate.

Primary: rule-based scorer (always available, no training required)
Secondary: XGBoost classifier (used when model artefact exists)

Artefact saved to: models/churn_model.pkl
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.core.logging_config import get_logger
from src.ml.feature_engineering import FEATURE_NAMES

logger = get_logger(__name__)

CHURN_MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "models/churn_model.pkl"))

_XGBOOST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

_model: Optional[XGBClassifier] = None


# ─── Rule-based scorer ────────────────────────────────────────────────────────

def calculate_churn_risk_rules(features: dict) -> float:
    """
    Rule-based churn risk scorer.

    Each rule adds a weighted contribution; final score clamped to [0.0, 1.0].
    """
    score = 0.0

    days = features.get("days_since_contact", 0)
    if days > 30:
        score += 0.35
    elif days > 14:
        score += 0.20
    elif days > 7:
        score += 0.10

    ctr = features.get("ctr_trend_pct", 0.0)
    if ctr < -2.0:
        score += 0.25
    elif ctr < 0:
        score += 0.10

    churn_sigs = features.get("churn_signal_count", 0)
    if churn_sigs >= 2:
        score += 0.25
    elif churn_sigs >= 1:
        score += 0.15

    comp = features.get("competitor_mention_count", 0)
    if comp >= 1:
        score += 0.20

    esc = features.get("escalation_count", 0)
    if esc >= 1:
        score += 0.15

    sent = features.get("avg_sentiment_30d", 0.0)
    if sent < -0.4:
        score += 0.20
    elif sent < -0.2:
        score += 0.10

    if features.get("comm_count_30d", 0) == 0:
        score += 0.15

    return min(1.0, max(0.0, score))


# ─── Training ─────────────────────────────────────────────────────────────────

def train_churn_model(df: pd.DataFrame) -> XGBClassifier:
    """
    Train XGBoost churn classifier.

    Target: 1 if status in ['at_risk', 'churned'], else 0.
    Saves model to CHURN_MODEL_PATH.

    Parameters
    ----------
    df : DataFrame from get_feature_dataframe() — index = affiliate_id,
         must contain 'status' column and all FEATURE_NAMES columns.

    Returns
    -------
    Trained XGBClassifier.
    """
    global _model

    df = df.reset_index()  # bring affiliate_id back as column if it's the index
    y = df["status"].isin(["at_risk", "churned"]).astype(int)
    X = df[FEATURE_NAMES].fillna(0)

    logger.info(
        "Training churn model",
        extra={"samples": len(df), "churn_rate": f"{y.mean():.1%}"},
    )

    model = XGBClassifier(**_XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)

    CHURN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHURN_MODEL_PATH)
    logger.info("Churn model saved", extra={"path": str(CHURN_MODEL_PATH)})

    _model = model
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def _load_saved_model() -> Optional[XGBClassifier]:
    """Load model from disk if available."""
    global _model
    if _model is not None:
        return _model
    if CHURN_MODEL_PATH.exists():
        _model = joblib.load(CHURN_MODEL_PATH)
        return _model
    return None


def predict_churn_risk(
    affiliate_id: str,
    features: dict,
    model: Optional[XGBClassifier] = None,
    db=None,
) -> float:
    """
    Predict churn_risk_score for one affiliate.

    Uses XGBoost if a saved model exists; falls back to rule-based scorer.

    Parameters
    ----------
    affiliate_id : UUID string (used for logging only)
    features     : dict from build_feature_vector()
    model        : pre-loaded XGBClassifier (optional)
    db           : unused; kept for signature compatibility

    Returns
    -------
    float in [0.0, 1.0]
    """
    if model is None:
        model = _load_saved_model()

    if model is not None:
        try:
            X = pd.DataFrame([features])[FEATURE_NAMES].fillna(0)
            return float(model.predict_proba(X)[0, 1])
        except Exception as exc:
            logger.warning(
                "XGBoost predict failed — falling back to rules",
                extra={"error": str(exc)},
            )

    return calculate_churn_risk_rules(features)