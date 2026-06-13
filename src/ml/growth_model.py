"""
Growth Potential Model
======================
Predicts growth_potential_score (0.0–1.0) for each affiliate.

Primary: rule-based scorer (always available)
Secondary: XGBoost classifier (used when model artefact exists)

Artefact saved via model_store (local disk + optional S3).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from xgboost import XGBClassifier

from src.core.logging_config import get_logger
from src.ml.feature_engineering import FEATURE_NAMES
from src.ml.model_store import load_model, save_model

logger = get_logger(__name__)

_GROWTH_FILENAME = "growth_model.pkl"

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

def calculate_growth_potential_rules(features: dict) -> float:
    """
    Rule-based growth potential scorer.

    Each rule adds a weighted contribution; final score clamped to [0.0, 1.0].
    """
    score = 0.0

    pos_sigs = features.get("positive_signal_count", 0)
    if pos_sigs >= 3:
        score += 0.30
    elif pos_sigs >= 1:
        score += 0.15

    sent = features.get("avg_sentiment_30d", 0.0)
    if sent > 0.4:
        score += 0.25
    elif sent > 0.2:
        score += 0.12

    comms = features.get("comm_count_30d", 0)
    if comms >= 3:
        score += 0.15
    elif comms >= 1:
        score += 0.08

    rev = features.get("revenue_30d", 0.0)
    if rev > 20000:
        score += 0.20
    elif rev > 5000:
        score += 0.10

    ctr = features.get("ctr_trend_pct", 0.0)
    if ctr > 2.0:
        score += 0.20
    elif ctr > 0:
        score += 0.10

    # expansion_interest: check churn_signal_count inverse as proxy
    # (positive_signal_count already covers expansion_interest tags)
    if pos_sigs >= 1:
        score += 0.15

    trend = features.get("sentiment_trend", 0.0)
    if trend > 0.1:
        score += 0.10

    return min(1.0, max(0.0, score))


# ─── Training ─────────────────────────────────────────────────────────────────

def train_growth_model(df: pd.DataFrame) -> XGBClassifier:
    """
    Train XGBoost growth potential classifier.

    Target: 1 if status == 'high_growth', else 0.
    Saves model to GROWTH_MODEL_PATH.

    Parameters
    ----------
    df : DataFrame from get_feature_dataframe() — index = affiliate_id,
         must contain 'status' column and all FEATURE_NAMES columns.

    Returns
    -------
    Trained XGBClassifier.
    """
    global _model

    df = df.reset_index()
    y = (df["status"] == "high_growth").astype(int)
    X = df[FEATURE_NAMES].fillna(0)

    logger.info(
        "Training growth model",
        extra={"samples": len(df), "growth_rate": f"{y.mean():.1%}"},
    )

    model = XGBClassifier(**_XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)

    save_model(model, _GROWTH_FILENAME)

    _model = model
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def _load_saved_model() -> Optional[XGBClassifier]:
    """Load model from disk (or S3) if available."""
    global _model
    if _model is not None:
        return _model
    loaded = load_model(_GROWTH_FILENAME)
    if loaded is not None:
        _model = loaded
        return _model
    return None


def predict_growth_potential(
    affiliate_id: str,
    features: dict,
    model: Optional[XGBClassifier] = None,
    db=None,
) -> float:
    """
    Predict growth_potential_score for one affiliate.

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

    return calculate_growth_potential_rules(features)