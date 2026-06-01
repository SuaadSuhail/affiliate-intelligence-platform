"""
Growth Potential Model
======================
Predicts growth_potential_score (0.0–1.0) for each affiliate.

Primary: rule-based scorer (always available)
Secondary: XGBoost classifier (used when model artefact exists)

Artefact saved to: models/growth_model.pkl
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.ml.feature_engineering import FEATURE_NAMES

GROWTH_MODEL_PATH = Path(os.getenv("GROWTH_MODEL_PATH", "models/growth_model.pkl"))

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

    print(f"[growth_model] Training on {len(df)} samples | growth rate: {y.mean():.1%}")

    model = XGBClassifier(**_XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)

    GROWTH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, GROWTH_MODEL_PATH)
    print(f"[growth_model] Saved → {GROWTH_MODEL_PATH}")

    _model = model
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def _load_saved_model() -> Optional[XGBClassifier]:
    global _model
    if _model is not None:
        return _model
    if GROWTH_MODEL_PATH.exists():
        _model = joblib.load(GROWTH_MODEL_PATH)
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
            print(f"[growth_model] XGBoost predict failed ({exc}), using rules")

    return calculate_growth_potential_rules(features)