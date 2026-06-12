"""
Explainability
==============
SHAP-based feature importance for the churn and growth models.

Key function: get_shap_explanation(affiliate_id, features, model_type)
Returns a rich dict with top_factors sorted by |shap_value|.
"""

from __future__ import annotations

from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.core.logging_config import get_logger
from src.ml.feature_engineering import FEATURE_NAMES
from src.ml.churn_model import CHURN_MODEL_PATH
from src.ml.growth_model import GROWTH_MODEL_PATH

logger = get_logger(__name__)

ModelType = Literal["churn", "growth"]


def _load_model(model_type: ModelType):
    """Load the appropriate saved model or return None if not found."""
    path = CHURN_MODEL_PATH if model_type == "churn" else GROWTH_MODEL_PATH
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error(
            "Could not load model",
            extra={"model_type": model_type, "path": str(path), "error": str(exc)},
        )
        return None


def get_shap_explanation(
    affiliate_id: str,
    features: dict,
    model_type: ModelType,
) -> dict:
    """
    Compute SHAP explanation for one affiliate.

    Parameters
    ----------
    affiliate_id : UUID string
    features     : feature dict from build_feature_vector()
    model_type   : "churn" or "growth"

    Returns
    -------
    {
        affiliate_id  : str,
        model_type    : str,
        base_value    : float,
        prediction    : float,
        top_factors   : [
            {feature, shap_value, feature_value, direction},
            ...  top 5 by |shap_value|
        ]
    }
    direction: "increases_risk"/"decreases_risk" (churn)
               "increases_growth"/"decreases_growth" (growth)
    """
    model = _load_model(model_type)

    if model is None:
        # Return rule-based placeholder when model not trained yet
        from src.ml.churn_model import calculate_churn_risk_rules
        from src.ml.growth_model import calculate_growth_potential_rules
        pred = (
            calculate_churn_risk_rules(features)
            if model_type == "churn"
            else calculate_growth_potential_rules(features)
        )
        return {
            "affiliate_id": affiliate_id,
            "model_type": model_type,
            "base_value": 0.0,
            "prediction": round(pred, 4),
            "top_factors": [],
            "note": "SHAP unavailable — model not trained. Run POST /ml/train first.",
        }

    X = pd.DataFrame([features])[FEATURE_NAMES].fillna(0)

    try:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
        # XGBoost binary: may return list[class0, class1] or single array
        if isinstance(raw, list):
            shap_row = raw[1][0]
            base = float(explainer.expected_value[1])
        else:
            shap_row = raw[0]
            base = float(explainer.expected_value)
    except Exception as exc:
        logger.error("SHAP computation failed", extra={"error": str(exc)})
        shap_row = np.zeros(len(FEATURE_NAMES))
        base = 0.0

    prediction = float(model.predict_proba(X)[0, 1])

    if model_type == "churn":
        pos_label, neg_label = "increases_risk", "decreases_risk"
    else:
        pos_label, neg_label = "increases_growth", "decreases_growth"

    factors = [
        {
            "feature": fname,
            "shap_value": round(float(shap_row[i]), 6),
            "feature_value": float(features.get(fname, 0.0)),
            "direction": pos_label if shap_row[i] > 0 else neg_label,
        }
        for i, fname in enumerate(FEATURE_NAMES)
    ]
    top_factors = sorted(factors, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

    return {
        "affiliate_id": affiliate_id,
        "model_type": model_type,
        "base_value": round(base, 6),
        "prediction": round(prediction, 4),
        "top_factors": top_factors,
    }


# ─── Backward-compatible helpers (used by legacy router code) ─────────────────

def explain_affiliate(
    affiliate_id: str,
    model_type: ModelType = "churn",
    top_n: int = 10,
    db=None,
) -> dict:
    """
    Legacy interface: build features internally and return {feature: shap_value}.
    Used by existing API router code.
    """
    from src.ml.feature_engineering import build_feature_vector
    from src.storage.database import db_session

    def _explain(session):
        feats = build_feature_vector(affiliate_id, session)
        result = get_shap_explanation(affiliate_id, feats, model_type)
        # Return flat dict of feature→shap_value for backward compatibility
        if result.get("top_factors"):
            return {f["feature"]: f["shap_value"] for f in result["top_factors"]}
        return {}

    if db is not None:
        return _explain(db)
    with db_session() as session:
        return _explain(session)


def top_risk_drivers(
    affiliate_id: str,
    model_type: ModelType = "churn",
    db=None,
) -> list[str]:
    """Return ordered list of top feature names driving risk/growth up."""
    shap_dict = explain_affiliate(affiliate_id, model_type=model_type, db=db)
    drivers = {k: v for k, v in shap_dict.items() if v > 0}
    return sorted(drivers, key=lambda k: drivers[k], reverse=True)