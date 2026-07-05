"""
Explainability
==============
SHAP-based feature importance for the churn and growth models.

Key function: get_shap_explanation(affiliate_id, features, model_type)
Returns a rich dict with top_factors sorted by |shap_value|.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
import shap

from src.core.logging_config import get_logger
from src.ml.feature_engineering import FEATURE_NAMES
from src.ml.model_store import load_model as _store_load_model

logger = get_logger(__name__)

ModelType = Literal["churn", "growth"]

_CHURN_FILENAME = "churn_model.pkl"
_GROWTH_FILENAME = "growth_model.pkl"

# get_shap_explanation()'s "prediction" is always a fresh XGBoost
# predict_proba() call — a secondary, explainability-only model that is
# independent of affiliates.churn_risk_score / growth_potential_score (the
# persisted, rule-based scores that actually drive status/tier/recommend()).
# The two will generally disagree for the same affiliate; that's expected,
# not a data-freshness bug — see CLAUDE.md §5's "rule-based primary,
# XGBoost secondary" design. Every branch below carries this same flag/text
# so any consumer of this endpoint (not just AffiliateDetail.tsx) gets the
# disambiguation without relying on frontend copy alone.
_SECONDARY_MODEL_DESCRIPTION = (
    "Independent XGBoost model estimate, shown for feature-importance "
    "insight only. It is not the deterministic score used for "
    "recommendations, and will generally not match this affiliate's "
    "churn_risk_score / growth_potential_score."
)


def _load_model(model_type: ModelType):
    """Load the appropriate saved model or return None if not found."""
    filename = _CHURN_FILENAME if model_type == "churn" else _GROWTH_FILENAME
    try:
        return _store_load_model(filename)
    except Exception as exc:
        logger.error(
            "Could not load model",
            extra={"model_type": model_type, "model_file": filename, "error": str(exc)},
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
        affiliate_id        : str,
        model_type          : str,
        base_value          : float | None,  # None when explanation_unavailable
        prediction          : float,  # fresh XGBoost predict_proba() — see
                                       # is_secondary_model/model_description
        top_factors         : [
            {feature, shap_value, feature_value, direction},
            ...  top 5 by |shap_value|
        ],
        explanation_unavailable : bool,
        is_secondary_model      : bool,  # always True today — this endpoint
                                          # never reflects the persisted,
                                          # rule-based churn/growth score
        model_description       : str,
        note                    : str | None,
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
            "base_value": None,
            "prediction": round(pred, 4),
            "top_factors": [],
            "explanation_unavailable": True,
            "is_secondary_model": True,
            "model_description": _SECONDARY_MODEL_DESCRIPTION,
            "note": "SHAP unavailable — model not trained. Run POST /ml/train first.",
        }

    X = pd.DataFrame([features])[FEATURE_NAMES].fillna(0)

    # Computed before the SHAP try/except, deliberately: predict_proba() does
    # not depend on shap.TreeExplainer at all, so a SHAP-specific failure
    # below must never take a perfectly working prediction down with it —
    # only the explanation becomes unavailable, not the prediction itself.
    prediction = float(model.predict_proba(X)[0, 1])

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
        # Do NOT substitute np.zeros()/base=0.0 here — that produces a
        # response indistinguishable from "every feature genuinely
        # contributes nothing", which is exactly how a real incompatibility
        # (shap/xgboost base_score serialization mismatch — see
        # requirements.txt) went unnoticed and looked like a valid result
        # for every affiliate. Callers get an explicit, checkable
        # explanation_unavailable flag instead of plausible-looking fake data.
        # Kept at ERROR (not downgraded): a genuine SHAP failure after the
        # Part 1 version fix should now be rare, so it's still worth paging
        # on, not routine noise.
        logger.error(
            "SHAP computation failed — returning explanation_unavailable "
            "rather than fabricated zero values",
            extra={
                "affiliate_id": affiliate_id,
                "model_type": model_type,
                "error": str(exc),
            },
        )
        return {
            "affiliate_id": affiliate_id,
            "model_type": model_type,
            "base_value": None,
            "prediction": round(prediction, 4),
            "top_factors": [],
            "explanation_unavailable": True,
            "is_secondary_model": True,
            "model_description": _SECONDARY_MODEL_DESCRIPTION,
            "note": f"SHAP explanation unavailable — computation failed: {exc}",
        }

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
        "explanation_unavailable": False,
        "is_secondary_model": True,
        "model_description": _SECONDARY_MODEL_DESCRIPTION,
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