"""
Explainability
==============
Computes SHAP values for both the churn and growth models.

Returns per-affiliate dicts of {feature_name: shap_value} that are
stored in score_history.shap_values (JSONB).

Usage
-----
    from src.ml.explainability import explain_affiliate

    shap_dict = explain_affiliate(affiliate_id="...", model_type="churn")
    # → {"days_since_last_contact": 0.32, "tag_churn_signal": 0.18, ...}
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import shap

from src.ml.feature_engineering import build_features, feature_columns
from src.ml.churn_model import _load_model as load_churn
from src.ml.growth_model import _load_model as load_growth


ModelType = Literal["churn", "growth"]


def compute_shap_values(
    df_features: pd.DataFrame,
    model_type: ModelType = "churn",
) -> pd.DataFrame:
    """
    Compute SHAP values for all rows in df_features.

    Parameters
    ----------
    df_features : DataFrame produced by build_features()
    model_type  : "churn" or "growth"

    Returns
    -------
    DataFrame with same index as df_features; columns = feature names;
    values = SHAP contributions (class-1 logit space for XGBoost TreeExplainer)
    """
    feat_cols = feature_columns()
    X = df_features[feat_cols].fillna(0)

    model = load_churn() if model_type == "churn" else load_growth()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # TreeExplainer returns shape (n_samples, n_features) for binary XGBoost
    # If it returns a list (older API), take index 1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return pd.DataFrame(shap_values, columns=feat_cols, index=df_features.index)


def explain_affiliate(
    affiliate_id: str,
    model_type: ModelType = "churn",
    top_n: int = 10,
) -> dict:
    """
    Return a SHAP explanation dict for a single affiliate.

    Parameters
    ----------
    affiliate_id : UUID string
    model_type   : "churn" or "growth"
    top_n        : number of top features to include (by absolute SHAP value)

    Returns
    -------
    Dict: {feature_name: shap_value, ...} — top_n features sorted by |shap|
    """
    df = build_features(affiliate_ids=[affiliate_id])
    if df.empty:
        return {}

    shap_df = compute_shap_values(df, model_type=model_type)
    row = shap_df.iloc[0]
    sorted_features = row.abs().sort_values(ascending=False).head(top_n).index
    return {feat: round(float(row[feat]), 6) for feat in sorted_features}


def explain_all(model_type: ModelType = "churn") -> dict[str, dict]:
    """
    Compute SHAP explanations for all affiliates.

    Returns
    -------
    Dict: {affiliate_id: {feature: shap_value, ...}}
    """
    df = build_features()
    if df.empty:
        return {}

    shap_df = compute_shap_values(df, model_type=model_type)
    feat_cols = feature_columns()
    result: dict[str, dict] = {}

    for i, row in shap_df.iterrows():
        affiliate_id = df.loc[i, "affiliate_id"]
        sorted_features = row.abs().sort_values(ascending=False).head(10).index
        result[affiliate_id] = {
            feat: round(float(row[feat]), 6) for feat in sorted_features
        }

    return result


def top_risk_drivers(affiliate_id: str, model_type: ModelType = "churn") -> list[str]:
    """
    Return an ordered list of the top feature names driving the score,
    suitable for display in the API / agent responses.

    Example output:
        ["days_since_last_contact", "tag_churn_signal", "avg_sentiment_30d"]
    """
    shap_dict = explain_affiliate(affiliate_id, model_type=model_type)
    # Keep only features with positive SHAP (driving risk / growth up)
    drivers = {k: v for k, v in shap_dict.items() if v > 0}
    return sorted(drivers, key=drivers.get, reverse=True)


if __name__ == "__main__":
    from src.storage.database import db_session
    from src.storage.models import Affiliate

    with db_session() as db:
        first = db.query(Affiliate).first()
        if first:
            aid = str(first.id)
            print(f"\nChurn SHAP for {first.name}:")
            print(explain_affiliate(aid, model_type="churn"))
            print(f"\nGrowth SHAP for {first.name}:")
            print(explain_affiliate(aid, model_type="growth"))
