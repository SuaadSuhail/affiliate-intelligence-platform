"""
Churn Risk Model
================
Trains and serves an XGBoost classifier that outputs churn_risk_score (0–1).

Target
------
churn_label = 1 if affiliate churned within 90 days (see feature_engineering.py
for synthetic label generation on mock data)

Artefact
--------
Saved to CHURN_MODEL_PATH (default: models/churn_model.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.ml.feature_engineering import (
    build_features,
    feature_columns,
    generate_synthetic_labels,
)

load_dotenv()

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "models/churn_model.json"))
MODEL_VERSION = "1.0.0"

# ─── XGBoost hyperparameters ─────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

_model: Optional[XGBClassifier] = None


# ─── Training ─────────────────────────────────────────────────────────────────

def train(save: bool = True) -> XGBClassifier:
    """
    Build features, generate synthetic labels, train churn model.
    Saves artefact to MODEL_PATH.
    """
    global _model
    print("[churn_model] Building feature matrix …")
    df = build_features()
    df = generate_synthetic_labels(df)

    if df.empty:
        raise ValueError("No training data found. Run the ETL pipeline first.")

    feat_cols = feature_columns()
    X = df[feat_cols].fillna(0)
    y = df["churn_label"]

    print(f"[churn_model] Training on {len(df)} samples | churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"[churn_model] Test AUC: {auc:.4f}")
    except ValueError:
        print("[churn_model] AUC not computable (single class in test split)")

    if len(set(y_test)) > 1:
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))
        print(f"[churn_model] Model saved: {MODEL_PATH}")

    _model = model
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def _load_model() -> XGBClassifier:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            print(f"[churn_model] Model not found at {MODEL_PATH} — training now …")
            return train()
        _model = XGBClassifier()
        _model.load_model(str(MODEL_PATH))
        print(f"[churn_model] Model loaded from {MODEL_PATH}")
    return _model


def predict(affiliate_ids: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Predict churn_risk_score for one or all affiliates.

    Parameters
    ----------
    affiliate_ids : optional list of affiliate UUID strings; if None scores all

    Returns
    -------
    DataFrame with columns: affiliate_id, churn_risk_score, features (dict)
    """
    model = _load_model()
    df = build_features(affiliate_ids=affiliate_ids)

    if df.empty:
        return pd.DataFrame(columns=["affiliate_id", "churn_risk_score", "features"])

    feat_cols = feature_columns()
    X = df[feat_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    result = df[["affiliate_id"]].copy()
    result["churn_risk_score"] = np.round(probs, 4)
    result["features"] = X.to_dict(orient="records")
    return result


def predict_one(affiliate_id: str) -> dict:
    """Return churn_risk_score + feature dict for a single affiliate."""
    df = predict([affiliate_id])
    if df.empty:
        return {"affiliate_id": affiliate_id, "churn_risk_score": 0.5, "features": {}}
    row = df.iloc[0]
    return {
        "affiliate_id": row["affiliate_id"],
        "churn_risk_score": float(row["churn_risk_score"]),
        "features": row["features"],
    }


if __name__ == "__main__":
    model = train()
    results = predict()
    print("\nChurn scores:")
    print(results[["affiliate_id", "churn_risk_score"]].to_string())
