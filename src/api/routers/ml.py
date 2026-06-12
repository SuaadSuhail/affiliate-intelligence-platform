"""
ML Router
=========
Endpoints for training models, running scoring, and explaining predictions.

POST /ml/train                  — train churn + growth XGBoost models
POST /ml/score                  — score all affiliates and persist results
GET  /ml/scores                 — list current affiliate scores (worst first)
GET  /ml/explain/{affiliate_id} — SHAP feature importance for one affiliate
GET  /ml/dashboard              — portfolio health summary + full scores list
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.storage.database import get_db, db_session
from src.storage.models import Affiliate, ScoreHistory

router = APIRouter()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TrainResult(BaseModel):
    churn_model: str
    growth_model: str
    samples_used: int


class ScoreResult(BaseModel):
    affiliates_scored: int
    avg_health_score: float
    at_risk_count: int
    high_growth_count: int


class AffiliateScore(BaseModel):
    affiliate_id: str
    name: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str


class ShapExplanation(BaseModel):
    affiliate_id: str
    model_type: str
    base_value: float
    prediction: float
    top_factors: list[ShapFactor]
    note: Optional[str] = None


class ExplainResult(BaseModel):
    affiliate_id: str
    churn: ShapExplanation
    growth: ShapExplanation


class DashboardStats(BaseModel):
    total_affiliates: int
    avg_health_score: float
    at_risk_count: int
    high_growth_count: int
    churned_count: int
    scores: list[AffiliateScore]


# ─── Train ────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResult, dependencies=[Depends(get_api_key)])
def train_models(db: Session = Depends(get_db)) -> TrainResult:
    """
    Train both XGBoost models (churn + growth) on current affiliate data.
    Saves artefacts to models/churn_model.pkl and models/growth_model.pkl.
    """
    from src.ml.feature_engineering import get_feature_dataframe
    from src.ml.churn_model import train_churn_model
    from src.ml.growth_model import train_growth_model

    df = get_feature_dataframe(db)
    if df.empty:
        raise HTTPException(status_code=400, detail="No affiliate data found. Run /ingest/full first.")

    n = len(df)

    try:
        train_churn_model(df)
        churn_status = "trained"
    except Exception as exc:
        churn_status = f"error: {exc}"

    try:
        train_growth_model(df)
        growth_status = "trained"
    except Exception as exc:
        growth_status = f"error: {exc}"

    return TrainResult(churn_model=churn_status, growth_model=growth_status, samples_used=n)


# ─── Score ────────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResult, dependencies=[Depends(get_api_key)])
def score_all_affiliates(db: Session = Depends(get_db)) -> ScoreResult:
    """
    Score all affiliates using rule-based or XGBoost model (whichever is available).
    Updates churn_risk_score, growth_potential_score, health_score in PostgreSQL
    and appends to score_history. Skips affiliates already scored today.
    """
    from src.ml.score_updater import update_all_scores

    try:
        result = update_all_scores(db)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")

    return ScoreResult(**result)


# ─── Scores list ──────────────────────────────────────────────────────────────

@router.get("/scores", response_model=list[AffiliateScore])
def get_scores(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[AffiliateScore]:
    """
    Return affiliate scores sorted by health_score ascending — worst affiliates first.
    """
    affiliates = (
        db.query(Affiliate)
        .order_by(Affiliate.health_score.asc())
        .limit(limit)
        .all()
    )
    return [
        AffiliateScore(
            affiliate_id=str(a.id),
            name=a.name,
            churn_risk_score=round(a.churn_risk_score or 0.5, 4),
            growth_potential_score=round(a.growth_potential_score or 0.5, 4),
            health_score=round(a.health_score or 50.0, 1),
        )
        for a in affiliates
    ]


# ─── Explain ──────────────────────────────────────────────────────────────────

@router.get("/explain/{affiliate_id}", response_model=ExplainResult)
def explain(affiliate_id: str, db: Session = Depends(get_db)) -> ExplainResult:
    """
    Return SHAP-based feature importances for one affiliate (churn + growth).
    Requires models to be trained first via POST /ml/train.
    """
    try:
        _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    aff = db.query(Affiliate).filter(Affiliate.id == _uuid.UUID(affiliate_id)).first()
    if not aff:
        raise HTTPException(status_code=404, detail=f"Affiliate {affiliate_id} not found")

    from src.ml.feature_engineering import build_feature_vector
    from src.ml.explainability import get_shap_explanation

    features = build_feature_vector(affiliate_id, db)

    churn_exp = get_shap_explanation(affiliate_id, features, "churn")
    growth_exp = get_shap_explanation(affiliate_id, features, "growth")

    def _to_shap(raw: dict) -> ShapExplanation:
        factors = [ShapFactor(**f) for f in raw.get("top_factors", [])]
        return ShapExplanation(
            affiliate_id=raw["affiliate_id"],
            model_type=raw["model_type"],
            base_value=raw.get("base_value", 0.0),
            prediction=raw.get("prediction", 0.0),
            top_factors=factors,
            note=raw.get("note"),
        )

    return ExplainResult(
        affiliate_id=affiliate_id,
        churn=_to_shap(churn_exp),
        growth=_to_shap(growth_exp),
    )


# ─── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=DashboardStats)
def dashboard(db: Session = Depends(get_db)) -> DashboardStats:
    """
    Portfolio health summary with full scores list.
    Shows at_risk, high_growth, and churned affiliate counts.
    """
    affiliates = db.query(Affiliate).order_by(Affiliate.health_score.asc()).all()

    if not affiliates:
        return DashboardStats(
            total_affiliates=0,
            avg_health_score=0.0,
            at_risk_count=0,
            high_growth_count=0,
            churned_count=0,
            scores=[],
        )

    n = len(affiliates)
    avg_health = round(sum(a.health_score or 50.0 for a in affiliates) / n, 1)
    at_risk = sum(1 for a in affiliates if (a.churn_risk_score or 0.5) > 0.5)
    high_growth = sum(1 for a in affiliates if (a.growth_potential_score or 0.5) > 0.5)
    churned = sum(1 for a in affiliates if (a.churn_risk_score or 0.5) > 0.8)

    scores = [
        AffiliateScore(
            affiliate_id=str(a.id),
            name=a.name,
            churn_risk_score=round(a.churn_risk_score or 0.5, 4),
            growth_potential_score=round(a.growth_potential_score or 0.5, 4),
            health_score=round(a.health_score or 50.0, 1),
        )
        for a in affiliates
    ]

    return DashboardStats(
        total_affiliates=n,
        avg_health_score=avg_health,
        at_risk_count=at_risk,
        high_growth_count=high_growth,
        churned_count=churned,
        scores=scores,
    )