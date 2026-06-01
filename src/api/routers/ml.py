"""
ML Router
=========
Endpoints for training models, running scoring, and explaining predictions.

POST /ml/train                  — train churn + growth XGBoost models
POST /ml/score                  — score all affiliates and persist results
GET  /ml/scores                 — list current affiliate scores
GET  /ml/explain/{affiliate_id} — SHAP feature importance for one affiliate
GET  /ml/dashboard              — aggregate health stats across all affiliates
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.database import get_db
from src.storage.models import Affiliate, ScoreHistory

router = APIRouter()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TrainResult(BaseModel):
    churn_model: str
    growth_model: str


class ScoreResult(BaseModel):
    affiliates_scored: int
    score_history_entries: int


class AffiliateScore(BaseModel):
    affiliate_id: str
    name: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float


class ExplainResult(BaseModel):
    affiliate_id: str
    churn_drivers: list[str]
    growth_drivers: list[str]
    churn_shap: dict
    growth_shap: dict


class DashboardStats(BaseModel):
    total_affiliates: int
    avg_health_score: float
    avg_churn_risk: float
    avg_growth_potential: float
    high_risk_count: int       # churn_risk_score > 0.7
    high_growth_count: int     # growth_potential_score > 0.7
    score_history_entries: int


# ─── Train ────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResult)
def train_models() -> TrainResult:
    """
    Train both XGBoost models (churn + growth) on the current affiliate data.
    Saves model artefacts to models/churn_model.json and models/growth_model.json.
    Requires at least one affiliate in the database.
    """
    from src.ml.churn_model import train as train_churn
    from src.ml.growth_model import train as train_growth

    try:
        train_churn(save=True)
        churn_status = "trained"
    except Exception as exc:
        churn_status = f"error: {exc}"

    try:
        train_growth(save=True)
        growth_status = "trained"
    except Exception as exc:
        growth_status = f"error: {exc}"

    return TrainResult(churn_model=churn_status, growth_model=growth_status)


# ─── Score ────────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResult)
def score_all_affiliates(db: Session = Depends(get_db)) -> ScoreResult:
    """
    Run churn + growth prediction for every affiliate and persist results.

    Updates affiliates.churn_risk_score, growth_potential_score, health_score
    and appends a new ScoreHistory row for each affiliate.
    Auto-trains models if model artefacts are not found.
    """
    from src.ml.churn_model import predict as churn_predict
    from src.ml.growth_model import predict as growth_predict

    try:
        churn_df = churn_predict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Churn prediction error: {exc}")

    try:
        growth_df = growth_predict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Growth prediction error: {exc}")

    # Merge on affiliate_id
    merged = churn_df.merge(growth_df[["affiliate_id", "growth_potential_score"]], on="affiliate_id", how="inner")

    updates = 0
    history_entries = 0
    for _, row in merged.iterrows():
        try:
            aff_uuid = _uuid.UUID(str(row["affiliate_id"]))
        except ValueError:
            continue

        aff = db.query(Affiliate).filter(Affiliate.id == aff_uuid).first()
        if not aff:
            continue

        c_score = float(row["churn_risk_score"])
        g_score = float(row["growth_potential_score"])
        h_score = round(((1 - c_score) * 0.6 + g_score * 0.4) * 100, 1)

        aff.churn_risk_score = c_score
        aff.growth_potential_score = g_score
        aff.health_score = h_score
        updates += 1

        entry = ScoreHistory(
            affiliate_id=aff.id,
            churn_risk_score=c_score,
            growth_potential_score=g_score,
            health_score=h_score,
            features=row.get("features", {}) if isinstance(row.get("features"), dict) else {},
            shap_values={},
        )
        db.add(entry)
        history_entries += 1

    db.commit()
    return ScoreResult(affiliates_scored=updates, score_history_entries=history_entries)


# ─── Scores list ──────────────────────────────────────────────────────────────

@router.get("/scores", response_model=list[AffiliateScore])
def get_scores(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[AffiliateScore]:
    """Return current churn/growth/health scores for all affiliates, sorted by health score."""
    affiliates = (
        db.query(Affiliate)
        .order_by(Affiliate.health_score.desc())
        .limit(limit)
        .all()
    )
    return [
        AffiliateScore(
            affiliate_id=str(a.id),
            name=a.name,
            churn_risk_score=a.churn_risk_score or 0.5,
            growth_potential_score=a.growth_potential_score or 0.5,
            health_score=a.health_score or 50.0,
        )
        for a in affiliates
    ]


# ─── Explain ──────────────────────────────────────────────────────────────────

@router.get("/explain/{affiliate_id}", response_model=ExplainResult)
def explain(affiliate_id: str, db: Session = Depends(get_db)) -> ExplainResult:
    """
    Return SHAP-based feature importances for one affiliate.
    Identifies which features are driving their churn risk and growth potential scores.
    """
    try:
        _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    aff = db.query(Affiliate).filter(Affiliate.id == _uuid.UUID(affiliate_id)).first()
    if not aff:
        raise HTTPException(status_code=404, detail=f"Affiliate {affiliate_id} not found")

    from src.ml.explainability import explain_affiliate, top_risk_drivers

    try:
        churn_shap = explain_affiliate(affiliate_id, model_type="churn")
    except Exception as exc:
        churn_shap = {"error": str(exc)}

    try:
        growth_shap = explain_affiliate(affiliate_id, model_type="growth")
    except Exception as exc:
        growth_shap = {"error": str(exc)}

    try:
        churn_drivers = top_risk_drivers(affiliate_id, model_type="churn")
    except Exception:
        churn_drivers = []

    try:
        growth_drivers = top_risk_drivers(affiliate_id, model_type="growth")
    except Exception:
        growth_drivers = []

    return ExplainResult(
        affiliate_id=affiliate_id,
        churn_drivers=churn_drivers,
        growth_drivers=growth_drivers,
        churn_shap=churn_shap if isinstance(churn_shap, dict) else {},
        growth_shap=growth_shap if isinstance(growth_shap, dict) else {},
    )


# ─── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=DashboardStats)
def dashboard(db: Session = Depends(get_db)) -> DashboardStats:
    """
    Return aggregate health statistics across all affiliates.
    Use this to get a quick overview of the affiliate portfolio health.
    """
    affiliates = db.query(Affiliate).all()
    if not affiliates:
        return DashboardStats(
            total_affiliates=0,
            avg_health_score=0.0,
            avg_churn_risk=0.0,
            avg_growth_potential=0.0,
            high_risk_count=0,
            high_growth_count=0,
            score_history_entries=db.query(ScoreHistory).count(),
        )

    n = len(affiliates)
    avg_health = round(sum(a.health_score or 50.0 for a in affiliates) / n, 1)
    avg_churn = round(sum(a.churn_risk_score or 0.5 for a in affiliates) / n, 4)
    avg_growth = round(sum(a.growth_potential_score or 0.5 for a in affiliates) / n, 4)
    high_risk = sum(1 for a in affiliates if (a.churn_risk_score or 0.5) > 0.7)
    high_growth = sum(1 for a in affiliates if (a.growth_potential_score or 0.5) > 0.7)
    history_count = db.query(ScoreHistory).count()

    return DashboardStats(
        total_affiliates=n,
        avg_health_score=avg_health,
        avg_churn_risk=avg_churn,
        avg_growth_potential=avg_growth,
        high_risk_count=high_risk,
        high_growth_count=high_growth,
        score_history_entries=history_count,
    )