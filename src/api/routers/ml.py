"""
ML Router
=========
Endpoints for training models, running scoring, and explaining predictions.

POST /ml/train                  — start model training in background
POST /ml/score                  — start affiliate scoring in background
GET  /ml/scores                 — list current affiliate scores (worst first)
GET  /ml/explain/{affiliate_id} — SHAP feature importance for one affiliate
GET  /ml/dashboard              — portfolio health summary + full scores list
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.api.task_store import set_task
from src.core.logging_config import get_logger
from src.rulebook.recommend import categorize
from src.storage.database import SessionLocal, get_db
from src.storage.models import Affiliate, ScoreHistory

logger = get_logger(__name__)
router = APIRouter()


def _or_default(value: Optional[float], default: float) -> float:
    """None-safe fallback that doesn't misfire on a real 0.0 — `value or default`
    would wrongly substitute default when value is 0.0, since 0.0 is falsy."""
    return default if value is None else value


def _latest_evidence_by_affiliate(db: Session, affiliate_ids: list) -> dict:
    """One query, most-recent-first: the last evidence_bundle written for
    each affiliate. Affiliates with no score_history rows are simply absent
    from the returned dict."""
    rows = (
        db.query(ScoreHistory)
        .filter(ScoreHistory.affiliate_id.in_(affiliate_ids))
        .order_by(ScoreHistory.affiliate_id, ScoreHistory.scored_at.desc())
        .all()
    )
    latest: dict = {}
    for row in rows:
        if row.affiliate_id not in latest:
            latest[row.affiliate_id] = row.evidence_bundle
    return latest


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TaskAccepted(BaseModel):
    status: str
    task_id: str
    message: str


class AffiliateScore(BaseModel):
    affiliate_id: str
    name: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    # Evidence bundle from this affiliate's most recent score_history row —
    # the facts behind the scores above, not just the numbers themselves.
    evidence_bundle: Optional[list[str]] = None
    # Leak signal, kept separate from and visible alongside the scores above —
    # not folded into churn/growth/health. See src.scraping.leakage_scraper.
    has_active_leak: bool = False
    # SEO signal, same principle — never fed into the tier. See
    # src.seo.analyze.derive_search_trend.
    search_trend: str = "stable"


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str


class ShapExplanation(BaseModel):
    affiliate_id: str
    model_type: str
    # None only when explanation_unavailable is True — a SHAP computation
    # failure (e.g. a library version incompatibility) must not be papered
    # over with a fabricated 0.0 that looks identical to a real result.
    base_value: Optional[float] = None
    prediction: float
    top_factors: list[ShapFactor]
    # Explicit, checkable signal that no real SHAP breakdown exists for this
    # affiliate/model right now (model not trained, or the SHAP computation
    # itself failed) — top_factors is empty in both cases. prediction is
    # still real: it comes from predict_proba()/the rule-based fallback,
    # neither of which depends on SHAP succeeding.
    explanation_unavailable: bool = False
    # This endpoint's prediction is always a fresh, independent XGBoost
    # estimate — never the persisted, rule-based churn_risk_score /
    # growth_potential_score that actually drives status/recommendations.
    # Carried explicitly so any consumer (not just AffiliateDetail.tsx) can
    # disambiguate without relying on frontend copy alone.
    is_secondary_model: bool = True
    model_description: Optional[str] = None
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


# ─── Background task functions ────────────────────────────────────────────────

def _run_training_task(task_id: str) -> None:
    """Train both XGBoost models in a background thread."""
    set_task(task_id, "running")
    logger.info("Training task started", extra={"task_id": task_id})
    db = SessionLocal()
    try:
        from src.ml.feature_engineering import get_feature_dataframe
        from src.ml.churn_model import train_churn_model
        from src.ml.growth_model import train_growth_model

        df = get_feature_dataframe(db)
        if df.empty:
            set_task(task_id, "failed", error="No affiliate data found. Run /ingest/full first.")
            return

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

        result = {"churn_model": churn_status, "growth_model": growth_status, "samples_used": n}
        set_task(task_id, "complete", result=result)
        logger.info("Training task complete", extra={"task_id": task_id, **result})
    except Exception as exc:
        logger.error("Training task failed", extra={"task_id": task_id, "error": str(exc)})
        set_task(task_id, "failed", error=str(exc))
    finally:
        db.close()


def _run_scoring_task(task_id: str) -> None:
    """Score all affiliates in a background thread."""
    set_task(task_id, "running")
    logger.info("Scoring task started", extra={"task_id": task_id})
    db = SessionLocal()
    try:
        from src.ml.score_updater import update_all_scores

        result = update_all_scores(db)
        db.commit()
        set_task(task_id, "complete", result=result)
        logger.info("Scoring task complete", extra={"task_id": task_id})
    except Exception as exc:
        db.rollback()
        logger.error("Scoring task failed", extra={"task_id": task_id, "error": str(exc)})
        set_task(task_id, "failed", error=str(exc))
    finally:
        db.close()


# ─── Train ────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TaskAccepted, dependencies=[Depends(get_api_key)])
async def train_models(background_tasks: BackgroundTasks) -> TaskAccepted:
    """
    Start training churn + growth XGBoost models in the background.
    Returns immediately with a task_id. Poll GET /task/{task_id} for status.
    """
    task_id = str(uuid4())
    set_task(task_id, "pending")
    background_tasks.add_task(_run_training_task, task_id)
    return TaskAccepted(
        status="accepted",
        task_id=task_id,
        message=f"Training started in background. Poll GET /task/{task_id} for status.",
    )


# ─── Score ────────────────────────────────────────────────────────────────────

@router.post("/score", response_model=TaskAccepted, dependencies=[Depends(get_api_key)])
async def score_all_affiliates(background_tasks: BackgroundTasks) -> TaskAccepted:
    """
    Score all affiliates in the background.
    Returns immediately with a task_id. Poll GET /task/{task_id} for status.
    """
    task_id = str(uuid4())
    set_task(task_id, "pending")
    background_tasks.add_task(_run_scoring_task, task_id)
    return TaskAccepted(
        status="accepted",
        task_id=task_id,
        message=f"Scoring started in background. Poll GET /task/{task_id} for status.",
    )


# ─── Scores list ──────────────────────────────────────────────────────────────

@router.get("/scores", response_model=list[AffiliateScore])
def get_scores(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[AffiliateScore]:
    """Return affiliate scores sorted by health_score ascending — worst affiliates first."""
    affiliates = (
        db.query(Affiliate)
        .order_by(Affiliate.health_score.asc())
        .limit(limit)
        .all()
    )
    evidence_by_affiliate = _latest_evidence_by_affiliate(db, [a.id for a in affiliates])
    return [
        AffiliateScore(
            affiliate_id=str(a.id),
            name=a.name,
            churn_risk_score=round(_or_default(a.churn_risk_score, 0.5), 4),
            growth_potential_score=round(_or_default(a.growth_potential_score, 0.5), 4),
            health_score=round(_or_default(a.health_score, 50.0), 1),
            evidence_bundle=evidence_by_affiliate.get(a.id),
            has_active_leak=bool(a.has_active_leak),
            search_trend=a.search_trend or "stable",
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
            base_value=raw.get("base_value"),
            prediction=raw.get("prediction", 0.0),
            top_factors=factors,
            explanation_unavailable=raw.get("explanation_unavailable", False),
            is_secondary_model=raw.get("is_secondary_model", True),
            model_description=raw.get("model_description"),
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
    """Portfolio health summary with full scores list."""
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
    avg_health = round(sum(_or_default(a.health_score, 50.0) for a in affiliates) / n, 1)

    # Mutually-exclusive tiers via the rulebook — same substitution as the
    # other call sites Phase 1 fixed (src.rulebook.recommend.categorize);
    # this function was missed then because it wasn't named in that phase.
    tiers = [
        categorize(_or_default(a.churn_risk_score, 0.5), _or_default(a.growth_potential_score, 0.5))
        for a in affiliates
    ]
    at_risk = sum(1 for t in tiers if t == "at_risk")
    high_growth = sum(1 for t in tiers if t == "high_growth")
    churned = sum(1 for t in tiers if t == "churned")

    evidence_by_affiliate = _latest_evidence_by_affiliate(db, [a.id for a in affiliates])
    scores = [
        AffiliateScore(
            affiliate_id=str(a.id),
            name=a.name,
            churn_risk_score=round(_or_default(a.churn_risk_score, 0.5), 4),
            growth_potential_score=round(_or_default(a.growth_potential_score, 0.5), 4),
            health_score=round(_or_default(a.health_score, 50.0), 1),
            evidence_bundle=evidence_by_affiliate.get(a.id),
            has_active_leak=bool(a.has_active_leak),
            search_trend=a.search_trend or "stable",
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