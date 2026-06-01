"""
Score Updater
=============
Scores all affiliates and persists results to PostgreSQL.

Runs the full scoring pipeline for every affiliate:
  1. Build feature vector
  2. Predict churn_risk_score
  3. Predict growth_potential_score
  4. Compute health_score
  5. Update affiliates table
  6. Insert into score_history (skips if already scored today)
"""

from __future__ import annotations

from datetime import datetime, date, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.storage.models import Affiliate, ScoreHistory
from src.ml.feature_engineering import build_feature_vector
from src.ml.churn_model import predict_churn_risk
from src.ml.growth_model import predict_growth_potential


def update_all_scores(db: Session) -> dict:
    """
    Score every affiliate and persist results.

    Skips affiliates already scored today (idempotent within a day).

    Parameters
    ----------
    db : active SQLAlchemy session (caller owns commit/rollback)

    Returns
    -------
    {
        affiliates_scored    : int,
        avg_health_score     : float,
        at_risk_count        : int,   (churn_risk_score > 0.5)
        high_growth_count    : int,   (growth_potential_score > 0.5)
    }
    """
    affiliates = db.query(Affiliate).all()
    today = date.today()

    scored = 0
    health_scores: list[float] = []

    for aff in affiliates:
        # Skip if already scored today
        existing = (
            db.query(ScoreHistory)
            .filter(
                ScoreHistory.affiliate_id == aff.id,
                func.date(ScoreHistory.scored_at) == today,
            )
            .first()
        )
        if existing:
            continue

        affiliate_id = str(aff.id)

        # 1. Build feature vector
        features = build_feature_vector(affiliate_id, db)

        # 2 & 3. Predict scores
        churn_score = predict_churn_risk(affiliate_id, features)
        growth_score = predict_growth_potential(affiliate_id, features)

        # 4. Compute health_score (CLAUDE.md formula)
        health = round(((1 - churn_score) * 0.6 + growth_score * 0.4) * 100, 1)

        # 5. Update affiliate record
        aff.churn_risk_score = round(churn_score, 4)
        aff.growth_potential_score = round(growth_score, 4)
        aff.health_score = health

        # 6. Insert score history
        entry = ScoreHistory(
            affiliate_id=aff.id,
            churn_risk_score=round(churn_score, 4),
            growth_potential_score=round(growth_score, 4),
            health_score=health,
            scored_at=datetime.now(timezone.utc),
        )
        db.add(entry)

        health_scores.append(health)
        scored += 1

    affiliates_all = db.query(Affiliate).all()
    at_risk = sum(1 for a in affiliates_all if (a.churn_risk_score or 0.0) > 0.5)
    high_growth = sum(1 for a in affiliates_all if (a.growth_potential_score or 0.0) > 0.5)
    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0.0

    return {
        "affiliates_scored": scored,
        "avg_health_score": avg_health,
        "at_risk_count": at_risk,
        "high_growth_count": high_growth,
    }