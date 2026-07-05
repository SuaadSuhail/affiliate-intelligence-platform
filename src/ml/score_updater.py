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
  6. Get the rulebook's evidence bundle for this affiliate's inputs
  7. Write an audit_log entry linking the evidence bundle back to its inputs
  8. Insert into score_history, evidence bundle included (skips if already scored today)
"""

from __future__ import annotations

from datetime import datetime, date, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.audit.log import write_audit_entry
from src.core.logging_config import get_logger
from src.rulebook.recommend import categorize, recommend
from src.storage.models import Affiliate, LeakedCode, ScoreHistory
from src.ml.feature_engineering import build_feature_vector
from src.ml.churn_model import calculate_churn_risk_rules
from src.ml.growth_model import calculate_growth_potential_rules

logger = get_logger(__name__)


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
        at_risk_count        : int,   (mutually-exclusive tier, see src.rulebook.recommend.categorize)
        high_growth_count    : int,   (mutually-exclusive tier, see src.rulebook.recommend.categorize)
    }
    """
    affiliates = db.query(Affiliate).all()
    today = date.today()

    logger.info("Scoring run started", extra={"total_affiliates": len(affiliates)})

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

        # 2 & 3. Predict scores (rule-based primary — CLAUDE.md §5)
        churn_score = calculate_churn_risk_rules(features)
        growth_score = calculate_growth_potential_rules(features)

        # 4. Compute health_score (CLAUDE.md formula)
        health = round(((1 - churn_score) * 0.6 + growth_score * 0.4) * 100, 1)

        logger.debug(
            "Affiliate scored",
            extra={
                "name": aff.name,
                "churn_risk": round(churn_score, 4),
                "growth_potential": round(growth_score, 4),
                "health_score": health,
            },
        )

        # 5. Update affiliate record
        aff.churn_risk_score = round(churn_score, 4)
        aff.growth_potential_score = round(growth_score, 4)
        aff.health_score = health

        # 6. Evidence bundle — the specific facts behind this run's scores,
        # not just the scores themselves (leaks included as a visible, separate
        # fact; see src.rulebook.recommend for why they don't affect the tier).
        recent_leaks = (
            db.query(LeakedCode)
            .filter(LeakedCode.affiliate_id == aff.id)
            .order_by(LeakedCode.found_at.desc())
            .limit(5)
            .all()
        )
        rec = recommend(aff, features, recent_leaks)

        # Audit trail: link this affiliate's stored evidence_bundle back to the
        # exact feature inputs and tier that produced it (see src.audit.log).
        write_audit_entry(
            db,
            stage="rulebook",
            record_type="affiliate",
            record_id=aff.id,
            rule_or_tool="recommend",
            input_snapshot=features,
            output_snapshot={
                "tier": categorize(aff.churn_risk_score, aff.growth_potential_score),
                "evidence": rec.evidence,
            },
        )

        # 7. Insert score history
        entry = ScoreHistory(
            affiliate_id=aff.id,
            churn_risk_score=round(churn_score, 4),
            growth_potential_score=round(growth_score, 4),
            health_score=health,
            evidence_bundle=rec.evidence,
            scored_at=datetime.now(timezone.utc),
        )
        db.add(entry)

        health_scores.append(health)
        scored += 1

    affiliates_all = db.query(Affiliate).all()
    # Mutually-exclusive tiers via the rulebook — a churned affiliate is not
    # also double-counted as at_risk (see src/rulebook/recommend.categorize).
    tiers_all = [
        categorize(a.churn_risk_score or 0.0, a.growth_potential_score or 0.0)
        for a in affiliates_all
    ]
    at_risk = sum(1 for t in tiers_all if t == "at_risk")
    high_growth = sum(1 for t in tiers_all if t == "high_growth")
    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0.0

    logger.info(
        "Scoring run complete",
        extra={
            "affiliates_scored": scored,
            "avg_health_score": avg_health,
            "at_risk_count": at_risk,
            "high_growth_count": high_growth,
        },
    )
    return {
        "affiliates_scored": scored,
        "avg_health_score": avg_health,
        "at_risk_count": at_risk,
        "high_growth_count": high_growth,
    }