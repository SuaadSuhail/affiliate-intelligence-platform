"""
Feature Engineering
===================
Builds a 12-feature vector for each affiliate from PostgreSQL data.

Feature groups
--------------
Activity    : days_since_contact, revenue_30d, ctr_trend_pct
Communication: avg_sentiment_30d, comm_count_30d, churn_signal_count,
               positive_signal_count, escalation_count, competitor_mention_count
Derived     : sentiment_trend, response_rate, days_since_positive
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.storage.models import Affiliate, Communication

# ─── Feature names in fixed order ────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Activity
    "days_since_contact",
    "revenue_30d",
    "ctr_trend_pct",
    # Communication (30-day window)
    "avg_sentiment_30d",
    "comm_count_30d",
    "churn_signal_count",
    "positive_signal_count",
    "escalation_count",
    "competitor_mention_count",
    # Derived
    "sentiment_trend",
    "response_rate",
    "days_since_positive",
]

_POSITIVE_TAGS = {"enthusiastic", "positive_sentiment", "expansion_interest"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _derive_status(aff: Affiliate) -> str:
    """Derive a status label from existing score columns."""
    if (aff.churn_risk_score or 0.0) > 0.7:
        return "at_risk"
    if (aff.growth_potential_score or 0.0) > 0.7:
        return "high_growth"
    return "active"


# ─── Core feature builder ─────────────────────────────────────────────────────

def build_feature_vector(affiliate_id: str, db: Session) -> dict:
    """
    Build a feature vector for one affiliate.

    Parameters
    ----------
    affiliate_id : UUID string
    db           : active SQLAlchemy session

    Returns
    -------
    Dict with keys: affiliate_id, affiliate_name, status, + all 12 FEATURE_NAMES.
    """
    import uuid
    aff = db.query(Affiliate).filter(Affiliate.id == uuid.UUID(affiliate_id)).first()
    if aff is None:
        return {
            "affiliate_id": affiliate_id,
            "affiliate_name": "Unknown",
            "status": "active",
            **{f: 0.0 for f in FEATURE_NAMES},
        }

    now = _now()
    cutoff_30d = now - timedelta(days=30)
    cutoff_15d = now - timedelta(days=15)

    # All communications for this affiliate
    all_comms = (
        db.query(Communication)
        .filter(Communication.affiliate_id == aff.id)
        .all()
    )

    # Last-30-day communications
    recent_comms = [
        c for c in all_comms
        if c.occurred_at and _make_aware(c.occurred_at) >= cutoff_30d
    ]

    # ── GROUP 1: Activity ─────────────────────────────────────────────────────

    # last_contact_at is the new schema field (was last_contact_date in old schema)
    lc_field = getattr(aff, "last_contact_at", None) or getattr(aff, "last_contact_date", None)
    if lc_field:
        lc = _make_aware(lc_field)
        days_since_contact = max(0, (now - lc).days)
    else:
        days_since_contact = int(getattr(aff, "days_since_contact", 30) or 30)

    # revenue_30d (new schema) or fall back to monthly_revenue (old schema)
    revenue_30d = float(getattr(aff, "revenue_30d", None) or getattr(aff, "monthly_revenue", None) or 0.0)

    # ctr_trend_pct is not stored in this schema — default to 0.0
    ctr_trend_pct = 0.0

    # ── GROUP 2: Communication features ──────────────────────────────────────

    comm_count_30d = len(recent_comms)

    sentiments_30d = [
        c.sentiment_score for c in recent_comms if c.sentiment_score is not None
    ]
    avg_sentiment_30d = (
        sum(sentiments_30d) / len(sentiments_30d) if sentiments_30d else 0.0
    )

    def _has_tag(comm: Communication, tag: str) -> bool:
        return tag in (comm.tags or [])

    churn_signal_count = sum(1 for c in recent_comms if _has_tag(c, "churn_signal"))
    positive_signal_count = sum(
        1 for c in recent_comms if any(_has_tag(c, t) for t in _POSITIVE_TAGS)
    )
    escalation_count = sum(1 for c in recent_comms if _has_tag(c, "escalation") or _has_tag(c, "escalation_risk"))
    competitor_mention_count = sum(
        1 for c in recent_comms if _has_tag(c, "competitor_mention")
    )

    # ── GROUP 3: Derived features ─────────────────────────────────────────────

    # sentiment_trend: last 15d avg minus previous 15d avg
    last_15d_comms = [
        c for c in recent_comms
        if c.occurred_at and _make_aware(c.occurred_at) >= cutoff_15d
    ]
    prev_15d_comms = [
        c for c in recent_comms
        if c.occurred_at and _make_aware(c.occurred_at) < cutoff_15d
    ]
    last_sents = [c.sentiment_score for c in last_15d_comms if c.sentiment_score is not None]
    prev_sents = [c.sentiment_score for c in prev_15d_comms if c.sentiment_score is not None]
    avg_last = sum(last_sents) / len(last_sents) if last_sents else 0.0
    avg_prev = sum(prev_sents) / len(prev_sents) if prev_sents else 0.0
    sentiment_trend = round(avg_last - avg_prev, 4) if (last_sents or prev_sents) else 0.0

    # response_rate: new schema has no direction column — default to 0.5
    total = len(all_comms)
    response_rate = 0.5 if total == 0 else min(1.0, total / max(1, days_since_contact or 1) / 10)

    # days_since_positive: days since last positive/enthusiastic comm
    positive_comms = [
        c for c in all_comms
        if any(_has_tag(c, t) for t in {"enthusiastic", "positive_sentiment", "satisfaction_high"})
        and c.occurred_at
    ]
    if positive_comms:
        last_positive = max(_make_aware(c.occurred_at) for c in positive_comms)
        days_since_positive = max(0, (now - last_positive).days)
    else:
        days_since_positive = days_since_contact

    return {
        "affiliate_id": affiliate_id,
        "affiliate_name": aff.name,
        "status": _derive_status(aff),
        # Activity
        "days_since_contact": days_since_contact,
        "revenue_30d": round(revenue_30d, 2),
        "ctr_trend_pct": ctr_trend_pct,
        # Communication
        "avg_sentiment_30d": round(avg_sentiment_30d, 4),
        "comm_count_30d": comm_count_30d,
        "churn_signal_count": churn_signal_count,
        "positive_signal_count": positive_signal_count,
        "escalation_count": escalation_count,
        "competitor_mention_count": competitor_mention_count,
        # Derived
        "sentiment_trend": sentiment_trend,
        "response_rate": round(response_rate, 4),
        "days_since_positive": days_since_positive,
    }


def build_all_features(db: Session) -> list[dict]:
    """
    Build feature vectors for all affiliates.

    Returns
    -------
    List of dicts — each with affiliate_id, affiliate_name, status,
    and all 12 feature values.
    """
    affiliates = db.query(Affiliate).all()
    return [build_feature_vector(str(aff.id), db) for aff in affiliates]


def get_feature_dataframe(db: Session) -> pd.DataFrame:
    """
    Call build_all_features() and return a DataFrame with affiliate_id as index.

    Returns
    -------
    pd.DataFrame — columns: affiliate_name, status, + 12 feature columns
    Index: affiliate_id
    """
    rows = build_all_features(db)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.set_index("affiliate_id")
    return df