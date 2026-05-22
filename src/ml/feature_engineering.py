"""
Feature Engineering
===================
Builds the feature DataFrame used by both the churn and growth models.

Each row = one affiliate. Features are aggregated from:
  - affiliates table  (static profile features)
  - communications table  (tag counts, sentiment aggregates, recency)

Feature groups
--------------
Profile      : tier_encoded, days_since_join, monthly_revenue
Recency      : days_since_last_contact
Sentiment    : avg_sentiment_30d, avg_sentiment_all, negative_ratio
Tag counts   : one column per NLP tag (30-day window + all-time)
Engagement   : comm_frequency_30d, comm_frequency_prev_30d,
               comm_frequency_change_ratio
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.storage.database import db_session
from src.storage.models import Affiliate, Communication
from src.ingestion.nlp_processor import ALL_TAGS

# ─── Tier encoding ────────────────────────────────────────────────────────────
TIER_MAP = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}

# ─── Reference date (override in tests) ──────────────────────────────────────
_NOW: Optional[datetime] = None


def _now() -> datetime:
    if _NOW is not None:
        return _NOW
    return datetime.now(timezone.utc)


# ─── Core builder ─────────────────────────────────────────────────────────────

def build_features(
    affiliate_ids: Optional[list[str]] = None,
    db: Optional[Session] = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame.

    Parameters
    ----------
    affiliate_ids : optional filter; if None, processes all affiliates
    db            : optional existing Session; if None opens its own

    Returns
    -------
    pd.DataFrame with one row per affiliate and columns:
        affiliate_id, + all feature columns (no target columns)
    """

    def _build(session: Session) -> pd.DataFrame:
        query = session.query(Affiliate)
        if affiliate_ids:
            query = query.filter(Affiliate.id.in_(affiliate_ids))
        affiliates = query.all()

        rows = []
        now = _now()
        window_30d = now - timedelta(days=30)
        window_60d = now - timedelta(days=60)

        for aff in affiliates:
            comms = (
                session.query(Communication)
                .filter(Communication.affiliate_id == aff.id)
                .all()
            )

            # ── Profile features ──────────────────────────────────────────────
            days_since_join = max(0, (now.date() - aff.join_date).days)
            tier_encoded = TIER_MAP.get(aff.tier, 1)

            # ── Recency ───────────────────────────────────────────────────────
            if aff.last_contact_date:
                lc = aff.last_contact_date
                if lc.tzinfo is None:
                    lc = lc.replace(tzinfo=timezone.utc)
                days_since_last_contact = max(0, (now - lc).days)
            else:
                days_since_last_contact = days_since_join

            # ── Sentiment aggregates ──────────────────────────────────────────
            recent_comms = [
                c for c in comms
                if c.occurred_at and _make_aware(c.occurred_at) >= window_30d
            ]
            sentiments_30d = [
                c.sentiment_score for c in recent_comms if c.sentiment_score is not None
            ]
            sentiments_all = [
                c.sentiment_score for c in comms if c.sentiment_score is not None
            ]
            avg_sentiment_30d = float(np.mean(sentiments_30d)) if sentiments_30d else 0.0
            avg_sentiment_all = float(np.mean(sentiments_all)) if sentiments_all else 0.0
            negative_ratio = (
                sum(1 for s in sentiments_all if s <= -0.05) / len(sentiments_all)
                if sentiments_all else 0.0
            )

            # ── Tag counts (all-time) ─────────────────────────────────────────
            tag_counts_all: dict[str, int] = {f"tag_{t}": 0 for t in ALL_TAGS}
            for comm in comms:
                for tag in (comm.tags or []):
                    key = f"tag_{tag}"
                    if key in tag_counts_all:
                        tag_counts_all[key] += 1

            # ── Tag counts (30-day window) ────────────────────────────────────
            tag_counts_30d: dict[str, int] = {f"tag30_{t}": 0 for t in ALL_TAGS}
            for comm in recent_comms:
                for tag in (comm.tags or []):
                    key = f"tag30_{tag}"
                    if key in tag_counts_30d:
                        tag_counts_30d[key] += 1

            # ── Communication frequency ───────────────────────────────────────
            comms_30d = len(recent_comms)
            comms_prev_30d = len([
                c for c in comms
                if c.occurred_at
                and window_60d <= _make_aware(c.occurred_at) < window_30d
            ])
            freq_change_ratio = (
                (comms_30d - comms_prev_30d) / max(comms_prev_30d, 1)
            )

            row = {
                "affiliate_id": str(aff.id),
                # Profile
                "tier_encoded": tier_encoded,
                "days_since_join": days_since_join,
                "monthly_revenue": aff.monthly_revenue or 0.0,
                # Recency
                "days_since_last_contact": days_since_last_contact,
                # Sentiment
                "avg_sentiment_30d": round(avg_sentiment_30d, 4),
                "avg_sentiment_all": round(avg_sentiment_all, 4),
                "negative_ratio": round(negative_ratio, 4),
                # Engagement
                "comms_30d": comms_30d,
                "comms_prev_30d": comms_prev_30d,
                "freq_change_ratio": round(freq_change_ratio, 4),
                # Tag counts (all-time)
                **tag_counts_all,
                # Tag counts (30d)
                **tag_counts_30d,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    if db is not None:
        return _build(db)
    with db_session() as session:
        return _build(session)


def _make_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def feature_columns() -> list[str]:
    """Return the ordered list of feature column names (excludes affiliate_id)."""
    base = [
        "tier_encoded", "days_since_join", "monthly_revenue",
        "days_since_last_contact",
        "avg_sentiment_30d", "avg_sentiment_all", "negative_ratio",
        "comms_30d", "comms_prev_30d", "freq_change_ratio",
    ]
    tag_all = [f"tag_{t}" for t in ALL_TAGS]
    tag_30d = [f"tag30_{t}" for t in ALL_TAGS]
    return base + tag_all + tag_30d


# ─── Synthetic label generation (for training on mock data) ───────────────────

def generate_synthetic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic churn and growth labels for training on mock data.

    Rule-based heuristic (replace with real labelled data in production):
      churn_label  = 1 if churn_risk_score in DB > 0.55 else 0
      growth_label = 1 if growth_potential_score in DB > 0.55 else 0

    We pull these from the DB since mock CSV already has plausible scores.
    """
    with db_session() as session:
        affiliates = {
            str(a.id): a for a in session.query(Affiliate).all()
        }

    df = df.copy()
    df["churn_label"] = df["affiliate_id"].apply(
        lambda aid: int(affiliates[aid].churn_risk_score > 0.55) if aid in affiliates else 0
    )
    df["growth_label"] = df["affiliate_id"].apply(
        lambda aid: int(affiliates[aid].growth_potential_score > 0.55) if aid in affiliates else 0
    )
    return df


if __name__ == "__main__":
    df = build_features()
    print(f"Feature matrix shape: {df.shape}")
    print(df[["affiliate_id", "tier_encoded", "monthly_revenue",
              "days_since_last_contact", "avg_sentiment_30d"]].to_string())
