"""
SQLAlchemy ORM models.

Tables
------
affiliates      — one row per affiliate partner
communications  — every email / call / chat / ticket
score_history   — time-series of health scores + SHAP snapshots
"""

import uuid
from datetime import datetime, date

from sqlalchemy import (
    Column, String, Float, Date, DateTime, Text,
    ForeignKey, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─── Affiliates ──────────────────────────────────────────────────────────────

class Affiliate(Base):
    __tablename__ = "affiliates"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    company = Column(String(255))
    tier = Column(String(20), nullable=False, default="bronze")  # bronze|silver|gold|platinum
    join_date = Column(Date, nullable=False, default=date.today)
    country = Column(String(100))
    niche = Column(String(100))           # e.g. finance, travel, SaaS, e-commerce
    traffic_source = Column(String(100))  # SEO | PPC | Social | Email | Influencer

    monthly_revenue = Column(Float, default=0.0)

    # Model outputs — refreshed each time score pipeline runs
    churn_risk_score = Column(Float, default=0.5)
    growth_potential_score = Column(Float, default=0.5)
    health_score = Column(Float, default=50.0)

    last_contact_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    communications = relationship(
        "Communication", back_populates="affiliate", cascade="all, delete-orphan"
    )
    score_history = relationship(
        "ScoreHistory", back_populates="affiliate", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_affiliates_email", "email"),
        Index("ix_affiliates_tier", "tier"),
        Index("ix_affiliates_churn_risk", "churn_risk_score"),
        Index("ix_affiliates_growth", "growth_potential_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<Affiliate id={self.id} name={self.name!r} "
            f"tier={self.tier} health={self.health_score:.1f}>"
        )

    @property
    def health_score_computed(self) -> float:
        """Re-compute health score from current model scores."""
        return round(
            ((1 - self.churn_risk_score) * 0.6 + self.growth_potential_score * 0.4) * 100,
            1,
        )


# ─── Communications ──────────────────────────────────────────────────────────

class Communication(Base):
    __tablename__ = "communications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )
    channel = Column(String(20), nullable=False)     # email | call | chat | ticket
    direction = Column(String(10), nullable=False)   # inbound | outbound
    subject = Column(String(500))
    content = Column(Text, nullable=False)

    # NLP outputs
    sentiment_score = Column(Float)                  # VADER compound: -1.0 to 1.0
    sentiment_label = Column(String(20))             # positive | neutral | negative
    tags = Column(JSONB, nullable=False, default=list)  # ["growth_intent", ...]
    embedding_id = Column(String(255))               # ChromaDB document ID

    occurred_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    affiliate = relationship("Affiliate", back_populates="communications")

    __table_args__ = (
        Index("ix_comms_affiliate_id", "affiliate_id"),
        Index("ix_comms_occurred_at", "occurred_at"),
        Index("ix_comms_channel", "channel"),
        Index("ix_comms_tags", "tags", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return (
            f"<Communication id={self.id} affiliate={self.affiliate_id} "
            f"channel={self.channel} tags={self.tags}>"
        )


# ─── ScoreHistory ─────────────────────────────────────────────────────────────

class ScoreHistory(Base):
    __tablename__ = "score_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )

    churn_risk_score = Column(Float, nullable=False)
    growth_potential_score = Column(Float, nullable=False)
    health_score = Column(Float, nullable=False)

    # Snapshot of the feature vector used to produce these scores
    features = Column(JSONB, nullable=False, default=dict)

    # SHAP values: {feature_name: shap_value} for explainability
    shap_values = Column(JSONB, nullable=False, default=dict)

    model_version = Column(String(50), nullable=False, default="1.0.0")
    scored_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    affiliate = relationship("Affiliate", back_populates="score_history")

    __table_args__ = (
        Index("ix_score_history_affiliate_id", "affiliate_id"),
        Index("ix_score_history_scored_at", "scored_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ScoreHistory affiliate={self.affiliate_id} "
            f"churn={self.churn_risk_score:.2f} growth={self.growth_potential_score:.2f} "
            f"health={self.health_score:.1f} at={self.scored_at}>"
        )
