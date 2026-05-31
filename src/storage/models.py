"""
SQLAlchemy ORM models.

Tables
------
affiliates      — one row per affiliate partner
communications  — every email / call / api_event
score_history   — time-series of health scores
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text,
    ForeignKey, Index, Numeric, Enum,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─── Affiliates ──────────────────────────────────────────────────────────────

class Affiliate(Base):
    __tablename__ = "affiliates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    status = Column(
        Enum("active", "at_risk", "churned", "high_growth", name="affiliate_status"),
        nullable=False,
        default="active",
    )

    # Model outputs
    churn_risk_score = Column(Float, nullable=False, default=0.5)
    growth_potential_score = Column(Float, nullable=False, default=0.5)
    health_score = Column(Float, nullable=False, default=50.0)

    # Revenue / engagement signals
    revenue_30d = Column(Numeric(10, 2), nullable=False, default=0.0)
    ctr_trend_pct = Column(Float, nullable=False, default=0.0)

    # Contact tracking
    last_contact_at = Column(DateTime(timezone=True), nullable=True)
    days_since_contact = Column(Integer, nullable=False, default=0)

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
        Index("ix_affiliates_status", "status"),
        Index("ix_affiliates_churn_risk", "churn_risk_score"),
        Index("ix_affiliates_growth", "growth_potential_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<Affiliate id={self.id} name={self.name!r} "
            f"status={self.status} health={self.health_score:.1f}>"
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
    source = Column(
        Enum("email", "call", "api_event", name="communication_source"),
        nullable=False,
    )
    raw_text = Column(Text, nullable=False)

    # NLP outputs
    tags = Column(ARRAY(String), nullable=False, default=list)
    sentiment_score = Column(Float, nullable=False, default=0.0)
    embedding_id = Column(String(255), nullable=True)

    occurred_at = Column(DateTime(timezone=True), nullable=False)

    # Relationships
    affiliate = relationship("Affiliate", back_populates="communications")

    __table_args__ = (
        Index("ix_comms_affiliate_id", "affiliate_id"),
        Index("ix_comms_occurred_at", "occurred_at"),
        Index("ix_comms_source", "source"),
    )

    def __repr__(self) -> str:
        return (
            f"<Communication id={self.id} affiliate={self.affiliate_id} "
            f"source={self.source} tags={self.tags}>"
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