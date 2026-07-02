"""
SQLAlchemy ORM models.

Tables
------
affiliates      — one row per affiliate partner
communications  — every email / call / api_event
score_history   — time-series of health scores
leaked_codes    — promo/discount code leak detection events
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
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

    # Promo code currently assigned to this affiliate
    active_promo_code = Column(String(64), nullable=True)

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
    leaked_codes = relationship(
        "LeakedCode", back_populates="affiliate", cascade="all, delete-orphan"
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


# ─── Embeddings ───────────────────────────────────────────────────────────────

class Embedding(Base):
    """One row per communication chunk — stores the 384-dim pgvector embedding."""

    __tablename__ = "embeddings"

    id = Column(String, primary_key=True)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=True,
    )
    affiliate_name = Column(String, nullable=True)
    source = Column(String, nullable=True)
    chunk_text = Column(Text, nullable=True)
    tags = Column(ARRAY(String), nullable=True, default=list)
    occurred_at = Column(DateTime(timezone=True), nullable=True)
    embedding = Column(Vector(384), nullable=True)

    __table_args__ = (
        Index("ix_embeddings_affiliate_id", "affiliate_id"),
    )

    def __repr__(self) -> str:
        return f"<Embedding id={self.id} affiliate={self.affiliate_id}>"


# ─── LeakedCodes ──────────────────────────────────────────────────────────────

class LeakedCode(Base):
    """One row per detected promo/discount code leak event."""

    __tablename__ = "leaked_codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )

    code = Column(String(64), nullable=False)
    site = Column(String(128), nullable=False)       # site name, e.g. "voucherslug-mock"
    source_url = Column(Text, nullable=False)         # full page URL or fixture file path
    raw_snippet = Column(Text, nullable=True)         # HTML/text the code was found in
    scan_type = Column(String(16), nullable=False, default="scheduled")  # "scheduled" | "on_demand"
    found_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    affiliate = relationship("Affiliate", back_populates="leaked_codes")

    __table_args__ = (
        Index("ix_leaked_codes_affiliate_id", "affiliate_id"),
        Index("ix_leaked_codes_code", "code"),
    )

    def __repr__(self) -> str:
        return (
            f"<LeakedCode id={self.id} affiliate={self.affiliate_id} "
            f"code={self.code!r} site={self.site!r}>"
        )