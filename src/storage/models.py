"""
SQLAlchemy ORM models — single source of truth for the database schema.

Tables
------
affiliates      one row per affiliate partner
communications  every ingested email / call / API event
score_history   daily score snapshots (one row per affiliate per day)

Schema reference: CLAUDE.md § 3
"""

import uuid
from datetime import datetime, date, timezone

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Shared base ───────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enum type definitions ─────────────────────────────────────────────────────

AffiliateStatus = Enum(
    "active",
    "at_risk",
    "churned",
    "high_growth",
    name="affiliate_status",
)

CommunicationSource = Enum(
    "email",
    "call",
    "api_event",
    name="communication_source",
)


# ─────────────────────────────────────────────────────────────────────────────
# Affiliate
# ─────────────────────────────────────────────────────────────────────────────

class Affiliate(Base):
    __tablename__ = "affiliates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)

    # Lifecycle status — drives dashboard colouring and alert routing
    status = Column(AffiliateStatus, nullable=False, default="active")

    # ML model outputs — refreshed by the scoring pipeline
    churn_risk_score = Column(Float, nullable=False, default=0.0)
    growth_potential_score = Column(Float, nullable=False, default=0.0)
    health_score = Column(Float, nullable=False, default=0.0)

    # Revenue and engagement metrics
    revenue_30d = Column(Numeric(10, 2), nullable=False, default=0.0)
    ctr_trend_pct = Column(Float, nullable=False, default=0.0)  # % change in CTR

    # Contact recency — last_contact_at is set externally;
    # days_since_contact is recomputed automatically before every save.
    last_contact_at = Column(DateTime(timezone=True), nullable=True)
    days_since_contact = Column(Integer, nullable=False, default=0)

    # Audit
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    communications = relationship(
        "Communication",
        back_populates="affiliate",
        cascade="all, delete-orphan",
    )
    score_history = relationship(
        "ScoreHistory",
        back_populates="affiliate",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_affiliates_status", "status"),
        Index("ix_affiliates_health_score", "health_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<Affiliate id={self.id} name={self.name!r} "
            f"status={self.status} health={self.health_score:.2f}>"
        )


# ── Event: recompute days_since_contact before every insert / update ──────────

def _compute_days_since_contact(
    mapper,      # noqa: ARG001  (required by SQLAlchemy event signature)
    connection,  # noqa: ARG001
    target: Affiliate,
) -> None:
    """Recompute days_since_contact from last_contact_at before saving."""
    if target.last_contact_at is None:
        target.days_since_contact = 0
        return
    now = datetime.now(timezone.utc)
    lc = target.last_contact_at
    if lc.tzinfo is None:
        lc = lc.replace(tzinfo=timezone.utc)
    target.days_since_contact = max(0, (now - lc).days)


event.listen(Affiliate, "before_insert", _compute_days_since_contact)
event.listen(Affiliate, "before_update", _compute_days_since_contact)


# ─────────────────────────────────────────────────────────────────────────────
# Communication
# ─────────────────────────────────────────────────────────────────────────────

class Communication(Base):
    __tablename__ = "communications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )

    source = Column(CommunicationSource, nullable=False)
    raw_text = Column(Text, nullable=False)

    # NLP outputs — populated by the ingestion pipeline
    tags = Column(ARRAY(String()), nullable=False, default=list)
    sentiment_score = Column(Float, nullable=False, default=0.0)
    embedding_id = Column(String(255), nullable=True)  # ChromaDB document ID

    occurred_at = Column(DateTime(timezone=True), nullable=False)

    # Relationship
    affiliate = relationship("Affiliate", back_populates="communications")

    __table_args__ = (
        Index("ix_communications_affiliate_id", "affiliate_id"),
        Index("ix_communications_occurred_at", "occurred_at"),
        Index("ix_communications_source", "source"),
    )

    def __repr__(self) -> str:
        return (
            f"<Communication id={self.id} affiliate_id={self.affiliate_id} "
            f"source={self.source} sentiment={self.sentiment_score:.3f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ScoreHistory
# ─────────────────────────────────────────────────────────────────────────────

class ScoreHistory(Base):
    """
    One row per affiliate per day.

    The unique constraint on (affiliate_id, scored_at) ensures we never
    accumulate more than one snapshot per day — the scoring pipeline should
    upsert via merge_score() rather than blindly inserting.
    """
    __tablename__ = "score_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )

    scored_at = Column(Date, nullable=False, default=date.today)
    churn_risk_score = Column(Float, nullable=False)
    growth_potential_score = Column(Float, nullable=False)
    health_score = Column(Float, nullable=False)

    # Relationship
    affiliate = relationship("Affiliate", back_populates="score_history")

    __table_args__ = (
        # One score snapshot per affiliate per day
        UniqueConstraint(
            "affiliate_id",
            "scored_at",
            name="uq_score_history_affiliate_day",
        ),
        Index("ix_score_history_affiliate_id", "affiliate_id"),
        Index("ix_score_history_scored_at", "scored_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ScoreHistory affiliate={self.affiliate_id} "
            f"date={self.scored_at} "
            f"churn={self.churn_risk_score:.3f} "
            f"growth={self.growth_potential_score:.3f} "
            f"health={self.health_score:.2f}>"
        )