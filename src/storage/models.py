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
    Boolean, Column, String, Float, Integer, DateTime, Text,
    ForeignKey, Index, Numeric, Enum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
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

    # First-class, queryable leak signal — kept separate from and visible
    # alongside churn/growth scores, not folded into them (src.rulebook.recommend
    # already keeps the tier leak-independent; this makes that same signal
    # visible at the storage/query layer too). Recomputed by
    # src.scraping.leakage_scraper.check_leakage from the leaked_codes table
    # on every scan — see that module for what "active" means here.
    has_active_leak = Column(Boolean, nullable=False, default=False)

    # Keyword this affiliate is tracked against for SEO rank checks — mirrors
    # active_promo_code's role for leak detection (identifies what to look
    # for; matching happens against src.seo.api_client's fetched rows).
    tracked_keyword = Column(String(255), nullable=True)

    # First-class, queryable SEO signal — kept separate from and visible
    # alongside churn/growth scores, not folded into them, same principle as
    # has_active_leak. A string, not a boolean: 'declining' | 'stable' |
    # 'improving' is inherently three-state, and collapsing it to a boolean
    # would lose the "improving" case. Recomputed by
    # src.seo.checker.check_seo via src.seo.analyze.derive_search_trend.
    search_trend = Column(String(20), nullable=False, default="stable")

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
    approval_requests = relationship(
        "ApprovalRequest", back_populates="affiliate", cascade="all, delete-orphan"
    )
    seo_signals = relationship(
        "SeoSignal", back_populates="affiliate", cascade="all, delete-orphan"
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

    # The rulebook's evidence list (src.rulebook.recommend.Recommendation.evidence)
    # for this affiliate's inputs at scoring time — the specific facts behind the
    # score, not just the score itself. Nullable: historical rows predate this
    # column and are not backfilled.
    evidence_bundle = Column(JSONB, nullable=True)

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
        Index(
            "embeddings_vector_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": "10"},
        ),
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


# ─── ApprovalRequests ───────────────────────────────────────────────────────

class ApprovalRequest(Base):
    """
    One row per action that would leave the system (send email, notify
    partner, ...). Nothing external fires until a human approves the row
    here — see src/api/routers/approvals.py and src/notifications/sender.py.
    """

    __tablename__ = "approval_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind = Column(String(32), nullable=False)  # "email" for now; room for more kinds later
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )
    payload = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="waiting_for_review")
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    decided_at = Column(DateTime(timezone=True), nullable=True)
    decided_by = Column(String(128), nullable=True)

    # Relationships
    affiliate = relationship("Affiliate", back_populates="approval_requests")

    __table_args__ = (
        Index("ix_approval_requests_affiliate_id", "affiliate_id"),
        Index("ix_approval_requests_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<ApprovalRequest id={self.id} kind={self.kind!r} "
            f"status={self.status!r} affiliate={self.affiliate_id}>"
        )


# ─── AuditLog ─────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Append-only record linking any stored recommendation/decision/outcome
    back to the specific record and rule/tool that produced it. The single
    write path is src.audit.log.write_audit_entry() — see src/audit/log.py.

    record_id is a plain UUID column, not a foreign key: record_type varies
    (affiliate, approval_request, ...), so it's a polymorphic reference, not
    a link to one fixed table.
    """

    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    stage = Column(String(20), nullable=False)  # signals | rulebook | agent | approval
    record_type = Column(String(32), nullable=False)  # e.g. "affiliate", "approval_request"
    record_id = Column(UUID(as_uuid=True), nullable=False)
    rule_or_tool = Column(String(64), nullable=False)  # e.g. "recommend", "check_leakage", "approvals.approve"
    input_snapshot = Column(JSONB, nullable=False)
    output_snapshot = Column(JSONB, nullable=False)

    __table_args__ = (
        Index("ix_audit_log_record_type_record_id", "record_type", "record_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} stage={self.stage!r} rule_or_tool={self.rule_or_tool!r} "
            f"record={self.record_type}:{self.record_id}>"
        )


# ─── SeoSignals ───────────────────────────────────────────────────────────────

class SeoSignal(Base):
    """
    One row per keyword rank check — mirrors LeakedCode's pattern of storing
    real evidence per row, not just a rolled-up score. See src/seo/checker.py
    for the write path and src/seo/analyze.py for how these feed
    affiliates.search_trend.
    """

    __tablename__ = "seo_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    affiliate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("affiliates.id", ondelete="CASCADE"),
        nullable=False,
    )

    keyword = Column(String(255), nullable=False)
    rank = Column(Integer, nullable=False)
    # previous_rank - rank at check time; None if no prior rank was on record
    # for this keyword. See src.seo.analyze for the sign convention.
    rank_change = Column(Integer, nullable=True)
    search_volume = Column(Integer, nullable=True)
    checked_at = Column(DateTime(timezone=True), nullable=False)

    # Relationships
    affiliate = relationship("Affiliate", back_populates="seo_signals")

    __table_args__ = (
        Index("ix_seo_signals_affiliate_id", "affiliate_id"),
        Index("ix_seo_signals_keyword", "keyword"),
    )

    def __repr__(self) -> str:
        return (
            f"<SeoSignal id={self.id} affiliate={self.affiliate_id} "
            f"keyword={self.keyword!r} rank={self.rank}>"
        )