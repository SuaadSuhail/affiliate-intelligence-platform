"""
Storage layer tests.

Covers
------
- Affiliate model: correct column defaults on insert
- Communication model: FK relationship to Affiliate
- ScoreHistory: unique constraint on (affiliate_id, scored_at)
- ChromaDB: add_document and search_similar return correct results

Infrastructure requirements
----------------------------
PostgreSQL tests  : docker-compose up -d postgres
ChromaDB tests    : use an in-memory ephemeral client (no server needed)

Tests that need PostgreSQL are wrapped in the `pg_session` fixture.
If PostgreSQL is not reachable they are automatically skipped.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

# ── Skip helpers ──────────────────────────────────────────────────────────────

def _postgres_available() -> bool:
    try:
        from src.storage.database import DATABASE_URL
        eng = create_engine(DATABASE_URL, pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_available(),
    reason="PostgreSQL not reachable — run: docker-compose up -d postgres",
)


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pg_engine():
    """
    Session-scoped engine.

    Drops all tables and recreates them with the current schema at the
    start of every test session.  This guarantees tests always run
    against the schema defined in models.py, regardless of any stale
    tables left by a previous scaffold or migration.
    """
    from src.storage.database import DATABASE_URL
    from src.storage.models import Base

    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.drop_all(bind=eng)   # remove any stale schema
    Base.metadata.create_all(bind=eng)  # create fresh from current models
    yield eng
    eng.dispose()


@pytest.fixture
def pg_session(pg_engine):
    """
    Function-scoped session wrapped in a transaction that is always
    rolled back — inserts made during a test never persist.

    flush() is used instead of commit() so generated IDs are visible
    within the test without permanently writing to the database.

    The try/except in teardown silences the benign SQLAlchemy warning
    that fires when a test has already rolled back (e.g. after catching
    an IntegrityError) before the fixture teardown runs.
    """
    connection = pg_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def chroma_col(monkeypatch):
    """
    Provides an in-memory ephemeral ChromaDB collection and patches the
    vector_store module to use it — no running ChromaDB server required.
    """
    import chromadb
    import src.storage.vector_store as vs

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        "test_affiliate_comms",
        metadata={"hnsw:space": "cosine"},
    )
    # Inject into the module so all public functions use the test collection
    monkeypatch.setattr(vs, "_collection", col)
    monkeypatch.setattr(vs, "_client", client)
    yield col
    # Cleanup: delete the in-memory collection (ephemeral, but be explicit)
    try:
        client.delete_collection("test_affiliate_comms")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────────────────────

def make_affiliate(**kwargs) -> "Affiliate":
    from src.storage.models import Affiliate

    defaults = {"name": f"Test Affiliate {uuid.uuid4().hex[:6]}"}
    defaults.update(kwargs)
    return Affiliate(**defaults)


def make_communication(affiliate_id, **kwargs) -> "Communication":
    from src.storage.models import Communication

    defaults = {
        "affiliate_id": affiliate_id,
        "source": "email",
        "raw_text": "This is a test communication.",
        "occurred_at": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return Communication(**defaults)


def make_score(affiliate_id, scored_at=None, **kwargs) -> "ScoreHistory":
    from src.storage.models import ScoreHistory

    defaults = {
        "affiliate_id": affiliate_id,
        "scored_at": scored_at or date.today(),
        "churn_risk_score": 0.3,
        "growth_potential_score": 0.7,
        "health_score": 62.0,
    }
    defaults.update(kwargs)
    return ScoreHistory(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Affiliate model — defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestAffiliateDefaults:

    @requires_postgres
    def test_default_status_is_active(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert aff.status == "active"

    @requires_postgres
    def test_default_scores_are_zero(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert aff.churn_risk_score == 0.0
        assert aff.growth_potential_score == 0.0
        assert aff.health_score == 0.0

    @requires_postgres
    def test_default_revenue_and_ctr_are_zero(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert float(aff.revenue_30d) == 0.0
        assert aff.ctr_trend_pct == 0.0

    @requires_postgres
    def test_days_since_contact_zero_when_no_contact(self, pg_session):
        """days_since_contact should default to 0 when last_contact_at is None."""
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert aff.days_since_contact == 0

    @requires_postgres
    def test_days_since_contact_computed_on_save(self, pg_session):
        """Event listener should compute days_since_contact from last_contact_at."""
        past = datetime.now(timezone.utc) - timedelta(days=5)
        aff = make_affiliate(last_contact_at=past)
        pg_session.add(aff)
        pg_session.flush()
        # Allow ±1 day tolerance for timing edge cases
        assert 4 <= aff.days_since_contact <= 6

    @requires_postgres
    def test_updated_at_is_set_on_insert(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert aff.updated_at is not None

    @requires_postgres
    def test_id_is_uuid(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()
        assert isinstance(aff.id, uuid.UUID)

    @requires_postgres
    def test_name_is_required(self, pg_session):
        """Inserting an Affiliate with no name should raise IntegrityError."""
        from src.storage.models import Affiliate

        aff = Affiliate()  # name is NULL
        pg_session.add(aff)
        with pytest.raises(IntegrityError):
            pg_session.flush()
        pg_session.rollback()

    @requires_postgres
    def test_valid_status_values_accepted(self, pg_session):
        for status in ("active", "at_risk", "churned", "high_growth"):
            aff = make_affiliate(status=status)
            pg_session.add(aff)
            pg_session.flush()
            assert aff.status == status
            pg_session.expunge(aff)  # detach so next iteration is clean


# ─────────────────────────────────────────────────────────────────────────────
# 2. Communication model — relationship
# ─────────────────────────────────────────────────────────────────────────────

class TestCommunicationRelationship:

    @requires_postgres
    def test_communication_links_to_affiliate(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        comm = make_communication(affiliate_id=aff.id)
        pg_session.add(comm)
        pg_session.flush()

        assert comm.affiliate_id == aff.id
        assert comm.affiliate.name == aff.name

    @requires_postgres
    def test_communication_source_values(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        for src in ("email", "call", "api_event"):
            comm = make_communication(affiliate_id=aff.id, source=src)
            pg_session.add(comm)
            pg_session.flush()
            assert comm.source == src
            pg_session.expunge(comm)

    @requires_postgres
    def test_communication_tags_stored_as_array(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        tags = ["churn_signal", "urgency", "payment_issue"]
        comm = make_communication(affiliate_id=aff.id, tags=tags)
        pg_session.add(comm)
        pg_session.flush()

        # Re-query to confirm the array round-trips through PostgreSQL
        fetched = pg_session.get(type(comm), comm.id)
        assert fetched.tags == tags

    @requires_postgres
    def test_affiliate_communications_back_reference(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        for i in range(3):
            pg_session.add(
                make_communication(
                    affiliate_id=aff.id,
                    raw_text=f"Communication {i}",
                )
            )
        pg_session.flush()

        pg_session.refresh(aff)
        assert len(aff.communications) == 3

    @requires_postgres
    def test_communication_default_sentiment_zero(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        comm = make_communication(affiliate_id=aff.id)
        pg_session.add(comm)
        pg_session.flush()

        assert comm.sentiment_score == 0.0

    @requires_postgres
    def test_embedding_id_is_nullable(self, pg_session):
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        comm = make_communication(affiliate_id=aff.id)
        pg_session.add(comm)
        pg_session.flush()

        assert comm.embedding_id is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. ScoreHistory — unique constraint
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreHistoryUniqueConstraint:

    @requires_postgres
    def test_one_score_per_affiliate_per_day(self, pg_session):
        """Two ScoreHistory rows for the same affiliate on the same day → IntegrityError."""
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        test_date = date(2026, 6, 1)

        pg_session.add(make_score(aff.id, scored_at=test_date))
        pg_session.flush()  # first insert succeeds

        pg_session.add(
            make_score(aff.id, scored_at=test_date, churn_risk_score=0.9)
        )
        with pytest.raises(IntegrityError):
            pg_session.flush()  # duplicate (affiliate_id, scored_at) → violation

        pg_session.rollback()  # required after IntegrityError to restore session

    @requires_postgres
    def test_same_affiliate_different_days_allowed(self, pg_session):
        """The same affiliate can have scores on different dates."""
        aff = make_affiliate()
        pg_session.add(aff)
        pg_session.flush()

        for delta in range(3):
            pg_session.add(
                make_score(aff.id, scored_at=date(2026, 6, delta + 1))
            )
        pg_session.flush()  # all three should succeed without error

        pg_session.refresh(aff)
        assert len(aff.score_history) == 3

    @requires_postgres
    def test_different_affiliates_same_day_allowed(self, pg_session):
        """Two different affiliates can each have a score on the same date."""
        aff1 = make_affiliate()
        aff2 = make_affiliate()
        pg_session.add_all([aff1, aff2])
        pg_session.flush()

        test_date = date(2026, 6, 15)
        pg_session.add(make_score(aff1.id, scored_at=test_date))
        pg_session.add(make_score(aff2.id, scored_at=test_date))
        pg_session.flush()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 4. ChromaDB vector store
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:

    def test_add_document_then_get_by_affiliate(self, chroma_col):
        """add_document stores a document retrievable by affiliate_id."""
        from src.storage.vector_store import add_document, get_by_affiliate

        aff_id = str(uuid.uuid4())
        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="The affiliate is asking about payment discrepancies.",
            affiliate_id=aff_id,
            affiliate_name="Alice Test",
            source="email",
            tags=["payment_issue"],
            occurred_at="2026-06-01T10:00:00",
        )

        docs = get_by_affiliate(affiliate_id=aff_id)
        assert len(docs) == 1
        assert docs[0]["source"] == "email"
        assert "payment_issue" in docs[0]["tags"]

    def test_add_multiple_documents_for_one_affiliate(self, chroma_col):
        """get_by_affiliate respects the limit parameter."""
        from src.storage.vector_store import add_document, get_by_affiliate

        aff_id = str(uuid.uuid4())
        for i in range(5):
            add_document(
                doc_id=f"comm_{uuid.uuid4()}",
                text=f"Communication number {i} about campaigns.",
                affiliate_id=aff_id,
                affiliate_name="Bob Test",
                source="call",
                tags=["growth_intent"],
                occurred_at=f"2026-06-0{i + 1}T10:00:00",
            )

        all_docs = get_by_affiliate(aff_id, limit=10)
        assert len(all_docs) == 5

        limited = get_by_affiliate(aff_id, limit=3)
        assert len(limited) == 3

    def test_search_similar_returns_relevant_result(self, chroma_col):
        """search_similar should return the semantically closest document."""
        from src.storage.vector_store import add_document, search_similar

        aff_id = str(uuid.uuid4())
        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="I want to cancel my account. Very disappointed with service.",
            affiliate_id=aff_id,
            affiliate_name="Carol Test",
            source="email",
            tags=["churn_signal", "satisfaction_low"],
            occurred_at="2026-06-01T09:00:00",
        )
        # Unrelated doc — should rank lower
        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="Looking forward to scaling up our campaigns next quarter.",
            affiliate_id=str(uuid.uuid4()),
            affiliate_name="Dave Test",
            source="call",
            tags=["growth_intent"],
            occurred_at="2026-06-01T11:00:00",
        )

        results = search_similar("affiliate wants to leave and cancel", n_results=2)
        assert len(results) >= 1
        # The churn-related document should rank first
        assert results[0]["affiliate_name"] == "Carol Test"

    def test_search_similar_with_affiliate_filter(self, chroma_col):
        """filter_affiliate_id restricts results to one affiliate."""
        from src.storage.vector_store import add_document, search_similar

        aff_a = str(uuid.uuid4())
        aff_b = str(uuid.uuid4())

        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="Payment was delayed again this month.",
            affiliate_id=aff_a,
            affiliate_name="Affiliate A",
            source="email",
            tags=["payment_issue"],
            occurred_at="2026-06-01T08:00:00",
        )
        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="Payment issue unresolved for two weeks.",
            affiliate_id=aff_b,
            affiliate_name="Affiliate B",
            source="email",
            tags=["payment_issue"],
            occurred_at="2026-06-01T08:30:00",
        )

        results = search_similar(
            "payment problem",
            n_results=5,
            filter_affiliate_id=aff_a,
        )
        assert all(r["affiliate_id"] == aff_a for r in results)

    def test_search_similar_with_tag_filter(self, chroma_col):
        """filter_tags removes results that do not contain all required tags."""
        from src.storage.vector_store import add_document, search_similar

        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="Urgent: payment is missing, I will escalate this.",
            affiliate_id=str(uuid.uuid4()),
            affiliate_name="Escalation Aff",
            source="email",
            tags=["payment_issue", "urgency", "escalation_risk"],
            occurred_at="2026-06-02T10:00:00",
        )
        add_document(
            doc_id=f"comm_{uuid.uuid4()}",
            text="Just a routine check-in about our commission payment.",
            affiliate_id=str(uuid.uuid4()),
            affiliate_name="Routine Aff",
            source="email",
            tags=["payment_issue"],
            occurred_at="2026-06-02T11:00:00",
        )

        results = search_similar(
            "payment",
            n_results=5,
            filter_tags=["urgency"],
        )
        # Only the escalation document has the "urgency" tag
        assert len(results) == 1
        assert "urgency" in results[0]["tags"]

    def test_delete_by_affiliate_removes_all_docs(self, chroma_col):
        """delete_by_affiliate removes every document for the given affiliate."""
        from src.storage.vector_store import add_document, delete_by_affiliate, get_by_affiliate

        aff_id = str(uuid.uuid4())
        for i in range(3):
            add_document(
                doc_id=f"comm_{uuid.uuid4()}",
                text=f"Document {i} for deletion test.",
                affiliate_id=aff_id,
                affiliate_name="Delete Test Aff",
                source="api_event",
                tags=[],
                occurred_at="2026-06-03T12:00:00",
            )

        assert len(get_by_affiliate(aff_id)) == 3
        delete_by_affiliate(aff_id)
        assert len(get_by_affiliate(aff_id)) == 0

    def test_upsert_overwrites_existing_document(self, chroma_col):
        """Calling add_document with the same doc_id updates the document."""
        from src.storage.vector_store import add_document, get_by_affiliate

        aff_id = str(uuid.uuid4())
        doc_id = f"comm_{uuid.uuid4()}"

        add_document(
            doc_id=doc_id,
            text="Original text.",
            affiliate_id=aff_id,
            affiliate_name="Upsert Aff",
            source="email",
            tags=["growth_intent"],
            occurred_at="2026-06-01T10:00:00",
        )
        # Upsert with updated text
        add_document(
            doc_id=doc_id,
            text="Updated text with new information.",
            affiliate_id=aff_id,
            affiliate_name="Upsert Aff",
            source="email",
            tags=["growth_intent", "new_opportunity"],
            occurred_at="2026-06-01T10:00:00",
        )

        docs = get_by_affiliate(aff_id)
        assert len(docs) == 1  # still one doc, not two
        assert "new_opportunity" in docs[0]["tags"]