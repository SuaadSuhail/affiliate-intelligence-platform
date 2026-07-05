"""
Agent Tests
===========
Tests for LangChain tools (query_database, semantic_search, get_affiliate_summary,
get_portfolio_health) and agent initialisation.

draft_email and full agent runs are NOT tested here — they make real API calls.

Run:
    pytest tests/test_agent.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_affiliate(
    name="Test Affiliate", churn=0.4, growth=0.6, health=60.0, days=5,
    has_active_leak=False, search_trend="stable",
):
    from src.storage.models import Affiliate
    a = Affiliate()
    a.id = uuid.uuid4()
    a.name = name
    a.status = "active"
    a.churn_risk_score = churn
    a.growth_potential_score = growth
    a.health_score = health
    a.revenue_30d = 10000.0
    a.ctr_trend_pct = 0.0
    a.days_since_contact = days
    a.last_contact_at = datetime.now(timezone.utc)
    a.has_active_leak = has_active_leak
    a.search_trend = search_trend
    return a


def _mock_db_with_affiliates(affiliates):
    """Return a mock SessionLocal() that yields the given affiliates list."""
    mock_db = MagicMock()
    q = MagicMock()
    q.all.return_value = affiliates
    q.filter.return_value = q
    q.filter_by.return_value = q
    q.order_by.return_value = q
    q.limit.return_value = q
    q.first.return_value = affiliates[0] if affiliates else None
    q.count.return_value = len(affiliates)
    mock_db.query.return_value = q
    mock_db.execute.return_value = MagicMock(
        fetchmany=lambda n: [],
        keys=lambda: [],
    )
    return mock_db


# ─── Test 1: query_database — valid SELECT ────────────────────────────────────

def test_query_database_valid_select():
    """A valid SELECT should return formatted rows or a 'no rows' message."""
    from src.agent.tools import query_database

    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [("Sarah Chen", 72.4, 0.18)]
    mock_result.keys.return_value = ["name", "health_score", "churn_risk_score"]

    mock_db = MagicMock()
    mock_db.execute.return_value = mock_result

    with patch("src.agent.tools._get_db", return_value=mock_db):
        result = query_database.invoke("SELECT name, health_score FROM affiliates LIMIT 5")

    assert "name" in result.lower() or "sarah" in result.lower() or "no rows" in result.lower()


# ─── Test 2: query_database — non-SELECT rejected ─────────────────────────────

def test_query_database_rejects_non_select():
    """Non-SELECT queries must return a safe error string (not raise)."""
    from src.agent.tools import query_database

    result_drop = query_database.invoke("DROP TABLE affiliates")
    assert "only select" in result_drop.lower() or "not allowed" in result_drop.lower() or "blocked" in result_drop.lower()

    result_update = query_database.invoke("UPDATE affiliates SET health_score=0")
    assert "only select" in result_update.lower() or "not allowed" in result_update.lower() or "blocked" in result_update.lower()


# ─── Test 3: get_affiliate_summary — known affiliate ─────────────────────────

def test_get_affiliate_summary_found():
    """get_affiliate_summary must return a profile block for a known affiliate."""
    from src.agent.tools import get_affiliate_summary
    from src.storage.models import Communication

    aff = _make_affiliate(name="Sarah Chen", churn=0.18, growth=0.82, health=72.4)
    mock_db = _mock_db_with_affiliates([aff])

    # Communications sub-query returns empty list
    comm_q = MagicMock()
    comm_q.filter.return_value = comm_q
    comm_q.order_by.return_value = comm_q
    comm_q.limit.return_value = comm_q
    comm_q.all.return_value = []

    def query_side(model):
        if model is Communication:
            return comm_q
        return mock_db.query.return_value

    mock_db.query.side_effect = query_side

    with (
        patch("src.agent.tools._get_db", return_value=mock_db),
        # build_feature_vector is imported locally inside the tool function
        patch("src.ml.feature_engineering.build_feature_vector",
              side_effect=Exception("no model")),
    ):
        result = get_affiliate_summary.invoke("Sarah Chen")

    assert "Sarah Chen" in result
    assert "72.4" in result or "health" in result.lower()


def test_get_affiliate_summary_drivers_use_tagged_history_not_untagged_noise():
    """
    Real DB: an affiliate whose single most recent communication is untagged
    (fresh / not yet NLP-processed) but who has real tagged history further
    back must still surface that tagged history as churn/growth drivers —
    not "Insufficient data" — and the summary must flag that more recent,
    untagged contact exists so a reader isn't misled into thinking there's
    no recent contact at all. Uses a throwaway affiliate so pre-existing
    communication history elsewhere in the DB can't affect the assertions.
    """
    from datetime import datetime, timedelta, timezone

    from src.agent.tools import get_affiliate_summary
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, Communication

    db = SessionLocal()
    aff = None
    try:
        aff = Affiliate(name=f"Driver Test Affiliate {uuid.uuid4()}", status="active")
        db.add(aff)
        db.flush()

        now = datetime.now(timezone.utc)
        tagged_old = Communication(
            affiliate_id=aff.id,
            source="email",
            raw_text="We are considering leaving for a competitor platform.",
            tags=["churn_signal", "competitor_mention"],
            sentiment_score=-0.5,
            occurred_at=now - timedelta(days=10),
        )
        untagged_new = Communication(
            affiliate_id=aff.id,
            source="email",
            raw_text="Quick fresh note, not yet processed.",
            tags=[],
            sentiment_score=0.0,
            occurred_at=now - timedelta(days=1),
        )
        db.add(tagged_old)
        db.add(untagged_new)
        db.commit()

        result = get_affiliate_summary.func(aff.name)

        assert "churn_signal" in result
        assert "competitor_mention" in result
        assert "hasn't been NLP-tagged yet" in result
    finally:
        if aff is not None:
            db.query(Communication).filter(Communication.affiliate_id == aff.id).delete(
                synchronize_session=False
            )
            db.query(Affiliate).filter(Affiliate.id == aff.id).delete(synchronize_session=False)
            db.commit()
        db.close()


# ─── Test 4: get_affiliate_summary — not found ───────────────────────────────

def test_get_affiliate_summary_not_found():
    """An unknown name must return a clear 'not found' message."""
    from src.agent.tools import get_affiliate_summary
    from src.storage.models import Affiliate

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = None
    mock_db.query.return_value = q

    with patch("src.agent.tools._get_db", return_value=mock_db):
        result = get_affiliate_summary.invoke("Nonexistent Person XYZ")

    assert "not found" in result.lower()


# ─── Test 5: get_portfolio_health — returns stats ────────────────────────────

def test_get_portfolio_health_returns_stats():
    """get_portfolio_health must return a summary including totals and names."""
    from src.agent.tools import get_portfolio_health
    from src.storage.models import ScoreHistory

    affiliates = [
        _make_affiliate("Sarah Chen", churn=0.18, growth=0.82, health=72.4),
        _make_affiliate("Tom Bauer",  churn=0.88, growth=0.12, health=14.4, days=51),
    ]

    mock_db = MagicMock()

    def query_side(model):
        q = MagicMock()
        if model is ScoreHistory:
            q.count.return_value = 10
        else:
            q.all.return_value = affiliates
            q.count.return_value = 2
        return q

    mock_db.query.side_effect = query_side

    with patch("src.agent.tools._get_db", return_value=mock_db):
        result = get_portfolio_health.invoke("")

    assert "2" in result  # total affiliates
    assert "Tom Bauer" in result or "sarah" in result.lower() or "portfolio" in result.lower()
    assert "health" in result.lower()


def test_get_portfolio_health_surfaces_leak_and_seo_signals():
    """
    get_portfolio_health must expose portfolio-level leak/SEO visibility —
    not just the churn/growth tier — so a "which affiliates have warning
    signs" question has all three signal types in view from one tool call.
    These are counts/names only, deliberately not folded into avg_churn,
    avg_health, or the tier counts (see src.rulebook.recommend / Tier B).
    """
    from src.agent.tools import get_portfolio_health
    from src.storage.models import ScoreHistory

    affiliates = [
        _make_affiliate("Marcus Williams", churn=0.60, growth=0.10, health=28.0,
                         has_active_leak=True, search_trend="declining"),
        _make_affiliate("Rachel Torres", churn=0.20, growth=1.00, health=88.0,
                         has_active_leak=True, search_trend="stable"),
        _make_affiliate("Clean Affiliate", churn=0.10, growth=0.50, health=70.0,
                         has_active_leak=False, search_trend="stable"),
    ]

    mock_db = MagicMock()

    def query_side(model):
        q = MagicMock()
        if model is ScoreHistory:
            q.count.return_value = 5
        else:
            q.all.return_value = affiliates
            q.count.return_value = 3
        return q

    mock_db.query.side_effect = query_side

    with patch("src.agent.tools._get_db", return_value=mock_db):
        result = get_portfolio_health.invoke("")

    assert "Active promo-code leaks: 2" in result
    assert "Declining SEO trend:     1" in result
    assert "Marcus Williams" in result
    assert "Rachel Torres" in result
    # Leak/SEO visibility must not silently reword the tier counts above it.
    assert "not a tier" in result.lower()


# ─── Test 6: semantic_search — returns results ───────────────────────────────

def test_semantic_search_returns_results():
    """semantic_search must call the vector store and format the results."""
    from src.agent.tools import semantic_search
    import numpy as np

    fake_results = [
        {
            "id": "comm_abc_chunk_0",
            "text": "I'm frustrated about the delayed payment.",
            "affiliate_name": "Tom Bauer",
            "source": "email",
            "tags": ["frustrated", "churn_signal"],
            "occurred_at": "2026-05-01T00:00:00",
            "distance": 0.15,
        }
    ]

    mock_vs_instance = MagicMock()
    mock_vs_instance.search_similar.return_value = fake_results
    mock_db = MagicMock()

    with (
        patch("src.agent.tools._get_db", return_value=mock_db),
        patch("src.storage.pgvector_store.PGVectorStore", return_value=mock_vs_instance),
        patch("src.ingestion.embedding_generator.model") as mock_model,
    ):
        mock_model.encode.return_value = np.zeros(384)
        result = semantic_search.invoke("frustrated affiliate payment issue")

    assert "Tom Bauer" in result or "frustrated" in result.lower()
    assert len(result) > 10


# ─── Test 7: agent initialises when OPENAI_API_KEY set ───────────────────────

def test_agent_initialises_with_api_key():
    """_get_agent must not raise when OPENAI_API_KEY is set (using langgraph API)."""
    import src.agent.agent as agent_mod

    fake_key = "sk-test-fake-key-for-unit-tests-only"

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": fake_key}),
        patch("langchain_openai.ChatOpenAI") as mock_llm_cls,
        patch("langgraph.prebuilt.create_react_agent") as mock_create,
    ):
        mock_llm_cls.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        # Reset singleton so build is triggered fresh with the patched key
        agent_mod._agent = None
        agent_mod._init_error = None
        agent_mod._agent_key = None

        agent = agent_mod._get_agent()
        assert agent is not None


# ─── Test 8: agent tool list — get_leakage_status in, check_promo_leakage out ──

def test_tools_list_contains_get_leakage_status_not_check_promo_leakage():
    """The agent's bound tool list (src.agent.tools.TOOLS — what _build_agent()
    passes straight through to create_react_agent) must offer read-only
    get_leakage_status and must not offer check_promo_leakage, which used to
    trigger a live scan + DB write from agent reasoning."""
    from src.agent import tools as tools_mod

    tool_names = [t.name for t in tools_mod.TOOLS]

    assert "get_leakage_status" in tool_names
    assert "check_promo_leakage" not in tool_names
    # Fully removed, not just unbound — nothing should still define it.
    assert not hasattr(tools_mod, "check_promo_leakage")


def test_tools_list_contains_get_seo_status():
    """The agent's bound tool list must offer read-only get_seo_status —
    same read-only, no-live-trigger pattern as get_leakage_status."""
    from src.agent import tools as tools_mod

    tool_names = [t.name for t in tools_mod.TOOLS]
    assert "get_seo_status" in tool_names


# ─── Test 9: no bound tool has a side effect other than draft_email's insert ──

def test_no_bound_tool_has_side_effects_beyond_draft_email_approval_insert():
    """Static check: none of the agent's bound tools may write to the DB or
    make a live external call (a scrape, a scan, an SEO check) — except
    draft_email, which is permitted exactly one DB write (the
    approval_requests insert filed for human review) and must not itself
    perform any live external action (no scraping/scanning call, e.g.
    check_leakage or check_seo)."""
    import inspect

    from src.agent import tools as tools_mod

    # .execute( is deliberately excluded — query_database legitimately calls
    # db.execute() for SELECTs, already SELECT-only/keyword-blocked and
    # covered by test_query_database_rejects_non_select. ORM-level mutation
    # calls (.add/.commit/.delete) are the actual write signal here.
    write_markers = (".add(", ".commit(", ".delete(")
    live_action_markers = (
        "check_leakage", "check_seo", "requests.", "httpx.", "playwright",
    )

    for t in tools_mod.TOOLS:
        source = inspect.getsource(t.func)

        if t.name == "draft_email":
            assert any(m in source for m in write_markers), (
                "draft_email is expected to contain the approval_requests insert"
            )
            assert not any(m in source for m in live_action_markers), (
                f"draft_email must not perform a live external action: {source[:200]}"
            )
            continue

        found_writes = [m for m in write_markers if m in source]
        found_live = [m for m in live_action_markers if m in source]
        assert not found_writes, f"{t.name} must not write to the DB, found: {found_writes}"
        assert not found_live, f"{t.name} must not perform a live external action, found: {found_live}"