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

def _make_affiliate(name="Test Affiliate", churn=0.4, growth=0.6, health=60.0, days=5):
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


# ─── Test 6: semantic_search — returns results ───────────────────────────────

def test_semantic_search_returns_results():
    """semantic_search must call the vector store and format the results."""
    from src.agent.tools import semantic_search
    import numpy as np

    fake_results = [
        {
            "id": "comm_abc_chunk_0",
            "text": "I'm frustrated about the delayed payment.",
            "metadata": {
                "affiliate_name": "Tom Bauer",
                "source": "email",
                "tags": "|frustrated|churn_signal|",
            },
            "distance": 0.15,
        }
    ]

    with (
        patch("src.agent.tools.vector_store") as mock_vs,
        patch("src.ingestion.embedding_generator.model") as mock_model,
    ):
        mock_model.encode.return_value = np.zeros(384)
        mock_vs.search_similar.return_value = fake_results

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