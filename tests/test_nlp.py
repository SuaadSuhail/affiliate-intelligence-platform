"""
NLP Processor Tests
===================
Tests for calculate_sentiment(), detect_tags(), and process_all_communications().

Run:
    pytest tests/test_nlp.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.nlp_processor import (
    SENTIMENT_LEXICON,
    _nlp,
    calculate_sentiment,
    detect_tags,
    process_all_communications,
    process_single_communication,
)
from src.storage.models import Affiliate, Communication


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_mock_db(days_since: int = 0) -> MagicMock:
    """Return a mock SQLAlchemy session with a stub Affiliate query."""
    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.days_since_contact = days_since

    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = aff
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query
    return mock_db, aff


# ─── Test 1: calculate_sentiment — positive text ──────────────────────────────

def test_calculate_sentiment_positive():
    """A clearly positive text should return a score > 0.3."""
    text = "We are absolutely thrilled with the results! Fantastic progress and happy to report excellent growth."
    score = calculate_sentiment(text)
    assert score > 0.3, f"Expected positive score > 0.3, got {score}"


# ─── Test 2: calculate_sentiment — negative text ─────────────────────────────

def test_calculate_sentiment_negative():
    """A clearly negative / churn-risk text should return a score < -0.3."""
    text = "I am frustrated and disappointed. We are cancelling and switching to another platform immediately."
    score = calculate_sentiment(text)
    assert score < -0.3, f"Expected negative score < -0.3, got {score}"


# ─── Test 3: detect_tags — churn_signal ──────────────────────────────────────

def test_detect_tags_churn_signal():
    """A churn-signal email should receive the 'churn_signal' tag."""
    text = (
        "Hi, I'm leaving your network and switching to a competitor. "
        "I've been cancelling campaigns one by one."
    )
    doc = _nlp(text)
    mock_db, _ = _make_mock_db(days_since=3)
    tags = detect_tags(
        doc=doc,
        sentiment_score=calculate_sentiment(text),
        text_lower=text.lower(),
        source="email",
        affiliate_id=uuid.uuid4(),
        db=mock_db,
    )
    assert "churn_signal" in tags, f"Expected 'churn_signal' in {tags}"


# ─── Test 4: detect_tags — enthusiastic ──────────────────────────────────────

def test_detect_tags_enthusiastic():
    """An enthusiastic email should receive the 'enthusiastic' tag."""
    text = (
        "We're so excited about this! The campaign is thrilling and we love this platform. "
        "Can't wait to expand further!"
    )
    doc = _nlp(text)
    mock_db, _ = _make_mock_db(days_since=2)
    tags = detect_tags(
        doc=doc,
        sentiment_score=calculate_sentiment(text),
        text_lower=text.lower(),
        source="email",
        affiliate_id=uuid.uuid4(),
        db=mock_db,
    )
    assert "enthusiastic" in tags, f"Expected 'enthusiastic' in {tags}"


# ─── Test 5: detect_tags — competitor_mention (Awin) ─────────────────────────

def test_detect_tags_competitor_mention_awin():
    """A message mentioning 'Awin' should receive the 'competitor_mention' tag."""
    text = "I've been speaking with Awin and they offered much better terms for us."
    doc = _nlp(text)
    mock_db, _ = _make_mock_db(days_since=4)
    tags = detect_tags(
        doc=doc,
        sentiment_score=calculate_sentiment(text),
        text_lower=text.lower(),
        source="email",
        affiliate_id=uuid.uuid4(),
        db=mock_db,
    )
    assert "competitor_mention" in tags, f"Expected 'competitor_mention' in {tags}"


# ─── Test 6: process_all_communications — counts ─────────────────────────────

def test_process_all_communications_counts():
    """
    process_all_communications should return correct total_processed count and
    a non-empty tag_summary when given untagged communications.
    """
    # Build two stub Communication objects
    def _make_comm(text: str) -> Communication:
        comm = Communication()
        comm.id = uuid.uuid4()
        comm.affiliate_id = uuid.uuid4()
        comm.source = "email"
        comm.raw_text = text
        comm.tags = []
        comm.sentiment_score = 0.0
        comm.occurred_at = datetime.now(timezone.utc)
        return comm

    comm1 = _make_comm(
        "I'm thrilled about the new campaign launch! Looking forward to expand our reach."
    )
    comm2 = _make_comm(
        "I'm frustrated and disappointed. We are leaving and switching to another platform."
    )

    # Mock the db session
    aff = Affiliate()
    aff.id = uuid.uuid4()
    aff.days_since_contact = 3

    mock_db = MagicMock()

    # query(Communication).filter(...).all() → [comm1, comm2]
    comm_query_mock = MagicMock()
    comm_filter_mock = MagicMock()
    comm_filter_mock.all.return_value = [comm1, comm2]

    # query(Affiliate).filter(...).first() → aff
    aff_query_mock = MagicMock()
    aff_filter_mock = MagicMock()
    aff_filter_mock.first.return_value = aff

    def query_side_effect(model):
        if model is Communication:
            return comm_query_mock
        if model is Affiliate:
            return aff_query_mock
        return MagicMock()

    mock_db.query.side_effect = query_side_effect
    comm_query_mock.filter.return_value = comm_filter_mock
    aff_query_mock.filter.return_value = aff_filter_mock

    result = process_all_communications(mock_db)

    assert result["total_processed"] == 2, (
        f"Expected total_processed=2, got {result['total_processed']}"
    )
    assert result["total_tagged"] >= 1, (
        f"Expected at least 1 tagged record, got {result['total_tagged']}"
    )
    assert isinstance(result["tag_summary"], dict), "tag_summary should be a dict"
    assert len(result["tag_summary"]) > 0, "tag_summary should have at least one entry"