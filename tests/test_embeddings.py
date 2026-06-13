"""
Embedding Generator Tests
=========================
Tests for chunk_text(), embed_communication(), embed_all_communications(),
and the GET /search endpoint.

Run:
    pytest tests/test_embeddings.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.ingestion.embedding_generator import chunk_text


# ─── Test 1: chunk_text splits long text correctly ────────────────────────────

def test_chunk_text_splits_long_text():
    """Text with more than chunk_size words is split into multiple chunks."""
    # 250 words — default chunk_size=200 → expect at least 2 chunks
    words = ["word"] * 250
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1, "Expected multiple chunks for 250-word text"
    # Each chunk must be non-empty
    for chunk in chunks:
        assert chunk.strip(), "Chunk must not be empty"


# ─── Test 2: chunk_text returns single chunk for short text ───────────────────

def test_chunk_text_short_text_single_chunk():
    """Text with fewer words than chunk_size is returned as one chunk."""
    text = "This is a short text with only a few words."
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


# ─── Test 3: chunk_text overlap works correctly ───────────────────────────────

def test_chunk_text_overlap():
    """Consecutive chunks share the expected number of overlap words."""
    # Build text: words w0, w1, ..., w299 (300 words)
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) >= 2, "Expected at least 2 chunks for 300 words with size=100"

    # The last 20 words of chunk 0 should be the first 20 words of chunk 1
    tail_of_first = chunks[0].split()[-20:]
    head_of_second = chunks[1].split()[:20]
    assert tail_of_first == head_of_second, (
        f"Overlap mismatch:\n  tail={tail_of_first}\n  head={head_of_second}"
    )


# ─── Test 4: embed_communication creates correct chunks in ChromaDB ───────────

def test_embed_communication_chunk_count():
    """
    embed_communication should call vs.add_document once per chunk and
    set comm.embedding_id to the first chunk's doc_id.
    """
    from src.ingestion.embedding_generator import embed_communication
    from src.storage.models import Communication

    # Build a stub Communication with 250 words of text (→ 2 chunks at size=200)
    comm = Communication()
    comm.id = uuid.uuid4()
    comm.affiliate_id = uuid.uuid4()
    comm.source = "email"
    comm.raw_text = " ".join(["word"] * 250)
    comm.tags = ["enthusiastic", "expansion_interest"]
    comm.sentiment_score = 0.7
    comm.occurred_at = datetime.now(timezone.utc)
    comm.embedding_id = None

    # Mock DB — affiliate lookup returns a stub
    from src.storage.models import Affiliate
    aff = Affiliate()
    aff.name = "Test Affiliate"
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = aff

    # Mock VectorStore
    mock_vs = MagicMock()

    # Mock model.encode to return a zero vector quickly
    with patch("src.ingestion.embedding_generator.model") as mock_model:
        mock_model.encode.return_value = np.zeros(384)
        result = embed_communication(comm, mock_db, mock_vs)

    expected_chunks = len(chunk_text(comm.raw_text))
    assert result["chunks_created"] == expected_chunks
    assert mock_vs.add_document.call_count == expected_chunks
    assert result["embedding_id"] == f"{comm.id}_chunk_0"
    assert comm.embedding_id == f"{comm.id}_chunk_0"


# ─── Test 5: embed_all_communications only processes unembedded records ────────

def test_embed_all_communications_skips_already_embedded():
    """
    embed_all_communications must only process records where embedding_id
    is None, and correctly report already_embedded count.
    """
    from src.ingestion.embedding_generator import embed_all_communications
    from src.storage.models import Communication

    def _make_comm(has_embedding: bool) -> Communication:
        c = Communication()
        c.id = uuid.uuid4()
        c.affiliate_id = uuid.uuid4()
        c.source = "email"
        c.raw_text = "Short text."
        c.tags = []
        c.sentiment_score = 0.0
        c.occurred_at = datetime.now(timezone.utc)
        c.embedding_id = "existing_id" if has_embedding else None
        return c

    comm_embedded = _make_comm(has_embedding=True)
    comm_new = _make_comm(has_embedding=False)

    # Mock DB
    from src.storage.models import Affiliate
    aff = Affiliate()
    aff.name = "Someone"

    mock_db = MagicMock()

    def query_side_effect(model_cls):
        q = MagicMock()
        if model_cls is Communication:
            def filter_side_effect(condition):
                f = MagicMock()
                # isnot(None) → return 1 already-embedded
                f.count.return_value = 1
                # is_(None) → return [comm_new]
                f.all.return_value = [comm_new]
                return f
            q.filter.side_effect = filter_side_effect
        elif model_cls is Affiliate:
            q.filter.return_value.first.return_value = aff
        return q

    mock_db.query.side_effect = query_side_effect

    mock_vs = MagicMock()

    with patch("src.ingestion.embedding_generator.model") as mock_model:
        mock_model.encode.return_value = np.zeros(384)
        result = embed_all_communications(mock_db, mock_vs)

    assert result["total_processed"] == 1, (
        f"Expected 1 processed, got {result['total_processed']}"
    )
    assert result["already_embedded"] == 1, (
        f"Expected 1 already_embedded, got {result['already_embedded']}"
    )
    # add_document should only have been called for the new record
    assert mock_vs.add_document.call_count == 1


# ─── Test 6: GET /search returns results ──────────────────────────────────────

def test_search_endpoint_returns_results():
    """GET /search should return a list of result dicts for a simple query."""
    from src.api.main import app

    fake_results = [
        {
            "id": "abc_chunk_0",
            "text": "The affiliate has gone quiet and hasn't responded in weeks.",
            "metadata": {
                "affiliate_id": str(uuid.uuid4()),
                "affiliate_name": "Test Affiliate",
                "source": "email",
                "tags": "|gone_silent|unresponsive|",
                "occurred_at": "2026-05-01T00:00:00",
            },
            "distance": 0.12,
        }
    ]

    with (
        patch("src.ingestion.embedding_generator.model") as mock_model,
        patch("src.ingestion.embedding_generator.vector_store") as mock_vs,
    ):
        mock_model.encode.return_value = np.zeros(384)
        mock_vs.search_similar.return_value = fake_results

        client = TestClient(app)
        response = client.get("/search?q=disengaged+affiliate")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "abc_chunk_0"
    assert "document" in data[0]
    assert "distance" in data[0]