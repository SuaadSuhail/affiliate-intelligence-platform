"""
pgvector Store
==============
Replaces ChromaDB with PostgreSQL + pgvector for communication embeddings.

All vector search operations run inside the same PostgreSQL instance used for
structured data — no separate container required.

Collection: embeddings table (vector(384), ivfflat cosine index)
Model:      sentence-transformers/all-MiniLM-L6-v2 (384 dims)
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional

from sqlalchemy.orm import Session

from src.core.logging_config import get_logger
from src.storage.models import Embedding

logger = get_logger(__name__)


class PGVectorStore:
    """Vector store backed by PostgreSQL + pgvector."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        text: str,
        affiliate_id: str,
        affiliate_name: str,
        source: str,
        tags: list[str],
        occurred_at,
        embedding_vector: list[float],
    ) -> None:
        """Upsert one embedding chunk into the embeddings table."""
        existing = self.db.get(Embedding, doc_id)
        if existing:
            existing.chunk_text = text
            existing.tags = tags
            existing.embedding = embedding_vector
        else:
            aff_uuid = (
                _uuid.UUID(affiliate_id)
                if isinstance(affiliate_id, str)
                else affiliate_id
            )
            doc = Embedding(
                id=doc_id,
                affiliate_id=aff_uuid,
                affiliate_name=affiliate_name,
                source=source,
                chunk_text=text,
                tags=tags,
                occurred_at=occurred_at,
                embedding=embedding_vector,
            )
            self.db.add(doc)
        self.db.commit()

    def delete_by_affiliate(self, affiliate_id: str) -> None:
        """Delete all embeddings for one affiliate."""
        aff_uuid = (
            _uuid.UUID(affiliate_id)
            if isinstance(affiliate_id, str)
            else affiliate_id
        )
        self.db.query(Embedding).filter(
            Embedding.affiliate_id == aff_uuid
        ).delete()
        self.db.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter_tags: Optional[list[str]] = None,
        filter_affiliate_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search using cosine distance.

        Returns list of dicts with keys:
            id, text, affiliate_name, source, tags, occurred_at, distance
        """
        distance_col = Embedding.embedding.cosine_distance(query_embedding)
        q = self.db.query(Embedding, distance_col.label("distance"))

        if filter_affiliate_id:
            aff_uuid = (
                _uuid.UUID(filter_affiliate_id)
                if isinstance(filter_affiliate_id, str)
                else filter_affiliate_id
            )
            q = q.filter(Embedding.affiliate_id == aff_uuid)

        if filter_tags:
            for tag in filter_tags:
                q = q.filter(Embedding.tags.contains([tag]))

        rows = q.order_by(distance_col).limit(n_results).all()

        return [
            {
                "id": row[0].id,
                "text": row[0].chunk_text,
                "affiliate_name": row[0].affiliate_name,
                "source": row[0].source,
                "tags": row[0].tags or [],
                "occurred_at": str(row[0].occurred_at),
                "distance": float(row[1]) if row[1] is not None else 0.0,
            }
            for row in rows
        ]

    def get_by_affiliate(
        self,
        affiliate_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return all embedding chunks for one affiliate."""
        aff_uuid = (
            _uuid.UUID(affiliate_id)
            if isinstance(affiliate_id, str)
            else affiliate_id
        )
        results = (
            self.db.query(Embedding)
            .filter(Embedding.affiliate_id == aff_uuid)
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "text": r.chunk_text,
                "tags": r.tags or [],
                "source": r.source,
            }
            for r in results
        ]