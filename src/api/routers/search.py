"""
Search & Communications Router
================================
Endpoints for listing communications and semantic search.

GET /communications             — list all communications with tags + sentiment
GET /communications/{comm_id}  — single communication by UUID
GET /search                     — semantic search over embedded communications
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.database import get_db
from src.storage.models import Communication

router = APIRouter()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class CommunicationOut(BaseModel):
    id: str
    affiliate_id: str
    channel: str
    direction: str
    subject: Optional[str]
    content_preview: str
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    tags: list[str]
    embedding_id: Optional[str]
    occurred_at: Optional[str]

    @classmethod
    def from_orm(cls, c: Communication) -> "CommunicationOut":
        raw = c.content or ""
        return cls(
            id=str(c.id),
            affiliate_id=str(c.affiliate_id),
            channel=c.channel,
            direction=c.direction,
            subject=c.subject,
            content_preview=raw[:200] + ("…" if len(raw) > 200 else ""),
            sentiment_score=c.sentiment_score,
            sentiment_label=c.sentiment_label,
            tags=c.tags or [],
            embedding_id=c.embedding_id,
            occurred_at=c.occurred_at.isoformat() if c.occurred_at else None,
        )


class SearchResultItem(BaseModel):
    id: str
    document: str
    metadata: dict
    distance: float


# ─── Communications list ──────────────────────────────────────────────────────

@router.get("/communications", response_model=list[CommunicationOut])
def list_communications(
    channel: Optional[str] = Query(None, description="Filter: email|call|chat|ticket"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[CommunicationOut]:
    """List all communications with tags and sentiment scores."""
    q = db.query(Communication).order_by(Communication.occurred_at.desc())
    if channel:
        q = q.filter(Communication.channel == channel.lower())
    comms = q.offset(offset).limit(limit).all()
    return [CommunicationOut.from_orm(c) for c in comms]


@router.get("/communications/{comm_id}", response_model=CommunicationOut)
def get_communication(
    comm_id: str,
    db: Session = Depends(get_db),
) -> CommunicationOut:
    """Return a single communication by UUID."""
    try:
        uid = _uuid.UUID(comm_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    comm = db.query(Communication).filter(Communication.id == uid).first()
    if not comm:
        raise HTTPException(status_code=404, detail=f"Communication {comm_id} not found")
    return CommunicationOut.from_orm(comm)


# ─── Semantic search ──────────────────────────────────────────────────────────

@router.get("/search", response_model=list[SearchResultItem])
def search(
    q: str = Query(..., description="Natural-language search query"),
    affiliate_id: Optional[str] = Query(None, description="Filter to one affiliate UUID"),
    n: int = Query(5, ge=1, le=20, description="Number of results"),
) -> list[SearchResultItem]:
    """
    Semantic search over embedded communications.

    Uses sentence-transformers/all-MiniLM-L6-v2 to encode the query, then
    returns the closest matching communication chunks from ChromaDB.
    """
    from src.ingestion.embedding_generator import get_generator

    gen = get_generator()
    try:
        results = gen.search_communications(
            query=q,
            n_results=n,
            affiliate_id=affiliate_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")

    return [
        SearchResultItem(
            id=r["id"],
            document=r["document"],
            metadata=r["metadata"],
            distance=r["distance"],
        )
        for r in results
    ]