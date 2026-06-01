"""
FastAPI Application
===================
REST API for the Affiliate Intelligence Platform.

Endpoints
---------
GET  /health                          — service health check
GET  /affiliates                      — list all affiliates with scores
GET  /affiliates/{id}                 — single affiliate + score history
GET  /affiliates/{id}/communications  — paginated communications
POST /affiliates/{id}/score           — trigger re-scoring
POST /agent/chat                      — chat with the ReAct agent
POST /process/nlp                     — run NLP on all untagged communications
POST /process/embeddings              — embed all unembedded communications
POST /process/full                    — run NLP + embeddings end-to-end
GET  /communications                  — list all communications with tags
GET  /communications/{comm_id}        — single communication by UUID
GET  /search                          — semantic search over communications

Run:
    uvicorn src.api.main:app --reload --port 8080
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.database import get_db, health_check as db_health, init_db
from src.storage.models import Affiliate, Communication, ScoreHistory

app = FastAPI(
    title="Affiliate Intelligence Platform",
    description=(
        "Agentic AI CRM that produces a 360° health score for every affiliate, "
        "predicting churn risk and growth potential."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Initialise DB on startup ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    init_db()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class AffiliateOut(BaseModel):
    id: str
    name: str
    status: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    revenue_30d: float
    ctr_trend_pct: float
    days_since_contact: int
    last_contact_at: Optional[str]

    @classmethod
    def from_orm(cls, a: Affiliate) -> "AffiliateOut":
        return cls(
            id=str(a.id),
            name=a.name,
            status=a.status,
            churn_risk_score=a.churn_risk_score or 0.5,
            growth_potential_score=a.growth_potential_score or 0.5,
            health_score=a.health_score or 50.0,
            revenue_30d=float(a.revenue_30d or 0.0),
            ctr_trend_pct=a.ctr_trend_pct or 0.0,
            days_since_contact=a.days_since_contact or 0,
            last_contact_at=a.last_contact_at.isoformat() if a.last_contact_at else None,
        )


class CommunicationOut(BaseModel):
    id: str
    affiliate_id: str
    source: str
    raw_text_preview: str
    tags: list[str]
    sentiment_score: float
    embedding_id: Optional[str]
    occurred_at: Optional[str]

    @classmethod
    def from_orm(cls, c: Communication) -> "CommunicationOut":
        raw = c.raw_text or ""
        return cls(
            id=str(c.id),
            affiliate_id=str(c.affiliate_id),
            source=c.source,
            raw_text_preview=raw[:200] + ("…" if len(raw) > 200 else ""),
            tags=c.tags or [],
            sentiment_score=c.sentiment_score or 0.0,
            embedding_id=c.embedding_id,
            occurred_at=c.occurred_at.isoformat() if c.occurred_at else None,
        )


class ScoreHistoryOut(BaseModel):
    id: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    scored_at: str

    @classmethod
    def from_orm(cls, s: ScoreHistory) -> "ScoreHistoryOut":
        return cls(
            id=str(s.id),
            churn_risk_score=s.churn_risk_score,
            growth_potential_score=s.growth_potential_score,
            health_score=s.health_score,
            scored_at=s.scored_at.isoformat() if s.scored_at else "",
        )


class AffiliateDetail(AffiliateOut):
    score_history: list[ScoreHistoryOut] = []


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None


class ScoreResponse(BaseModel):
    affiliate_id: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    scored_at: str


class NLPResult(BaseModel):
    total_processed: int
    total_tagged: int
    tag_summary: dict[str, int]


class EmbeddingResult(BaseModel):
    total_processed: int
    total_chunks_created: int
    already_embedded: int


class FullPipelineResult(BaseModel):
    etl: dict
    nlp: NLPResult
    embeddings: EmbeddingResult


class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: dict
    distance: float


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_affiliate_or_404(affiliate_id: str, db: Session) -> Affiliate:
    try:
        uid = _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    aff = db.query(Affiliate).filter(Affiliate.id == uid).first()
    if not aff:
        raise HTTPException(status_code=404, detail=f"Affiliate {affiliate_id} not found")
    return aff


def _get_comm_or_404(comm_id: str, db: Session) -> Communication:
    try:
        uid = _uuid.UUID(comm_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    comm = db.query(Communication).filter(Communication.id == uid).first()
    if not comm:
        raise HTTPException(status_code=404, detail=f"Communication {comm_id} not found")
    return comm


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    """Service health check — verifies PostgreSQL connectivity."""
    pg_ok = db_health()
    return {
        "status": "ok" if pg_ok else "degraded",
        "postgres": "up" if pg_ok else "down",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Affiliates ───────────────────────────────────────────────────────────────

@app.get("/affiliates", response_model=list[AffiliateOut], tags=["Affiliates"])
def list_affiliates(
    status: Optional[str] = Query(None, description="Filter by status: active|at_risk|churned|high_growth"),
    sort_by: str = Query("health_score", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[AffiliateOut]:
    """List all affiliates with optional filtering and sorting."""
    q = db.query(Affiliate)
    if status:
        q = q.filter(Affiliate.status == status.lower())

    sort_col = getattr(Affiliate, sort_by, Affiliate.health_score)
    q = q.order_by(sort_col.desc() if order == "desc" else sort_col.asc())
    affiliates = q.offset(offset).limit(limit).all()
    return [AffiliateOut.from_orm(a) for a in affiliates]


@app.get("/affiliates/{affiliate_id}", response_model=AffiliateDetail, tags=["Affiliates"])
def get_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> AffiliateDetail:
    """Get a single affiliate with full score history."""
    aff = _get_affiliate_or_404(affiliate_id, db)
    history = (
        db.query(ScoreHistory)
        .filter(ScoreHistory.affiliate_id == aff.id)
        .order_by(ScoreHistory.scored_at.desc())
        .limit(20)
        .all()
    )
    detail = AffiliateDetail(
        **AffiliateOut.from_orm(aff).model_dump(),
        score_history=[ScoreHistoryOut.from_orm(s) for s in history],
    )
    return detail


@app.get(
    "/affiliates/{affiliate_id}/communications",
    response_model=list[CommunicationOut],
    tags=["Affiliates"],
)
def get_affiliate_communications(
    affiliate_id: str,
    source: Optional[str] = Query(None, description="Filter: email|call|api_event"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[CommunicationOut]:
    """Get paginated communications for an affiliate."""
    aff = _get_affiliate_or_404(affiliate_id, db)
    q = (
        db.query(Communication)
        .filter(Communication.affiliate_id == aff.id)
        .order_by(Communication.occurred_at.desc())
    )
    if source:
        q = q.filter(Communication.source == source.lower())
    comms = q.offset(offset).limit(limit).all()
    return [CommunicationOut.from_orm(c) for c in comms]


# ─── NLP Processing ───────────────────────────────────────────────────────────

@app.post("/process/nlp", response_model=NLPResult, tags=["NLP"])
def run_nlp_processing(
    db: Session = Depends(get_db),
) -> NLPResult:
    """
    Run NLP tagging on all untagged communications.

    Reads every Communication with tags=[], applies the spaCy-based
    20-tag detection pipeline and lexicon sentiment scoring, then
    persists the results back to PostgreSQL.
    """
    from src.ingestion.nlp_processor import process_all_communications

    result = process_all_communications(db)
    db.commit()
    return NLPResult(
        total_processed=result["total_processed"],
        total_tagged=result["total_tagged"],
        tag_summary=result["tag_summary"],
    )


# ─── Communications ───────────────────────────────────────────────────────────

@app.get("/communications", response_model=list[CommunicationOut], tags=["Communications"])
def list_communications(
    source: Optional[str] = Query(None, description="Filter: email|call|api_event"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[CommunicationOut]:
    """List all communications with tags and sentiment scores."""
    q = db.query(Communication).order_by(Communication.occurred_at.desc())
    if source:
        q = q.filter(Communication.source == source.lower())
    comms = q.offset(offset).limit(limit).all()
    return [CommunicationOut.from_orm(c) for c in comms]


@app.get("/communications/{comm_id}", response_model=CommunicationOut, tags=["Communications"])
def get_communication(
    comm_id: str,
    db: Session = Depends(get_db),
) -> CommunicationOut:
    """Return a single communication by UUID."""
    comm = _get_comm_or_404(comm_id, db)
    return CommunicationOut.from_orm(comm)


# ─── Embedding Processing ─────────────────────────────────────────────────────

@app.post("/process/embeddings", response_model=EmbeddingResult, tags=["Embeddings"])
def run_embedding_processing(
    db: Session = Depends(get_db),
) -> EmbeddingResult:
    """
    Embed all communications that do not yet have an embedding_id.

    Chunks each communication's raw_text, encodes with all-MiniLM-L6-v2,
    and stores vectors + metadata in ChromaDB.  Writes the first chunk's
    doc_id back to communications.embedding_id in PostgreSQL.
    """
    from src.ingestion.embedding_generator import (
        embed_all_communications,
        vector_store as emb_vs,
    )

    result = embed_all_communications(db, emb_vs)
    db.commit()
    return EmbeddingResult(**result)


@app.post("/process/full", response_model=FullPipelineResult, tags=["Embeddings"])
def run_full_pipeline(
    db: Session = Depends(get_db),
) -> FullPipelineResult:
    """
    Run the complete pipeline end-to-end:
      1. ETL  — reload affiliates + communications from mock data files
      2. NLP  — tag all untagged communications
      3. Embed — embed all unembedded communications

    Safe to call multiple times; each step is idempotent.
    """
    from src.ingestion.nlp_processor import process_all_communications
    from src.ingestion.embedding_generator import (
        embed_all_communications,
        vector_store as emb_vs,
    )

    # Step 1 — ETL (manages its own session internally)
    etl_result: dict = {"status": "skipped"}
    try:
        from src.ingestion.etl_pipeline import run_full_pipeline as _etl
        _etl()
        etl_result = {"status": "complete"}
    except Exception as exc:
        etl_result = {"status": "error", "detail": str(exc)}

    # Step 2 — NLP
    # Refresh session after ETL wrote its own commits
    db.expire_all()
    nlp_raw = process_all_communications(db)
    db.commit()

    # Step 3 — Embeddings
    emb_raw = embed_all_communications(db, emb_vs)
    db.commit()

    return FullPipelineResult(
        etl=etl_result,
        nlp=NLPResult(**nlp_raw),
        embeddings=EmbeddingResult(**emb_raw),
    )


# ─── Semantic search ──────────────────────────────────────────────────────────

@app.get("/search", response_model=list[SearchResultItem], tags=["Search"])
def search(
    q: str = Query(..., description="Natural-language search query"),
    affiliate_id: Optional[str] = Query(None, description="Filter to one affiliate UUID"),
    tags: Optional[str] = Query(None, description="Comma-separated tag names to filter by"),
    n: int = Query(5, ge=1, le=20, description="Number of results"),
) -> list[SearchResultItem]:
    """
    Semantic search over embedded communications.

    Returns the closest matching communication chunks ranked by cosine
    similarity to the query.  Optionally filter by affiliate_id and/or tags.
    """
    from src.ingestion.embedding_generator import model, vector_store as emb_vs

    query_embedding = model.encode(q).tolist()

    tag_list: Optional[list[str]] = (
        [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    )

    results = emb_vs.search_similar(
        query_embedding=query_embedding,
        n_results=n,
        affiliate_id=affiliate_id,
        tags=tag_list,
    )
    return [SearchResultItem(**r) for r in results]


# ─── Scoring ──────────────────────────────────────────────────────────────────

@app.post("/affiliates/{affiliate_id}/score", response_model=ScoreResponse, tags=["Scoring"])
def score_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    """
    Trigger full re-scoring (churn + growth) for a single affiliate.
    Updates the affiliate row and appends to score_history.
    """
    from src.ml.churn_model import predict_one as churn_one
    from src.ml.growth_model import predict_one as growth_one

    aff = _get_affiliate_or_404(affiliate_id, db)
    aid = str(aff.id)

    churn = churn_one(aid)
    growth = growth_one(aid)

    c_score = churn["churn_risk_score"]
    g_score = growth["growth_potential_score"]
    h_score = round(((1 - c_score) * 0.6 + g_score * 0.4) * 100, 1)

    aff.churn_risk_score = c_score
    aff.growth_potential_score = g_score
    aff.health_score = h_score

    entry = ScoreHistory(
        affiliate_id=aff.id,
        churn_risk_score=c_score,
        growth_potential_score=g_score,
        health_score=h_score,
    )
    db.add(entry)
    db.commit()

    return ScoreResponse(
        affiliate_id=aid,
        churn_risk_score=c_score,
        growth_potential_score=g_score,
        health_score=h_score,
        scored_at=datetime.utcnow().isoformat(),
    )


# ─── Agent chat ───────────────────────────────────────────────────────────────

@app.post("/agent/chat", response_model=ChatResponse, tags=["Agent"])
def agent_chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the LangChain ReAct agent."""
    from src.agent.agent import chat

    try:
        response = chat(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(
        response=response,
        session_id=request.session_id,
    )