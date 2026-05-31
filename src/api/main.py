"""
FastAPI Application
===================
REST API for the Affiliate Intelligence Platform.

Endpoints
---------
GET  /health                          — service health check
GET  /affiliates                      — list affiliates with scores
GET  /affiliates/{id}                 — single affiliate + score history
GET  /affiliates/{id}/communications  — paginated communications
POST /affiliates/{id}/score           — trigger re-scoring
POST /agent/chat                      — chat with the ReAct agent
POST /ingest/csv                      — upload affiliates CSV

Run:
    uvicorn src.api.main:app --reload --port 8080
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.database import get_db, health_check as db_health, init_db
from src.storage.vector_store import health_check as chroma_health
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


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Create all DB tables on startup (idempotent)."""
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
    last_contact_at: Optional[str]
    days_since_contact: int
    updated_at: Optional[str]

    @classmethod
    def from_orm(cls, a: Affiliate) -> "AffiliateOut":
        return cls(
            id=str(a.id),
            name=a.name,
            status=a.status,
            churn_risk_score=a.churn_risk_score,
            growth_potential_score=a.growth_potential_score,
            health_score=a.health_score,
            revenue_30d=float(a.revenue_30d or 0.0),
            ctr_trend_pct=a.ctr_trend_pct or 0.0,
            last_contact_at=a.last_contact_at.isoformat() if a.last_contact_at else None,
            days_since_contact=a.days_since_contact or 0,
            updated_at=a.updated_at.isoformat() if a.updated_at else None,
        )


class CommunicationOut(BaseModel):
    id: str
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
            source=c.source,
            raw_text_preview=raw[:200] + ("…" if len(raw) > 200 else ""),
            tags=c.tags or [],
            sentiment_score=c.sentiment_score or 0.0,
            embedding_id=c.embedding_id,
            occurred_at=c.occurred_at.isoformat() if c.occurred_at else None,
        )


class ScoreHistoryOut(BaseModel):
    id: str
    scored_at: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float

    @classmethod
    def from_orm(cls, s: ScoreHistory) -> "ScoreHistoryOut":
        return cls(
            id=str(s.id),
            scored_at=str(s.scored_at),
            churn_risk_score=s.churn_risk_score,
            growth_potential_score=s.growth_potential_score,
            health_score=s.health_score,
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_affiliate_or_404(affiliate_id: str, db: Session) -> Affiliate:
    try:
        uid = _uuid.UUID(affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    aff = db.query(Affiliate).filter(Affiliate.id == uid).first()
    if not aff:
        raise HTTPException(status_code=404, detail=f"Affiliate {affiliate_id} not found")
    return aff


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    """Service health check — verifies PostgreSQL and ChromaDB connectivity."""
    pg_ok = db_health()
    chroma_ok = chroma_health()
    return {
        "status": "ok" if (pg_ok and chroma_ok) else "degraded",
        "postgres": "up" if pg_ok else "down",
        "chromadb": "up" if chroma_ok else "down",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Affiliates ───────────────────────────────────────────────────────────────

@app.get("/affiliates", response_model=list[AffiliateOut], tags=["Affiliates"])
def list_affiliates(
    status: Optional[str] = Query(
        None, description="Filter by status: active|at_risk|churned|high_growth"
    ),
    min_health: Optional[float] = Query(None, description="Minimum health score"),
    max_churn: Optional[float] = Query(None, description="Maximum churn risk (0–1)"),
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
    if min_health is not None:
        q = q.filter(Affiliate.health_score >= min_health)
    if max_churn is not None:
        q = q.filter(Affiliate.churn_risk_score <= max_churn)

    sort_col = getattr(Affiliate, sort_by, Affiliate.health_score)
    q = q.order_by(sort_col.desc() if order == "desc" else sort_col.asc())
    return [AffiliateOut.from_orm(a) for a in q.offset(offset).limit(limit).all()]


@app.get("/affiliates/{affiliate_id}", response_model=AffiliateDetail, tags=["Affiliates"])
def get_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> AffiliateDetail:
    """Single affiliate with full score history (most recent 20 snapshots)."""
    aff = _get_affiliate_or_404(affiliate_id, db)
    history = (
        db.query(ScoreHistory)
        .filter(ScoreHistory.affiliate_id == aff.id)
        .order_by(ScoreHistory.scored_at.desc())
        .limit(20)
        .all()
    )
    return AffiliateDetail(
        **AffiliateOut.from_orm(aff).model_dump(),
        score_history=[ScoreHistoryOut.from_orm(s) for s in history],
    )


@app.get(
    "/affiliates/{affiliate_id}/communications",
    response_model=list[CommunicationOut],
    tags=["Affiliates"],
)
def get_communications(
    affiliate_id: str,
    source: Optional[str] = Query(None, description="Filter: email|call|api_event"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[CommunicationOut]:
    """Paginated communications for one affiliate."""
    aff = _get_affiliate_or_404(affiliate_id, db)
    q = (
        db.query(Communication)
        .filter(Communication.affiliate_id == aff.id)
        .order_by(Communication.occurred_at.desc())
    )
    if source:
        q = q.filter(Communication.source == source.lower())
    return [CommunicationOut.from_orm(c) for c in q.offset(offset).limit(limit).all()]


# ─── Scoring ──────────────────────────────────────────────────────────────────

@app.post("/affiliates/{affiliate_id}/score", response_model=ScoreResponse, tags=["Scoring"])
def score_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    """Trigger full re-scoring (churn + growth) for a single affiliate."""
    # ML imports are lazy to avoid loading heavy models on every request
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

    from datetime import date
    entry = ScoreHistory(
        affiliate_id=aff.id,
        scored_at=date.today(),
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


# ─── Agent ────────────────────────────────────────────────────────────────────

@app.post("/agent/chat", response_model=ChatResponse, tags=["Agent"])
def agent_chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the LangChain ReAct agent.

    Example inputs:
      "Who are my highest churn risk affiliates?"
      "Draft a retention email for affiliate <id>"
    """
    # Lazy import — agent initialisation loads heavy LLM/embedding models
    from src.agent.agent import chat

    try:
        response = chat(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(response=response, session_id=request.session_id)


# ─── Ingestion ────────────────────────────────────────────────────────────────

@app.post("/ingest/full", tags=["Ingestion"])
def ingest_full(db: Session = Depends(get_db)) -> dict:
    """
    Run the full ETL pipeline: load affiliates from CSV then communications
    from emails.txt and transcripts.txt.  The entire operation is atomic —
    either both jobs commit or the whole transaction is rolled back.

    Returns a summary of records created / updated for both jobs.
    """
    from src.ingestion.etl_pipeline import run_full_etl

    try:
        return run_full_etl(db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ETL error: {exc}")


@app.post("/ingest/affiliates", tags=["Ingestion"])
def ingest_affiliates_only(db: Session = Depends(get_db)) -> dict:
    """
    Run Job 1 only: load affiliates from data/mock/affiliates.csv.
    Upserts by name — safe to run multiple times.
    """
    from src.ingestion.etl_pipeline import run_affiliate_etl

    try:
        result = run_affiliate_etl(db)
        db.commit()
        return result
    except FileNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"ETL error: {exc}")


@app.post("/ingest/communications", tags=["Ingestion"])
def ingest_communications_only(db: Session = Depends(get_db)) -> dict:
    """
    Run Job 2 only: load communications from emails.txt and transcripts.txt.
    Upserts by (affiliate_id, occurred_at) — safe to run multiple times.
    Affiliates must already exist before calling this endpoint.
    """
    from src.ingestion.etl_pipeline import run_communications_etl

    try:
        result = run_communications_etl(db)
        db.commit()
        return result
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"ETL error: {exc}")


@app.post("/ingest/csv", tags=["Ingestion"])
async def ingest_csv(file: UploadFile = File(...)) -> dict:
    """
    Upload a CSV file of affiliates.
    Expected columns: name, status, revenue_30d, ctr_trend_pct
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    # Lazy import — etl_pipeline has heavy dependencies not needed at startup
    from src.ingestion.etl_pipeline import ingest_csv_content

    content = await file.read()
    result = ingest_csv_content(content.decode("utf-8"))
    return {"status": "success", "filename": file.filename, **result}