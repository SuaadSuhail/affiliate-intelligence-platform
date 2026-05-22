"""
FastAPI Application
===================
REST API for the Affiliate Intelligence Platform.

Endpoints
---------
GET  /health                     — service health check
GET  /affiliates                 — list all affiliates with scores
GET  /affiliates/{id}            — single affiliate + score history
GET  /affiliates/{id}/communications  — paginated communications
POST /affiliates/{id}/score      — trigger re-scoring
POST /agent/chat                 — chat with the ReAct agent
POST /ingest/csv                 — upload affiliates CSV

Run:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.storage.database import get_db, health_check as db_health, init_db
from src.storage.vector_store import vector_store
from src.storage.models import Affiliate, Communication, ScoreHistory
from src.ingestion.etl_pipeline import ingest_csv_content

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
    email: str
    company: Optional[str]
    tier: str
    join_date: str
    country: Optional[str]
    niche: Optional[str]
    traffic_source: Optional[str]
    monthly_revenue: float
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    last_contact_date: Optional[str]

    @classmethod
    def from_orm(cls, a: Affiliate) -> "AffiliateOut":
        return cls(
            id=str(a.id),
            name=a.name,
            email=a.email,
            company=a.company,
            tier=a.tier,
            join_date=str(a.join_date),
            country=a.country,
            niche=a.niche,
            traffic_source=a.traffic_source,
            monthly_revenue=a.monthly_revenue or 0.0,
            churn_risk_score=a.churn_risk_score or 0.5,
            growth_potential_score=a.growth_potential_score or 0.5,
            health_score=a.health_score or 50.0,
            last_contact_date=a.last_contact_date.isoformat() if a.last_contact_date else None,
        )


class CommunicationOut(BaseModel):
    id: str
    channel: str
    direction: str
    subject: Optional[str]
    content_preview: str
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    tags: list[str]
    occurred_at: Optional[str]

    @classmethod
    def from_orm(cls, c: Communication) -> "CommunicationOut":
        return cls(
            id=str(c.id),
            channel=c.channel,
            direction=c.direction,
            subject=c.subject,
            content_preview=(c.content or "")[:200] + ("…" if len(c.content or "") > 200 else ""),
            sentiment_score=c.sentiment_score,
            sentiment_label=c.sentiment_label,
            tags=c.tags or [],
            occurred_at=c.occurred_at.isoformat() if c.occurred_at else None,
        )


class ScoreHistoryOut(BaseModel):
    id: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    shap_values: dict
    model_version: str
    scored_at: str

    @classmethod
    def from_orm(cls, s: ScoreHistory) -> "ScoreHistoryOut":
        return cls(
            id=str(s.id),
            churn_risk_score=s.churn_risk_score,
            growth_potential_score=s.growth_potential_score,
            health_score=s.health_score,
            shap_values=s.shap_values or {},
            model_version=s.model_version,
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


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    """Service health check — verifies PostgreSQL and ChromaDB connectivity."""
    pg_ok = db_health()
    chroma_ok = vector_store.health_check()
    return {
        "status": "ok" if (pg_ok and chroma_ok) else "degraded",
        "postgres": "up" if pg_ok else "down",
        "chromadb": "up" if chroma_ok else "down",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Affiliates ───────────────────────────────────────────────────────────────

@app.get("/affiliates", response_model=list[AffiliateOut], tags=["Affiliates"])
def list_affiliates(
    tier: Optional[str] = Query(None, description="Filter by tier: bronze|silver|gold|platinum"),
    niche: Optional[str] = Query(None, description="Filter by niche (partial match)"),
    min_health: Optional[float] = Query(None, description="Minimum health score"),
    max_churn: Optional[float] = Query(None, description="Maximum churn risk score"),
    sort_by: str = Query("health_score", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[AffiliateOut]:
    """List all affiliates with optional filtering and sorting."""
    q = db.query(Affiliate)
    if tier:
        q = q.filter(Affiliate.tier == tier.lower())
    if niche:
        q = q.filter(Affiliate.niche.ilike(f"%{niche}%"))
    if min_health is not None:
        q = q.filter(Affiliate.health_score >= min_health)
    if max_churn is not None:
        q = q.filter(Affiliate.churn_risk_score <= max_churn)

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
        **AffiliateOut.from_orm(aff).dict(),
        score_history=[ScoreHistoryOut.from_orm(s) for s in history],
    )
    return detail


@app.get(
    "/affiliates/{affiliate_id}/communications",
    response_model=list[CommunicationOut],
    tags=["Affiliates"],
)
def get_communications(
    affiliate_id: str,
    channel: Optional[str] = Query(None, description="Filter: email|call|chat|ticket"),
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
    if channel:
        q = q.filter(Communication.channel == channel.lower())
    comms = q.offset(offset).limit(limit).all()
    return [CommunicationOut.from_orm(c) for c in comms]


@app.post("/affiliates/{affiliate_id}/score", response_model=ScoreResponse, tags=["Scoring"])
def score_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    """
    Trigger full re-scoring (churn + growth + SHAP) for a single affiliate.
    Updates the affiliate row and appends to score_history.
    """
    from src.ml.churn_model import predict_one as churn_one
    from src.ml.growth_model import predict_one as growth_one
    from src.ml.explainability import explain_affiliate

    aff = _get_affiliate_or_404(affiliate_id, db)
    aid = str(aff.id)

    churn = churn_one(aid)
    growth = growth_one(aid)

    c_score = churn["churn_risk_score"]
    g_score = growth["growth_potential_score"]
    h_score = round(((1 - c_score) * 0.6 + g_score * 0.4) * 100, 1)

    try:
        churn_shap = explain_affiliate(aid, model_type="churn")
        growth_shap = explain_affiliate(aid, model_type="growth")
    except Exception:
        churn_shap = {}
        growth_shap = {}

    aff.churn_risk_score = c_score
    aff.growth_potential_score = g_score
    aff.health_score = h_score

    entry = ScoreHistory(
        affiliate_id=aff.id,
        churn_risk_score=c_score,
        growth_potential_score=g_score,
        health_score=h_score,
        features=churn.get("features", {}),
        shap_values={"churn": churn_shap, "growth": growth_shap},
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
    """
    Send a message to the LangChain ReAct agent.

    Example messages:
    - "Who are my top 3 churn risks right now?"
    - "Summarise the health of Priya Sharma"
    - "Draft a retention email for Tom Bauer"
    - "Which affiliates have mentioned competitor networks recently?"
    """
    from src.agent.agent import chat

    try:
        response = chat(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(
        response=response,
        session_id=request.session_id,
    )


# ─── Ingestion ────────────────────────────────────────────────────────────────

@app.post("/ingest/csv", tags=["Ingestion"])
async def ingest_csv(file: UploadFile = File(...)) -> dict:
    """
    Upload a CSV file of affiliates to ingest into the platform.
    Expected columns: name, email, company, tier, join_date, country,
                      niche, traffic_source, monthly_revenue
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")
    content = await file.read()
    csv_text = content.decode("utf-8")
    result = ingest_csv_content(csv_text)
    return {
        "status": "success",
        "filename": file.filename,
        **result,
    }
