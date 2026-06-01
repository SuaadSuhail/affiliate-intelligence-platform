"""
FastAPI Application
===================
REST API for the Affiliate Intelligence Platform.

Endpoints
---------
GET  /health                          — service health check
GET  /affiliates                      — list all affiliates with scores
GET  /affiliates/{id}                 — single affiliate + score history
GET  /affiliates/{id}/communications  — paginated communications for one affiliate
POST /affiliates/{id}/score           — trigger re-scoring for one affiliate
POST /agent/chat                      — chat with the LangChain ReAct agent

POST /ingest/full                     — run full ETL from mock data files
POST /ingest/affiliates               — re-ingest affiliates CSV
POST /ingest/communications           — re-ingest emails + transcripts
POST /ingest/csv                      — upload affiliates CSV file

POST /process/nlp                     — tag all untagged communications
POST /process/embeddings              — embed all unembedded communications
POST /process/full                    — NLP + embeddings end-to-end

GET  /communications                  — list all communications with tags
GET  /communications/{id}             — single communication by UUID
GET  /search                          — semantic search over communications

POST /ml/train                        — train churn + growth XGBoost models
POST /ml/score                        — score all affiliates and persist
GET  /ml/scores                       — list current affiliate scores
GET  /ml/explain/{affiliate_id}       — SHAP feature importances
GET  /ml/dashboard                    — aggregate health statistics

Run:
    uvicorn src.api.main:app --port 8080 --reload
"""

from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.storage.database import get_db, health_check as db_health, init_db
from src.storage.models import Affiliate, Communication, ScoreHistory

# ─── Routers ──────────────────────────────────────────────────────────────────

from src.api.routers.ingest import router as ingest_router
from src.api.routers.process import router as process_router
from src.api.routers.search import router as search_router
from src.api.routers.ml import router as ml_router

# ─── App ──────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

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

# ─── Include routers ──────────────────────────────────────────────────────────

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(process_router, prefix="/process", tags=["Processing"])
app.include_router(search_router, tags=["Search"])
app.include_router(ml_router, prefix="/ml", tags=["ML"])


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    routes = [r.path for r in app.routes]
    logger.info(f"Registered {len(routes)} routes: {routes}")
    print(f"[startup] {len(routes)} routes registered.")


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
    from src.storage.vector_store import vector_store
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
        **AffiliateOut.from_orm(aff).model_dump(),
        score_history=[ScoreHistoryOut.from_orm(s) for s in history],
    )
    return detail


@app.get(
    "/affiliates/{affiliate_id}/communications",
    response_model=list[dict],
    tags=["Affiliates"],
)
def get_affiliate_communications(
    affiliate_id: str,
    channel: Optional[str] = Query(None, description="Filter: email|call|chat|ticket"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
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
    return [
        {
            "id": str(c.id),
            "channel": c.channel,
            "direction": c.direction,
            "subject": c.subject,
            "content_preview": (c.content or "")[:200],
            "sentiment_score": c.sentiment_score,
            "sentiment_label": c.sentiment_label,
            "tags": c.tags or [],
            "embedding_id": c.embedding_id,
            "occurred_at": c.occurred_at.isoformat() if c.occurred_at else None,
        }
        for c in comms
    ]


# ─── Per-affiliate scoring ─────────────────────────────────────────────────────

@app.post("/affiliates/{affiliate_id}/score", response_model=ScoreResponse, tags=["Affiliates"])
def score_affiliate(
    affiliate_id: str,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    """Trigger re-scoring for a single affiliate."""
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
        features=churn.get("features", {}),
        shap_values={},
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

    return ChatResponse(response=response, session_id=request.session_id)