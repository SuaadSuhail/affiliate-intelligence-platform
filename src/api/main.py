"""
FastAPI Application
===================
REST API + HTML frontend for the Affiliate Intelligence Platform.

Endpoints
---------
GET  /                            — chat UI (served from templates/index.html)
GET  /health                      — service health check
GET  /affiliates                  — list affiliates (new schema)
GET  /affiliates/{id}             — single affiliate + score history
GET  /affiliates/{id}/communications — paginated communications
POST /affiliates/{id}/score       — trigger re-scoring

POST /ingest/full|affiliates|communications|csv
POST /process/nlp|embeddings|full
GET  /communications, /communications/{id}, /search
POST /ml/train|score
GET  /ml/scores|explain/{id}|dashboard
POST /agent/chat|quick
GET  /agent/demo

Run:
    uvicorn src.api.main:app --port 8080 --reload
"""

from __future__ import annotations

import time
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structured JSON logging before any other src imports so that
# module-level loggers (embedding_generator, nlp_processor, etc.) pick it up.
from src.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

from src.storage.database import get_db, health_check as db_health, init_db
from src.storage.models import Affiliate, Communication, ScoreHistory

# ─── Routers ──────────────────────────────────────────────────────────────────

from src.api.routers.ingest import router as ingest_router
from src.api.routers.process import router as process_router
from src.api.routers.search import router as search_router
from src.api.routers.ml import router as ml_router
from src.api.routers.agent import router as agent_router

# ─── Paths ────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent
_STATIC_DIR = _BASE / "static"
_TEMPLATES_DIR = _BASE / "templates"

# ─── App ──────────────────────────────────────────────────────────────────────

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

# Static files and templates
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ─── Request logging middleware ───────────────────────────────────────────────

class _RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "HTTP request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


app.add_middleware(_RequestLoggingMiddleware)

# ─── Include routers ──────────────────────────────────────────────────────────

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(process_router, prefix="/process", tags=["Processing"])
app.include_router(search_router, tags=["Search"])
app.include_router(ml_router, prefix="/ml", tags=["ML"])
app.include_router(agent_router, prefix="/agent", tags=["Agent"])


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    routes = [r.path for r in app.routes]
    logger.info("Application startup complete", extra={"routes_registered": len(routes)})


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def index(request: Request) -> HTMLResponse:
    """Serve the main chat UI."""
    return templates.TemplateResponse(request=request, name="index.html")


# ─── Pydantic schemas (new schema) ────────────────────────────────────────────

class AffiliateOut(BaseModel):
    id: str
    name: str
    status: str
    churn_risk_score: float
    growth_potential_score: float
    health_score: float
    revenue_30d: float
    days_since_contact: int
    last_contact_at: Optional[str]

    @classmethod
    def from_orm(cls, a: Affiliate) -> "AffiliateOut":
        return cls(
            id=str(a.id),
            name=a.name,
            status=a.status or "active",
            churn_risk_score=round(a.churn_risk_score or 0.5, 4),
            growth_potential_score=round(a.growth_potential_score or 0.5, 4),
            health_score=round(a.health_score or 50.0, 1),
            revenue_30d=float(a.revenue_30d or 0.0),
            days_since_contact=int(a.days_since_contact or 0),
            last_contact_at=(
                a.last_contact_at.isoformat() if a.last_contact_at else None
            ),
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
    """Service health check — verifies PostgreSQL and ChromaDB."""
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
    status: Optional[str] = Query(None, description="Filter by status: active|at_risk|churned|high_growth"),
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
    if status:
        q = q.filter(Affiliate.status == status.lower())
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
    return AffiliateDetail(
        **AffiliateOut.from_orm(aff).model_dump(),
        score_history=[ScoreHistoryOut.from_orm(s) for s in history],
    )


@app.get(
    "/affiliates/{affiliate_id}/communications",
    response_model=list[dict],
    tags=["Affiliates"],
)
def get_affiliate_communications(
    affiliate_id: str,
    source: Optional[str] = Query(None, description="Filter: email|call|api_event"),
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
    if source:
        q = q.filter(Communication.source == source.lower())
    comms = q.offset(offset).limit(limit).all()
    return [
        {
            "id": str(c.id),
            "source": c.source,
            "raw_text_preview": (c.raw_text or "")[:200],
            "sentiment_score": c.sentiment_score,
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
    from src.ml.feature_engineering import build_feature_vector
    from src.ml.churn_model import predict_churn_risk
    from src.ml.growth_model import predict_growth_potential

    aff = _get_affiliate_or_404(affiliate_id, db)
    aid = str(aff.id)

    features = build_feature_vector(aid, db)
    c_score = predict_churn_risk(aid, features)
    g_score = predict_growth_potential(aid, features)
    h_score = round(((1 - c_score) * 0.6 + g_score * 0.4) * 100, 1)

    aff.churn_risk_score = round(c_score, 4)
    aff.growth_potential_score = round(g_score, 4)
    aff.health_score = h_score

    entry = ScoreHistory(
        affiliate_id=aff.id,
        churn_risk_score=round(c_score, 4),
        growth_potential_score=round(g_score, 4),
        health_score=h_score,
        scored_at=datetime.utcnow(),
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