# Affiliate Intelligence Platform

> **Agentic AI CRM & Sales Communication Optimisation Platform**  
> Produces a 360° health score for every affiliate — predicting churn risk and growth potential.

---

## Architecture Overview

```
CSV / Email / Transcript
        │
        ▼
┌─────────────────────┐
│   ETL Pipeline      │  ← etl_pipeline.py: ingest, clean, route
│   NLP Processor     │  ← nlp_processor.py: 20 tags + VADER sentiment
│   Embedding Gen.    │  ← embedding_generator.py: sentence-transformers
└────────┬────────────┘
         │
    ┌────▼─────────────────────┐
    │  PostgreSQL              │  ← affiliates, communications, score_history
    │  ChromaDB                │  ← communications_embeddings, affiliate_profiles
    └────┬─────────────────────┘
         │
┌────────▼────────────┐
│  Feature Engineering│  ← build feature vectors from DB aggregates
│  Churn Model        │  ← XGBoost → churn_risk_score (0–1)
│  Growth Model       │  ← XGBoost → growth_potential_score (0–1)
│  SHAP Explainability│  ← per-affiliate feature attribution
└────────┬────────────┘
         │
┌────────▼────────────┐
│  LangChain Agent    │  ← ReAct agent with 6 tools
│  FastAPI REST API   │  ← /affiliates, /agent/chat, /ingest
└─────────────────────┘
```

## Health Score Formula

```
health_score = ((1 - churn_risk_score) × 0.6 + growth_potential_score × 0.4) × 100
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/SuaadSuhail/affiliate-intelligence-platform.git
cd affiliate-intelligence-platform
```

### 2. Create the conda environment & configure

```bash
conda env create -f environment.yml
conda activate affiliate-intelligence
python -m spacy download en_core_web_sm
cp .env.example .env          # then fill in your values
```

> **`.env` values you must set:** `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, `CHROMA_TOKEN`  
> All other values have working defaults for local development.

### 3. Start infrastructure

```bash
docker-compose up -d postgres chromadb
```

### 4. Seed mock data

```bash
python src/ingestion/etl_pipeline.py
```

### 5. Train ML models (uses seeded data)

```bash
python src/ml/churn_model.py
python src/ml/growth_model.py
```

### 6. Start API

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## Project Structure

```
affiliate-intelligence-platform/
├── CLAUDE.md               ← Single source of truth (read this first)
├── docker-compose.yml
├── environment.yml         ← Conda environment (pinned deps)
├── .env.example
├── requirements.txt
├── data/
│   └── mock/
│       ├── affiliates.csv
│       ├── emails.txt
│       └── transcripts.txt
├── src/
│   ├── ingestion/
│   │   ├── etl_pipeline.py         ← Orchestrates full ingestion
│   │   ├── nlp_processor.py        ← 20-tag tagger + sentiment
│   │   └── embedding_generator.py  ← ChromaDB upsert via sentence-transformers
│   ├── storage/
│   │   ├── models.py               ← SQLAlchemy ORM models
│   │   ├── database.py             ← Session factory + init_db()
│   │   └── vector_store.py         ← ChromaDB client wrapper
│   ├── ml/
│   │   ├── feature_engineering.py  ← Aggregate DB → feature DataFrame
│   │   ├── churn_model.py          ← Train / predict churn_risk_score
│   │   ├── growth_model.py         ← Train / predict growth_potential_score
│   │   └── explainability.py       ← SHAP value computation
│   ├── agent/
│   │   ├── tools.py                ← 6 LangChain tools
│   │   └── agent.py                ← ReAct agent executor
│   └── api/
│       └── main.py                 ← FastAPI application
└── tests/
    └── test_tagging.py
```

---

## NLP Tags

The platform detects 20 tags per communication. See [CLAUDE.md](CLAUDE.md) for full
detection rules. Summary:

`churn_signal` · `growth_intent` · `payment_issue` · `technical_issue` ·
`satisfaction_high` · `satisfaction_low` · `competitor_mention` · `escalation_risk` ·
`support_request` · `feature_request` · `pricing_concern` · `fraud_risk` ·
`high_engagement` · `low_engagement` · `compliance_issue` · `new_opportunity` ·
`seasonal_pattern` · `relationship_warm` · `urgency` · `question_asked`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/affiliates` | List all affiliates with current scores |
| GET | `/affiliates/{id}` | Single affiliate + score history |
| GET | `/affiliates/{id}/communications` | Paginated comms |
| POST | `/affiliates/{id}/score` | Trigger re-scoring |
| POST | `/agent/chat` | `{"message": "..."}` → agent response |
| POST | `/ingest/csv` | Upload affiliates CSV |
| GET | `/health` | Service health check |

---

## Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.x |
| Relational DB | PostgreSQL 16 |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| NLP | spaCy + VADER Sentiment |
| ML | XGBoost + scikit-learn |
| Explainability | SHAP |
| Agent | LangChain ReAct + GPT-4o |
| Containerisation | Docker Compose |
