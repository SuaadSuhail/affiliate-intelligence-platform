# Affiliate Intelligence Platform

An agentic AI system that produces a **360° health score** for every affiliate partner by combining structured CRM data, NLP analysis of email and call communications, semantic vector search, and XGBoost ML models. A LangChain ReAct agent answers plain-English questions, surfaces at-risk affiliates, and drafts personalised re-engagement emails autonomously — all through a browser-based chat interface.

---

## Demo

The agent chains five tools — SQL query, semantic search, affiliate profile, email drafting, and portfolio stats — to answer complex questions in a single response.

**"Which affiliates need urgent attention?"**
```
1. Tom Bauer       — health 14.4  | churn 88% | status: churned | 54 days silent
2. James O'Brien   — health 27.2  | churn 74% | status: at_risk | 33 days silent
3. Marcus Williams — health 37.0  | churn 61% | status: at_risk | 24 days silent

Tools used: query_database
```

**"What is happening with Tom Bauer?"**
```
Tom Bauer has a health score of 14.4/100 (critical). His churn risk is 88%
and he is currently marked as churned. He has been silent for 54 days.
His last communication expressed frustration — tags: escalation, frustrated.

Recommended action: Schedule urgent retention call within 48 hours.

Tools used: get_affiliate_summary
```

**"Draft a re-engagement email for Tom Bauer."**
```
Subject: We want to make this right, Tom

Hi Tom,

I noticed it's been a while since we last spoke, and I wanted to reach out
personally. I understand there were some frustrations with the platform —
I'd like to address those directly.

Could we schedule a 20-minute call this week? I'll come prepared with
specific steps we can take to resolve the issues you raised.

[Your Name] — Partner Success Team

Tools used: get_affiliate_summary, draft_email
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. Data Sources                                         │
│     affiliates.csv · emails.txt · transcripts.txt · API  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  2. Ingestion & Processing                               │
│     ETL pipeline · spaCy NLP (21 tags) · embeddings     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  3. Storage                                              │
│     PostgreSQL (affiliates, communications, scores)      │
│     pgvector  (384-dim communication embeddings)         │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼──────────┐   ┌─────────────▼──────────────────┐
│  4. ML Prediction  │   │  5. Agentic AI Core             │
│  XGBoost churn +   │   │  LangChain ReAct agent          │
│  growth models     │   │  5 tools · gpt-4o-mini          │
│  SHAP explanations │   │  conversation history           │
└─────────┬──────────┘   └─────────────┬──────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  6. API & Frontend                                       │
│     FastAPI (21 endpoints) · chat UI · portfolio panel   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | Python 3.11, FastAPI, Uvicorn |
| Agent framework | LangChain / LangGraph, OpenAI gpt-4o-mini |
| ML models | XGBoost, SHAP, scikit-learn |
| NLP | spaCy (`en_core_web_sm`), custom sentiment lexicon |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Vector store | pgvector (cosine similarity, PostgreSQL extension) |
| Database | PostgreSQL, SQLAlchemy ORM |
| Infrastructure | Docker, Docker Compose |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Testing | pytest |

---

## Key Features

- **360° health score** — composite metric combining churn risk (0–1) and growth potential (0–1) into a 0–100 score: `((1 − churn) × 0.6 + growth × 0.4) × 100`
- **21-tag NLP classification** — each communication is tagged across four groups (engagement, sentiment, intent, relationship) using spaCy NER plus a 40-word custom sentiment lexicon
- **Semantic search** — pgvector embedding search over all email and call transcript content; returns the most semantically relevant communications for any natural-language query
- **SHAP explainability** — every XGBoost prediction includes top-5 SHAP feature importances identifying the specific drivers of churn risk or growth potential for each affiliate
- **ReAct agent with 5 tools** — the LangChain agent autonomously decides which tools to call, chains multiple results together, and produces a coherent answer with source attribution
- **Browser chat interface** — two-panel UI with live portfolio stats, affiliate health bars, conversation history, tools-used attribution, and suggested questions

---

## Project Structure

```
affiliate-intelligence-platform/
├── src/
│   ├── agent/
│   │   ├── agent.py            ← LangGraph ReAct agent, run_agent()
│   │   └── tools.py            ← 5 tool definitions (@tool decorated)
│   ├── api/
│   │   ├── main.py             ← FastAPI app, router wiring, GET /
│   │   ├��─ routers/            ← ingest, process, search, ml, agent
│   │   ├── templates/          ← Jinja2 chat interface (index.html)
│   │   └── static/             ← CSS and static assets
│   ├── ingestion/
│   │   ├── etl_pipeline.py     ← CSV + flat-file data loading
│   │   ├── nlp_processor.py    ← spaCy tagging + sentiment scoring
│   │   └── embedding_generator.py ← chunk, encode, store in pgvector
│   ├── ml/
│   │   ├── feature_engineering.py ← 12-feature vector builder
│   │   ├── churn_model.py      ← XGBoost churn + rule-based fallback
│   │   ├── growth_model.py     ← XGBoost growth + rule-based fallback
│   │   ├── explainability.py   ← SHAP TreeExplainer, top-5 factors
│   │   └── score_updater.py    ← daily scoring pipeline, score_history
│   └── storage/
│       ├── models.py           ← SQLAlchemy ORM models
│       ├── database.py         ← engine, session factory, get_db()
│       └── pgvector_store.py   ← pgvector wrapper, add/search
├── data/mock/                  ← 10 affiliate profiles, 13 communications
├── tests/                      ← pytest suite (24 tests across 4 files)
├── models/                     ← XGBoost artefacts (gitignored)
└── docker-compose.yml          ← PostgreSQL (with pgvector extension)
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.11 with [conda](https://docs.conda.io/)
- OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SuaadSuhail/affiliate-intelligence-platform.git
cd affiliate-intelligence-platform

# 2. Create the Python environment
conda create -n affiliate-intelligence python=3.11
conda activate affiliate-intelligence
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure environment variables
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...

# 4. Start the infrastructure
docker compose up -d
# PostgreSQL → :5432  |  App → :8080

# 5. Run migrations (required on first setup)
curl -X POST http://localhost:8080/admin/migrate \
  -H "X-Api-Key: change-me-in-production"
```

### Run the Data Pipeline

```bash
# Load affiliates and communications from mock data files
curl -X POST http://localhost:8080/ingest/full \
  -H "X-Api-Key: change-me-in-production"

# Run NLP tagging on all communications
curl -X POST http://localhost:8080/process/nlp \
  -H "X-Api-Key: change-me-in-production"

# Generate and index communication embeddings
curl -X POST http://localhost:8080/process/embeddings \
  -H "X-Api-Key: change-me-in-production"

# Train churn and growth XGBoost models
curl -X POST http://localhost:8080/ml/train \
  -H "X-Api-Key: change-me-in-production"

# Score all affiliates
curl -X POST http://localhost:8080/ml/score \
  -H "X-Api-Key: change-me-in-production"
```

### Open the Interface

Navigate to **[http://localhost:8080](http://localhost:8080)**

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/full` | Run full ETL from mock data files |
| `POST` | `/ingest/csv` | Upload affiliates CSV |
| `POST` | `/process/nlp` | Tag all untagged communications |
| `POST` | `/process/embeddings` | Generate and store embeddings |
| `POST` | `/ml/train` | Train churn + growth XGBoost models |
| `POST` | `/ml/score` | Score all affiliates, persist results |
| `GET` | `/ml/dashboard` | Portfolio aggregate statistics |
| `GET` | `/ml/scores` | Affiliate scores sorted worst-first |
| `GET` | `/ml/explain/{id}` | SHAP feature importances for one affiliate |
| `GET` | `/affiliates` | List affiliates with filtering and sorting |
| `GET` | `/search?q=...` | Semantic search over communications |
| `POST` | `/agent/chat` | Chat with the ReAct agent (with history) |
| `POST` | `/agent/quick` | Single-turn agent query |
| `GET` | `/agent/demo` | Run three preset demo questions |
| `GET` | `/agent/health` | Check agent status |
| `POST` | `/admin/migrate` | Run database migrations |
| `GET` | `/admin/migration-status` | Check migration version |
| `GET` | `/task/{task_id}` | Poll background task status |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Model storage

Models are saved locally to `models/` by default.

For production, set `USE_S3=true` in `.env` and provide `S3_BUCKET`. Models are automatically uploaded after training and downloaded on first use if not present locally.

Works with any S3-compatible storage:
- **AWS S3** — leave `S3_ENDPOINT_URL` empty
- **DigitalOcean Spaces** — `https://nyc3.digitaloceanspaces.com`
- **Cloudflare R2** — `https://<acct>.r2.cloudflarestorage.com`
- **Backblaze B2** — `https://s3.us-west-002.backblazeb2.com`

---

## Database migrations

This project uses [Alembic](https://alembic.sqlalchemy.org/) for schema management. Migrations are run **manually** — never automatically on startup. This is intentional: in production, multiple app instances may start simultaneously behind a load balancer, and auto-migration would cause database lock contention and data corruption risk.

### How migrations work

Alembic tracks which migrations have been applied in a table called `alembic_version` in PostgreSQL. Each migration file has an `upgrade()` and `downgrade()` function. Running `upgrade head` applies all pending migrations in order.

### First time setup (new environment)

```bash
# 1. Start the app
docker compose up -d

# 2. Wait for the app to be healthy
curl http://localhost:8080/health

# 3. Run migrations
curl -X POST http://localhost:8080/admin/migrate \
  -H "X-Api-Key: your-api-key"

# 4. Verify — response should be:
# {"status": "complete", "message": "All migrations applied successfully"}
```

### Check current migration version

```bash
curl http://localhost:8080/admin/migration-status \
  -H "X-Api-Key: your-api-key"
```

### After a schema change (development)

```bash
# 1. Generate the migration file
alembic revision --autogenerate -m "describe what changed"

# 2. Review the generated file in alembic/versions/ and confirm it looks correct

# 3. Apply it
curl -X POST http://localhost:8080/admin/migrate \
  -H "X-Api-Key: your-api-key"
```

### Deploying a schema change to production

Follow this sequence exactly — order matters:

1. Deploy the new version of the app
2. Wait for the app to be healthy: `curl https://your-domain.com/health`
3. Run migrations **once** via the admin endpoint:
   ```bash
   curl -X POST https://your-domain.com/admin/migrate \
     -H "X-Api-Key: your-production-api-key"
   ```
4. Verify the response shows `"status": "complete"`
5. Only then send real traffic to the new version

### CLI commands (direct access)

```bash
alembic upgrade head    # apply all pending
alembic downgrade -1    # rollback one migration
alembic current         # show current version
alembic history         # show all migrations
```

### Important rules

- Never run migrations on multiple instances simultaneously
- Always back up your database before running migrations in production
- Always review autogenerated migration files before applying — Alembic can miss complex changes
- Never delete files from `alembic/versions/` even after they have been applied

---

## Frontend

The React frontend is in a separate repository:
https://github.com/SuaadSuhail/affiliate-intelligence-frontend

---

## Security

All write endpoints require an API key header:

```
X-Api-Key: your-api-key
```

Set `API_SECRET_KEY` in your `.env` file.
Default for local dev: `change-me-in-production`

Set `ALLOWED_ORIGINS` to your domain in production.

In development (`APP_ENV=development`) auth is bypassed automatically.

---

## Tests

```bash
pytest tests/ -v
```

| File | Tests | Coverage |
|---|---|---|
| `tests/test_nlp.py` | 6 | Sentiment scoring, tag detection (churn signal, competitor mention, enthusiasm), bulk processing |
| `tests/test_embeddings.py` | 6 | `chunk_text` splits and overlap, embed pipeline, semantic search endpoint |
| `tests/test_ml.py` | 5 | Feature vector structure, rule-based scorers, score updater idempotency, SHAP explanation format |
| `tests/test_agent.py` | 7 | SQL validation (SELECT-only), affiliate summary found/not-found, portfolio stats, semantic search, agent initialisation |

---

## Background

This project was built to demonstrate the application of agentic AI to affiliate relationship management. It combines classical machine learning, large language models, and semantic search into a unified system that enables proactive, data-driven decision making across an affiliate partner portfolio.