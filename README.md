# Affiliate Intelligence Platform

An agentic AI system that produces a **360° health score** for every affiliate partner by combining structured CRM data, NLP analysis of email and call communications, semantic vector search, and XGBoost ML models. A LangChain ReAct agent answers plain-English questions, surfaces at-risk affiliates, and drafts personalised re-engagement emails autonomously — all through a browser-based chat interface.

---

## Demo

The agent chains seven tools — SQL query, semantic search, affiliate profile, email drafting (to an approval queue, never a direct send), portfolio stats, promo-leak status, and SEO status — to answer complex questions in a single response.

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
│     ChromaDB  (384-dim communication embeddings)         │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴───────────┬──────────────────────────┐
          │                          │                          │
┌─────────▼──────────┐   ┌───────────▼──────────┐   ┌───────────▼───────────┐
│  4. ML Prediction  │   │  5. Leakage Detector │   │  6. Agentic AI Core   │
│  XGBoost churn +   │   │  Playwright scrape + │   │  LangChain ReAct agent│
│  growth models     │   │  code match · persist│   │  6 tools · gpt-4o-mini│
│  SHAP explanations │   │  nightly + on-demand │   │  conversation history │
└─────────┬──────────┘   └───────────┬──────────┘   └───────────┬───────────┘
          │                          │                          │
          └──────────────┬───────────┴──────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  7. API & Frontend                                       │
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
| Vector store | pgvector (cosine similarity, same PostgreSQL instance — ChromaDB was removed) |
| Database | PostgreSQL, SQLAlchemy ORM |
| Infrastructure | Docker, Docker Compose |
| Frontend | Vanilla HTML / CSS / JS (no framework) |
| Web scraping | Playwright (headless Chromium) |
| HTML parsing | BeautifulSoup4, lxml |
| Scheduling | APScheduler |
| Testing | pytest |

---

## Key Features

- **360° health score** — composite metric combining churn risk (0–1) and growth potential (0–1) into a 0–100 score: `((1 − churn) × 0.6 + growth × 0.4) × 100`
- **Deterministic rulebook** — a single, pure, exhaustively-tested module (`src/rulebook/`) is the only place in the codebase allowed to turn churn/growth scores into a tier or recommendation; every other call site imports from it instead of keeping its own threshold copy
- **21-tag NLP classification** — each communication is tagged across four groups (engagement, sentiment, intent, relationship) using spaCy NER plus a 40-word custom sentiment lexicon
- **Semantic search** — pgvector cosine-similarity search over all email and call transcript content; returns the most semantically relevant communications for any natural-language query
- **SHAP explainability, honestly labelled** — every XGBoost prediction includes top-5 SHAP feature importances; a computation failure returns an explicit `explanation_unavailable` flag instead of fabricated zero values, and every response is clearly marked as a secondary, independent model estimate — not the persisted score that actually drives recommendations
- **ReAct agent with 7 tools** — the LangChain agent autonomously decides which tools to call, chains multiple results together, and produces a coherent answer with source attribution; every tool is read-only or draft-only, enforced by a static-analysis regression test
- **Promo code leakage detection** — scheduled (nightly) and on-demand scans of monitored voucher sites for unauthorised use of affiliate promo codes, with every match linked to its source URL as evidence, and a first-class `has_active_leak` flag on the affiliate itself
- **SEO rank tracking** — scheduled (weekly) and on-demand checks of a tracked keyword per affiliate, deriving a `declining`/`stable`/`improving` search trend kept visible alongside, never folded into, the health score
- **Human approval gate** — nothing that would leave the system (currently: an email) fires without a person clicking Approve; a drafted email sits in a queue as `waiting_for_review` until then
- **Audit trail** — every stored recommendation, signal check, and approval decision is linked back to the exact inputs and rule/tool that produced it, queryable via `GET /audit`
- **Queryable log persistence** — structured JSON logs are written to a rotating file in addition to stdout, readable via `GET /admin/logs` — a demo-appropriate stand-in for a real log aggregator
- **Browser chat interface** — two-panel UI with live portfolio stats, affiliate health bars, conversation history, tools-used attribution, and suggested questions

---

## Project Structure

```
affiliate-intelligence-platform/
├── src/
│   ├── agent/
│   │   ├── agent.py            ← LangGraph ReAct agent, run_agent()
│   │   └── tools.py            ← 7 tool definitions (@tool decorated)
│   ├── api/
│   │   ├── main.py             ← FastAPI app, router wiring, GET /
│   │   ├── routers/            ← ingest, process, search, ml, agent, admin,
│   │   │                          leakage, approvals, audit, seo
│   │   ├── templates/          ← Jinja2 chat interface (index.html)
│   │   └── static/             ← CSS and static assets
│   ├── audit/
│   │   └── log.py              ← write_audit_entry() — single append-only write path
│   ├── core/
│   │   └── logging_config.py   ← structured JSON logging: stdout + rotating file
│   ├── ingestion/
│   │   ├── etl_pipeline.py     ← CSV + flat-file data loading, demo leak/SEO seeding
│   │   ├── nlp_processor.py    ← spaCy tagging + sentiment scoring
│   │   └── embedding_generator.py ← chunk, encode, store via pgvector
│   ├── ml/
│   │   ├── feature_engineering.py ← 12-feature vector builder
│   │   ├── churn_model.py      ← XGBoost churn + rule-based fallback
│   │   ├── growth_model.py     ← XGBoost growth + rule-based fallback
│   │   ├── explainability.py   ← SHAP TreeExplainer, top-5 factors, honest-failure handling
│   │   ├── model_store.py      ← local disk + optional S3 model persistence
│   │   └── score_updater.py    ← daily scoring pipeline, score_history, audit wiring
│   ├── notifications/
│   │   └── sender.py           ← send_email() placeholder — called only from POST /approvals/{id}/approve
│   ├── rulebook/
│   │   └── recommend.py        ← categorize()/recommend() — single source of truth for tier/reason/evidence
│   ├── scheduling/
│   │   └── jobs.py             ← APScheduler: nightly leakage scan (03:00 UTC),
│   │                              weekly SEO scan (Mon 04:00 UTC)
│   ├── scraping/
│   │   ├── site_config.py      ← SiteConfig registry (fixture + live sites)
│   │   ├── fetcher.py          ← Playwright fetch, robots.txt + rate limiting
│   │   ├── extractor.py        ← selector + regex candidate code extraction
│   │   ├── leakage_scraper.py  ← check_leakage() orchestrator, dedup window
│   │   └── fixtures/           ← offline HTML/JS fixtures for safe demo scans
│   ├── seo/
│   │   ├── api_client.py       ← fetch_seo_data(), fixture-first (mirrors fetcher.py)
│   │   ├── analyze.py          ← derive_search_trend(), pure signal-detection
│   │   └── checker.py          ← check_seo() orchestrator
│   └── storage/
│       ├── models.py           ← SQLAlchemy ORM models
│       ├── database.py         ← engine, session factory, get_db()
│       └── pgvector_store.py   ← pgvector wrapper, add/search
├── data/mock/                  ← 10 affiliate profiles, 13 communications, SEO rank fixture
├── tests/                      ← pytest suite (121 tests across 12 files)
├── models/                     ← XGBoost artefacts (gitignored)
├── logs/                       ← rotating structured JSON log file (gitignored)
└── docker-compose.yml          ← PostgreSQL (with pgvector) service
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

# 2. Create the Python environment (used for tests, Alembic CLI, and other local tooling)
conda create -n affiliate-intelligence python=3.11
conda activate affiliate-intelligence
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium

# 3. Configure environment variables
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...

# 4. Start everything with Docker Compose
docker compose up -d
# PostgreSQL → :5432  |  API → :8080 (auto-reloads on src/ changes via bind mount)
```

Docker Compose is the only supported way to run the API server — the `app` service
already mounts `./src` with `--reload`, so code changes take effect without a rebuild
or restart. Running `uvicorn` locally against the same port will conflict with the
Docker container; the conda environment above is for running tests and Alembic CLI
commands, not for serving the API.

### Run the Data Pipeline

```bash
# Load affiliates and communications from mock data files.
# Also seeds a demo leak: two affiliates in affiliates.csv (Rachel Torres,
# Marcus Williams) ship with an active_promo_code that matches a code already
# present in the scraping fixtures, and this step runs one real leakage scan
# over them — so GET /leakage/results and has_active_leak on those two
# affiliates are already populated, no separate manual scan needed.
# Also seeds demo SEO signals: four affiliates ship with a tracked_keyword
# matching data/mock/seo/rank_tracking_mock.json, and this step runs one real
# SEO rank check over them — Sarah Chen and Marcus Williams are engineered to
# show search_trend=declining, so GET /seo/results and search_trend are
# already populated too.
curl -X POST http://localhost:8080/ingest/full

# Run NLP tagging on all communications
curl -X POST http://localhost:8080/process/nlp

# Generate and index communication embeddings
curl -X POST http://localhost:8080/process/embeddings

# Train churn and growth XGBoost models
curl -X POST http://localhost:8080/ml/train

# Score all affiliates
curl -X POST http://localhost:8080/ml/score
```

### Open the Interface

Navigate to **[http://localhost:8080](http://localhost:8080)**

---

## API Reference

Not exhaustive — see `GET /docs` (Swagger UI) for the complete, live list of
all routes with request/response schemas.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/full` | Run full ETL from mock data files (also seeds demo leak/SEO data) |
| `POST` | `/ingest/csv` | Upload affiliates CSV |
| `POST` | `/process/nlp` | Tag all untagged communications |
| `POST` | `/process/embeddings` | Generate and store embeddings |
| `POST` | `/ml/train` | Train churn + growth XGBoost models |
| `POST` | `/ml/score` | Score all affiliates, persist results |
| `GET` | `/ml/dashboard` | Portfolio aggregate statistics |
| `GET` | `/ml/scores` | Affiliate scores sorted worst-first |
| `GET` | `/ml/explain/{id}` | SHAP feature importances for one affiliate (a secondary, independent model estimate — see CLAUDE.md) |
| `POST` | `/leakage/scan` | Run a full promo-code leakage scan (background task) |
| `GET` | `/leakage/results` | All recorded leak detections, newest first |
| `POST` | `/seo/scan` | Run a full SEO rank check (background task) |
| `GET` | `/seo/results` | All recorded SEO rank signals, newest first |
| `GET` | `/approvals?status=` | List approval requests, optionally filtered by status |
| `POST` | `/approvals/{id}/approve` | Approve a pending request — the only path that dispatches the real action |
| `POST` | `/approvals/{id}/reject` | Reject a pending request |
| `GET` | `/audit?record_type=&record_id=&stage=` | Decision-linked audit trail (API-key gated) |
| `GET` | `/admin/logs?level=&search=&limit=` | Recent structured log entries from disk (API-key gated) |
| `GET` | `/affiliates` | List affiliates with filtering and sorting |
| `GET` | `/search?q=...` | Semantic search over communications |
| `POST` | `/agent/chat` | Chat with the ReAct agent (with history) |
| `POST` | `/agent/quick` | Single-turn agent query |
| `GET` | `/agent/demo` | Run three preset demo questions |
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

## Security

All write endpoints require an API key header:

```
X-Api-Key: your-secret-key
```

Set `API_SECRET_KEY` in your `.env` file. Set `ALLOWED_ORIGINS` to your domain in production.

In development (`APP_ENV=development`) auth is bypassed automatically.

---

## Tests

```bash
pytest tests/ -v
```

121 tests across 12 files (120 passing, 1 real-DB test self-skips when its
precondition already holds — see `tests/test_audit.py`):

| File | Tests | Coverage |
|---|---|---|
| `tests/test_nlp.py` | 6 | Sentiment scoring, tag detection (churn signal, competitor mention, enthusiasm), bulk processing |
| `tests/test_embeddings.py` | 6 | `chunk_text` splits and overlap, embed pipeline, semantic search endpoint |
| `tests/test_ml.py` | 13 | Feature vector structure, rule-based scorers, score updater idempotency, SHAP explanation format + honest-failure handling |
| `tests/test_agent.py` | 12 | SQL validation, affiliate summary, portfolio health (incl. leak/SEO visibility), agent tool side-effect regression test |
| `tests/test_agent_multisignal.py` | 1 | Real (non-mocked) LLM call confirming multi-signal warning-sign questions surface both multi- and single-signal affiliates |
| `tests/test_rulebook.py` | 22 | `categorize()`/`recommend()` boundary values, leak evidence without tier changes |
| `tests/test_approvals.py` | 8 | `draft_email` files to the approval queue (never sends), approve/reject lifecycle, 409 on double-decision, audit wiring |
| `tests/test_audit.py` | 4 | `GET /audit` filtering, referential integrity of `record_id` across wiring points |
| `tests/test_etl_pipeline.py` | 19 | Ingest idempotency (scores, communications), demo leak/SEO seeding, mock-only guards |
| `tests/test_leakage_scraper.py` | 11 | Extractor/matcher, end-to-end scan, dedup window, `has_active_leak` recompute |
| `tests/test_logging_config.py` | 8 | File handler output, level/search filtering, rotation correctness |
| `tests/test_seo.py` | 11 | `derive_search_trend()` boundaries, end-to-end scan, exact-measurement dedup guard |

---

## Dependency pinning

`xgboost` and `shap` are pinned to exact versions (`xgboost==2.1.4`,
`shap==0.49.1`) in `requirements.txt`, not the unbounded `>=` ranges used
for everything else. `xgboost>=3.0.0` changed how it serializes a trained
model's `base_score`, which breaks `shap.TreeExplainer` construction on
this project's Python version (3.10, pinned by the Playwright base image).
See `requirements.txt`'s inline comment and CLAUDE.md's "SHAP/XGBoost
version incompatibility" section for the full story before loosening
either pin.

---

## Background

This project was built to demonstrate the application of agentic AI to affiliate relationship management. It combines classical machine learning, large language models, and semantic search into a unified system that enables proactive, data-driven decision making across an affiliate partner portfolio.