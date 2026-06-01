# Affiliate Intelligence Platform — Project Context

> **Single source of truth for Claude Code.** Every architectural decision, schema,
> tag definition, and model target is documented here. Update this file whenever the
> design changes.

---

## 1. System Objective

Produce a **360° health score** for every affiliate that predicts both:

| Target | Range | Meaning |
|---|---|---|
| `churn_risk_score` | 0.0 – 1.0 | Probability the affiliate disengages / leaves within 90 days |
| `growth_potential_score` | 0.0 – 1.0 | Probability the affiliate significantly increases revenue within 90 days |

Both scores feed a composite `health_score` (0–100):

```
health_score = round(((1 - churn_risk_score) * 0.6 + growth_potential_score * 0.4) * 100, 1)
```

---

## 2. NLP Tags (20 total)

Each communication (email, call transcript, chat) is run through `nlp_processor.py`
which assigns zero or more tags as a JSON list stored in `communications.tags`.

### Detection Method Key
- **KW** = keyword/phrase matching on lowercased text
- **RE** = regex pattern
- **ML** = spaCy classifier or rule-based entity detection
- **SENT** = sentiment score threshold from VADER/TextBlob

| # | Tag | Detection | Trigger Condition |
|---|---|---|---|
| 1 | `churn_signal` | KW + SENT | Words: *cancel*, *leaving*, *switching*, *done with*, *closing account*, *move on*, *not working for us* + negative sentiment ≤ -0.4 |
| 2 | `growth_intent` | KW | Words: *scale*, *expand*, *ramp up*, *increase volume*, *new campaign*, *bigger budget*, *double*, *grow*, *opportunity* |
| 3 | `payment_issue` | KW + RE | Words: *payment*, *commission*, *invoice*, *not received*, *missing*, *delayed*, *discrepancy*; amount regex `\$[\d,]+` present |
| 4 | `technical_issue` | KW | Words: *bug*, *broken*, *error*, *not working*, *tracking issue*, *pixel*, *postback*, *API down*, *integration fail* |
| 5 | `satisfaction_high` | SENT + KW | VADER compound ≥ 0.5 OR words: *love*, *excellent*, *amazing*, *best*, *great work*, *fantastic*, *thrilled* |
| 6 | `satisfaction_low` | SENT + KW | VADER compound ≤ -0.3 OR words: *disappointed*, *frustrated*, *unhappy*, *poor*, *terrible*, *waste of time* |
| 7 | `competitor_mention` | KW | Named competitors: *Commission Junction*, *ShareASale*, *Impact*, *Awin*, *Rakuten*, *PartnerStack*, *Partnerize*, plus generic: *competitor*, *other network*, *your rival* |
| 8 | `escalation_risk` | KW + SENT | Words: *manager*, *escalate*, *legal*, *lawyer*, *report*, *complain*, *BBB*, *sue*, *formal complaint* + negative sentiment |
| 9 | `support_request` | KW | Words: *help*, *question*, *how do I*, *can you assist*, *need support*, *ticket*, *not sure how*, *please advise* |
| 10 | `feature_request` | KW | Words: *feature*, *wish*, *would be great if*, *can you add*, *missing functionality*, *request*, *suggestion*, *enhancement* |
| 11 | `pricing_concern` | KW + RE | Words: *commission rate*, *payout*, *low rate*, *lower than*, *not worth it*, *increase my rate*, *better terms*; percentage regex `\d+%` |
| 12 | `fraud_risk` | KW + RE | Words: *fake*, *bot traffic*, *proxy*, *VPN*, *refund spike*, *chargeback*, *suspicious*, *invalid clicks*; IP regex present |
| 13 | `high_engagement` | ML | Communication frequency > 2 per week OR response time < 4 hours (computed at feature engineering stage, tag set retroactively) |
| 14 | `low_engagement` | ML | No communication for 14+ days OR single-word replies (computed at feature engineering stage) |
| 15 | `compliance_issue` | KW | Words: *policy violation*, *terms of service*, *TOS*, *prohibited*, *not allowed*, *ban*, *trademark*, *brand bidding* |
| 16 | `new_opportunity` | KW | Words: *new traffic source*, *new channel*, *partnership*, *collab*, *new audience*, *influencer*, *podcast*, *newsletter*, *social media push* |
| 17 | `seasonal_pattern` | KW + RE | Words: *Black Friday*, *Cyber Monday*, *Q4*, *holiday season*, *summer sale*, *back to school*, *Christmas*, *BFCM*; month/quarter regex |
| 18 | `relationship_warm` | SENT + KW | VADER compound ≥ 0.3 AND words: *thanks*, *appreciate*, *great chatting*, *always a pleasure*, *looking forward*, *enjoyed our call*, *cheers* |
| 19 | `urgency` | KW + RE | Words: *urgent*, *ASAP*, *immediately*, *right away*, *by end of day*, *today*, *deadline*; exclamation marks ≥ 2 |
| 20 | `question_asked` | RE | Sentence ending with `?` OR words: *can you*, *could you*, *would you*, *do you know*, *is there a way* |

---

## 3. PostgreSQL Schema

Database: `affiliate_intelligence`

### 3.1 `affiliates`

```sql
CREATE TABLE affiliates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255)   NOT NULL,
    email           VARCHAR(255)   UNIQUE NOT NULL,
    company         VARCHAR(255),
    tier            VARCHAR(20)    NOT NULL DEFAULT 'bronze',   -- bronze | silver | gold | platinum
    join_date       DATE           NOT NULL,
    country         VARCHAR(100),
    niche           VARCHAR(100),                               -- e.g. finance, travel, SaaS, e-commerce
    traffic_source  VARCHAR(100),                               -- SEO | PPC | Social | Email | Influencer
    monthly_revenue DECIMAL(12,2)  DEFAULT 0.00,               -- last full-month revenue generated
    churn_risk_score     FLOAT     DEFAULT 0.50,               -- 0.0–1.0
    growth_potential_score FLOAT   DEFAULT 0.50,               -- 0.0–1.0
    health_score         FLOAT     DEFAULT 50.0,               -- 0–100
    last_contact_date    TIMESTAMP,
    created_at      TIMESTAMP      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP      NOT NULL DEFAULT NOW()
);
```

**Indexes:** `email`, `tier`, `churn_risk_score DESC`, `growth_potential_score DESC`

### 3.2 `communications`

```sql
CREATE TABLE communications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id    UUID           NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    channel         VARCHAR(20)    NOT NULL,   -- email | call | chat | ticket
    direction       VARCHAR(10)    NOT NULL,   -- inbound | outbound
    subject         VARCHAR(500),
    content         TEXT           NOT NULL,
    sentiment_score FLOAT,                     -- VADER compound: -1.0 to 1.0
    sentiment_label VARCHAR(20),               -- positive | neutral | negative
    tags            JSONB          NOT NULL DEFAULT '[]',
    embedding_id    VARCHAR(255),              -- ChromaDB document ID reference
    occurred_at     TIMESTAMP      NOT NULL,
    created_at      TIMESTAMP      NOT NULL DEFAULT NOW()
);
```

**Indexes:** `affiliate_id`, `occurred_at DESC`, `channel`, GIN index on `tags`

### 3.3 `score_history`

```sql
CREATE TABLE score_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id    UUID           NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    churn_risk_score      FLOAT    NOT NULL,
    growth_potential_score FLOAT   NOT NULL,
    health_score          FLOAT    NOT NULL,
    features        JSONB          NOT NULL DEFAULT '{}',  -- raw feature vector snapshot
    shap_values     JSONB          NOT NULL DEFAULT '{}',  -- {feature: shap_value} for explainability
    model_version   VARCHAR(50)    NOT NULL DEFAULT '1.0.0',
    scored_at       TIMESTAMP      NOT NULL DEFAULT NOW()
);
```

**Indexes:** `affiliate_id`, `scored_at DESC`

---

## 4. ChromaDB Collection Structure

### 4.1 `communications_embeddings`

Stores one document per communication.

| Field | Value |
|---|---|
| **collection** | `communications_embeddings` |
| **model** | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| **document** | Full `content` text |
| **id** | `comm_{uuid}` |
| **metadata** | `affiliate_id`, `channel`, `direction`, `sentiment_label`, `tags` (pipe-joined string), `occurred_at` (ISO) |

Used for: semantic search over past communications, RAG context for agent.

### 4.2 `affiliate_profiles`

Stores one document per affiliate — a concatenated summary of their profile + recent comms.

| Field | Value |
|---|---|
| **collection** | `affiliate_profiles` |
| **model** | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| **document** | `f"{name} | {company} | {niche} | {traffic_source} | tier={tier} | revenue={monthly_revenue}"` |
| **id** | `aff_{uuid}` |
| **metadata** | `affiliate_id`, `tier`, `niche`, `churn_risk_score`, `growth_potential_score`, `health_score` |

Used for: finding similar affiliates, clustering, agent profile lookups.

---

## 5. ML Model Architecture

### Churn Model (`churn_model.py`)
- **Algorithm**: XGBoostClassifier
- **Target**: `churn_risk_score` (binary label: churned within 90 days)
- **Key features**: days_since_last_contact, churn_signal_count, satisfaction_low_count,
  competitor_mention_count, payment_issue_count, escalation_risk_count,
  avg_sentiment_score (30d), comm_frequency_decline_ratio, days_since_join,
  monthly_revenue, tier_encoded

### Growth Model (`growth_model.py`)
- **Algorithm**: XGBoostClassifier
- **Target**: `growth_potential_score` (binary label: revenue grew ≥ 20% in 90 days)
- **Key features**: growth_intent_count, new_opportunity_count, satisfaction_high_count,
  high_engagement_count, seasonal_pattern_count, feature_request_count,
  avg_sentiment_score (30d), comm_frequency_growth_ratio, monthly_revenue,
  tier_encoded, relationship_warm_count

---

## 6. Agent Architecture

**Framework**: LangChain ReAct (Reason + Act)

### Tools available to the agent

| Tool | Function |
|---|---|
| `query_affiliates` | SQL query against PostgreSQL affiliates + score data |
| `search_communications` | Semantic search over ChromaDB communications_embeddings |
| `summarise_affiliate` | Generates a narrative health summary for one affiliate |
| `draft_email` | Drafts a personalised outreach email for a given affiliate + goal |
| `flag_risk` | Creates an urgent risk flag record for an affiliate |
| `run_scoring` | Triggers re-scoring for one or all affiliates |

---

## 7. API Endpoints (FastAPI)

Run with: `uvicorn src.api.main:app --port 8080 --reload`

Endpoints are defined in `src/api/routers/` and wired into `src/api/main.py`
via `app.include_router(...)`. Startup logs the total registered route count.

| Method | Path | Router file | Description |
|---|---|---|---|
| GET | `/health` | main.py | Service health check (PostgreSQL + ChromaDB) |
| GET | `/affiliates` | main.py | List affiliates with filtering + sorting |
| GET | `/affiliates/{id}` | main.py | Single affiliate with score history |
| GET | `/affiliates/{id}/communications` | main.py | Paginated communications |
| POST | `/affiliates/{id}/score` | main.py | Trigger re-scoring for one affiliate |
| POST | `/agent/chat` | main.py | Chat with the LangChain ReAct agent |
| POST | `/ingest/full` | routers/ingest.py | Run full ETL from mock data files |
| POST | `/ingest/affiliates` | routers/ingest.py | Re-ingest affiliates CSV only |
| POST | `/ingest/communications` | routers/ingest.py | Re-ingest emails + transcripts |
| POST | `/ingest/csv` | routers/ingest.py | Upload affiliates CSV file |
| POST | `/process/nlp` | routers/process.py | Tag all untagged communications |
| POST | `/process/embeddings` | routers/process.py | Embed all unembedded communications |
| POST | `/process/full` | routers/process.py | NLP + embeddings end-to-end |
| GET | `/communications` | routers/search.py | List all communications with tags |
| GET | `/communications/{id}` | routers/search.py | Single communication by UUID |
| GET | `/search` | routers/search.py | Semantic search; params: `q`, `affiliate_id`, `n` |
| POST | `/ml/train` | routers/ml.py | Train churn + growth XGBoost models |
| POST | `/ml/score` | routers/ml.py | Score all affiliates and persist results |
| GET | `/ml/scores` | routers/ml.py | List current affiliate scores |
| GET | `/ml/explain/{id}` | routers/ml.py | SHAP feature importances for one affiliate |
| GET | `/ml/dashboard` | routers/ml.py | Aggregate health stats across all affiliates |

**Total: 21 routes**

---

## 8. Environment Variables

See `.env.example` for all required variables. Never commit `.env`.

---

## 9. Running Locally

```bash
docker compose up -d                          # start PostgreSQL + ChromaDB
conda activate affiliate-intelligence         # or: pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/ingestion/etl_pipeline.py          # seed mock data (10 affiliates, 14 comms)
uvicorn src.api.main:app --port 8080 --reload # start API on :8080
curl -X POST http://localhost:8080/process/nlp        # tag all communications
curl -X POST http://localhost:8080/process/embeddings # embed all communications
```

---

## 10. Infrastructure

### Docker services

| Service | Image | Host port | Notes |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `5432` | healthcheck: `pg_isready` |
| `chromadb` | `chromadb/chroma:latest` | `8001` | container internal port 8000; **NO auth**; **NO healthcheck** |
| `app` | custom build | `8080` | depends on postgres (healthy) + chromadb (started) |

**chromadb environment (only these two — do not add auth vars):**
```
IS_PERSISTENT=TRUE
PERSIST_DIRECTORY=/chroma/chroma
```

**app environment:**
```
CHROMA_HOST=chromadb
CHROMA_PORT=8001
DATABASE_URL=postgresql://...@postgres:5432/affiliate_intelligence
OPENAI_API_KEY=...
```

### Known fixes — permanently committed to git

The `docker-compose.yml` on `feature/nlp-tagging` branch has all fixes applied.
Do **not** re-add any of these:

- **chromadb 1.5.9 does not support token auth** — `CHROMA_SERVER_AUTH_*` env vars
  cause container startup failure; they have been removed.
- **chromadb healthcheck removed** — the `/api/v1/heartbeat` path does not exist in
  chromadb ≥ 1.0; `app` uses `service_started` instead of `service_healthy`.
- **Port conflict resolved** — app on `8080`, chromadb on host port `8001` (container `8000`).
- **`version:` attribute removed** — obsolete in Docker Compose v2; omit entirely.

### Daily startup sequence

```bash
docker compose up -d           # start PostgreSQL + ChromaDB
conda activate affiliate-intelligence
docker compose ps              # verify: postgres (healthy), chromadb (up), app (up)
```

---

## 11. Git Conventions

### Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, demo-ready code only |
| `develop` | Integration branch — all features merge here |
| `feature/*` | One branch per module |

### Commit message format

```
<type>(<scope>): <description>
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `data` | Mock data or schema changes |
| `ml` | Model training, evaluation, SHAP |
| `docs` | README, CLAUDE.md, comments |
| `refactor` | Restructure, no behaviour change |
| `test` | Tests |

**Examples:**
```
feat(ingestion): add CSV ETL pipeline for affiliates table
ml(churn): train XGBoost churn risk model with SHAP output
feat(agent): add draft_email tool to LangChain agent
```

### Rules
- Always commit after completing each logical unit of work
- Never commit `.env` files
- Never commit trained model `.pkl` or `.joblib` files
- Never commit `chroma_db/` data folders

---

## 12. Built Modules

### API layer

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/api/main.py`, `src/api/routers/ingest.py`, `src/api/routers/process.py`, `src/api/routers/search.py`, `src/api/routers/ml.py` |

**Structure:** `main.py` imports and wires all four routers via `app.include_router()`.
On startup it logs the total registered route count. Endpoint logic lives in
`src/api/routers/` — **not** in the ingestion/ML module files.

| Router | Prefix | Endpoints |
|---|---|---|
| `ingest.py` | `/ingest` | `POST /full`, `/affiliates`, `/communications`, `/csv` |
| `process.py` | `/process` | `POST /nlp`, `/embeddings`, `/full` |
| `search.py` | *(none)* | `GET /communications`, `/communications/{id}`, `/search` |
| `ml.py` | `/ml` | `POST /train`, `/score` · `GET /scores`, `/explain/{id}`, `/dashboard` |

**Total routes: 21** (verified via `/openapi.json`)

---

### ETL pipeline

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/etl_pipeline.py` |

**What it does:** Reads mock CSV and flat-text files from `data/mock/`, runs NLP tagging
inline (`process_text()`), generates embeddings via `EmbeddingGenerator`, and upserts
everything into PostgreSQL and ChromaDB. `run_full_pipeline()` is the single entry point.

**Key functions:** `run_full_pipeline()`, `ingest_affiliates_csv(path)`,
`ingest_communications_file(path)`, `index_affiliate_profiles()`, `ingest_csv_content(csv_str)`

**Output:** 10 affiliates + 14 communications in PostgreSQL; affiliate profiles in ChromaDB

**Idempotent:** upserts by email (affiliates); creates new comm rows on each run

**Fixes applied:**
- Non-UUID IDs in mock CSV (`aff-001`) now handled gracefully — falls back to `uuid4()`
- Offset-aware vs offset-naive datetime comparison fixed in `last_contact_date` update

---

### NLP processor

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/nlp_processor.py` |

**What it does:** VADER-based sentiment scoring plus keyword/regex 20-tag detection.
Called inline by the ETL pipeline during `ingest_communications_file()`. Also callable
directly via `POST /process/nlp` for re-tagging.

**Key function:** `process_text(text: str) -> NLPResult` — returns `sentiment_score`,
`sentiment_label`, and `tags` list.

---

### Embedding generator

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/embedding_generator.py` |

**What it does:** Wraps `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Called
inline by the ETL pipeline to embed each communication and affiliate profile into
ChromaDB. Also callable via `POST /process/embeddings`.

**Key class / functions:** `EmbeddingGenerator` (lazy singleton via `get_generator()`),
`index_communication()`, `index_affiliate_profile()`, `search_communications()`

---

### ML models

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/ml/churn_model.py`, `src/ml/growth_model.py`, `src/ml/feature_engineering.py`, `src/ml/explainability.py` |

**What it does:**
- `feature_engineering.py` — builds feature matrix from PostgreSQL (tag counts, sentiment,
  revenue, days_since_contact, tier encoding); generates synthetic training labels
- `churn_model.py` — XGBoost classifier → `churn_risk_score` (0–1); auto-trains if
  no model artefact found
- `growth_model.py` — XGBoost classifier → `growth_potential_score` (0–1)
- `explainability.py` — SHAP values per affiliate; `explain_affiliate()`,
  `top_risk_drivers()`, `explain_all()`

**Artefacts:** `models/churn_model.json`, `models/growth_model.json` (never committed)

**API:** `POST /ml/train`, `POST /ml/score`, `GET /ml/scores`,
`GET /ml/explain/{id}`, `GET /ml/dashboard`

---

### Infrastructure fixes applied

| File | Fix |
|---|---|
| `src/storage/vector_store.py` | Removed `chromadb.auth.token.TokenAuthClientProvider` — module does not exist in chromadb ≥ 1.0; client now connects without auth settings |
| `src/ingestion/etl_pipeline.py` | Handle non-UUID IDs in mock CSV gracefully; fix offset-aware vs offset-naive datetime comparison |
| `docker-compose.yml` | Removed `version:` attribute, all `CHROMA_SERVER_AUTH_*` env vars, and chromadb healthcheck; `app` chromadb dependency changed to `service_started`; app port fixed to `8080`, `CHROMA_PORT` fixed to `8001` |

---

### Current data state

| | |
|---|---|
| Affiliates in PostgreSQL | 10 |
| Communications in PostgreSQL | 14 (expanded mock data) |
| Communications tagged | 14 / 14 |
| Communications embedded | 14 / 14 |
| ChromaDB collection | `communications_embeddings` + `affiliate_profiles` |
| Semantic search | Working via `GET /search` |
