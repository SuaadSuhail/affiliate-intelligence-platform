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

## 2. NLP Tags (21 total)

Each communication is processed by `src/ingestion/nlp_processor.py`, which runs spaCy
(`en_core_web_sm`) and a custom `SENTIMENT_LEXICON` to assign tags stored as a
`TEXT[]` array in `communications.tags`.

### Sentiment scoring
Sentiment is **lexicon-based** (not VADER). `calculate_sentiment(text)` averages the
scores of all SENTIMENT_LEXICON words found in the text. 40 words total: 20 negative
(-0.3 to -0.9) and 20 positive (+0.3 to +0.9). Result clamped to [-1.0, +1.0].

### Detection method key
- **KW** = keyword/phrase matching on lowercased text
- **ML** = spaCy NER entity matching
- **SENT** = lexicon sentiment score threshold
- **DB** = database lookup on the `affiliates` table

### ENGAGEMENT group

| Tag | Detection | Trigger Condition |
|---|---|---|
| `responsive` | KW + SENT | `source = email` AND `sentiment_score > 0.1` |
| `proactive_outreach` | KW | *just wanted to reach out*, *checking in*, *wanted to share*, *thought you'd like to know* |
| `campaign_active` | KW | *live*, *launched*, *running*, *went live*, *campaign is active*, *pushing the campaign* |
| `unresponsive` | DB | `affiliates.days_since_contact > 5` |
| `disengaged_tone` | KW + SENT | *slow*, *quiet*, *not much happening*, *haven't been able*, *things are slow*, *been quiet* AND `sentiment < -0.2` |
| `gone_silent` | DB | `affiliates.days_since_contact > 14` |

### SENTIMENT group

| Tag | Detection | Trigger Condition |
|---|---|---|
| `positive_sentiment` | SENT | `sentiment_score > 0.3` |
| `enthusiastic` | SENT + KW | `sentiment_score > 0.6` OR *excited*, *thrilled*, *can't wait*, *love this*, *amazing* |
| `neutral_sentiment` | SENT | `-0.2 ≤ sentiment_score ≤ 0.3` |
| `frustrated` | SENT + KW | `sentiment_score < -0.4` OR *frustrated*, *disappointing*, *not working*, *let down*, *expected better* |
| `complaint` | KW | *complaint*, *unacceptable*, *not acceptable*, *raising a complaint*, *formally complain* |

### INTENT group

| Tag | Detection | Trigger Condition |
|---|---|---|
| `upsell_signal` | KW | *new product*, *can we add*, *interested in*, *another brand*, *additional* |
| `expansion_interest` | KW | *scale*, *grow*, *more volume*, *increase*, *bigger*, *expand* |
| `new_campaign_intent` | KW | *new campaign*, *launch*, *plan to run*, *ready to start*, *want to start* |
| `churn_signal` | KW | *leaving*, *switching*, *cancelling*, *stopping*, *moving to*, *other platform*, *looking elsewhere* |
| `competitor_mention` | ML + KW | spaCy NER ORG entities + keyword list: *awin*, *rakuten*, *impact*, *cj affiliate*, *partnerize*, *tradedoubler*, *webgains* |
| `stalled_deal` | KW + SENT | *still waiting*, *no update*, *heard nothing*, *no response*, *chasing* AND `sentiment < 0.0` |

### RELATIONSHIP group

| Tag | Detection | Trigger Condition |
|---|---|---|
| `escalation` | KW | *escalate*, *speak to manager*, *your manager*, *senior*, *urgent*, *asap*, *immediately* OR (`frustrated` AND `complaint` both present) |
| `follow_up_needed` | KW | *let me know*, *waiting to hear*, *please confirm*, *can you*, *could you*, *please check*, *get back to me* |
| `action_committed` | KW | *i will*, *we will*, *will send*, *will do*, *by end of*, *done by*, *will have it* |
| `question_asked` | ML | Any sentence in `doc.sents` ends with `?` |

---

## 3. PostgreSQL Schema

Database: `affiliate_intelligence`

> **Note:** The schema below reflects the **actual live database** (confirmed via
> `sqlalchemy.inspect`). It differs from the original scaffold design — several
> columns were renamed or restructured during the storage-layer build.

### 3.1 `affiliates`

```sql
CREATE TABLE affiliates (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                   VARCHAR(255)  NOT NULL,
    status                 affiliate_status NOT NULL DEFAULT 'active',
                           -- ENUM: active | at_risk | churned | high_growth
    churn_risk_score       FLOAT         NOT NULL DEFAULT 0.50,   -- 0.0–1.0
    growth_potential_score FLOAT         NOT NULL DEFAULT 0.50,   -- 0.0–1.0
    health_score           FLOAT         NOT NULL DEFAULT 50.0,   -- 0–100
    revenue_30d            NUMERIC(10,2) NOT NULL DEFAULT 0.00,
    ctr_trend_pct          FLOAT         NOT NULL DEFAULT 0.0,
    last_contact_at        TIMESTAMPTZ,
    days_since_contact     INTEGER       NOT NULL DEFAULT 0,
    updated_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
```

**Indexes:** `status`, `churn_risk_score DESC`, `growth_potential_score DESC`

### 3.2 `communications`

```sql
CREATE TABLE communications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id    UUID             NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    source          communication_source NOT NULL,
                    -- ENUM: email | call | api_event
    raw_text        TEXT             NOT NULL,
    tags            VARCHAR[]        NOT NULL DEFAULT '{}',  -- TEXT ARRAY (not JSONB)
    sentiment_score FLOAT            NOT NULL DEFAULT 0.0,  -- lexicon score: -1.0 to 1.0
    embedding_id    VARCHAR(255),                           -- ChromaDB document ID
    occurred_at     TIMESTAMPTZ      NOT NULL
);
```

**Indexes:** `affiliate_id`, `occurred_at DESC`, `source`

### 3.3 `score_history`

```sql
CREATE TABLE score_history (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id           UUID   NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    churn_risk_score       FLOAT  NOT NULL,
    growth_potential_score FLOAT  NOT NULL,
    health_score           FLOAT  NOT NULL,
    scored_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Indexes:** `affiliate_id`, `scored_at DESC`

---

## 4. pgvector Embedding Store

**ChromaDB has been replaced by pgvector.** All embeddings are stored in the same PostgreSQL
instance used for structured data — no separate container required.

### 4.1 `embeddings` table

One row per communication chunk (200-word overlapping segments).

```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,                    -- "{comm_uuid}_chunk_{i}"
    affiliate_id UUID REFERENCES affiliates(id) ON DELETE CASCADE,
    affiliate_name TEXT,
    source TEXT,                            -- email | call | api_event
    chunk_text TEXT,
    tags TEXT[],
    occurred_at TIMESTAMPTZ,
    embedding vector(384)                   -- all-MiniLM-L6-v2
);
CREATE INDEX embeddings_vector_idx ON embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
CREATE INDEX embeddings_affiliate_id_idx ON embeddings (affiliate_id);
```

### 4.2 `PGVectorStore` class

`src/storage/pgvector_store.py` — replaces `vector_store.py`:

| Method | Description |
|---|---|
| `add_document(doc_id, text, affiliate_id, ...)` | Upsert one chunk |
| `search_similar(query_embedding, n_results, ...)` | Cosine distance search; returns `[{id, text, affiliate_name, source, tags, occurred_at, distance}]` |
| `get_by_affiliate(affiliate_id, limit)` | All chunks for one affiliate |
| `delete_by_affiliate(affiliate_id)` | Remove all chunks for one affiliate |

**Usage:** Every caller creates `PGVectorStore(db)` with the current SQLAlchemy session — no singleton.

---

## 5. ML Model Architecture

### Design principle
With only 10 affiliates, XGBoost cannot be relied on as a primary predictor.
The system uses **rule-based scoring as primary** and XGBoost as secondary
(activated after `POST /ml/train`).

### Feature vector (12 features — see § 12 for full details)
Activity: `days_since_contact`, `revenue_30d`, `ctr_trend_pct`
Communication (30d): `avg_sentiment_30d`, `comm_count_30d`, tag counts (5 features)
Derived: `sentiment_trend`, `response_rate`, `days_since_positive`

### Churn Model (`churn_model.py`)
- **Primary**: `calculate_churn_risk_rules(features)` — weighted rule-based scorer
- **Secondary**: `train_churn_model(df)` → XGBClassifier (n_estimators=50, max_depth=3)
- **Target**: `status in ['at_risk', 'churned']` (derived from `churn_risk_score > 0.7`)
- **Artefact**: `models/churn_model.pkl`

### Growth Model (`growth_model.py`)
- **Primary**: `calculate_growth_potential_rules(features)` — weighted rule-based scorer
- **Secondary**: `train_growth_model(df)` → XGBClassifier (same params)
- **Target**: `status == 'high_growth'` (derived from `growth_potential_score > 0.7`)
- **Artefact**: `models/growth_model.pkl`

### Explainability (`explainability.py`)
- SHAP TreeExplainer on the XGBoost model
- `get_shap_explanation()` returns top 5 factors with `{feature, shap_value, feature_value, direction}`
- Falls back to rule-based prediction summary if model not trained

### Score updater (`score_updater.py`)
- `update_all_scores(db)` — runs the full pipeline for every affiliate
- Health score: `round(((1 - churn_risk) × 0.6 + growth × 0.4) × 100, 1)`
- Idempotent within a day (skips affiliates already scored today)

---

## 6. Agent Architecture

**Framework**: LangChain ReAct (Reason + Act)

### Tools available to the agent

| Tool | Function |
|---|---|
| `query_database` | Raw SQL SELECT against PostgreSQL; validates SELECT-only, max 20 rows |
| `semantic_search` | pgvector cosine search over communications via `PGVectorStore.search_similar()` |
| `get_affiliate_summary` | Full affiliate profile: scores, recent comms, risk signals, recommended action |
| `draft_email` | LLM-generated re-engagement email; template fallback when API key missing |
| `get_portfolio_health` | Whole-portfolio aggregate stats: health, churn, growth counts |

---

## 7. API Endpoints (FastAPI)

Run with: `uvicorn src.api.main:app --port 8080 --reload`

Endpoints are defined in `src/api/routers/` and wired into `src/api/main.py`
via `app.include_router(...)`. Startup logs the total registered route count.

| Method | Path | Router file | Description |
|---|---|---|---|
| GET | `/health` | main.py | Service health check (PostgreSQL + ChromaDB) |
| GET | `/task/{task_id}` | main.py | Poll status of a background task |
| GET | `/affiliates` | main.py | List affiliates with filtering + sorting |
| GET | `/affiliates/{id}` | main.py | Single affiliate with score history |
| GET | `/affiliates/{id}/communications` | main.py | Paginated communications |
| POST | `/affiliates/{id}/score` | main.py | Trigger re-scoring for one affiliate |
| POST | `/agent/chat` | main.py | Chat with the LangChain ReAct agent |
| POST | `/ingest/full` | routers/ingest.py | Start ETL in background; returns task_id |
| POST | `/ingest/affiliates` | routers/ingest.py | Re-ingest affiliates CSV only |
| POST | `/ingest/communications` | routers/ingest.py | Re-ingest emails + transcripts |
| POST | `/ingest/csv` | routers/ingest.py | Upload affiliates CSV file |
| POST | `/process/nlp` | routers/process.py | Tag all untagged communications |
| POST | `/process/embeddings` | routers/process.py | Embed all unembedded communications |
| POST | `/process/full` | routers/process.py | Start NLP + embeddings in background; returns task_id |
| GET | `/communications` | routers/search.py | List all communications with tags |
| GET | `/communications/{id}` | routers/search.py | Single communication by UUID |
| GET | `/search` | routers/search.py | Semantic search; params: `q`, `affiliate_id`, `n` |
| POST | `/ml/train` | routers/ml.py | Start model training in background; returns task_id |
| POST | `/ml/score` | routers/ml.py | Start affiliate scoring in background; returns task_id |
| GET | `/ml/scores` | routers/ml.py | List current affiliate scores |
| GET | `/ml/explain/{id}` | routers/ml.py | SHAP feature importances for one affiliate |
| GET | `/ml/dashboard` | routers/ml.py | Aggregate health stats across all affiliates |

**Total: 22 routes** (added `GET /task/{task_id}`)

### Background task pattern

`POST /ml/train`, `POST /ml/score`, `POST /process/full`, and `POST /ingest/full`
are now **non-blocking** and return immediately:

```json
{"status": "accepted", "task_id": "uuid", "message": "...Poll GET /task/{task_id}..."}
```

Task lifecycle: `pending → running → complete | failed`

State is held in `src/api/task_store.py` (in-memory dict; resets on process restart).
Each background function creates its own `SessionLocal()` — it never shares the
HTTP request's DB session (which is closed before the task runs).

Poll response schema:
```json
{"task_id": "...", "status": "complete", "result": {...}, "error": null}
```

---

## 8. Environment Variables

See `.env.example` for all required variables. Never commit `.env`.

---

## 9. Running Locally

```bash
docker compose up -d                          # start PostgreSQL + ChromaDB
conda activate affiliate-intelligence         # or: pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn src.api.main:app --port 8080 --reload # start API on :8080

# Full pipeline — run in order:
curl -X POST http://localhost:8080/ingest/full       # seed 10 affiliates + 7 comms
curl -X POST http://localhost:8080/process/nlp       # tag all 7 communications
curl -X POST http://localhost:8080/process/embeddings # embed all communications
curl -X POST http://localhost:8080/ml/train          # train XGBoost models
curl -X POST http://localhost:8080/ml/score          # score all affiliates
curl http://localhost:8080/ml/dashboard              # verify pipeline state
```

---

## 10. Infrastructure

### Docker services

| Service | Image | Host port | Notes |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `5432` | pgvector extension pre-installed; healthcheck: `pg_isready` |
| `app` | custom build | `8080` | depends on postgres (healthy) only |

> **ChromaDB container removed.** All vector storage runs inside PostgreSQL via the
> `pgvector` extension. The `pgvector/pgvector:pg16` image is a drop-in replacement for
> `postgres:16-alpine` with pgvector pre-installed — data format is identical.

**app environment (no CHROMA_* vars):**
```
DATABASE_URL=postgresql://...@postgres:5432/affiliate_intelligence
OPENAI_API_KEY=...
APP_ENV=production
API_SECRET_KEY=...
```

### Known fixes — permanently committed to git

The `docker-compose.yml` has all fixes applied. Do **not** re-add any of these:

- **`version:` attribute removed** — obsolete in Docker Compose v2; omit entirely.
- **ChromaDB removed** — replaced by pgvector; do not re-add chromadb service or CHROMA_* env vars.
- **ChromaDB `health_check()` uses `list_collections()`** — `client.heartbeat()` calls
  `/api/v1/heartbeat` which does not exist in ChromaDB ≥ 1.0, causing the `/health`
  endpoint to always report `"chromadb": "down"` even when the container is healthy.
  Fixed in `src/storage/vector_store.py` by replacing `heartbeat()` with
  `list_collections()`, which verifies real connectivity. Do not revert to `heartbeat()`.

### Daily startup sequence

```bash
# 1. Start Docker Desktop (if not already running)
# 2. Start containers
docker compose up -d

# 3. Activate Python environment
conda activate affiliate-intelligence

# 4. Verify all containers are up
docker compose ps
```

Expected output from `docker compose ps`:
```
NAME            STATUS          PORTS
aip_postgres    Up (healthy)    0.0.0.0:5432->5432/tcp
aip_app         Up              0.0.0.0:8080->8080/tcp
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

### Storage layer

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/storage/database.py`, `src/storage/models.py`, `src/storage/pgvector_store.py` |

**What it does:**
- `database.py` — creates the SQLAlchemy engine, `SessionLocal` factory, and FastAPI
  `get_db()` dependency; exposes `db_session()` context manager for scripts
- `models.py` — ORM models for `Affiliate`, `Communication`, `ScoreHistory`, and `Embedding`
  (pgvector) that match the live DB schema (see §§ 3–4); uses `ARRAY(String)` for `tags`,
  `Enum` types for `status` / `source`, `Vector(384)` for embeddings
- `pgvector_store.py` — pgvector wrapper; `PGVectorStore(db)` is instantiated per-request
  with the active SQLAlchemy session; replaces the old `vector_store.py` ChromaDB singleton

**Key functions:** `get_db()`, `db_session()`, `init_db()`, `health_check()`,
`PGVectorStore.add_document()`, `PGVectorStore.search_similar()`,
`PGVectorStore.get_by_affiliate()`, `PGVectorStore.delete_by_affiliate()`

---

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

**What it does:** Loads raw data into PostgreSQL only. NLP tagging and embedding
generation are separate steps (`POST /process/nlp`, `POST /process/embeddings`).
- `ingest_affiliates_csv(path)` — upserts Affiliate rows by **name** (no email column in new schema); maps `monthly_revenue → revenue_30d`; derives `status` from churn/growth scores; computes `days_since_contact`
- `ingest_communications_file(path)` — inserts Communication rows with `raw_text` and `source` (mapped from block `channel`); leaves `tags=[]` and `sentiment_score=0.0`
- `run_full_pipeline()` — calls both in order (3 steps: init DB, affiliates, comms)

**Key functions:** `run_full_pipeline()`, `ingest_affiliates_csv(path)`,
`ingest_communications_file(path)`, `ingest_csv_content(csv_str)`

**Output:** 10 affiliates + 7 communications in PostgreSQL (raw text only, no tags)

**Idempotent:** upserts by name (affiliates); creates new comm rows on each run

**Fixes applied:**
- Removed `process_text` import — ETL and NLP are separate steps
- Removed `get_generator` / `EmbeddingGenerator` imports — embeddings are a separate step
- New schema fields: `raw_text` (not `content`), `source` (not `channel`), `last_contact_at` (not `last_contact_date`), no `direction`/`subject`/`sentiment_label`
- Non-UUID mock IDs (`aff-001`) handled gracefully — falls back to `uuid4()`
- `_find_affiliate_by_mock_id` now looks up by name (email column removed from schema)

---

### NLP processor

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/nlp_processor.py` |
| **Tests** | `tests/test_nlp.py` — 6 tests, all passing |

**What it does:** Reads all communications where `tags = []`, runs each through the
spaCy pipeline and `SENTIMENT_LEXICON`, detects applicable tags, and writes
`tags[]` and `sentiment_score` back to the `communications` table.

**SENTIMENT_LEXICON:** 40 words — 20 negative (-0.3 to -0.9) covering churn /
frustration signals; 20 positive (+0.3 to +0.9) covering growth / enthusiasm signals.
Scoring = average of matched word scores, clamped to [-1.0, +1.0].

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `calculate_sentiment` | `(text: str) -> float` | Lexicon-based sentiment score |
| `detect_tags` | `(doc, sentiment_score, text_lower, source, affiliate_id, db) -> list[str]` | All 21 tag rules; no duplicates |
| `process_single_communication` | `(comm: Communication, db: Session) -> dict` | Full pipeline for one record |
| `process_all_communications` | `(db: Session) -> dict` | Bulk tags all untagged comms; returns summary |

**API endpoints:** `POST /process/nlp`, `GET /communications`, `GET /communications/{id}`

**Depends on:** Storage layer (models + DB session), ETL pipeline must run first to
populate communications

**Output:** `tags[]` and `sentiment_score` written to every `communications` row;
7/7 communications tagged in the mock dataset

---

### Embedding generator

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/embedding_generator.py` |
| **Tests** | `tests/test_embeddings.py` — 6 tests, all passing |

**What it does:** Chunks each communication's `raw_text` into 200-word overlapping
segments, encodes each chunk with `all-MiniLM-L6-v2`, and stores the vectors +
metadata in ChromaDB's `communications_embeddings` collection. Writes the first
chunk's doc_id back to `communications.embedding_id` in PostgreSQL.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — loaded once at module level,
produces 384-dimension vectors.

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `chunk_text` | `(text, chunk_size=200, overlap=50) -> list[str]` | Overlapping word-level chunking |
| `embed_communication` | `(comm, db, vs) -> dict` | Embed one record; returns `{comm_id, chunks_created, embedding_id}` |
| `embed_all_communications` | `(db, vs) -> dict` | Embed all where `embedding_id IS NULL`; returns `{total_processed, total_chunks_created, already_embedded}` |

**ChromaDB metadata per chunk:**
- `affiliate_id`, `affiliate_name`, `source`, `occurred_at`, `tags` (pipe-joined display string)
- `tag_{name} = True` for each tag — individual boolean fields used for `$eq` filtering
  (chromadb 1.x does not support `$contains` on metadata string fields)

**API endpoints:** `POST /process/embeddings`, `POST /process/full`, `GET /search`

**Depends on:** NLP processor must run first so `tags[]` are available for metadata

**Output:** 7/7 communications embedded; 13 total chunks stored in ChromaDB
(`communications_embeddings` collection)

---

### ML models

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/ml/feature_engineering.py`, `src/ml/churn_model.py`, `src/ml/growth_model.py`, `src/ml/explainability.py`, `src/ml/score_updater.py` |
| **Tests** | `tests/test_ml.py` — 5 tests, all passing |

**Features (12 total across 3 groups):**

| Group | Features |
|---|---|
| Activity | `days_since_contact`, `revenue_30d`, `ctr_trend_pct` |
| Communication (30d) | `avg_sentiment_30d`, `comm_count_30d`, `churn_signal_count`, `positive_signal_count`, `escalation_count`, `competitor_mention_count` |
| Derived | `sentiment_trend`, `response_rate`, `days_since_positive` |

**What each file does:**
- `feature_engineering.py` — `build_feature_vector(affiliate_id, db)` computes all 12 features
  for one affiliate; `build_all_features(db)` iterates all affiliates; `get_feature_dataframe(db)`
  returns a pandas DataFrame indexed by `affiliate_id`
- `churn_model.py` — `calculate_churn_risk_rules(features)` (rule-based, always available) +
  `train_churn_model(df)` (XGBoost, saved to `models/churn_model.pkl`) +
  `predict_churn_risk(affiliate_id, features)` (uses XGBoost if model exists, rules as fallback)
- `growth_model.py` — identical structure for `growth_potential_score`;
  `calculate_growth_potential_rules()`, `train_growth_model()`, `predict_growth_potential()`
- `explainability.py` — `get_shap_explanation(affiliate_id, features, model_type)` returns
  `{affiliate_id, model_type, base_value, prediction, top_factors[5]}` with per-factor
  `{feature, shap_value, feature_value, direction}`;
  also exposes legacy `explain_affiliate()` and `top_risk_drivers()` for router compatibility
- `score_updater.py` — `update_all_scores(db)` scores every affiliate, writes results to
  `affiliates` table and inserts into `score_history`; skips affiliates already scored today

**Important design decision:** With only 10 affiliates, XGBoost produces unreliable predictions.
Rule-based scorers are the primary method; XGBoost is secondary (activated after `POST /ml/train`).

**Artefacts:** `models/churn_model.pkl`, `models/growth_model.pkl` — tracked in `.gitignore`, never committed

**API:** `POST /ml/train`, `POST /ml/score`, `GET /ml/scores` (worst-first),
`GET /ml/explain/{id}`, `GET /ml/dashboard`

---

### Infrastructure fixes applied

| File | Fix |
|---|---|
| `src/storage/vector_store.py` | Removed `chromadb.auth.token.TokenAuthClientProvider` — module does not exist in chromadb ≥ 1.0; client connects without auth |
| `docker-compose.yml` | Removed `version:` attribute, all `CHROMA_SERVER_AUTH_*` env vars, and chromadb healthcheck; `app` dependency changed to `service_started`; app port `8080`, `CHROMA_PORT=8001` for host access |
| `docker-compose.yml` (networking) | `CHROMA_PORT` inside Docker containers must be `8000` (internal), not `8001` (host mapping). See § 10 Docker networking note. |

### Pipeline fixes (post-merge)

Applied after merging `feature/nlp-tagging` and `feature/ml-models` into `develop`.
These files had stale imports and old schema field names that caused runtime failures:

| File | Fix |
|---|---|
| `src/api/routers/process.py` | Replaced `process_text` import with `process_all_communications`; replaced `get_generator()` loop with `embed_all_communications(db, vs)` — old functions no longer exist after the nlp-tagging merge |
| `src/ingestion/etl_pipeline.py` | Removed stale `process_text` and `get_generator` imports; rewrote for new schema: `raw_text` (not `content`), `source` (not `channel`), removed `direction`/`subject`/`sentiment_label`; upserts by `name` (no `email` column); `last_contact_at` not `last_contact_date` |
| `src/ml/feature_engineering.py` | Fixed `aff.last_contact_date` → `aff.last_contact_at`; replaced `c.direction` reference (field removed from schema) with a `response_rate` proxy calculation |
| `src/ml/score_updater.py` | Removed `features`, `shap_values`, `model_version` from `ScoreHistory` constructor — those columns were removed in the new `score_history` schema |

---

### LangChain agent

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/agent/tools.py`, `src/agent/agent.py`, `src/api/routers/agent.py` |
| **Tests** | `tests/test_agent.py` — 7 tests, all passing |

**LLM:** `gpt-4o-mini`, temperature=0, via `langgraph.prebuilt.create_react_agent`
(langchain 1.3.x does not ship `create_openai_functions_agent` — use langgraph prebuilt instead)

**Tools (5):**

| Tool | Description |
|---|---|
| `query_database` | Raw SQL SELECT against PostgreSQL; validates SELECT-only, max 20 rows |
| `semantic_search` | pgvector cosine search over communications via `PGVectorStore(db).search_similar()` |
| `get_affiliate_summary` | Full affiliate profile: scores, recent comms, risk signals, recommended action |
| `draft_email` | LLM-generated re-engagement email; template fallback when API key missing |
| `get_portfolio_health` | Whole-portfolio aggregate stats: health, churn, growth counts |

**API endpoints:** `POST /agent/chat` (with history), `POST /agent/quick` (single-turn), `GET /agent/demo`, `GET /agent/health`

**Important implementation notes:**
- All tools create a fresh `SessionLocal()` per call (not shared) and close it in `finally`
- `get_affiliate_summary` derives risk signals from communication tags — NOT from SHAP (loading
  XGBoost via joblib inside a LangGraph tool context causes a segfault in uvicorn)
- Agent singleton tracks `_agent_key` (the OPENAI_API_KEY active at build time); if the key
  changes between requests, the singleton resets and rebuilds — errors never cache permanently
- `_invoke_agent()` wraps the LangGraph `agent.invoke()` call with tenacity retry:
  `RateLimitError` and `APITimeoutError` trigger exponential backoff (1s→10s), up to 3 attempts;
  on final failure returns `_UNAVAILABLE_MSG` instead of raising
- `draft_email` LLM is instantiated with `timeout=30` to prevent indefinite hangs
- `GET /agent/health` returns `{agent_ready, openai_key_configured, model, last_error}` with no
  API call — useful for readiness checks without spending tokens
- Demo endpoint (`GET /agent/demo`) runs 3 questions sequentially; requires models trained first
- **`SYSTEM_PROMPT` scope restriction**: the prompt ends with `IMPORTANT SCOPE RULES` that instruct the agent to refuse any question not related to affiliate management (general knowledge, news, politics, geography, etc.) and respond with a fixed out-of-scope message. The agent must only use tool data — never its own general knowledge.

**Depends on:** All pipeline steps must have run: `/ingest/full` → `/process/nlp` →
`/process/embeddings` → `/ml/train` → `/ml/score`

---

### Frontend

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/api/templates/index.html`, `src/api/static/` |
| **Served at** | `http://localhost:8080` |

**Type:** Single-page HTML/CSS/JS, no external frameworks, served by FastAPI with Jinja2.

**Layout:** Two-panel on desktop, stacked on mobile:
- **Left panel (320px):** App header, 4 stats cards (total affiliates, avg health, at-risk, high-growth) pulled from `GET /ml/dashboard`; affiliate list with health bars (red/amber/green) and status badges, sorted worst-first from `GET /affiliates`
- **Right panel:** Chat interface with typing indicator, conversation history, per-response tools-used chips, and 4 suggested question chips

**Behaviour:**
- Page load fetches dashboard stats + affiliate list concurrently
- Click an affiliate row → pre-fills chat input with a question about that affiliate
- Click a suggestion chip → pre-fills input
- Conversation history sent with every message (last 10 turns, `[{role, content}]`)
- `tools_used` displayed as small chips under each agent response
- Status dot changes gold while agent is thinking

**Technical notes:**
- Starlette 1.1.0 requires `templates.TemplateResponse(request=request, name="index.html")` (not the old positional-dict form)
- `Jinja2` and `aiofiles` added to requirements.txt
- Main.py `AffiliateOut` updated to new schema (`status`, `revenue_30d`, `days_since_contact` — no `email`/`tier`/`company`)
- `ScoreHistoryOut` updated (no `shap_values`/`model_version` in new schema)
- `list_affiliates` filter updated (no `tier`/`niche` filter, use `status` instead)
- `POST /agent/chat` fetch includes `X-Api-Key: change-me-in-production` header (required after auth hardening); `/ml/dashboard` and `/affiliates` are GET routes and remain headerless

---

### Current verified pipeline state

Full end-to-end pipeline tested and working on `feature/data-persistence` branch:

| Step | Endpoint | Result |
|---|---|---|
| Migrate | `POST /admin/migrate` | `revision: ccc1c19d5237` — pgvector extension + embeddings table |
| Ingest | `POST /ingest/full` | 10 affiliates + 21 communications loaded |
| NLP | `POST /process/nlp` | 21/21 communications tagged |
| Embeddings | `POST /process/embeddings` | 21 embedded, 39 chunks in pgvector |
| Search | `GET /search?q=...` | Cosine similarity results from PostgreSQL |
| Agent | `POST /agent/quick` | Correctly identifies at-risk affiliates |
| Tests | `pytest tests/ -v` | 24/24 passing |

---

### Promo code leakage detector

| | |
|---|---|
| **Status** | Complete — `feature/promo-leakage-detector` branch |
| **Files** | `src/scraping/site_config.py`, `src/scraping/fetcher.py`, `src/scraping/extractor.py`, `src/scraping/matcher.py`, `src/scraping/leakage_scraper.py`, `src/scraping/__init__.py`, `src/scraping/fixtures/voucherslug_mock.html`, `src/scraping/fixtures/dealsden_mock.html`, `src/scraping/fixtures/csr_shell_mock.html`, `src/scraping/fixtures/csr_shell_mock.js`, `src/scheduling/jobs.py`, `src/scheduling/__init__.py`, `src/storage/models.py` (additions), `src/agent/tools.py` (addition), `src/api/routers/leakage.py` |
| **Migration** | `alembic/versions/9d115196b272_add_leaked_codes_table_and_affiliate_.py` |
| **Tests** | `tests/test_leakage_scraper.py` — 8 tests, all passing |

**What it does:** Detects when an affiliate's promo code appears on monitored
voucher or deal aggregator sites without authorisation. Two callers share one
code path — `check_leakage(db, scan_type="scheduled")` from the nightly
APScheduler job, and `check_leakage(db, scan_type="on_demand")` triggered via
the API or the agent tool. A single entry-point was chosen deliberately: the
fetch → extract → match → persist pipeline is identical in both contexts and
duplicating it would create two things to keep in sync.

**Schema additions (`src/storage/models.py`):**
- `affiliates.active_promo_code` — `VARCHAR(64) NULLABLE`; the code this
  affiliate is currently authorised to share. `NULL` means no code registered;
  those affiliates are skipped silently during every scan.
- `leaked_codes` table — one row per detection event:

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | `gen_random_uuid()` |
| `affiliate_id` | UUID FK → `affiliates.id` | CASCADE delete |
| `code` | VARCHAR(64) | The promo code found |
| `site` | VARCHAR(128) | Site name from `SiteConfig.name` |
| `source_url` | TEXT | Full page URL or fixture `file://` path |
| `raw_snippet` | TEXT NULLABLE | Surrounding HTML kept for audit |
| `scan_type` | VARCHAR(16) | `"scheduled"` or `"on_demand"` |
| `found_at` | TIMESTAMPTZ | UTC timestamp of detection |

Indexes on `affiliate_id` and `code`.

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `check_leakage` | `(db, scan_type, affiliate_id=None) -> dict` | Orchestrator: fetch → extract → match → dedup → persist; commits once at the end |
| `fetch_html` | `(url, kind="live") -> str` | Playwright-based fetcher for SSR and CSR pages |
| `extract_candidate_codes` | `(html, site) -> list[CandidateCode]` | Two-pass selector + regex extraction |
| `match_candidates_to_affiliates` | `(candidates, affiliate_codes, site_name, source_url) -> list[LeakMatch]` | Case-normalised exact matching |
| `run_scheduled_leakage_scan` | `() -> None` | Wraps `check_leakage` for APScheduler; creates its own `SessionLocal()` |
| `start_scheduler` | `() -> BackgroundScheduler` | Idempotent — returns running scheduler if already started |

**Fetcher design (`src/scraping/fetcher.py`):**

Playwright's headless Chromium was chosen over a `requests`-only approach
because voucher aggregator sites are frequently client-side rendered — a plain
HTTP GET of the HTML shell returns an empty `<div id="app">` with no codes.
Playwright handles both SSR and CSR pages with one mechanism.

Three behaviours for `kind="live"` (http/https) only:
- **robots.txt** — checked before every fetch via `urllib.robotparser`; fails
  closed: if `robots.txt` cannot be retrieved for any reason the fetch is
  aborted and the site goes to `sites_failed`. `file://` URLs are exempt (no
  robots.txt concept for local files).
- **Rate limit** — `_MIN_GAP_SECONDS = 3.0` enforced per host via
  `_last_request` dict; subsequent requests to the same host sleep until the
  gap is satisfied.
- **CSR fixtures** — `csr_shell_mock.html` is a `file://` URL pointing at an
  empty `<div id="app">`. Playwright navigates `file://` URLs natively and
  executes the adjacent `csr_shell_mock.js`, which injects the rendered voucher
  card. This confirms the fetcher correctly handles JS-driven pages without any
  special casing.

For `kind="fixture"` (the two static SSR fixtures), the file is read directly
from disk — Playwright is not invoked, keeping the fixture tests fast.

**Extractor design (`src/scraping/extractor.py`):**

Two passes run in sequence; results are deduplicated by uppercased code value
before returning.

1. **Selector pass** — CSS selectors from `SiteConfig.code_selectors` target
   precise elements; sibling/parent traversal via `SiteConfig.merchant_selectors`
   captures the merchant name. High-confidence, site-specific.
2. **Regex fallback** — `_CODE_PATTERN` scans `soup.body.get_text()` for any
   code-shaped token not already found by the selector pass. Catches codes on
   pages that don't match the expected HTML structure.

Two bugs were found and fixed during development — both are recorded here because
they explain why the code looks the way it does:

- **`\b` word-boundary regex failing on hyphenated codes** — `\b` considers
  `-` a non-word character, so it fires at every interior hyphen boundary.
  `TOMB-EXCL20` was silently truncated to `EXCL20` (wrong match) or `TOMB-`
  (incomplete). Fixed by replacing `\b` with negative lookarounds that use the
  token's own character class `[A-Z0-9\-]`:
  `(?<![A-Z0-9\-])([A-Z0-9][A-Z0-9\-]{3,14})(?![A-Z0-9\-])`.

- **`<title>` false positive** — `soup.get_text()` includes the `<head>`
  block. A title such as `<title>CSR Shell — Mock JS-Rendered Page</title>`
  produced the spurious candidate `JS-R` because "JS-Rendered" contains a
  capital-letter hyphen sequence that passes the code-shape heuristic. Fixed by
  scoping the regex fallback to `soup.body.get_text()` only.

**Dedup window (`src/scraping/leakage_scraper.py`):**

`_DEDUP_WINDOW_HOURS = 20`. Before persisting a new `LeakedCode` row, the
orchestrator queries whether the same `(affiliate_id, code, site)` triple
already exists with `found_at` within the past 20 hours. If so, the row is
skipped. This prevents the nightly job (03:00 UTC) from flooding the table with
identical rows across consecutive runs while a leak is ongoing; 20 hours was
chosen to give a comfortable margin below 24 hours so back-to-back days never
accidentally deduplicate across runs.

**Scheduler (`src/scheduling/jobs.py`):**

APScheduler `BackgroundScheduler(timezone="UTC")` with a `CronTrigger(hour=3,
minute=0)`. `start_scheduler()` is called once from `src/api/main.py`
`startup_event()` and is idempotent — if the scheduler is already running it
returns it unchanged. `run_scheduled_leakage_scan()` creates its own
`SessionLocal()` and closes it in `finally` (never shares the HTTP request
session).

**Agent tool (`src/agent/tools.py`):**

`check_promo_leakage` is Tool 6, added to the existing `TOOLS` list. It runs an
on-demand scan immediately for a specific affiliate UUID. The tool docstring
explains to the agent how to obtain the UUID first (`query_database` → SELECT
id WHERE name ILIKE).

Updated tool count — **Tools (6):**

| Tool | Description |
|---|---|
| `query_database` | Raw SQL SELECT against PostgreSQL; validates SELECT-only, max 20 rows |
| `semantic_search` | pgvector cosine search over communications via `PGVectorStore(db).search_similar()` |
| `get_affiliate_summary` | Full affiliate profile: scores, recent comms, risk signals, recommended action |
| `draft_email` | LLM-generated re-engagement email; template fallback when API key missing |
| `get_portfolio_health` | Whole-portfolio aggregate stats: health, churn, growth counts |
| `check_promo_leakage` | Live leakage scan for one affiliate's promo code across all configured sites |

**API endpoints (`src/api/routers/leakage.py`):**

Router registered in `main.py` under prefix `/leakage`. POST route requires
`X-Api-Key` (via `Depends(get_api_key)`); GET routes are unprotected.

| Method | Path | Description |
|---|---|---|
| `POST` | `/leakage/scan` | Start a full portfolio leakage scan as a background task; returns `task_id` |
| `GET` | `/leakage/results` | All `LeakedCode` rows, newest first |
| `GET` | `/leakage/results/{affiliate_id}` | All leak events for one affiliate; returns `[]` (not 404) if none |

**Dockerfile base image change:**

`python:3.11-slim` → `mcr.microsoft.com/playwright/python:v1.61.0-jammy`

This is a meaningful infrastructure change, not a cosmetic one. The slim Python
image does not include a browser binary; adding `pip install playwright` alone
installs the Python bindings but not Chromium. Running `playwright install
chromium` inside the build would work, but the MCR image already has Chromium
baked in at `/ms-playwright/` and sets `PLAYWRIGHT_BROWSERS_PATH=/ms-playwright`
so the Python package finds it automatically. The MCR image ships Python 3.10.12
(no 3.11 tag exists for this Playwright version); the codebase is compatible
with 3.10+.

**New dependencies (added to `requirements.txt`):**
- `playwright>=1.44.0` — headless Chromium for SSR and CSR page rendering
- `beautifulsoup4>=4.12.0` — HTML parsing for selector and regex extraction passes
- `lxml>=5.2.0` — faster BeautifulSoup parser backend
- `apscheduler>=3.10.0` — background scheduler for the nightly leakage job

**Test coverage (`tests/test_leakage_scraper.py`, 8 tests):**

| Test | What it covers |
|---|---|
| `test_extractor_finds_leaked_code_in_fixture` | CSS selector path; asserts TESTLEAK20 + TOMB-EXCL20 with correct merchant context |
| `test_extractor_no_false_positive_on_clean_fixture` | Exactly 3 codes from dealsden; no cross-contamination from other fixtures |
| `test_extractor_handles_csr_rendered_content` | Playwright executes the JS shell and CSRLEAK99 is injected and found |
| `test_extractor_title_text_not_falsely_matched` | Regression for `JS-R` false positive from `<title>`; confirms `soup.body` fix |
| `test_matcher_exact_match_only` | Case normalisation works; trailing space matches; TESTLEAK21 ≠ TESTLEAK20 |
| `test_check_leakage_end_to_end_writes_expected_rows` | Real DB; one LeakedCode row with correct fields; cleans up in `finally` |
| `test_check_leakage_dedup_window_prevents_duplicate` | Real DB; two back-to-back scans → 1 leak row total |
| `test_check_leakage_isolates_site_failures` | Injected broken site goes to `sites_failed`; working sites still produce leaks |

**Deliberately out of scope for this branch:**

- **No live voucher sites are enabled.** `SITES` in `src/scraping/site_config.py`
  contains only the three fixture entries. A commented-out live `SiteConfig`
  example is present in that file with a checklist (robots.txt verification,
  rate limit confirmation, ToS review) that must be completed before any real
  site is added.
- **No numeric integration into `churn_risk_score`.** `check_leakage()` writes
  only to `leaked_codes` — it never touches `affiliates.churn_risk_score`,
  `growth_potential_score`, or `health_score`. The intent is flag-now,
  score-later: once the detection pipeline is proven reliable, a future branch
  will add a scoring rule that increases `churn_risk_score` for affiliates with
  recent unresolved leaks.

---

## 13. Production Hardening

### Week 1 — Complete

- Structured JSON logging via `src/core/logging_config.py`
- API key authentication on all write endpoints via `src/api/auth.py`
- CORS restricted to `ALLOWED_ORIGINS` env var
- SQL injection hardening on `query_database` tool
- Startup validation for required env vars
- OpenAI retry logic with exponential backoff via tenacity (3 attempts, 1–10s backoff)
- 30-second timeout on `draft_email` LLM call
- `GET /agent/health` endpoint
- Background tasks for long-running operations: `POST /ml/train`, `/ml/score`, `POST /process/full`, `/ingest/full`
- Task status polling via `GET /task/{task_id}`
- Frontend pipeline buttons with live polling
- Model fixed to `gpt-4o-mini` via env var

### Week 2 — Complete

- Alembic database migrations ✓
- S3 model storage and versioning ✓
- pgvector replaces ChromaDB — single-database architecture ✓

### Data persistence — Complete

- Alembic migrations replacing `create_all()` — schema versioned and auditable
- Manual migration via `POST /admin/migrate` — safe for multi-instance production
- `GET /admin/migration-status` endpoint — shows current `alembic_version` revision
- S3-compatible model storage with local fallback (`src/ml/model_store.py`)
- pgvector replacing ChromaDB entirely — no separate vector database container
- Single PostgreSQL database for all storage (structured data + vectors)
- `embeddings` table with `vector(384)` column and `ivfflat` cosine index
- 24/24 tests passing

### Planned next

- AWS/Railway deployment
- CloudWatch or structured log aggregation
- RDS PostgreSQL with automated backups
- Rate limiting on public endpoints

---

### Structured JSON logging

| | |
|---|---|
| **Status** | Complete — `feature/production-hardening` branch |
| **File** | `src/core/logging_config.py` |

**What was added:**

A central `src/core/logging_config.py` module replaces all `print()` calls across the codebase
with Python's `logging` module emitting **single-line JSON** to stdout.

**Output format:**
```json
{
  "timestamp": "2026-06-12T17:34:09.549022+00:00",
  "level": "INFO",
  "module": "src.storage.database",
  "message": "Tables created / verified.",
  "extra": {}
}
```

**Key functions:**
- `configure_logging()` — called once at startup in `src/api/main.py` before any other imports; sets log level from `LOG_LEVEL` env var (default `INFO`); suppresses noisy third-party loggers (`uvicorn.access`, `httpx`, `sentence_transformers`); safe to call multiple times (no duplicate handlers)
- `get_logger(name)` — imported and called at module level in every file (`logger = get_logger(__name__)`)

**Log levels applied:**
| Level | When used |
|---|---|
| `logger.debug()` | Per-record detail: individual affiliate scores, SQL query results, per-comm embedding |
| `logger.info()` | Normal operations: tables created, pipeline steps, scoring complete, HTTP requests |
| `logger.warning()` | Recoverable issues: affiliate not found (skipped), XGBoost fallback to rules |
| `logger.error()` | Failures: DB health check failed, model load error, SHAP computation failed, ChromaDB unreachable |

**Files updated (12 total):**
- `src/storage/database.py` — init_db, health_check
- `src/storage/vector_store.py` — health_check
- `src/ingestion/etl_pipeline.py` — all pipeline steps, per-affiliate/comm logging
- `src/ingestion/nlp_processor.py` — spaCy load, process_all summary
- `src/ingestion/embedding_generator.py` — model load, embed_all summary
- `src/ml/churn_model.py` — training, save, fallback warning
- `src/ml/growth_model.py` — training, save, fallback warning
- `src/ml/explainability.py` — model load error, SHAP failure
- `src/ml/score_updater.py` — scoring run start/complete, per-affiliate debug
- `src/agent/tools.py` — SQL query debug, error handling
- `src/agent/agent.py` — agent init, init error
- `src/api/main.py` — startup complete, request middleware

**Request logging middleware** (`_RequestLoggingMiddleware` in `main.py`):
Logs every HTTP request with method, path, status code, and duration in milliseconds:
```json
{"level":"INFO","module":"src.api.main","message":"HTTP request",
 "extra":{"method":"GET","path":"/health","status_code":200,"duration_ms":526.1}}
```

**Environment variable:** `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` (default: `INFO`)

---

### API security hardening

| | |
|---|---|
| **Status** | Complete — `feature/production-hardening` branch |
| **Files** | `src/api/auth.py` (new), `src/api/main.py`, `src/api/routers/*.py`, `src/agent/tools.py` |

**What was added:**

#### 1. CORS — env-driven origin allowlist

`src/api/main.py` reads `ALLOWED_ORIGINS` from the environment (comma-separated) instead of using `allow_origins=["*"]`.

- Default (dev): `http://localhost:8080,http://localhost:3000`
- `allow_methods` narrowed from `["*"]` to `["GET", "POST"]`
- `.env.example` has `ALLOWED_ORIGINS=http://localhost:8080,http://localhost:3000`

#### 2. API key authentication

`src/api/auth.py` — FastAPI `Depends()` dependency:

- Reads `X-API-Key` header; compares to `API_SECRET_KEY` env var
- Returns HTTP 401 if header is missing or wrong
- Returns HTTP 500 if `API_SECRET_KEY` is not set in production
- **Bypassed when `APP_ENV=development`** (default for local dev via uvicorn)

**Protected routes (require X-API-Key in production):**

| Router | Routes protected |
|---|---|
| `ingest.py` | All 4 POST routes (router-level dependency) |
| `process.py` | All 3 POST routes (router-level dependency) |
| `ml.py` | `POST /ml/train`, `POST /ml/score` |
| `agent.py` | `POST /agent/chat`, `POST /agent/quick` |

**Unprotected routes** (no auth needed): all GET routes, `GET /agent/demo`, `GET /health`

#### 3. SQL injection hardening

`src/agent/tools.py` — `query_database` tool now blocks dangerous SQL keywords in addition to the existing SELECT-only check:

- Pattern: `\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|EXEC|EXECUTE)\b` (word-boundary, case-insensitive)
- Returns a safe error string (not an exception) so the agent can handle it gracefully
- Logs a `logger.warning()` with the blocked keyword and truncated query

#### 4. Startup validation

`src/api/main.py` `startup_event()` now validates env vars before initialising the DB:

- Missing `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB` → raises `RuntimeError` (app refuses to start)
- `OPENAI_API_KEY` missing or `"placeholder"` → `logger.warning()` (non-fatal; agent endpoints will fail at call time)

#### 5. docker-compose.yml updates

- `CHROMA_PORT=8001` → `CHROMA_PORT=8000` (container-to-container uses internal port)
- Added to app environment: `APP_ENV`, `API_SECRET_KEY`, `ALLOWED_ORIGINS`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- Docker default: `APP_ENV=production` (auth enforced); override with `APP_ENV=development` to skip auth

**Test auth locally (with Docker running):**
```bash
# Should return 401 (auth required in production mode)
curl -X POST http://localhost:8080/ingest/full

# Should return 200 (correct key)
curl -X POST http://localhost:8080/ingest/full \
  -H "X-Api-Key: change-me-in-production"

# Health check still open (no auth)
curl http://localhost:8080/health
```

---

### OpenAI reliability

| | |
|---|---|
| **Status** | Complete — `feature/production-hardening` branch |
| **Files** | `src/agent/agent.py`, `src/agent/tools.py` |

**What was added:**

- **tenacity retry** on `_invoke_agent()`: `RateLimitError` and `APITimeoutError` trigger exponential backoff (1s–10s), up to 3 attempts; final failure returns `_UNAVAILABLE_MSG` instead of raising
- **`_agent_key` tracking**: singleton resets and rebuilds automatically if `OPENAI_API_KEY` changes between requests — errors never cache permanently
- **30-second timeout** on the `draft_email` `ChatOpenAI` instance (`timeout=30`)
- **`GET /agent/health`** returns `{agent_ready, openai_key_configured, model, last_error}` with no API call — safe for readiness probes
- **Model fixed to `gpt-4o-mini`**: both agent and tools use `gpt-4o-mini`; `OPENAI_MODEL` env var removed from `.env.example` to prevent accidental override

---

### Background tasks

| | |
|---|---|
| **Status** | Complete — `feature/production-hardening` branch |
| **Files** | `src/api/task_store.py` (new), `src/api/routers/ml.py`, `src/api/routers/process.py`, `src/api/routers/ingest.py`, `src/api/main.py`, `src/api/templates/index.html` |

**What was added:**

`POST /ml/train`, `POST /ml/score`, `POST /process/full`, and `POST /ingest/full` all returned from blocking HTTP worker threads. Moved to FastAPI `BackgroundTasks` so they return immediately with a `task_id`.

**`src/api/task_store.py`** — 25-line in-memory store:
- `set_task(task_id, status, result, error)` — sets task state
- `get_task(task_id)` — retrieves task by id
- Resets on process restart (acceptable for demo/dev)

**`GET /task/{task_id}`** — added to `main.py`; returns `{task_id, status, result, error}` or 404.

**Background task pattern** — each task function:
1. Is named `_run_<operation>_task(task_id: str)`
2. Calls `set_task(task_id, "running")` at start
3. Creates its own `db = SessionLocal()` (never shares the HTTP request session, which closes before the task runs)
4. Calls `set_task(task_id, "complete", result=...)` on success or `set_task(task_id, "failed", error=...)` on failure
5. Closes `db` in `finally`

**Frontend pipeline buttons** — left panel now shows 4 pipeline control buttons (Ingest, Process, Train, Score). On click: sends POST, receives `task_id`, polls `GET /task/{task_id}` every 2 seconds, shows ⏳/✓/✗ status, and calls `loadData()` on completion to refresh the affiliate list.

**Task lifecycle:** `pending → running → complete | failed`

---

### Database migrations (Alembic) — production safe

| | |
|---|---|
| **Status** | Complete — `feature/data-persistence` branch |
| **Files** | `alembic.ini`, `alembic/env.py`, `alembic/versions/13ea16583831_initial_schema_affiliates_.py`, `src/api/routers/admin.py` |

**Design decision: migrations do NOT run on startup.**

In production, multiple app instances start simultaneously behind a load balancer. If each instance runs `alembic upgrade head` on startup, they race to acquire schema locks on PostgreSQL, causing deadlocks and potential data corruption.

**How it works:**
- App startup runs `SELECT 1` only — verifies the database is reachable, nothing more
- Migrations are triggered manually via `POST /admin/migrate` (requires API key)
- `GET /admin/migration-status` returns the current revision from `alembic_version`
- Migration files live in `alembic/versions/`

**`src/api/routers/admin.py`** — two protected endpoints:
- `POST /admin/migrate` — runs `alembic upgrade head`; returns `{status, message}` or `{status, error}`
- `GET /admin/migration-status` — returns `{current_revision, status}` via `MigrationContext`

**When to run migrations:**
1. First-time setup: after `docker compose up -d`, call `POST /admin/migrate` once
2. After schema changes: generate with `alembic revision --autogenerate -m "description"`, review the file, then call `POST /admin/migrate`
3. Production deployment: deploy new app version first, wait for health, then call `POST /admin/migrate` once

**CLI commands (direct access):**
```bash
alembic upgrade head    # apply all pending
alembic downgrade -1    # rollback one migration
alembic current         # show current version
alembic history         # show all migrations
```

**Never:**
- Run migrations on multiple instances simultaneously
- Delete files from `alembic/versions/`
- Skip reviewing autogenerated migration files
- Run migrations before the app is healthy

---

### S3 model storage

| | |
|---|---|
| **Status** | Complete — `feature/data-persistence` branch |
| **File** | `src/ml/model_store.py` (new) |

**What was added:**

- `src/ml/model_store.py` — thin abstraction over local disk + S3-compatible object storage:
  - `save_model(model, filename)` — writes with joblib locally, then uploads to S3 if `USE_S3=true`
  - `load_model(filename)` — returns from local disk; if missing and `USE_S3=true`, downloads from S3 first
  - `model_exists(filename)` — checks local disk first, then S3 via `head_object`
  - `_get_s3_client()` — creates a boto3 client; `S3_ENDPOINT_URL` makes it work with any S3-compatible platform (DigitalOcean Spaces, Cloudflare R2, Backblaze B2)
- `USE_S3=false` by default — works locally with no cloud credentials required
- `boto3` added to `requirements.txt`
- `churn_model.py` and `growth_model.py` updated: `joblib.dump/load` calls replaced with `save_model` / `load_model`; `CHURN_MODEL_PATH` / `GROWTH_MODEL_PATH` env vars and direct `os`/`Path`/`joblib` imports removed
- `models/*.json` added to `.gitignore`
- `.env.example` extended with all S3 variables (`USE_S3`, `S3_BUCKET`, `S3_MODEL_PREFIX`, `S3_ENDPOINT_URL`, `AWS_*`)
- README `Model storage` section added

---

### pgvector replaces ChromaDB — single-database architecture

| | |
|---|---|
| **Status** | Complete — `feature/data-persistence` branch |
| **Files** | `src/storage/pgvector_store.py` (new), `src/storage/models.py`, `src/ingestion/embedding_generator.py`, `src/agent/tools.py`, `src/api/routers/search.py`, `src/api/routers/process.py`, `src/api/main.py`, `docker-compose.yml` |

**What was changed:**

- **ChromaDB container removed** from `docker-compose.yml`; PostgreSQL image switched from `postgres:16-alpine` to `pgvector/pgvector:pg16` (same PG16, adds the vector extension)
- **New Alembic migration** (`ccc1c19d5237`): enables `CREATE EXTENSION IF NOT EXISTS vector`, creates `embeddings` table with `vector(384)` column, adds `ivfflat` cosine index with `lists=10`
- **`Embedding` ORM model** added to `src/storage/models.py` — maps to the `embeddings` table using `pgvector.sqlalchemy.Vector(384)`
- **`PGVectorStore` class** (`src/storage/pgvector_store.py`) replaces the ChromaDB `VectorStore` singleton:
  - Receives a `db: Session` — no module-level singleton, no separate HTTP client
  - `add_document(...)` upserts one chunk; `search_similar(query_embedding, n_results, ...)` returns `[{id, text, affiliate_name, source, tags, occurred_at, distance}]` using `Embedding.embedding.cosine_distance()`
- **`embedding_generator.py`** updated: removed `chromadb`/`VectorStore` imports; `add_document()` call signature changed — `embedding` param renamed to `embedding_vector`; `occurred_at` passed as a datetime object (not a string)
- **`tools.py` `semantic_search`** tool updated: creates `PGVectorStore(db)` inside the tool with its own `_get_db()` session; result format now flat (`affiliate_name`, `source`, `tags` at top level — no nested `metadata`)
- **`search.py`** router updated: `GET /search` creates `PGVectorStore(db)` using FastAPI `get_db()` dependency; maps `text` → `document` and builds `metadata` dict for backward-compatible `SearchResultItem` response
- **`process.py`** router updated: `POST /process/embeddings` and `_run_process_full_task` create `PGVectorStore(db)` instead of passing the old `vector_store` singleton
- **`main.py`** health endpoint simplified: removed ChromaDB check; returns `{status, postgres, timestamp}` only
- **`requirements.txt`**: `chromadb>=0.5.0` replaced with `pgvector>=0.3.0`
- **`.env.example`**: `CHROMA_*` variables removed
- **`src/ml/explainability.py`** bonus fix: `CHURN_MODEL_PATH` / `GROWTH_MODEL_PATH` imports replaced with `model_store.load_model()` (were broken since the S3 migration)
- **`src/ml/model_store.py`** bonus fix: `"filename"` key in `logger.warning()` extra dict renamed to `"model_file"` (`filename` is a reserved `LogRecord` field that caused a `KeyError`)
- **Tests updated**: `test_embeddings.py` and `test_agent.py` `semantic_search` tests now patch `src.storage.pgvector_store.PGVectorStore` instead of the removed `vector_store` singleton

**Verified live:**
```
GET /health → {"status": "ok", "postgres": "up"}   # no chromadb key
GET /search?q=affiliate+going+cold → Tom Bauer emails at top (cosine distance 0.63)
POST /agent/quick {"message": "which affiliates need attention?"} → correct ranking
pytest tests/ -v → 24/24 passed
```
---

## 14. Mock Data State

### Current verified pipeline state — `feature/better-mock-data`

| Step | Endpoint | Result |
|---|---|---|
| Ingest | `POST /ingest/full` | 10 affiliates + 13 communications loaded |
| NLP | `POST /process/nlp` | 13/13 communications tagged |
| Embeddings | `POST /process/embeddings` | 13 embedded, 13 chunks in pgvector |
| Train | `POST /ml/train` | churn + growth XGBoost models trained (10 samples) |
| Score | `POST /ml/score` | 10 affiliates scored via rule-based scorer |
| Agent | `POST /agent/quick` | Correctly identifies Tom, Marcus, James as urgent |

### 10 affiliates — differentiated profiles

| Affiliate | health | churn | growth | days | status |
|---|---|---|---|---|---|
| Rachel Torres | 100.0 | 0.00 | 1.00 | 6 | high_growth |
| Priya Sharma | 94.0 | 0.10 | 1.00 | 8 | high_growth |
| Sarah Chen | 94.0 | 0.10 | 1.00 | 9 | active |
| Aiko Tanaka | 94.0 | 0.10 | 1.00 | 11 | active |
| Fatima Al-Hassan | 91.2 | 0.10 | 0.93 | 10 | active |
| Nkechi Okonkwo | 65.2 | 0.10 | 0.28 | 12 | active |
| Carlos Mendez | 59.2 | 0.20 | 0.28 | 15 | active |
| James O'Brien | 33.0 | 0.45 | 0.00 | 30 | at_risk |
| Marcus Williams | 32.2 | 0.65 | 0.28 | 18 | at_risk |
| Tom Bauer | 15.0 | 0.75 | 0.00 | 40 | churned |

### 13 communications — tag coverage

**10 emails** (one per affiliate) + **3 transcripts** (Tom Bauer, Rachel Torres, Marcus Williams):

| Affiliate | Source | Key tags triggered |
|---|---|---|
| Tom Bauer | email (45d) | churn_signal, competitor_mention, disengaged_tone, frustrated |
| Tom Bauer | call (40d) | churn_signal, competitor_mention, stalled_deal, escalation, frustrated |
| James O'Brien | email (30d) | disengaged_tone, stalled_deal, follow_up_needed, gone_silent |
| Marcus Williams | email (20d) | complaint, escalation, frustrated, follow_up_needed |
| Marcus Williams | call (18d) | complaint, escalation, frustrated, action_committed |
| Carlos Mendez | email (15d) | neutral_sentiment, follow_up_needed |
| Nkechi Okonkwo | email (12d) | neutral_sentiment, action_committed |
| Fatima Al-Hassan | email (10d) | positive_sentiment, campaign_active, responsive |
| Aiko Tanaka | email (11d) | positive_sentiment, new_campaign_intent, question_asked |
| Sarah Chen | email (9d) | positive_sentiment, expansion_interest, upsell_signal |
| Priya Sharma | email (8d) | enthusiastic, positive_sentiment, expansion_interest, new_campaign_intent |
| Rachel Torres | email (7d) | enthusiastic, positive_sentiment, campaign_active, action_committed |
| Rachel Torres | call (6d) | enthusiastic, expansion_interest, action_committed, positive_sentiment |

### Design notes

- **`days_since_contact` is computed relative to today** — `last_contact_at = now() - timedelta(days=N)` in ETL, so dates stay accurate regardless of when the pipeline runs.
- **Communications outside the 30-day window** (Tom's comms at 40d and 45d) contribute to `days_since_contact` and `ctr_trend_pct` churn signals but NOT to `churn_signal_count` or `competitor_mention_count` (which only count 30-day window comms). The `comm_count_30d == 0` rule (+0.15) still fires for Tom.
- **Rule-based scoring is primary** (`score_updater.py` calls `calculate_churn_risk_rules` / `calculate_growth_potential_rules` directly). XGBoost is trained and available for `predict_churn_risk` / `predict_growth_potential` calls (e.g. explainability) but not used in the score update loop — with 10 samples it produces near-identical predictions for all affiliates.
- **Bug fixed** (`src/api/main.py` `AffiliateOut.from_orm`): `a.churn_risk_score or 0.5` replaced with `a.churn_risk_score if a.churn_risk_score is not None else 0.5` — the `or` form treats `0.0` as falsy and incorrectly returns `0.5` for affiliates with zero churn risk.

---

## 15. Known issues

- **`Embedding` model missing from `alembic/env.py` explicit import** — `env.py` imports
  `Affiliate`, `Communication`, `ScoreHistory`, `LeakedCode` by name to ensure they are
  registered on `Base.metadata` before autogenerate runs, but `Embedding` is absent.
  Alembic autogenerate may silently miss schema changes to the `embeddings` table.
  Pre-existing gap, unrelated to the promo leakage detector work. Needs its own fix on a
  separate branch.

- **`test_get_shap_explanation_structure` segfaults on macOS** — XGBoost loaded via joblib
  inside the pytest process triggers a dylib conflict, same root cause already noted
  elsewhere in this file for the uvicorn context. Confirmed pre-existing on `develop`,
  unrelated to the promo leakage detector work. Re-tested 2026-07-03: passed cleanly
  across 5 separate isolated runs with no segfault, suggesting this is
  intermittent/non-deterministic rather than a consistent failure. Not marked resolved
  — root cause (macOS XGBoost/joblib dylib conflict) has not been fixed, only observed
  to not always reproduce.

- **README Getting Started steps 4-5 are inconsistent** — step 4 (`docker compose up -d`)
  already starts the `app` service on port 8080; the step 5 alternative local `uvicorn`
  path would then hit a port conflict if both are followed in sequence as written.
  Pre-existing, unrelated to the promo leakage detector work.
