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
- Returns `explanation_unavailable: true` (never fabricated zero values) if the model isn't
  trained yet or the SHAP computation itself fails — see § "SHAP/XGBoost version incompatibility"
- Every response also carries `is_secondary_model: true` and a `model_description` — `prediction`
  is always a fresh, independent XGBoost estimate, not the persisted `churn_risk_score`/
  `growth_potential_score` that `recommend()`/status actually use; the two are expected to disagree

### Score updater (`score_updater.py`)
- `update_all_scores(db)` — runs the full pipeline for every affiliate
- Health score: `round(((1 - churn_risk) × 0.6 + growth × 0.4) × 100, 1)`
- Idempotent within a day (skips affiliates already scored today)

---

## 6. Agent Architecture

**Framework**: LangChain ReAct (Reason + Act)

### Tools available to the agent (7)

| Tool | Function |
|---|---|
| `query_database` | Raw SQL SELECT against PostgreSQL; validates SELECT-only, max 20 rows |
| `semantic_search` | pgvector cosine search over communications via `PGVectorStore.search_similar()` |
| `get_affiliate_summary` | Full affiliate profile: scores, recent comms, `recommend()`'s tier/evidence — see § Rulebook module |
| `draft_email` | Composes a re-engagement email and files it to the approval queue (`status='waiting_for_review'`) — never sends; see § Approval queue |
| `get_portfolio_health` | Whole-portfolio aggregate stats: health/churn/growth counts, plus leak and SEO-decline counts/names (visibility only, not folded into the tier counts) |
| `get_leakage_status` | Read-only — most recently recorded promo-code leak findings for one affiliate; cannot trigger a new scan |
| `get_seo_status` | Read-only — most recently recorded SEO rank signal for one affiliate; cannot trigger a new check |

`check_promo_leakage` (an earlier tool that ran a *live* scan on demand) was
removed entirely — an agent tool must never be able to fire a live external
action. Enforced by a static-analysis regression test
(`tests/test_agent.py::test_no_bound_tool_has_side_effects_beyond_draft_email_approval_insert`)
that inspects every tool's source for write/live-action markers.

---

## 7. API Endpoints (FastAPI)

Run with: `uvicorn src.api.main:app --port 8080 --reload`

Endpoints are defined in `src/api/routers/` and wired into `src/api/main.py`
via `app.include_router(...)`. Startup logs the total registered route count.

| Method | Path | Router file | Description |
|---|---|---|---|
| GET | `/health` | main.py | Service health check (PostgreSQL) |
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
docker compose up -d                          # start PostgreSQL (pgvector)
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

### Rulebook module

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/rulebook/recommend.py`, `src/rulebook/__init__.py` |
| **Tests** | `tests/test_rulebook.py` — 22 tests, all passing |

**What it does:** Single source of truth for "what tier is this affiliate
in, and why" — replaces three previously-inconsistent inline threshold
checks that had drifted apart (`get_affiliate_summary`'s 0.65/0.45/0.70,
`get_portfolio_health`'s 0.5/0.5/0.8, `score_updater.py`'s 0.5/0.5). Every
function is pure — no DB session, no I/O — so it's exhaustively
unit-testable against boundary values.

**Canonical thresholds** (the only place in the codebase allowed to
contain a churn/growth cutoff):

| Constant | Value |
|---|---|
| `CHURN_AT_RISK_THRESHOLD` | 0.50 |
| `CHURN_CRITICAL_THRESHOLD` | 0.80 |
| `GROWTH_HIGH_THRESHOLD` | 0.50 |

Imported directly by `src/agent/tools.py`, `src/ml/score_updater.py`,
`src/api/routers/ml.py` (`dashboard()`), and `src/ingestion/etl_pipeline.py`'s
`_derive_status()` — none of them keep a private copy anymore.

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `categorize` | `(churn_risk_score, growth_potential_score) -> str` | Pure tier classification: `churned \| at_risk \| high_growth \| active`, mutually exclusive, churn checked before growth |
| `recommend` | `(affiliate, features, leaks) -> Recommendation` | Full recommendation with reason code and an evidence bundle (list of human-readable facts) |

**Tier is a pure function of churn/growth only.** A promo-code leak (or any
other signal — SEO trend, etc.) never changes `tier` or the numeric score.
`recommend()` still surfaces a leak in the evidence bundle and appends
`_leak_detected` to the `reason_code` (e.g. `at_risk_leak_detected`) —
visible alongside the tier, never folded into it. This was an explicit
correction made during this module's build: an earlier draft had a leak
force `tier` to `at_risk` regardless of actual churn/growth, which violated
the same separation-of-signals principle applied everywhere else in this
system (`has_active_leak`, `search_trend`).

**Evidence bundle:** a `list[str]` of plain-English facts — the two
threshold comparisons always included, plus any non-zero feature from
`days_since_contact`, `churn_signal_count`, `competitor_mention_count`,
`escalation_count`, `comm_count_30d` when a feature vector is available,
plus a leak-count line when leaks exist. Persisted verbatim into
`score_history.evidence_bundle` (see `score_updater.py`) and surfaced
directly on `AffiliateDetail` via `latest_evidence_bundle`.

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
| **Tests** | `tests/test_agent.py` — 12 tests; `tests/test_agent_multisignal.py` — 1 test (real LLM call, see below); all passing |

**LLM:** `gpt-4o-mini`, temperature=0, via `langgraph.prebuilt.create_react_agent`
(langchain 1.3.x does not ship `create_openai_functions_agent` — use langgraph prebuilt instead)

**Tools (7):**

| Tool | Description |
|---|---|
| `query_database` | Raw SQL SELECT against PostgreSQL; validates SELECT-only, max 20 rows |
| `semantic_search` | pgvector cosine search over communications via `PGVectorStore(db).search_similar()` |
| `get_affiliate_summary` | Full affiliate profile: scores, `recommend()`'s tier/reason_code/evidence (see § Rulebook module), a leak note kept visually separate from the tier, an SEO Signal section reading `search_trend` directly |
| `draft_email` | Composes an email draft and files it into the approval queue (`ApprovalRequest`, `status='waiting_for_review'`) — never sends; see § Approval queue |
| `get_portfolio_health` | Whole-portfolio aggregate stats: health/churn/growth counts, plus leak and SEO-decline counts and names (visibility only, never folded into the counts) |
| `get_leakage_status` | Read-only — most recently recorded promo-code leak findings for one affiliate; cannot trigger a new scan |
| `get_seo_status` | Read-only — most recently recorded SEO rank signal for one affiliate; cannot trigger a new check |

`check_promo_leakage` (an earlier tool that ran a *live* scan on demand) was
removed entirely — an agent tool must never be able to fire a live external
action; see § Promo code leakage detector's has_active_leak note and
§ SEO ingestion for the read-only replacements.

**API endpoints:** `POST /agent/chat` (with history), `POST /agent/quick` (single-turn), `GET /agent/demo`, `GET /agent/health`

**Important implementation notes:**
- All tools create a fresh `SessionLocal()` per call (not shared) and close it in `finally`
- `get_affiliate_summary` derives churn/growth drivers from the most recent *tagged*
  communications within a 90-day lookback (not just the most recent N regardless of tag
  status) — NOT from SHAP (loading XGBoost via joblib inside a LangGraph tool context causes a
  segfault in uvicorn). If a more recent, untagged communication exists, the summary says so
  explicitly rather than silently basing drivers on older data.
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
- **"Warning signs" guidance**: the prompt explicitly instructs that "warning signs"/"risk"/
  "needs attention" span all three signal types this system tracks (churn/growth tier, leaks,
  SEO trend) — not just the tier. It gives a literal step-by-step procedure (call
  `get_portfolio_health`, union its three name lists, report every name found in at least one
  list with its signal count) because looser phrasing repeatedly caused the model to anchor on
  the worst-health list alone and silently drop single-signal names. `tests/test_agent_multisignal.py`
  is a real, non-mocked LLM call (a deliberate exception to the "no real API calls" convention
  used elsewhere in this test suite) verifying the model actually surfaces both a multi-signal
  and a single-signal affiliate.

**Depends on:** All pipeline steps must have run: `/ingest/full` → `/process/nlp` →
`/process/embeddings` → `/ml/train` → `/ml/score`

---

### Approval queue

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/api/routers/approvals.py` (new), `src/notifications/sender.py` (new), `src/storage/models.py` (`ApprovalRequest`), `src/agent/tools.py` (`draft_email` rewritten) |
| **Migration** | `alembic/versions/c44d4aa7e33a_add_approval_requests_table.py` |
| **Tests** | `tests/test_approvals.py` — 8 tests, all passing |

**What it does:** Nothing that would leave the system (currently: an email)
fires without a human clicking Approve. `draft_email` used to return raw
email text directly to the chat response; it now composes the draft and
inserts an `ApprovalRequest` row with `status='waiting_for_review'` instead
— the agent can draft, but it cannot dispatch.

**`approval_requests` table:**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | `gen_random_uuid()` |
| `kind` | VARCHAR(32) | `"email"` today; room for more kinds later |
| `affiliate_id` | UUID FK → `affiliates.id` | CASCADE delete |
| `payload` | JSONB | For email: `{to, subject, body, affiliate_id}` |
| `status` | VARCHAR(20) | `waiting_for_review` \| `approved` \| `rejected` |
| `created_at` | TIMESTAMPTZ | |
| `decided_at` | TIMESTAMPTZ NULLABLE | Set on approve/reject |
| `decided_by` | VARCHAR(128) NULLABLE | See placeholder note below |

Indexes on `affiliate_id` and `status`.

**`decided_by` is a fixed placeholder (`"api"`), not a real user identity.**
There is no per-user auth system in this app — `src.api.auth.get_api_key`
only confirms a valid shared API key was presented, it doesn't identify who
holds it. Echoing the raw key value into stored data would leak a secret
into the database, so a constant placeholder is used until real per-user
identity exists (see § Planned next).

**`send_email()` (`src/notifications/sender.py`) is a placeholder — no real
email provider is wired up.** It logs `"Would send email (no real provider
configured)"` via the existing structured logger. It is called from exactly
one place in the entire codebase: `POST /approvals/{id}/approve`. Enforced
by a static-analysis test (`test_send_email_only_referenced_from_approvals_router`)
that greps all of `src/` for any other reference.

**API endpoints (`src/api/routers/approvals.py`):**

| Method | Path | Description |
|---|---|---|
| `POST` | `/approvals` | Create a request directly (mainly for testing — `draft_email` is the real caller) |
| `GET` | `/approvals?status=` | List, newest first, optional status filter |
| `POST` | `/approvals/{id}/approve` | Approve — the only path that calls `send_email()`; writes an audit_log entry; returns 409 if already decided |
| `POST` | `/approvals/{id}/reject` | Reject — nothing fires; writes an audit_log entry; returns 409 if already decided |

**Idempotency / race safety:** approve/reject both check `status ==
'waiting_for_review'` before acting and return HTTP 409 if the request was
already decided (e.g. a double-click, or two people acting on the same
request) — the frontend (`Approvals.tsx`) treats a 409 as "someone already
decided this" and refreshes the list rather than surfacing a raw error.

---

### Audit log

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/audit/log.py` (new), `src/api/routers/audit.py` (new), `src/storage/models.py` (`AuditLog`) |
| **Migration** | `alembic/versions/6f6b2c550b26_add_audit_log_table.py` |
| **Tests** | `tests/test_audit.py` — 4 tests, all passing |

**What it does:** An append-only record linking any stored
recommendation/signal-check/decision back to the exact inputs and
rule/tool that produced it. `write_audit_entry()` is deliberately dumb — it
builds one row and `add()`s it to the given session; it never commits (the
caller already owns the transaction) and makes no decisions about what or
when to log.

**`audit_log` table:**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `timestamp` | TIMESTAMPTZ | |
| `stage` | VARCHAR(20) | `signals` \| `rulebook` \| `agent` \| `approval` |
| `record_type` | VARCHAR(32) | e.g. `"affiliate"`, `"approval_request"` — polymorphic, not a foreign key |
| `record_id` | UUID | Not a FK — `record_type` varies, so this can't point at one fixed table |
| `rule_or_tool` | VARCHAR(64) | e.g. `"recommend"`, `"check_leakage"`, `"check_seo"`, `"approvals.approve"` |
| `input_snapshot` | JSONB | |
| `output_snapshot` | JSONB | |

Index on `(record_type, record_id)`.

**Four wiring points** write to this table (`src/audit/log.py`'s own module
docstring says three — that predates the SEO work and is now stale; the
real count is four):
1. `src/ml/score_updater.py` — `stage="rulebook"`, one entry per affiliate
   per scoring run, linking the feature inputs to `recommend()`'s tier +
   evidence
2. `src/scraping/leakage_scraper.py` — `stage="signals"`, one entry per
   affiliate scanned, even when zero leaks are found ("checked and found
   nothing" is on record too)
3. `src/seo/checker.py` — `stage="signals"`, one entry per tracked
   affiliate, including a `"note": "keyword not found in source"` case for
   misses
4. `src/api/routers/approvals.py` — `stage="approval"`, on both approve and
   reject

**`GET /audit?record_type=&record_id=&stage=&limit=&offset=`** —
deliberately API-key gated, unlike most other GET routes in this app (a
design choice: this is a decision-trail endpoint, not general read access).

**Referential-integrity regression tests:** for each of the four wiring
points, a test confirms the `record_id` written actually resolves to a
real row in the table implied by `record_type` — not a schema constraint
(`record_id` stays polymorphic and unconstrained by design), just a
regression check against a future bug that writes an audit entry pointing
nowhere. Two live in `tests/test_audit.py` (score_updater, approvals); the
leakage and SEO checks live as extra assertions on each module's own
existing real-DB tests (`tests/test_leakage_scraper.py`, `tests/test_seo.py`)
rather than duplicating setup in `test_audit.py`.

**Not the same as `GET /admin/logs`** (see § Log persistence) — this table
is for decision-linked entries only (a deliberate subset); the log file is
for general operational visibility (every HTTP request, every log line).

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

### has_active_leak — first-class leak visibility

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/storage/models.py` (`Affiliate.has_active_leak`), `src/scraping/leakage_scraper.py`, `src/api/main.py`, `src/api/routers/ml.py`, `src/agent/tools.py`, `src/ingestion/etl_pipeline.py` (`seed_demo_leak_scan`) |
| **Migration** | `alembic/versions/c19c2f2fc727_add_has_active_leak_to_affiliates.py` |

**What it does:** Promotes the leak signal from an implicit join (querying
`leaked_codes` every time you want to know if an affiliate has one) to a
first-class, queryable boolean column — `affiliates.has_active_leak`,
`BOOLEAN NOT NULL DEFAULT false`. Kept separate from and visible alongside
`churn_risk_score`/`growth_potential_score`/`status`, never folded into any
of them — the same principle `src/rulebook/recommend.py` already enforces
at the tier level, now also enforced at the storage layer so a caller
doesn't have to know to join `leaked_codes` to get the same answer.

**Recomputed, not incrementally updated:** `check_leakage()` recomputes
`has_active_leak` from the FULL `leaked_codes` table for every affiliate
scanned on every call (`count(*) > 0`), not from just that run's new
matches — so the flag stays correct for an affiliate whose leak was found
in a prior scan and shows nothing new today (the dedup window skips
re-inserting the row, but the flag must still reflect that a leak is on
record).

**"Active" means "at least one `leaked_codes` row exists," full stop —
there is no resolution/expiry workflow.** Once true, it stays true
indefinitely. Clearing it today requires a manual DB operation on
`leaked_codes`. A resolution workflow (e.g. a `resolved_at` column, or an
admin "mark resolved" endpoint) would be needed before this flag can
self-correct — see § Planned next.

**Fast-path read optimisation:** `get_leakage_status` (agent tool) and
`get_affiliate_summary` check `has_active_leak` first and skip the
`leaked_codes` detail query entirely when it's `false` — a safe drop-in
since `has_active_leak=false` guarantees the detail query would return
nothing.

**Demo seed guard (`_all_sites_are_mock()` in `etl_pipeline.py`):**
`seed_demo_leak_scan()` — the demo-convenience step in `run_full_pipeline()`
that seeds Rachel Torres and Marcus Williams's leaks on every `POST
/ingest/full` — refuses to run and logs a warning if any site in
`src.scraping.site_config.SITES` is not a local `file://` fixture.
`SiteConfig.kind` is deliberately NOT what gates this: the `csr-shell-mock`
fixture is `kind="live"` for an unrelated technical reason (it needs the
Playwright browser-render path since its JS shell must execute), despite
being a completely safe local file. Only the URL scheme is trustworthy
here. This prevents a live external scan from becoming a silent side
effect of routine ingestion the moment a real site is ever added to `SITES`.

---

### SEO ingestion

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/seo/api_client.py`, `src/seo/analyze.py`, `src/seo/checker.py`, `src/seo/__init__.py`, `src/storage/models.py` (`SeoSignal`), `src/api/routers/seo.py`, `src/agent/tools.py` (`get_seo_status`), `src/scheduling/jobs.py`, `src/ingestion/etl_pipeline.py` (`seed_demo_seo_scan`) |
| **Migration** | `alembic/versions/3ec97360dce2_add_seo_signals_table_and_affiliate_seo_.py` |
| **Tests** | `tests/test_seo.py` — 11 tests, all passing |

**What it does:** Tracks search-ranking trend for a keyword per affiliate,
mirroring the promo-leak detector's mock-first pattern exactly: fixture
data checked in, a module-level flag distinguishes mock vs. live, and an
explicit, *enforced* guard prevents a live check from becoming a silent
side effect of routine ingestion.

**Mock data shape (`data/mock/seo/rank_tracking_mock.json`)** — modeled on
a SEMrush/Ahrefs-style Position Tracking API response: one object per
keyword with `keyword, position, previous_position, search_volume, url,
checked_at`. `checked_at` is a **fixed** timestamp in the fixture (not
computed relative to "now" the way the mock communications' "N days ago"
markers are) — this detail mattered later (see the duplication bug below).

**Schema additions:**
- `affiliates.tracked_keyword` — `VARCHAR(255) NULLABLE`; mirrors
  `active_promo_code`'s role (identifies what to look for). `NULL` = not
  tracked, skipped silently by every check.
- `affiliates.search_trend` — `VARCHAR(20) NOT NULL DEFAULT 'stable'` —
  `declining \| stable \| improving`. Chose a new string column over
  reusing `has_active_leak`'s boolean pattern: a three-state trend can't
  collapse into a boolean without losing "improving." **`'stable'` is also
  the default for an affiliate that has never been tracked at all** — it
  does not by itself mean "checked, and found stable." The frontend's
  `SeoTrendBadge` gates on real `seo_signals` rows existing, not on this
  value alone, for exactly this reason.
- `seo_signals` table — one row per keyword rank check (real evidence per
  row, same pattern as `leaked_codes`, not just a rolled-up score):

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `affiliate_id` | UUID FK → `affiliates.id` | CASCADE delete |
| `keyword` | VARCHAR(255) | |
| `rank` | INTEGER | Current position |
| `rank_change` | INTEGER NULLABLE | `previous_position - position`; positive = improved (moved to a lower/better position number); `NULL` if no prior rank was on record |
| `search_volume` | INTEGER NULLABLE | |
| `checked_at` | TIMESTAMPTZ | |

Indexes on `affiliate_id` and `keyword`.

**`derive_search_trend()` (`src/seo/analyze.py`, pure function):** looks
only at the most-recently-`checked_at` signal's `rank_change` — an old
declining signal never overrides a newer stable/improving one.
`DECLINING_THRESHOLD = -3`, `IMPROVING_THRESHOLD = 3`; empty list or `None`
rank_change → `"stable"`. Accepts both dicts and ORM objects (same
dual-mode pattern as `recommend()`'s leak handling).

**`check_seo()` (`src/seo/checker.py`) — no time-window dedup, by design,
with one narrower exact-measurement guard added later.** Unlike
`check_leakage()`'s 20-hour window, every call is treated as a legitimate
new rank measurement — even an unchanged rank is a real data point worth
recording in the time series. This surfaced a real bug: the mock fixture's
`checked_at` is *fixed*, so replaying it via `seed_demo_seo_scan()` on
every `POST /ingest/full` inserted a fresh, byte-for-byte duplicate
`SeoSignal` row every time — found live as 176 rows collapsing to 4 real
measurements. Fixed by adding an exact-measurement guard: skip inserting
only if a row with the identical `(affiliate_id, keyword, checked_at)`
already exists. This is **not** a time-window suppression — a real API
always returns its own current timestamp, so a genuinely new measurement
is never affected; it only catches literal replay of the same data point.

**`affiliates.search_trend` is recomputed from the affiliate's full
`seo_signals` history on every check** (same recompute-from-source-of-truth
pattern as `has_active_leak`), so it stays accurate for an affiliate whose
keyword momentarily wasn't found in a given run.

**Never feeds into `recommend()`'s tier**, same principle as leaks —
visible alongside the score, not folded into it.

**Demo seed guard (`_seo_source_is_mock()` in `etl_pipeline.py`):**
`seed_demo_seo_scan()` refuses to run and logs a warning if
`src.seo.api_client.LIVE_API_CONFIGURED` is `True` — mirrors
`_all_sites_are_mock()`'s enforcement for the leak checker, adapted to
SEO's simpler reality (one global flag instead of a list of sites to
inspect).

**Agent tool (`get_seo_status`):** read-only, reports the most recently
recorded signal; cannot trigger a live check. `get_affiliate_summary` also
reads `search_trend` directly for its "SEO Signal" section, never through
`recommend()`.

**API endpoints (`src/api/routers/seo.py`):**

| Method | Path | Description |
|---|---|---|
| `POST` | `/seo/scan` | Full rank check as a background task; returns `task_id` |
| `GET` | `/seo/results` | All `SeoSignal` rows, newest first |
| `GET` | `/seo/results/{affiliate_id}` | Rows for one affiliate; `[]` (not 404) if none |

**Scheduler:** weekly, `CronTrigger(day_of_week="mon", hour=4, minute=0,
timezone="UTC")` — less frequent than the leak checker's nightly job,
since search rank genuinely doesn't move meaningfully day-to-day the way a
leak can appear overnight.

**Deliberately out of scope:** no live SEO API is wired up
(`LIVE_API_CONFIGURED = False`); `fetch_seo_data(kind="live")` raises
`NotImplementedError`.

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
- RDS PostgreSQL with automated backups
- Rate limiting on public endpoints
- A real log aggregator (CloudWatch/ELK/Datadog) — the file-based
  `logs/app.jsonl` + `GET /admin/logs` (see § "Log persistence") is a
  demo-appropriate stopgap, not a substitute for one at production scale
- **Live leak/SEO scraping — deliberately out of scope, not just
  unfinished.** `src.scraping.site_config.SITES` contains only fixture
  entries; no real voucher site has been enabled. `src.seo.api_client.LIVE_API_CONFIGURED`
  is `False`; no real SEO API is wired up. Both have an explicit, enforced
  guard (not just a docstring) preventing a live external call from
  becoming a side effect of routine ingestion the moment either is turned
  on — see § Promo code leakage detector and § SEO ingestion.
- **Per-user identity / real auth — not built.** `src.api.auth.get_api_key`
  only confirms a valid shared API key, it doesn't identify who holds it.
  `approval_requests.decided_by` is a fixed placeholder (`"api"`) as a
  result — see § Approval queue. A real identity system is needed before
  "who approved this" is a meaningful question.
- **`has_active_leak` has no resolution path.** Once set `true`, it stays
  `true` indefinitely — no admin action, expiry, or `resolved_at` column
  clears it, even after the underlying leak is fixed. Clearing it today
  requires a manual DB operation on `leaked_codes`. See § has_active_leak.

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

### Log persistence (file-based)

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/core/logging_config.py`, `src/api/routers/admin.py`, `docker-compose.yml` |
| **Tests** | `tests/test_logging_config.py` — 8 tests, all passing |

**What was added:**

`configure_logging()` now attaches a **second, additive** handler alongside
the existing stdout `StreamHandler` (which is untouched — `docker logs` keeps
working exactly as before): a `RotatingFileHandler` writing the same JSON
lines to `logs/app.jsonl` (project root; gitignored, already covered by the
pre-existing `logs/` entry in `.gitignore`).

**Rotation policy: size-based, 10MB per file, 5 backups (60MB ceiling).**
`RotatingFileHandler` was chosen over `TimedRotatingFileHandler` because the
concern driving this feature — not growing unbounded during a long-running
session — is fundamentally about disk size, not calendar time; a demo/interview
session could run continuously for hours without ever hitting a day boundary,
so time-based rotation wouldn't actually bound anything in that scenario.

**`GET /admin/logs`** — API-key gated the same way as `/admin/migrate` /
`/admin/migration-status`. Query params: `level` (optional, e.g. `WARNING`),
`limit` (default 100, max 1000), `search` (optional, case-insensitive
substring match over `message`). Returns `{count, entries}`, newest first.

This is **general operational visibility** — HTTP requests, pipeline steps,
blocked-SQL warnings, startup errors, etc. It does **not** replace `GET
/audit` (§ Audit log), which stays reserved for decision-linked entries tied
to the rulebook/signals/approval stages.

**Reading large/rotated files gracefully:** `read_log_entries()` scans each
candidate file (current file, then `.1`, `.2`, ... up to `backup_count`)
forward exactly once, keeping only the most recent `limit` matching entries
per file in a bounded `deque` — it never holds a whole file's parsed contents
in memory, and a request right after a rotation still finds older entries by
falling through into the freshly-rotated backup once the current file is
exhausted.

**Demo-appropriate, not production-grade.** This is file-based storage on a
single instance's disk — it does not survive the instance being replaced (only
container restarts, via the `./logs:/app/logs` bind mount added to
`docker-compose.yml`), doesn't aggregate across multiple instances, and has no
retention beyond the 60MB rotation ceiling. A real deployment should point at
CloudWatch, ELK, or Datadog instead of scaling this further.

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

**`src/api/routers/admin.py`** — three protected endpoints:
- `POST /admin/migrate` — runs `alembic upgrade head`; returns `{status, message}` or `{status, error}`
- `GET /admin/migration-status` — returns `{current_revision, status}` via `MigrationContext`
- `GET /admin/logs` — reads recent structured log entries from disk (see § "Log persistence")

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

### SHAP/XGBoost version incompatibility

| | |
|---|---|
| **Status** | Fixed |
| **Files** | `requirements.txt`, `src/ml/explainability.py`, `src/api/routers/ml.py` |
| **Tests** | `tests/test_ml.py` — 3 dedicated tests (unavailable-shape on SHAP failure, unavailable-shape on model-not-trained, real values on success) |

**What broke:** `GET /ml/explain/{id}` returned `top_factors` with 5 real
feature names and real `feature_value`s, but every single `shap_value` was
exactly `0.0` and `base_value` was exactly `0.0` too — for every affiliate,
both models. It looked like a complete, valid explanation. It wasn't: a
broad `except Exception` around the SHAP computation was silently catching
`ValueError: could not convert string to float: '[3E-1]'` and substituting
`np.zeros(len(FEATURE_NAMES))` / `base=0.0`. `xgboost>=3.0.0` serializes a
model's `base_score` as a bracketed-scientific-notation string (e.g.
`"[3E-1]"`); `shap.TreeExplainer`'s XGBoost model loader does a naive
`float(base_score)` written for the older plain-numeric-string format, and
raises on construction — before `shap_values()` is ever called.

**Version pins chosen: `xgboost==2.1.4`, `shap==0.49.1`** (`requirements.txt`
had unbounded `>=` minimums before this, which is what let the incompatible
pair install in the first place — now pinned exactly, with a comment
explaining why, so a future "helpful" `pip install --upgrade` doesn't
reintroduce this). shap fixed this exact bug in 0.50.0 (`shap/shap#4187` —
confirmed in its own changelog), but **shap 0.50.0+ requires Python
>=3.11**, and this project's container runs **Python 3.10.12** (pinned via
the Playwright base image — see § Promo code leakage detector's Dockerfile
note). Upgrading shap was the preferred fix in principle but isn't viable
without also upgrading the base image, so `xgboost` was downgraded to its
latest 2.x release instead — verified empirically against this codebase's
real models before picking it, not assumed from a changelog. **The old
model files still loaded** under the downgraded xgboost (with a
cross-version pickle warning), but produced *different* prediction values
than they had under 3.2.0 — confirming the warning was right and
retraining was mandatory, not optional. Retrained via `POST /ml/train`.

**Honest-failure pattern replacing silent zero-substitution:** the same
class of bug (a dependency incompatibility silently producing
plausible-looking fake output) could recur with any future dependency
bump, so the broad exception handler itself was narrowed.
`get_shap_explanation()` now returns, on any SHAP-computation failure:

```json
{
  "base_value": null,
  "prediction": 0.30,
  "top_factors": [],
  "explanation_unavailable": true,
  "note": "SHAP explanation unavailable — computation failed: <error>"
}
```

`prediction` is still real — it comes from `model.predict_proba()`,
computed *before* the SHAP try/except specifically so a SHAP-only failure
can never take a working prediction down with it. `explanation_unavailable`
is also set `true` on the pre-existing "model not trained" fallback, so a
caller has one single flag to check regardless of *why* no real
explanation exists. Logged at `logger.error` (not downgraded) — after this
fix, a genuine SHAP failure should be rare, so it's still worth paging on.

**`is_secondary_model` / `model_description` fields:** added because
`get_shap_explanation()`'s `prediction` is always a *fresh, independent*
XGBoost estimate — it is never the persisted, rule-based
`churn_risk_score`/`growth_potential_score` that `recommend()` and
`status`/tier actually use. The two numbers will generally disagree for
the same affiliate (confirmed live on every affiliate checked) — this is
expected, not a data-freshness bug; re-running `POST /ml/score` does not
and cannot make them converge, since they're two different scoring
algorithms over the same features. Every response carries
`is_secondary_model: true` and a `model_description` string so any
consumer of this endpoint (not just `AffiliateDetail.tsx`) gets the same
disambiguation without relying on frontend copy alone.

---

### Idempotency audit

| | |
|---|---|
| **Status** | Complete |
| **Files** | `src/ingestion/etl_pipeline.py`, `src/seo/checker.py`, `tests/test_etl_pipeline.py`, `tests/test_seo.py`, `tests/test_leakage_scraper.py` |

Three real bugs, all the same root cause, were found and fixed across
several passes — a pipeline step assuming it runs against a clean
database, silently producing wrong or duplicated state on re-run instead
of erroring or no-opping cleanly:

1. **`ingest_affiliates_csv()` reset real computed scores on every
   re-ingest.** It unconditionally set `churn_risk_score` /
   `growth_potential_score` / `health_score` to 0.5/0.5/50.0 defaults on
   every upsert, including for an already-scored existing affiliate. Fixed
   to only apply defaults to a brand-new affiliate, or when the CSV
   explicitly provides an override value (legacy re-import format).
2. **Six real-DB test cleanups silently no-op'd their own reset logic.**
   `check_leakage()`/`check_seo()` recompute `has_active_leak`/`search_trend`
   via a bulk `Query.update(synchronize_session=False)`, which bypasses
   the ORM identity map — a session's cached object goes stale. Test
   `finally` blocks reassigning that same object's attributes were
   comparing against the stale in-memory value, not the real DB row, and
   silently no-op'd whenever the two coincidentally matched. Fixed by
   adding `db.refresh(aff)` before any such reassignment, in every
   affected cleanup — and, in one follow-up correction, moving the
   refresh to *before all* reassignments in a block that had several (a
   refresh discards any not-yet-flushed pending change, so placing it
   between two reassignments silently reverted the first one).
3. **`ingest_communications_file()` had no dedup at all.** Every re-run
   (e.g. every `POST /ingest/full`) inserted a second, untagged copy of
   every affiliate's communications, with a freshly recomputed
   `occurred_at` — confirmed live at 78 rows collapsing to the real 13.
   This is what caused `get_affiliate_summary`'s churn/growth drivers to
   silently show "Insufficient data" for several affiliates: the "last 5
   communications" window filled up with untagged duplicates, burying
   real tagged history. Fixed with a dedup key of `(affiliate_id, source,
   raw_text)` — `occurred_at` can't be part of the identity key since it's
   derived at parse time and isn't stable across runs. Decision:
   re-ingesting the same content leaves its original `occurred_at` frozen
   rather than refreshing it — the mock files represent a fixed
   historical backstory, not a live moving window.
4. **`check_seo()`'s demo-seed path duplicated `seo_signals` the same
   way** (see § SEO ingestion above) — same root cause, same class of
   bug, found during the audit that specifically went looking for it.

**Every other `ingest_*`/`seed_*` function was audited for the same class
of problem and confirmed safe:** `ingest_csv_content()` (only ever
overwrites `revenue_30d`, direct-overwrite semantics are inherently
idempotent), `seed_demo_leak_scan()`/`check_leakage()` (20-hour dedup
window plus a recompute-from-source-of-truth `has_active_leak`),
`process_all_communications()` and the embedding-generation equivalent
(both filter to unprocessed rows only — `tags == []` / `embedding_id IS
NULL` — so a repeat run is a clean no-op; `PGVectorStore.add_document()`
is also a genuine upsert), `run_full_pipeline()`'s orchestration order (no
new interaction bug from combining the fixes above — each step's
idempotency is independent), and every Alembic migration (grepped for
`INSERT`/`op.execute` — all of them are schema DDL, none seed row data, so
there's nothing for Alembic's revision-tracking guarantee to bypass).

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
