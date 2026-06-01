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

| Method | Path | Tag | Description |
|---|---|---|---|
| GET | `/health` | System | Service health check |
| GET | `/affiliates` | Affiliates | List all affiliates with scores (filterable by `status`) |
| GET | `/affiliates/{id}` | Affiliates | Single affiliate with full score history |
| GET | `/affiliates/{id}/communications` | Affiliates | Paginated communications for one affiliate |
| POST | `/affiliates/{id}/score` | Scoring | Trigger re-scoring for a single affiliate |
| POST | `/agent/chat` | Agent | Chat with the LangChain ReAct agent |
| POST | `/process/nlp` | NLP | Run NLP tagging on all untagged communications |
| GET | `/communications` | Communications | List all communications with tags + sentiment |
| GET | `/communications/{id}` | Communications | Single communication by UUID |

---

## 8. Environment Variables

See `.env.example` for all required variables. Never commit `.env`.

---

## 9. Running Locally

```bash
docker-compose up -d          # start PostgreSQL + ChromaDB
conda activate affiliate-intelligence   # or: pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/ingestion/etl_pipeline.py   # seed mock data (10 affiliates, 7 comms)
uvicorn src.api.main:app --port 8080 --reload  # start API on :8080
curl -X POST http://localhost:8080/process/nlp # tag all communications
```

---

## 10. Infrastructure

### Docker services

| Service | Image | Host port | Notes |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `5432` | healthcheck: `pg_isready` |
| `chromadb` | `chromadb/chroma:latest` | `8001` | container internal port 8000; NO auth; NO healthcheck |
| `app` | custom build | `8080` | depends on postgres (healthy) + chromadb (started) |

**chromadb environment (only these two):**
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

### Known fixes applied

- **chromadb 1.5.9 does not support token auth** — never add `CHROMA_SERVER_AUTH_*`
  variables to the chromadb service. The `chromadb.auth.token` module does not exist
  in chromadb ≥ 1.0 and causes the container to fail on startup.
- **chromadb healthcheck removed** — the container starts successfully but fails the
  `curl /api/v1/heartbeat` check (v1 API path is gone in chromadb ≥ 1.0). No
  healthcheck is defined; app uses `service_started` instead of `service_healthy`.
- **Port conflict resolved** — app runs on `8080`, chromadb is exposed on host port
  `8001` (maps to container port `8000`). Never use `8000` for the app.
- **`version:` attribute removed** — the top-level `version: "3.9"` key is obsolete
  in Docker Compose v2 and causes a warning; omit it entirely.

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
aip_chromadb    Up              0.0.0.0:8001->8000/tcp
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
| **Files** | `src/storage/database.py`, `src/storage/models.py`, `src/storage/vector_store.py` |

**What it does:**
- `database.py` — creates the SQLAlchemy engine, `SessionLocal` factory, and FastAPI
  `get_db()` dependency; exposes `db_session()` context manager for scripts
- `models.py` — ORM models for `Affiliate`, `Communication`, `ScoreHistory` that match
  the live DB schema (see § 3); uses `ARRAY(String)` for `tags`, `Enum` types for
  `status` / `source`
- `vector_store.py` — thin ChromaDB wrapper; manages `communications_embeddings` and
  `affiliate_profiles` collections with cosine similarity; lazy HTTP client connection

**Key functions:** `get_db()`, `db_session()`, `init_db()`, `health_check()`,
`VectorStore.upsert_communication()`, `VectorStore.search_communications()`,
`VectorStore.upsert_affiliate_profile()`, `VectorStore.find_similar_affiliates()`

**Infrastructure fix:** Removed `chromadb.auth.token.TokenAuthClientProvider` from
`_get_client()` — that module does not exist in chromadb ≥ 1.0. Client now connects
without auth settings (token auth is configured at the ChromaDB server level via env).

---

### ETL pipeline

| | |
|---|---|
| **Status** | Complete |
| **File** | `src/ingestion/etl_pipeline.py` |

**What it does:** Reads the mock CSV and flat-text communication files from
`data/mock/` and loads them into PostgreSQL with full upsert logic:
- `run_affiliate_etl(db)` — reads `affiliates.csv` via Pandas; upserts by `name`;
  parses `last_contact_at`, computes `days_since_contact`
- `run_communications_etl(db)` — parses `[AFFILIATE: Name]` / `[DATE: ...]` header
  blocks from `emails.txt` and `transcripts.txt`; links to affiliate by name;
  upserts by `(affiliate_id, occurred_at)`; leaves `tags = []` and `embedding_id = None`
- `run_full_etl(db)` — calls both jobs in a single transaction; commits on success,
  rolls back on any failure

**API endpoints:** `POST /ingest/full`, `POST /ingest/affiliates`,
`POST /ingest/communications`

**Output:** 10 affiliates + 7 communications in PostgreSQL after a clean run

**Idempotent:** Re-running updates existing rows rather than duplicating them

**Depends on:** Storage layer (must call `init_db()` before first run)

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

**API endpoints:** `POST /process/nlp`, `GET /communications`,
`GET /communications/{id}`

**Depends on:** Storage layer (models + DB session), ETL pipeline must run first to
populate communications

**Output:** `tags[]` and `sentiment_score` written to every `communications` row;
7/7 communications tagged in the mock dataset
