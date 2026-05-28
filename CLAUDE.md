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

> **Implementation file:** `src/storage/models.py` — this section must stay in sync with it.

### 3.1 Enum types

```sql
CREATE TYPE affiliate_status     AS ENUM ('active', 'at_risk', 'churned', 'high_growth');
CREATE TYPE communication_source AS ENUM ('email', 'call', 'api_event');
```

### 3.2 `affiliates`

```sql
CREATE TABLE affiliates (
    id                      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    name                    VARCHAR(255) NOT NULL,
    status                  affiliate_status NOT NULL DEFAULT 'active',

    -- ML model outputs (refreshed by scoring pipeline)
    churn_risk_score        FLOAT        NOT NULL DEFAULT 0.0,   -- 0.0–1.0
    growth_potential_score  FLOAT        NOT NULL DEFAULT 0.0,   -- 0.0–1.0
    health_score            FLOAT        NOT NULL DEFAULT 0.0,   -- 0–100

    -- Revenue & engagement
    revenue_30d             NUMERIC(10,2) NOT NULL DEFAULT 0.0,  -- last 30-day revenue
    ctr_trend_pct           FLOAT        NOT NULL DEFAULT 0.0,   -- % change in CTR

    -- Contact recency
    last_contact_at         TIMESTAMPTZ,
    days_since_contact      INTEGER      NOT NULL DEFAULT 0,     -- recomputed on every save

    -- Audit
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);
```

**Indexes:** `ix_affiliates_status (status)`, `ix_affiliates_health_score (health_score)`

> `days_since_contact` is recomputed automatically by a SQLAlchemy `before_insert` /
> `before_update` event listener — never set it manually.

### 3.3 `communications`

```sql
CREATE TABLE communications (
    id              UUID                 PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id    UUID                 NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    source          communication_source NOT NULL,   -- email | call | api_event
    raw_text        TEXT                 NOT NULL,
    tags            TEXT[]               NOT NULL DEFAULT '{}',  -- NLP tag array
    sentiment_score FLOAT                NOT NULL DEFAULT 0.0,   -- VADER compound: -1.0 to 1.0
    embedding_id    VARCHAR(255),                                 -- ChromaDB document ID
    occurred_at     TIMESTAMPTZ          NOT NULL
);
```

**Indexes:** `ix_communications_affiliate_id`, `ix_communications_occurred_at`, `ix_communications_source`

### 3.4 `score_history`

```sql
CREATE TABLE score_history (
    id                      UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    affiliate_id            UUID  NOT NULL REFERENCES affiliates(id) ON DELETE CASCADE,
    scored_at               DATE  NOT NULL DEFAULT CURRENT_DATE,
    churn_risk_score        FLOAT NOT NULL,
    growth_potential_score  FLOAT NOT NULL,
    health_score            FLOAT NOT NULL,
    UNIQUE (affiliate_id, scored_at)   -- one snapshot per affiliate per day
);
```

**Indexes:** `ix_score_history_affiliate_id`, `ix_score_history_scored_at`

---

## 4. ChromaDB Collection Structure

> **Implementation file:** `src/storage/vector_store.py`

### 4.1 `affiliate_comms`

Single collection — one document per `Communication` row.

| Field | Value |
|---|---|
| **collection** | `affiliate_comms` |
| **similarity** | cosine (`hnsw:space = cosine`) |
| **embedding model** | ChromaDB default ONNXMiniLM (all-MiniLM-L6-v2, 384 dims) — auto-applied on upsert |
| **document** | Full `raw_text` of the communication |
| **id** | `comm_{uuid}` — matches `Communication.embedding_id` |
| **metadata.affiliate_id** | UUID string of the parent `Affiliate` row |
| **metadata.affiliate_name** | Human-readable affiliate name |
| **metadata.source** | `email` \| `call` \| `api_event` |
| **metadata.tags** | Pipe-joined tag string, e.g. `"churn_signal\|urgency"` |
| **metadata.occurred_at** | ISO-8601 datetime string |

Used for: semantic search over past communications, RAG context for the agent.

**Key public functions** (`src/storage/vector_store.py`):
- `add_document(doc_id, text, affiliate_id, affiliate_name, source, tags, occurred_at)`
- `search_similar(query, n_results, filter_tags, filter_affiliate_id) → list[dict]`
- `get_by_affiliate(affiliate_id, limit) → list[dict]`
- `delete_by_affiliate(affiliate_id)`
- `health_check() → bool`

---

## 5. ML Model Architecture

### Churn Model (`churn_model.py`)
- **Algorithm**: XGBoostClassifier
- **Target**: `churn_risk_score` (binary label: churned within 90 days)
- **Key features**: days_since_contact, churn_signal_count, satisfaction_low_count,
  competitor_mention_count, payment_issue_count, escalation_risk_count,
  avg_sentiment_score (30d), comm_frequency_decline_ratio,
  revenue_30d, ctr_trend_pct, status_encoded

### Growth Model (`growth_model.py`)
- **Algorithm**: XGBoostClassifier
- **Target**: `growth_potential_score` (binary label: revenue grew ≥ 20% in 90 days)
- **Key features**: growth_intent_count, new_opportunity_count, satisfaction_high_count,
  high_engagement_count, seasonal_pattern_count, feature_request_count,
  avg_sentiment_score (30d), comm_frequency_growth_ratio,
  revenue_30d, ctr_trend_pct, relationship_warm_count, status_encoded

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

| Method | Path | Description |
|---|---|---|
| GET | `/affiliates` | List all affiliates with scores |
| GET | `/affiliates/{id}` | Single affiliate with full score history |
| GET | `/affiliates/{id}/communications` | Paginated communications |
| POST | `/affiliates/{id}/score` | Trigger re-scoring |
| POST | `/agent/chat` | Chat with the ReAct agent |
| POST | `/ingest/csv` | Upload affiliates CSV |
| GET | `/health` | Service health check |

---

## 8. Environment Variables

See `.env.example` for all required variables. Never commit `.env`.

---

## 9. Running Locally

```bash
docker-compose up -d          # start PostgreSQL + ChromaDB
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/ingestion/etl_pipeline.py   # seed mock data
uvicorn src.api.main:app --reload --port 8080  # start API on :8080
```

---

## 10. Git Conventions

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
