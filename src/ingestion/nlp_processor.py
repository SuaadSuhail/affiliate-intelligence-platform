"""
NLP Processor
=============
Reads all untagged communications from PostgreSQL, runs spaCy on each one,
calculates sentiment using a custom SENTIMENT_LEXICON, applies 20 tag
detection rules, and writes the tags and sentiment score back to the database.

Usage
-----
    from src.ingestion.nlp_processor import process_all_communications
    from src.storage.database import db_session

    with db_session() as db:
        result = process_all_communications(db)
        print(result)
"""

from __future__ import annotations

import re
from typing import Optional

import spacy
from sqlalchemy.orm import Session

from src.storage.models import Affiliate, Communication

# ─── spaCy: load once at module level ────────────────────────────────────────

try:
    _nlp: spacy.Language = spacy.load("en_core_web_sm")
except OSError as exc:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    ) from exc

# ─── Sentiment Lexicon ───────────────────────────────────────────────────────

SENTIMENT_LEXICON: dict[str, float] = {
    # Negative signals (-0.3 to -0.9)
    "frustrated": -0.7,
    "disappointment": -0.6,
    "slow": -0.3,
    "unacceptable": -0.8,
    "complaint": -0.6,
    "leaving": -0.7,
    "switching": -0.6,
    "cancelling": -0.8,
    "stopping": -0.5,
    "unhappy": -0.7,
    "concerned": -0.4,
    "problem": -0.4,
    "issue": -0.4,
    "overdue": -0.5,
    "unanswered": -0.5,
    "silence": -0.3,
    "delay": -0.4,
    "unclear": -0.3,
    "confused": -0.4,
    "escalate": -0.6,
    # Positive signals (+0.3 to +0.9)
    "excited": 0.8,
    "excellent": 0.8,
    "thrilled": 0.9,
    "love": 0.7,
    "great": 0.6,
    "fantastic": 0.8,
    "growing": 0.5,
    "expanding": 0.5,
    "launch": 0.5,
    "ready": 0.4,
    "committed": 0.5,
    "interested": 0.4,
    "keen": 0.5,
    "positive": 0.5,
    "progress": 0.5,
    "successful": 0.7,
    "happy": 0.7,
    "pleased": 0.6,
    "looking forward": 0.6,
    "opportunity": 0.5,
}

# Competitor ORGs for spaCy NER cross-check
_COMPETITOR_NAMES: set[str] = {
    "awin", "rakuten", "impact", "cj affiliate",
    "partnerize", "tradedoubler", "webgains",
}

# ─── Core functions ───────────────────────────────────────────────────────────

def calculate_sentiment(text: str) -> float:
    """
    Score text using SENTIMENT_LEXICON.

    Averages the scores of all lexicon words/phrases found (case-insensitive).
    Returns 0.0 if no lexicon words are found. Result clamped to [-1.0, +1.0].
    """
    if not text or not text.strip():
        return 0.0

    lower = text.lower()
    found_scores: list[float] = []

    for word, score in SENTIMENT_LEXICON.items():
        if word in lower:
            found_scores.append(score)

    if not found_scores:
        return 0.0

    return max(-1.0, min(1.0, sum(found_scores) / len(found_scores)))


def detect_tags(
    doc: spacy.tokens.Doc,
    sentiment_score: float,
    text_lower: str,
    source: str,
    affiliate_id,
    db: Session,
) -> list[str]:
    """
    Apply all 20 tag rules to a parsed communication.

    Parameters
    ----------
    doc            : spaCy Doc object
    sentiment_score: float from calculate_sentiment()
    text_lower     : lowercased raw text
    source         : communication channel (email | call | api_event)
    affiliate_id   : UUID of the owning Affiliate
    db             : active SQLAlchemy session (for DB lookups)

    Returns
    -------
    Deduplicated list of matched tag strings.
    """
    tags: set[str] = set()

    # ── Lookup affiliate for engagement-based tags ────────────────────────────
    aff: Optional[Affiliate] = db.query(Affiliate).filter(
        Affiliate.id == affiliate_id
    ).first()
    days_since = aff.days_since_contact if aff else 0

    # ════════════════════════════════════════════════════════════════
    # ENGAGEMENT GROUP
    # ════════════════════════════════════════════════════════════════

    # 1. responsive: email + positive sentiment
    if source == "email" and sentiment_score > 0.1:
        tags.add("responsive")

    # 2. proactive_outreach
    _proactive_kws = [
        "just wanted to reach out", "checking in",
        "wanted to share", "thought you'd like to know",
    ]
    if any(kw in text_lower for kw in _proactive_kws):
        tags.add("proactive_outreach")

    # 3. campaign_active
    _campaign_kws = [
        "live", "launched", "running", "went live",
        "campaign is active", "pushing the campaign",
    ]
    if any(kw in text_lower for kw in _campaign_kws):
        tags.add("campaign_active")

    # 4. unresponsive: affiliate hasn't been contacted in > 5 days
    if days_since > 5:
        tags.add("unresponsive")

    # 5. disengaged_tone
    _disengaged_kws = [
        "slow", "quiet", "not much happening",
        "haven't been able", "things are slow", "been quiet",
    ]
    if any(kw in text_lower for kw in _disengaged_kws) and sentiment_score < -0.2:
        tags.add("disengaged_tone")

    # 6. gone_silent: 14+ days since contact
    if days_since > 14:
        tags.add("gone_silent")

    # ════════════════════════════════════════════════════════════════
    # SENTIMENT GROUP
    # ════════════════════════════════════════════════════════════════

    # 7. positive_sentiment
    if sentiment_score > 0.3:
        tags.add("positive_sentiment")

    # 8. enthusiastic
    _enthusiastic_kws = ["excited", "thrilled", "can't wait", "love this", "amazing"]
    if sentiment_score > 0.6 or any(kw in text_lower for kw in _enthusiastic_kws):
        tags.add("enthusiastic")

    # 9. neutral_sentiment
    if -0.2 <= sentiment_score <= 0.3:
        tags.add("neutral_sentiment")

    # 10. frustrated
    _frustrated_kws = [
        "frustrated", "disappointing", "not working",
        "let down", "expected better",
    ]
    if sentiment_score < -0.4 or any(kw in text_lower for kw in _frustrated_kws):
        tags.add("frustrated")

    # 11. complaint
    _complaint_kws = [
        "complaint", "unacceptable", "not acceptable",
        "raising a complaint", "formally complain",
    ]
    if any(kw in text_lower for kw in _complaint_kws):
        tags.add("complaint")

    # ════════════════════════════════════════════════════════════════
    # INTENT GROUP
    # ════════════════════════════════════════════════════════════════

    # 12. upsell_signal
    _upsell_kws = ["new product", "can we add", "interested in", "another brand", "additional"]
    if any(kw in text_lower for kw in _upsell_kws):
        tags.add("upsell_signal")

    # 13. expansion_interest
    _expansion_kws = ["scale", "grow", "more volume", "increase", "bigger", "expand"]
    if any(kw in text_lower for kw in _expansion_kws):
        tags.add("expansion_interest")

    # 14. new_campaign_intent
    _new_campaign_kws = [
        "new campaign", "launch", "plan to run",
        "ready to start", "want to start",
    ]
    if any(kw in text_lower for kw in _new_campaign_kws):
        tags.add("new_campaign_intent")

    # 15. churn_signal
    _churn_kws = [
        "leaving", "switching", "cancelling", "stopping",
        "moving to", "other platform", "looking elsewhere",
    ]
    if any(kw in text_lower for kw in _churn_kws):
        tags.add("churn_signal")

    # 16. competitor_mention: spaCy NER ORG entities matched against known list
    org_entities = {ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"}
    if org_entities & _COMPETITOR_NAMES:
        tags.add("competitor_mention")
    # Also keyword fallback for common names not picked up by NER
    if any(comp in text_lower for comp in _COMPETITOR_NAMES):
        tags.add("competitor_mention")

    # 17. stalled_deal
    _stalled_kws = [
        "still waiting", "no update", "heard nothing",
        "no response", "chasing",
    ]
    if any(kw in text_lower for kw in _stalled_kws) and sentiment_score < 0.0:
        tags.add("stalled_deal")

    # ════════════════════════════════════════════════════════════════
    # RELATIONSHIP GROUP
    # ════════════════════════════════════════════════════════════════

    # 18. escalation
    _escalation_kws = [
        "escalate", "speak to manager", "your manager",
        "senior", "urgent", "asap", "immediately",
    ]
    if (
        any(kw in text_lower for kw in _escalation_kws)
        or ("frustrated" in tags and "complaint" in tags)
    ):
        tags.add("escalation")

    # 19. follow_up_needed
    _followup_kws = [
        "let me know", "waiting to hear", "please confirm",
        "can you", "could you", "please check", "get back to me",
    ]
    if any(kw in text_lower for kw in _followup_kws):
        tags.add("follow_up_needed")

    # 20. action_committed
    _action_kws = [
        "i will", "we will", "will send", "will do",
        "by end of", "done by", "will have it",
    ]
    if any(kw in text_lower for kw in _action_kws):
        tags.add("action_committed")

    # 21. question_asked: any sentence ends with "?"
    if any(str(sent).strip().endswith("?") for sent in doc.sents):
        tags.add("question_asked")

    return list(tags)


def process_single_communication(
    comm: Communication,
    db: Session,
) -> dict:
    """
    Run the full NLP pipeline on one Communication record.

    Steps
    -----
    1. Parse raw_text with spaCy
    2. Calculate sentiment score via SENTIMENT_LEXICON
    3. Detect all applicable tags
    4. Persist tags + sentiment_score back to the DB row
    5. Return summary dict

    Parameters
    ----------
    comm : Communication ORM instance (must be in the current db session)
    db   : active SQLAlchemy session

    Returns
    -------
    dict with keys: id, tags_applied, sentiment_score
    """
    text = comm.raw_text or ""
    doc = _nlp(text)
    sentiment_score = calculate_sentiment(text)
    text_lower = text.lower()

    tags = detect_tags(
        doc=doc,
        sentiment_score=sentiment_score,
        text_lower=text_lower,
        source=comm.source,
        affiliate_id=comm.affiliate_id,
        db=db,
    )

    comm.tags = tags
    comm.sentiment_score = round(sentiment_score, 4)

    return {
        "id": str(comm.id),
        "tags_applied": tags,
        "sentiment_score": comm.sentiment_score,
    }


def process_all_communications(db: Session) -> dict:
    """
    Process every communication that has an empty tags array.

    Calls process_single_communication() for each untagged record,
    then commits the session once all records are updated.

    Parameters
    ----------
    db : active SQLAlchemy session (caller owns commit/rollback)

    Returns
    -------
    {
        total_processed : int,
        total_tagged    : int,   # records that received ≥ 1 tag
        tag_summary     : {tag_name: count}
    }
    """
    untagged = (
        db.query(Communication)
        .filter(Communication.tags == [])
        .all()
    )

    total_processed = 0
    total_tagged = 0
    tag_summary: dict[str, int] = {}

    for comm in untagged:
        result = process_single_communication(comm, db)
        total_processed += 1
        if result["tags_applied"]:
            total_tagged += 1
        for tag in result["tags_applied"]:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1

    return {
        "total_processed": total_processed,
        "total_tagged": total_tagged,
        "tag_summary": tag_summary,
    }