"""
NLP Processor
=============
Applies the 20-tag taxonomy defined in CLAUDE.md to a piece of text and
returns a sentiment score + list of matched tags.

Detection methods used per tag (see CLAUDE.md § 2 for full rules):
  KW   — keyword/phrase matching
  RE   — regex pattern
  SENT — VADER sentiment threshold
  ML   — spaCy rule-based (used for entity-level detection)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Lazy-load spaCy model (en_core_web_sm must be downloaded) ─────────────────
_nlp: Optional[spacy.Language] = None
_vader = SentimentIntensityAnalyzer()

ALL_TAGS: list[str] = [
    "churn_signal",
    "growth_intent",
    "payment_issue",
    "technical_issue",
    "satisfaction_high",
    "satisfaction_low",
    "competitor_mention",
    "escalation_risk",
    "support_request",
    "feature_request",
    "pricing_concern",
    "fraud_risk",
    "high_engagement",
    "low_engagement",
    "compliance_issue",
    "new_opportunity",
    "seasonal_pattern",
    "relationship_warm",
    "urgency",
    "question_asked",
]


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[nlp_processor] en_core_web_sm not found; running: python -m spacy download en_core_web_sm")
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ─── Keyword / phrase lists ────────────────────────────────────────────────────

_KW = {
    "churn_signal": [
        "cancel", "cancellation", "leaving", "switching", "done with",
        "closing account", "move on", "not working for us", "going elsewhere",
        "move my campaigns", "close my account",
    ],
    "growth_intent": [
        "scale", "expand", "ramp up", "increase volume", "new campaign",
        "bigger budget", "double", "grow", "opportunity", "triple", "scale up",
    ],
    "payment_issue": [
        "payment", "commission", "invoice", "not received", "missing",
        "delayed", "discrepancy", "outstanding", "unpaid", "overdue",
    ],
    "technical_issue": [
        "bug", "broken", "error", "not working", "tracking issue", "pixel",
        "postback", "api down", "integration fail", "firing inconsistently",
        "tracking system", "conversion data",
    ],
    "satisfaction_high": [
        "love", "excellent", "amazing", "best", "great work", "fantastic",
        "thrilled", "absolutely", "phenomenal", "incredible",
    ],
    "satisfaction_low": [
        "disappointed", "frustrated", "unhappy", "poor", "terrible",
        "waste of time", "not happy", "not great", "useless", "not satisfied",
    ],
    "competitor_mention": [
        "commission junction", "shareasale", "impact", "impact radius", "awin",
        "rakuten", "partnerstack", "partnerize", "competitor", "other network",
        "your rival", "another network",
    ],
    "escalation_risk": [
        "manager", "escalate", "legal", "lawyer", "report", "complain", "bbb",
        "sue", "formal complaint", "executive",
    ],
    "support_request": [
        "help", "how do i", "can you assist", "need support", "ticket",
        "not sure how", "please advise", "guidance", "question",
    ],
    "feature_request": [
        "feature", "wish", "would be great if", "can you add", "missing functionality",
        "request", "suggestion", "enhancement", "would love to see",
    ],
    "pricing_concern": [
        "commission rate", "payout", "low rate", "lower than", "not worth it",
        "increase my rate", "better terms", "competitive rate", "better rate",
    ],
    "fraud_risk": [
        "fake", "bot traffic", "proxy", "vpn", "refund spike", "chargeback",
        "suspicious", "invalid clicks", "bot",
    ],
    "compliance_issue": [
        "policy violation", "terms of service", "tos", "prohibited",
        "not allowed", "ban", "trademark", "brand bidding",
    ],
    "new_opportunity": [
        "new traffic source", "new channel", "partnership", "collab",
        "new audience", "influencer", "podcast", "newsletter",
        "social media push", "new platform", "untapped",
    ],
    "seasonal_pattern": [
        "black friday", "cyber monday", "q4", "holiday season", "summer sale",
        "back to school", "christmas", "bfcm", "q3", "seasonal",
    ],
    "relationship_warm": [
        "thanks", "appreciate", "great chatting", "always a pleasure",
        "looking forward", "enjoyed our call", "cheers", "really value",
        "means a lot",
    ],
    "urgency": [
        "urgent", "asap", "immediately", "right away", "by end of day",
        "today", "deadline", "right now",
    ],
}

# ─── Regex patterns ────────────────────────────────────────────────────────────

_RE_AMOUNT = re.compile(r"\$[\d,]+")
_RE_PERCENT = re.compile(r"\d+%")
_RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_RE_EXCLAIM = re.compile(r"!")
_RE_QUESTION_SENTENCE = re.compile(r"\?")
_RE_QUESTION_PHRASE = re.compile(
    r"\b(can you|could you|would you|do you know|is there a way)\b",
    re.IGNORECASE,
)


# ─── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class NLPResult:
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    tags: list[str] = field(default_factory=list)


# ─── Main processor ────────────────────────────────────────────────────────────

def process_text(text: str) -> NLPResult:
    """
    Run NLP pipeline on a single piece of text.

    Returns an NLPResult with:
      - sentiment_score  : VADER compound (-1.0 to 1.0)
      - sentiment_label  : positive | neutral | negative
      - tags             : list of matched tags (may be empty)
    """
    if not text or not text.strip():
        return NLPResult()

    lower = text.lower()
    result = NLPResult()

    # ── 1. Sentiment (VADER) ──────────────────────────────────────────────────
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    result.sentiment_score = round(compound, 4)
    if compound >= 0.05:
        result.sentiment_label = "positive"
    elif compound <= -0.05:
        result.sentiment_label = "negative"
    else:
        result.sentiment_label = "neutral"

    tags: set[str] = set()

    # ── 2. Keyword matching ───────────────────────────────────────────────────
    for tag, phrases in _KW.items():
        if any(phrase in lower for phrase in phrases):
            tags.add(tag)

    # ── 3. Churn signal — also requires negative sentiment ────────────────────
    if "churn_signal" in tags and compound > -0.2:
        tags.discard("churn_signal")

    # ── 4. Satisfaction high / low — sentiment override ───────────────────────
    if compound >= 0.5:
        tags.add("satisfaction_high")
    if compound <= -0.3:
        tags.add("satisfaction_low")

    # ── 5. Escalation risk — needs negative context ───────────────────────────
    if "escalation_risk" in tags and compound > 0.0:
        tags.discard("escalation_risk")

    # ── 6. Payment issue — boost with amount regex ────────────────────────────
    if "payment_issue" in tags and not _RE_AMOUNT.search(text):
        # keep tag even without amount — keyword match is sufficient
        pass

    # ── 7. Pricing concern — boost with percentage regex ──────────────────────
    if _RE_PERCENT.search(text) and "commission" in lower:
        tags.add("pricing_concern")

    # ── 8. Fraud risk — boost with IP address regex ───────────────────────────
    if _RE_IP.search(text):
        tags.add("fraud_risk")

    # ── 9. Urgency — also flag multiple exclamation marks ─────────────────────
    if len(_RE_EXCLAIM.findall(text)) >= 2:
        tags.add("urgency")

    # ── 10. Question asked ────────────────────────────────────────────────────
    if _RE_QUESTION_SENTENCE.search(text) or _RE_QUESTION_PHRASE.search(text):
        tags.add("question_asked")

    # ── 11. Relationship warm — positive sentiment required ───────────────────
    if "relationship_warm" in tags and compound < 0.3:
        tags.discard("relationship_warm")

    result.tags = sorted(tags)
    return result


def tag_counts(tags_list: list[list[str]]) -> dict[str, int]:
    """
    Given a list of tag arrays (one per communication), return counts per tag.

    Useful in feature_engineering.py.
    """
    counts: dict[str, int] = {t: 0 for t in ALL_TAGS}
    for tags in tags_list:
        for tag in tags:
            if tag in counts:
                counts[tag] += 1
    return counts


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I'm cancelling my account. I've been disappointed with everything.",
        "We want to scale up massively for BFCM — can you increase our commission rate to 28%?",
        "My payment is missing again! This is the third month. I'll have to escalate to legal.",
        "Thanks so much, I really appreciate the support — always a pleasure working with your team!",
    ]
    for s in samples:
        r = process_text(s)
        print(f"\nText : {s[:70]}...")
        print(f"Sent : {r.sentiment_score} ({r.sentiment_label})")
        print(f"Tags : {r.tags}")
