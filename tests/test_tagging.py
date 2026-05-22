"""
Test Suite: NLP Tagger
======================
Tests the 20-tag taxonomy defined in CLAUDE.md against known sample texts.

Run:
    pytest tests/test_tagging.py -v
"""

import pytest
from src.ingestion.nlp_processor import process_text, NLPResult, ALL_TAGS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def assert_tags(result: NLPResult, expected: list[str], forbidden: list[str] = None) -> None:
    """Assert that all expected tags are present and no forbidden tags appear."""
    for tag in expected:
        assert tag in result.tags, (
            f"Expected tag '{tag}' not found in {result.tags}"
        )
    if forbidden:
        for tag in forbidden:
            assert tag not in result.tags, (
                f"Forbidden tag '{tag}' unexpectedly found in {result.tags}"
            )


# ─── Sentiment tests ──────────────────────────────────────────────────────────

class TestSentiment:
    def test_positive_sentiment(self):
        r = process_text("This is absolutely fantastic! I love working with your team.")
        assert r.sentiment_label == "positive"
        assert r.sentiment_score > 0.0

    def test_negative_sentiment(self):
        r = process_text("I am extremely frustrated and very disappointed with this service.")
        assert r.sentiment_label == "negative"
        assert r.sentiment_score < 0.0

    def test_neutral_sentiment(self):
        r = process_text("Please send me the report for March.")
        assert r.sentiment_label == "neutral"

    def test_score_range(self):
        r = process_text("Some random text about affiliate marketing campaigns.")
        assert -1.0 <= r.sentiment_score <= 1.0


# ─── Churn signal ─────────────────────────────────────────────────────────────

class TestChurnSignal:
    def test_direct_cancellation(self):
        r = process_text(
            "I've decided I'm cancelling my account. This platform has been "
            "disappointing and I'm switching to a competitor."
        )
        assert_tags(r, ["churn_signal"])

    def test_churn_signal_requires_negative_sentiment(self):
        # "moving on" without negativity should NOT trigger churn_signal
        r = process_text(
            "We're moving on to the next phase of our campaign — really excited!"
        )
        assert "churn_signal" not in r.tags

    def test_closing_account_phrasing(self):
        r = process_text(
            "I'm done with this — I'll be closing my account at the end of the month. "
            "Nothing has worked for us here."
        )
        assert_tags(r, ["churn_signal"])


# ─── Growth intent ────────────────────────────────────────────────────────────

class TestGrowthIntent:
    def test_scale_up(self):
        r = process_text(
            "We want to scale up our campaigns significantly this quarter "
            "and expand into new markets."
        )
        assert_tags(r, ["growth_intent"])

    def test_double_revenue(self):
        r = process_text(
            "My goal is to double our revenue by Q4 with a bigger budget on PPC."
        )
        assert_tags(r, ["growth_intent"])

    def test_new_campaign(self):
        r = process_text(
            "I'm launching a new campaign next week targeting a different audience."
        )
        assert_tags(r, ["growth_intent"])


# ─── Payment issue ────────────────────────────────────────────────────────────

class TestPaymentIssue:
    def test_missing_commission(self):
        r = process_text(
            "My commission for March is still missing. The invoice was due two weeks ago."
        )
        assert_tags(r, ["payment_issue"])

    def test_delayed_payment(self):
        r = process_text(
            "Payment has been delayed again — this is the second month in a row "
            "my $1,200 has not arrived."
        )
        assert_tags(r, ["payment_issue"])

    def test_discrepancy(self):
        r = process_text(
            "There's a discrepancy in my payout — the amount is lower than my invoice."
        )
        assert_tags(r, ["payment_issue"])


# ─── Technical issue ──────────────────────────────────────────────────────────

class TestTechnicalIssue:
    def test_tracking_pixel(self):
        r = process_text(
            "My tracking pixel is not firing correctly on the checkout page. "
            "Something broke after the last update."
        )
        assert_tags(r, ["technical_issue"])

    def test_postback_failure(self):
        r = process_text(
            "The postback URL is returning errors and my conversion data "
            "is not recording properly."
        )
        assert_tags(r, ["technical_issue"])

    def test_api_issue(self):
        r = process_text("Your API is down — I can't pull any data right now.")
        assert_tags(r, ["technical_issue"])


# ─── Satisfaction high / low ──────────────────────────────────────────────────

class TestSatisfaction:
    def test_high_satisfaction_keywords(self):
        r = process_text(
            "I absolutely love working with your team — it's been a fantastic "
            "experience and the results have been amazing."
        )
        assert_tags(r, ["satisfaction_high"])

    def test_low_satisfaction_keywords(self):
        r = process_text(
            "I'm very disappointed with the service. The support has been "
            "terrible and I'm not happy at all."
        )
        assert_tags(r, ["satisfaction_low"])

    def test_high_satisfaction_via_sentiment(self):
        r = process_text(
            "Everything is going incredibly well. Best platform ever. Thrilled!"
        )
        assert r.sentiment_label == "positive"
        assert_tags(r, ["satisfaction_high"])


# ─── Competitor mention ───────────────────────────────────────────────────────

class TestCompetitorMention:
    def test_named_competitor(self):
        r = process_text(
            "I've been approached by ShareASale and they're offering better rates."
        )
        assert_tags(r, ["competitor_mention"])

    def test_impact_radius(self):
        r = process_text(
            "Impact Radius has reached out with an attractive commission structure."
        )
        assert_tags(r, ["competitor_mention"])

    def test_generic_competitor(self):
        r = process_text(
            "Your competitor seems to offer better analytics dashboards."
        )
        assert_tags(r, ["competitor_mention"])


# ─── Escalation risk ──────────────────────────────────────────────────────────

class TestEscalationRisk:
    def test_legal_threat(self):
        r = process_text(
            "If this isn't resolved I'll have no choice but to involve my lawyer "
            "and file a formal complaint. I'm extremely frustrated."
        )
        assert_tags(r, ["escalation_risk"])

    def test_escalation_without_negativity_excluded(self):
        # "manager" in a positive context should not trigger escalation_risk
        r = process_text(
            "I'll mention this to my manager — they'll be thrilled to hear about "
            "the great results we're getting."
        )
        assert "escalation_risk" not in r.tags


# ─── Pricing concern ──────────────────────────────────────────────────────────

class TestPricingConcern:
    def test_commission_rate(self):
        r = process_text(
            "The commission rate of 18% is too low for the volume I'm generating. "
            "Can we discuss better terms?"
        )
        assert_tags(r, ["pricing_concern"])

    def test_percentage_with_commission(self):
        r = process_text(
            "I'd like to request an increase to 30% commission on new conversions."
        )
        assert_tags(r, ["pricing_concern"])


# ─── Fraud risk ───────────────────────────────────────────────────────────────

class TestFraudRisk:
    def test_bot_traffic(self):
        r = process_text(
            "I'm seeing a lot of suspicious activity — possibly bot traffic "
            "or invalid clicks from an unknown source."
        )
        assert_tags(r, ["fraud_risk"])

    def test_refund_spike(self):
        r = process_text(
            "There's been a huge refund spike this month — 40% of conversions "
            "were reversed which is highly suspicious."
        )
        assert_tags(r, ["fraud_risk"])

    def test_ip_address_triggers_fraud_risk(self):
        r = process_text(
            "We detected unusual activity from IP address 192.168.1.100 "
            "with potential proxy usage."
        )
        assert_tags(r, ["fraud_risk"])


# ─── Compliance issue ─────────────────────────────────────────────────────────

class TestComplianceIssue:
    def test_tos_violation(self):
        r = process_text(
            "You have a potential policy violation on your account related to "
            "brand bidding — this is prohibited under our terms of service."
        )
        assert_tags(r, ["compliance_issue"])

    def test_trademark_issue(self):
        r = process_text(
            "Bidding on trademark keywords is not allowed and your account "
            "may be subject to a ban."
        )
        assert_tags(r, ["compliance_issue"])


# ─── Seasonal pattern ─────────────────────────────────────────────────────────

class TestSeasonalPattern:
    def test_bfcm(self):
        r = process_text(
            "We're planning a huge BFCM campaign — Black Friday and Cyber Monday "
            "are our biggest revenue days."
        )
        assert_tags(r, ["seasonal_pattern"])

    def test_q4(self):
        r = process_text(
            "Q4 is approaching and we want to start the holiday season preparation early."
        )
        assert_tags(r, ["seasonal_pattern"])


# ─── Urgency ─────────────────────────────────────────────────────────────────

class TestUrgency:
    def test_urgent_keyword(self):
        r = process_text(
            "This is URGENT — I need a response immediately. Please sort this ASAP."
        )
        assert_tags(r, ["urgency"])

    def test_multiple_exclamation_marks(self):
        r = process_text(
            "Please respond today! I have a deadline!! This cannot wait."
        )
        assert_tags(r, ["urgency"])


# ─── Question asked ───────────────────────────────────────────────────────────

class TestQuestionAsked:
    def test_question_mark(self):
        r = process_text("Can you send me the latest report?")
        assert_tags(r, ["question_asked"])

    def test_question_phrase_without_mark(self):
        r = process_text("I was wondering if you could help me with the tracking setup")
        assert_tags(r, ["question_asked"])

    def test_multiple_questions(self):
        r = process_text(
            "Do you know when payment will be processed? Is there a way to track it?"
        )
        assert_tags(r, ["question_asked"])


# ─── Relationship warm ────────────────────────────────────────────────────────

class TestRelationshipWarm:
    def test_appreciation(self):
        r = process_text(
            "Thanks so much for the quick response — I really appreciate your help. "
            "It's always a pleasure working with your team!"
        )
        assert_tags(r, ["relationship_warm"])

    def test_warm_without_positive_sentiment_excluded(self):
        r = process_text(
            "Thanks for nothing. I'm very disappointed and frustrated with "
            "how this has been handled."
        )
        # "thanks" present but negative sentiment should prevent relationship_warm
        assert "relationship_warm" not in r.tags


# ─── Multi-tag scenarios (end-to-end) ────────────────────────────────────────

class TestMultiTagScenarios:
    def test_churn_with_competitor_and_payment(self):
        """Mirrors email_002 from mock data."""
        text = (
            "I'm leaving — my payment has been missing for two months and "
            "ShareASale is offering me better rates. I've been very disappointed. "
            "If this isn't resolved I'll cancel my account immediately!"
        )
        r = process_text(text)
        assert_tags(r, ["churn_signal", "payment_issue", "competitor_mention", "urgency"])

    def test_growth_with_seasonal_and_pricing(self):
        """Mirrors email_001 / transcript_002 scenarios."""
        text = (
            "We want to scale up massively for BFCM and double our revenue by Q4. "
            "Can we increase our commission rate to 32% for the influencer channel?"
        )
        r = process_text(text)
        assert_tags(r, ["growth_intent", "seasonal_pattern", "pricing_concern", "question_asked"])

    def test_empty_text(self):
        r = process_text("")
        assert r.tags == []
        assert r.sentiment_score == 0.0
        assert r.sentiment_label == "neutral"

    def test_all_tags_are_valid(self):
        """Ensure every tag returned by process_text is in the known taxonomy."""
        texts = [
            "Cancel my account — switching to ShareASale immediately!!",
            "Scale up campaign, new opportunity, looking forward to BFCM.",
            "Payment missing, tracking pixel broken, please help?",
        ]
        for text in texts:
            r = process_text(text)
            for tag in r.tags:
                assert tag in ALL_TAGS, f"Unknown tag returned: '{tag}'"
