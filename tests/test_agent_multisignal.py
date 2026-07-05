"""
Agent Multi-Signal Regression Test
====================================
Deliberate exception to the "no real API call" convention noted in
test_agent.py — this specific regression can only be caught by actually
running the agent end-to-end, because the bug it guards against is a
prompt/tool-visibility gap (the agent choosing not to call a tool), not
something a mocked unit test can exercise.

Bug being guarded against: asked "which affiliates have multiple warning
signs", the agent previously called only get_portfolio_health and
get_affiliate_summary, never surfacing Marcus Williams or Rachel Torres —
both of whom have an active promo-code leak (Marcus also has a declining
SEO trend and at_risk churn status: the strongest multi-signal case in the
seeded demo data) — because get_portfolio_health had no leak/SEO visibility
at all, and the prompt never told the agent "warning signs" spans all three
signal types this system tracks.

Requires: a real OPENAI_API_KEY (skips otherwise) and a live PostgreSQL
database seeded via POST /ingest/full + POST /leakage/scan (or the demo
seed) + POST /seo/scan (or the demo seed), so Marcus Williams and Rachel
Torres actually have has_active_leak=True in the DB.

Run:
    pytest tests/test_agent_multisignal.py -v
"""

from __future__ import annotations

import os

import pytest


def _real_openai_key_configured() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and key != "placeholder"


@pytest.mark.skipif(
    not _real_openai_key_configured(),
    reason="OPENAI_API_KEY not configured — this test makes a real LLM call",
)
def test_agent_surfaces_multi_signal_affiliates_for_warning_signs_question():
    """
    Real DB + real LLM call: asking "which affiliates have multiple warning
    signs" must surface Marcus Williams and Rachel Torres (both have an
    active leak in the seeded demo data) with their leak status mentioned,
    not just churn/growth numbers for unrelated affiliates.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate

    db = SessionLocal()
    try:
        marcus = db.query(Affiliate).filter(Affiliate.name == "Marcus Williams").first()
        rachel = db.query(Affiliate).filter(Affiliate.name == "Rachel Torres").first()
        if marcus is None or rachel is None:
            pytest.skip("Marcus Williams / Rachel Torres not in DB — run POST /ingest/full first")
        if not (marcus.has_active_leak and rachel.has_active_leak):
            pytest.skip(
                "Demo leak seed not present (has_active_leak False) — run "
                "seed_demo_leak_scan / POST /leakage/scan first"
            )
    finally:
        db.close()

    from src.agent.agent import run_agent

    result = run_agent("Which affiliates have multiple warning signs?")
    response = result["response"]
    tools_used = result["tools_used"]

    assert "Marcus Williams" in response, (
        f"Expected Marcus Williams (leak + declining SEO + at_risk) in the "
        f"response, got: {response!r}"
    )
    assert "Rachel Torres" in response, (
        f"Expected Rachel Torres (active leak) in the response, got: {response!r}"
    )

    response_lower = response.lower()
    assert "leak" in response_lower, (
        f"Expected the leak signal to be mentioned, got: {response!r}"
    )

    # get_portfolio_health is the one tool call that gives the agent leak/SEO
    # visibility across the whole portfolio in a single call — confirm it was
    # actually used, not just get_affiliate_summary per affiliate.
    assert "get_portfolio_health" in tools_used, (
        f"Expected get_portfolio_health to be called for a portfolio-wide "
        f"warning-signs question, tools used: {tools_used}"
    )
