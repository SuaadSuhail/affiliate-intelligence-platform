"""
Agent Multi-Signal Regression Test
====================================
Deliberate exception to the "no real API call" convention noted in
test_agent.py — this specific regression can only be caught by actually
running the agent end-to-end, because the bug it guards against is a
prompt/tool-visibility gap (the agent choosing not to call a tool), not
something a mocked unit test can exercise.

Bug being guarded against (original): asked "which affiliates have multiple
warning signs", the agent previously called only get_portfolio_health and
get_affiliate_summary, never surfacing Marcus Williams or Rachel Torres —
both of whom have an active promo-code leak — because get_portfolio_health
had no leak/SEO visibility at all.

Bug being guarded against (current): once leak/SEO visibility and severity
ordering were added, asking the literal phrase "which affiliates have
multiple warning signs" turned out to be a genuinely ambiguous prompt for
the LLM — "multiple" is technically only true for Marcus Williams (the one
affiliate with 2+ signals), so the model would sometimes correctly answer
just Marcus and sometimes also include the single-signal note, flip-flopping
run to run (confirmed empirically: 2 of 4 real-LLM runs on that exact
phrasing omitted the single-signal names entirely — not a bug, a legitimately
ambiguous question). This test now asks "which affiliates need urgent
attention" instead — a question that isn't specifically about multiplicity,
so there's no such tension — and additionally asserts the response respects
severity ordering (Tom Bauer, at-risk tier, must be mentioned before Marcus
Williams, whose lower severity rank comes from weak growth rather than
elevated churn — see src.agent.tools.get_portfolio_health's
Combined Signal Groups section).

Requires: a real OPENAI_API_KEY (skips otherwise) and a live PostgreSQL
database seeded via POST /ingest/full + POST /leakage/scan (or the demo
seed) + POST /seo/scan (or the demo seed), so Marcus Williams and Rachel
Torres actually have has_active_leak=True in the DB.

Run:
    pytest tests/test_agent_multisignal.py -v
"""

from __future__ import annotations

import os
import re

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
    Real DB + real LLM call: asking "which affiliates need urgent attention"
    must surface every flagged affiliate — including single-signal ones
    (Tom Bauer, James O'Brien, Sarah Chen, Rachel Torres), not just the one
    multi-signal case (Marcus Williams) — and must respect severity ordering:
    Tom Bauer (at-risk tier, the most severe case in the seeded demo data)
    mentioned before Marcus Williams (lower severity — flagged by leak/SEO/
    weak growth, but churn itself is not elevated), even though Marcus has
    more signals overall.
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

    result = run_agent("Which affiliates need urgent attention?")
    response = result["response"]
    tools_used = result["tools_used"]

    for name in ("Tom Bauer", "James O'Brien", "Marcus Williams", "Sarah Chen", "Rachel Torres"):
        assert name in response, (
            f"Expected {name} in the response, got: {response!r}"
        )

    response_lower = response.lower()
    assert "leak" in response_lower, (
        f"Expected the leak signal to be mentioned, got: {response!r}"
    )

    # Severity must win over signal breadth: Tom Bauer (at-risk tier, the
    # most severe case) should read before Marcus Williams (lower severity
    # rank — leak/SEO/weak-growth, but churn itself isn't elevated) even
    # though Marcus has more signals overall. See get_portfolio_health's
    # Combined Signal Groups section — this is the exact ordering bug fixed
    # earlier in this session (breadth-over-severity), guarded against here
    # end-to-end through the real LLM rather than just the tool's own output.
    tom_bauer_pos = response.find("Tom Bauer")
    marcus_pos = response.find("Marcus Williams")
    assert tom_bauer_pos < marcus_pos, (
        f"Expected Tom Bauer (more severe) to be mentioned before Marcus "
        f"Williams (less severe, more signals) — severity should win over "
        f"signal count. Got: {response!r}"
    )

    # get_portfolio_health is the one tool call that gives the agent leak/SEO
    # visibility across the whole portfolio in a single call — confirm it was
    # actually used, not just get_affiliate_summary per affiliate.
    assert "get_portfolio_health" in tools_used, (
        f"Expected get_portfolio_health to be called for a portfolio-wide "
        f"warning-signs question, tools used: {tools_used}"
    )


@pytest.mark.skipif(
    not _real_openai_key_configured(),
    reason="OPENAI_API_KEY not configured — this test makes a real LLM call",
)
def test_agent_does_not_fabricate_names_for_full_list_request():
    """
    Real DB + real LLM call: asked for "the full list" after a capped
    "performing well" summary (3 of 5 shown), the agent previously
    fabricated two entirely fictional affiliates — "Liam Johnson" and
    "Emma Wilson", complete with plausible-looking but fake health/growth
    scores — to pad the list out to 5, instead of retrieving the real
    remaining two (Sarah Chen, Fatima Al-Hassan) via a tool call
    (get_portfolio_health with input_str="full", added specifically to
    close this gap — see its docstring and SYSTEM_PROMPT's GROUNDING
    section). This is a stronger check than "the right names are present":
    it asserts every bolded name-shaped token in the response is a real
    affiliate from the database, so an extra invented name would fail this
    test even if all the real ones were also correctly included.
    """
    from src.rulebook.recommend import categorize
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate

    db = SessionLocal()
    try:
        affiliates = db.query(Affiliate).all()
        if not affiliates:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")
        real_names = {a.name for a in affiliates}
        performing_well_names = {
            a.name
            for a in affiliates
            if (a.health_score or 50.0) >= 60.0
            and categorize(a.churn_risk_score or 0.0, a.growth_potential_score or 0.0)
            not in ("at_risk", "churned")
        }
        if len(performing_well_names) < 4:
            pytest.skip(
                "Fewer than 4 affiliates performing well in the current seed — "
                "need a genuinely capped (top-3-of-N, N>3) list to reproduce "
                "the scenario that caused fabrication"
            )
    finally:
        db.close()

    from src.agent.agent import run_agent

    # Recreate the exact scenario: a prior capped "top 3 of 5" turn, then a
    # follow-up asking for the rest — this is what triggered the fabrication.
    history = [
        {"role": "user", "content": "Which affiliates are performing well?"},
        {
            "role": "assistant",
            "content": (
                "Top 3 of 5 performing well: "
                + ", ".join(sorted(performing_well_names)[:3])
                + "."
            ),
        },
    ]
    result = run_agent("Give me the full list", conversation_history=history)
    response = result["response"]

    # Every bolded "**Name**" token the model used as a name must be a real
    # affiliate — anything else is fabrication, full stop.
    claimed_names = set(re.findall(r"\*\*([A-Z][a-zA-Z'\- ]+?)\*\*", response))
    fabricated = claimed_names - real_names
    assert not fabricated, (
        f"Agent stated affiliate name(s) not present in the database — "
        f"fabricated: {fabricated}. Full response: {response!r}"
    )

    for name in performing_well_names:
        assert name in response, (
            f"Expected real performing-well affiliate {name!r} in the full-list "
            f"response, got: {response!r}"
        )
