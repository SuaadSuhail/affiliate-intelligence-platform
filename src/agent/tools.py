"""
LangChain Tool Definitions
==========================
Five tools for the ReAct agent. Each docstring is used by LangChain
to decide when to call the tool — keep them descriptive.

Tools
-----
1. query_database        — raw SQL SELECT against PostgreSQL
2. semantic_search       — ChromaDB embedding search over communications
3. get_affiliate_summary — full profile for one affiliate
4. draft_email           — LLM-generated personalised email draft
5. get_portfolio_health  — whole-portfolio aggregate stats
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from langchain.tools import tool
from sqlalchemy import text

from src.storage.database import SessionLocal
from src.storage.models import Affiliate, Communication, ScoreHistory
from src.storage.vector_store import vector_store

def _get_db():
    """Return a fresh SessionLocal for each tool call."""
    return SessionLocal()


# Lazy LLM for draft_email (avoids import error when key not set)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENAI_API_KEY", "placeholder")
        if not api_key or api_key == "placeholder":
            return None
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


# ─── Tool 1: query_database ───────────────────────────────────────────────────

@tool
def query_database(sql_query: str) -> str:
    """Query the PostgreSQL database with a SELECT statement to get affiliate
    scores, health metrics, communication counts, or score history.
    Use this for precise filtered queries on structured data.
    Only SELECT statements are allowed.
    Example: SELECT name, health_score, churn_risk_score
    FROM affiliates ORDER BY health_score ASC LIMIT 5"""
    sql = sql_query.strip()
    if not sql.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are permitted.")

    db = _get_db()
    try:
        result = db.execute(text(sql))
        rows = result.fetchmany(20)
        if not rows:
            return "Query returned no rows."
        cols = list(result.keys())
        lines = [" | ".join(cols)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
        return "\n".join(lines)
    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        db.close()


# ─── Tool 2: semantic_search ──────────────────────────────────────────────────

@tool
def semantic_search(query: str) -> str:
    """Search through affiliate emails and call transcripts by meaning.
    Use this to find relevant communications without needing exact keywords.
    Input should be a natural language description of what you are looking for.
    Example: 'affiliate expressing frustration about platform performance'"""
    try:
        from src.ingestion.embedding_generator import model as embed_model
        embedding = embed_model.encode(query).tolist()
    except Exception as exc:
        return f"Embedding error: {exc}"

    try:
        results = vector_store.search_similar(embedding, n_results=5)
    except Exception as exc:
        return f"Search error: {exc}"

    if not results:
        return "No matching communications found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text_snippet = r.get("text", r.get("document", ""))[:300]
        score = round(1 - r.get("distance", 1.0), 3)
        tags_str = meta.get("tags", "").strip("|").replace("|", ", ")
        lines.append(
            f"[{i}] Affiliate: {meta.get('affiliate_name', meta.get('affiliate_id', '?'))} "
            f"| Source: {meta.get('source', '?')} "
            f"| Similarity: {score:.3f}\n"
            f"    Tags: {tags_str or 'none'}\n"
            f"    \"{text_snippet}…\""
        )
    return "\n\n".join(lines)


# ─── Tool 3: get_affiliate_summary ────────────────────────────────────────────

@tool
def get_affiliate_summary(affiliate_name: str) -> str:
    """Get a complete profile for one affiliate including their health score,
    churn risk, growth potential, SHAP explanation of risk factors, recent
    communication tags, and days since last contact.
    Use this when you need a full picture of one specific affiliate."""
    db = _get_db()
    try:
        aff = (
            db.query(Affiliate)
            .filter(Affiliate.name.ilike(f"%{affiliate_name.strip()}%"))
            .first()
        )
        if not aff:
            return f"Affiliate not found: '{affiliate_name}'. Check the name and try again."

        recent_comms = (
            db.query(Communication)
            .filter(Communication.affiliate_id == aff.id)
            .order_by(Communication.occurred_at.desc())
            .limit(5)
            .all()
        )

        comm_lines: list[str] = []
        for c in recent_comms:
            tags_str = ", ".join(c.tags) if c.tags else "none"
            date_str = c.occurred_at.strftime("%Y-%m-%d") if c.occurred_at else "?"
            snippet = (c.raw_text or "")[:120].replace("\n", " ")
            comm_lines.append(
                f"  • [{c.source.upper()}] {date_str} | tags: {tags_str}\n"
                f"    \"{snippet}…\""
            )

        # Key risk signals from recent communications (derived from tags)
        churn_factors: list[str] = []
        growth_factors: list[str] = []
        for c in recent_comms:
            for tag in (c.tags or []):
                if tag in ("churn_signal", "competitor_mention", "escalation", "frustrated", "gone_silent"):
                    if tag not in churn_factors:
                        churn_factors.append(tag)
                if tag in ("expansion_interest", "upsell_signal", "enthusiastic", "positive_sentiment", "new_campaign_intent"):
                    if tag not in growth_factors:
                        growth_factors.append(tag)

        c_risk = aff.churn_risk_score or 0.5
        g_pot = aff.growth_potential_score or 0.5
        if c_risk >= 0.65:
            action = "⚠️  URGENT: Schedule retention call within 48 hours."
        elif g_pot >= 0.70:
            action = "🚀  Growth opportunity: Propose scale-up plan."
        elif c_risk >= 0.45:
            action = "📞  Monitor: Follow up within 7 days."
        else:
            action = "✅  Healthy: Maintain regular check-in cadence."

        lines = [
            "═══ AFFILIATE HEALTH SUMMARY ═══",
            f"Name:             {aff.name}",
            f"Status:           {aff.status}",
            f"Revenue (30d):    ${float(aff.revenue_30d or 0):,.2f}",
            f"Days silent:      {aff.days_since_contact or 0}",
            "",
            "─── Scores ──────────────────────",
            f"Health Score:      {aff.health_score:.1f} / 100",
            f"Churn Risk:        {c_risk:.1%}",
            f"Growth Potential:  {g_pot:.1%}",
            "",
            "─── Recent Communications ───────",
        ]
        lines += (comm_lines or ["  No communications on record."])
        lines += [
            "",
            "─── Churn Risk Drivers ──────────",
            f"  {', '.join(churn_factors) if churn_factors else 'Insufficient data'}",
            "",
            "─── Growth Drivers ──────────────",
            f"  {', '.join(growth_factors) if growth_factors else 'Insufficient data'}",
            "",
            "─── Recommended Action ──────────",
            f"  {action}",
        ]
        return "\n".join(lines)
    finally:
        db.close()


# ─── Tool 4: draft_email ──────────────────────────────────────────────────────

@tool
def draft_email(input_str: str) -> str:
    """Draft a personalised re-engagement or follow-up email for an affiliate.
    Input should be a string containing: affiliate name, their current situation
    (scores, recent behaviour), and the desired tone (urgent, warm, neutral).
    Use this as the final step after understanding an affiliate's situation.
    Example input: 'affiliate_name: Tom Bauer, situation: 51 days silent,
    competitor mentioned, CTR declining -4.2%, tone: urgent but warm'"""
    # Parse input string
    affiliate_name = ""
    situation = ""
    tone = "warm"

    for part in input_str.split(","):
        part = part.strip()
        if part.lower().startswith("affiliate_name:"):
            affiliate_name = part.split(":", 1)[1].strip()
        elif part.lower().startswith("situation:"):
            situation = part.split(":", 1)[1].strip()
        elif part.lower().startswith("tone:"):
            tone = part.split(":", 1)[1].strip()

    if not affiliate_name:
        affiliate_name = input_str[:50]

    # Try LLM-generated email
    llm = _get_llm()
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            prompt_text = (
                f"Write a professional affiliate marketing re-engagement email.\n\n"
                f"Affiliate: {affiliate_name}\n"
                f"Situation: {situation}\n"
                f"Tone: {tone}\n\n"
                f"Requirements:\n"
                f"- Under 150 words\n"
                f"- Start with Subject: on the first line\n"
                f"- Then Body: on the next line\n"
                f"- Sound human and specific to the situation\n"
                f"- Include one concrete next step"
            )
            response = llm.invoke([HumanMessage(content=prompt_text)])
            email_text = response.content
            if "Subject:" not in email_text:
                email_text = f"Subject: Following up — {affiliate_name}\n\nBody:\n{email_text}"
            return f"=== EMAIL DRAFT ===\n\n{email_text}"
        except Exception as exc:
            pass  # fall through to template

    # Template fallback when LLM unavailable
    first_name = affiliate_name.split()[0] if affiliate_name else "there"
    return (
        f"=== EMAIL DRAFT (template fallback — LLM unavailable) ===\n\n"
        f"Subject: Following up — {affiliate_name}\n\n"
        f"Hi {first_name},\n\n"
        f"I wanted to reach out personally given recent activity on your account. "
        f"Situation context: {situation}.\n\n"
        f"I'd love to jump on a quick 20-minute call to discuss how we can best support you. "
        f"When works for you this week?\n\n"
        f"Tone: {tone}\n\n"
        f"[Your Name]\nPartner Success Team"
    )


# ─── Tool 5: get_portfolio_health ────────────────────────────────────────────

@tool
def get_portfolio_health(input_str: str = "") -> str:
    """Get a summary of the entire affiliate portfolio health including average
    health score, number of at-risk affiliates, high growth affiliates, and
    churned affiliates. Use this for portfolio-level questions."""
    db = _get_db()
    try:
        affiliates = db.query(Affiliate).all()
        if not affiliates:
            return "No affiliates found. Run POST /ingest/full first."

        n = len(affiliates)
        avg_health = sum(a.health_score or 50.0 for a in affiliates) / n
        avg_churn = sum(a.churn_risk_score or 0.5 for a in affiliates) / n
        avg_growth = sum(a.growth_potential_score or 0.5 for a in affiliates) / n

        at_risk = [a for a in affiliates if (a.churn_risk_score or 0.0) > 0.5]
        high_growth = [a for a in affiliates if (a.growth_potential_score or 0.0) > 0.5]
        churned = [a for a in affiliates if (a.churn_risk_score or 0.0) > 0.8]

        score_history_count = db.query(ScoreHistory).count()

        worst = sorted(affiliates, key=lambda a: a.health_score or 50.0)[:3]
        best = sorted(affiliates, key=lambda a: a.health_score or 50.0, reverse=True)[:3]

        lines = [
            "═══ PORTFOLIO HEALTH SUMMARY ═══",
            f"Total affiliates:    {n}",
            f"Avg health score:    {avg_health:.1f} / 100",
            f"Avg churn risk:      {avg_churn:.1%}",
            f"Avg growth potential:{avg_growth:.1%}",
            f"Score history rows:  {score_history_count}",
            "",
            f"At-risk (churn > 50%):  {len(at_risk)} affiliate(s)",
            f"High-growth (>50%):     {len(high_growth)} affiliate(s)",
            f"Critical (churn > 80%): {len(churned)} affiliate(s)",
            "",
            "─── Worst 3 (needs attention) ───",
        ]
        for a in worst:
            lines.append(f"  • {a.name}: health={a.health_score:.1f}, churn={a.churn_risk_score:.1%}, silent={a.days_since_contact}d")

        lines += ["", "─── Top 3 (performing well) ─────"]
        for a in best:
            lines.append(f"  • {a.name}: health={a.health_score:.1f}, growth={a.growth_potential_score:.1%}")

        if at_risk:
            lines += ["", "─── At-Risk Names ───────────────"]
            lines.append("  " + ", ".join(a.name for a in at_risk))

        return "\n".join(lines)
    finally:
        db.close()


# Expose tools list for agent setup
TOOLS = [
    query_database,
    semantic_search,
    get_affiliate_summary,
    draft_email,
    get_portfolio_health,
]