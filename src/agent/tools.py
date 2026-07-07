"""
LangChain Tool Definitions
==========================
Seven tools for the ReAct agent. Each docstring is used by LangChain
to decide when to call the tool — keep them descriptive.

Every tool here is read-only or draft-only. None may perform a live scan,
send anything externally, or otherwise act — see get_leakage_status,
get_seo_status, and draft_email below for how that boundary is enforced.

Tools
-----
1. query_database        — raw SQL SELECT against PostgreSQL
2. semantic_search       — pgvector embedding search over communications
3. get_affiliate_summary — full profile for one affiliate
4. draft_email           — composes an email draft and files it for human approval (never sends)
5. get_portfolio_health  — whole-portfolio aggregate stats
6. get_leakage_status     — reads recorded leak findings for one affiliate (no live scan)
7. get_seo_status         — reads recorded SEO rank signals for one affiliate (no live check)
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from sqlalchemy import text

from src.audit.log import write_audit_entry
from src.core.logging_config import get_logger
from src.rulebook.recommend import categorize, recommend
from src.storage.database import SessionLocal
from src.storage.models import (
    Affiliate,
    ApprovalRequest,
    Communication,
    LeakedCode,
    ScoreHistory,
    SeoSignal,
)

logger = get_logger(__name__)

_BLOCKED_KEYWORDS = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
    re.IGNORECASE,
)

# Same bands used for health-bar color-coding in src/api/templates/index.html
# (h < 40 = red, h >= 60 = green) — reused here so "performing well" and
# "needs attention" match an existing convention instead of introducing a new one.
PERFORMING_WELL_THRESHOLD = 60.0  # health_score >= this = performing well
NEEDS_ATTENTION_THRESHOLD = 40.0  # health_score < this = needs attention


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
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
    return _llm


# ─── Tool 1: query_database ───────────────────────────────────────────────────

@tool
def query_database(sql_query: str) -> str:
    """Query the PostgreSQL database with a SELECT statement to get affiliate
    scores, health metrics, communication counts, or score history.
    Use this for precise filtered queries on structured data.
    Only SELECT statements are allowed.
    IMPORTANT: health_score is on a 0-100 scale (not 0-1).
    Example queries:
      SELECT name, health_score, churn_risk_score, status
      FROM affiliates ORDER BY health_score ASC LIMIT 5
      -- finds lowest health scores (urgent attention needed below 40)

      SELECT name, health_score, churn_risk_score, status
      FROM affiliates WHERE health_score < 40 ORDER BY health_score ASC
      -- finds affiliates needing urgent attention"""
    sql = sql_query.strip()
    if not sql.upper().startswith("SELECT"):
        return "Only SELECT statements are permitted."

    blocked = _BLOCKED_KEYWORDS.search(sql)
    if blocked:
        keyword = blocked.group(0).upper()
        logger.warning("Blocked dangerous SQL keyword", extra={"keyword": keyword, "sql": sql[:200]})
        return f"Blocked: query contains forbidden keyword '{keyword}'. Only safe SELECT queries are permitted."

    logger.debug("Executing SQL query", extra={"sql": sql})

    db = _get_db()
    try:
        result = db.execute(text(sql))
        rows = result.fetchmany(20)

        if not rows:
            logger.debug("SQL query returned 0 rows")
            return "Query returned no rows."

        cols = list(result.keys())
        lines = [" | ".join(cols)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

        output = "\n".join(lines)
        logger.debug("SQL query complete", extra={"row_count": len(rows), "columns": cols})
        return output
    except Exception as exc:
        logger.error("SQL query failed", extra={"error": str(exc)})
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

    db = _get_db()
    try:
        from src.storage.pgvector_store import PGVectorStore
        vs = PGVectorStore(db)
        results = vs.search_similar(embedding, n_results=5)
    except Exception as exc:
        return f"Search error: {exc}"
    finally:
        db.close()

    if not results:
        return "No matching communications found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        text_snippet = r.get("text", "")[:300]
        score = round(1 - r.get("distance", 1.0), 3)
        tags_list = r.get("tags", [])
        tags_str = ", ".join(tags_list) if isinstance(tags_list, list) else str(tags_list)
        lines.append(
            f"[{i}] Affiliate: {r.get('affiliate_name', '?')} "
            f"| Source: {r.get('source', '?')} "
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

        # Key risk signals — pulled from the most recent *tagged* communications
        # within a 90-day lookback, not just the most recent N regardless of
        # tag status. `recent_comms` above (used for the raw "Recent
        # Communications" log) can legitimately be dominated by communications
        # that haven't been NLP-tagged yet (POST /process/nlp hasn't run
        # since, or genuinely fresh contact) — using it for drivers too would
        # silently bury real historical signal under "Insufficient data"
        # the moment untagged noise fills the top-5 window, which is exactly
        # what happened before this fix.
        _DRIVER_LOOKBACK_DAYS = 90
        driver_cutoff = datetime.now(timezone.utc) - timedelta(days=_DRIVER_LOOKBACK_DAYS)
        driver_comms = (
            db.query(Communication)
            .filter(
                Communication.affiliate_id == aff.id,
                Communication.occurred_at >= driver_cutoff,
                Communication.tags != [],
            )
            .order_by(Communication.occurred_at.desc())
            .limit(5)
            .all()
        )

        churn_factors: list[str] = []
        growth_factors: list[str] = []
        for c in driver_comms:
            for tag in (c.tags or []):
                if tag in ("churn_signal", "competitor_mention", "escalation", "frustrated", "gone_silent"):
                    if tag not in churn_factors:
                        churn_factors.append(tag)
                if tag in ("expansion_interest", "upsell_signal", "enthusiastic", "positive_sentiment", "new_campaign_intent"):
                    if tag not in growth_factors:
                        growth_factors.append(tag)

        # Transparency: if truly nothing tagged exists in the lookback window,
        # "Insufficient data" is now an honest signal, not a windowing
        # artifact. But if more recent contact exists and simply hasn't been
        # tagged yet, say so — otherwise a reader could mistake the drivers
        # below (based on older tagged data) for "no recent contact at all".
        driver_staleness_note = None
        if recent_comms:
            newest_overall = recent_comms[0]
            newest_driver_at = driver_comms[0].occurred_at if driver_comms else None
            if not newest_overall.tags and (
                newest_driver_at is None or newest_overall.occurred_at > newest_driver_at
            ):
                newest_date_str = (
                    newest_overall.occurred_at.strftime("%Y-%m-%d")
                    if newest_overall.occurred_at else "?"
                )
                driver_staleness_note = (
                    f"  Note: more recent contact exists ({newest_date_str}, see Recent "
                    "Communications above) but hasn't been NLP-tagged yet — the drivers "
                    "below are based on the latest tagged communication instead."
                )

        c_risk = aff.churn_risk_score or 0.5
        g_pot = aff.growth_potential_score or 0.5

        # features/leaks only enrich the evidence bundle — recommend() handles
        # None gracefully, so a failure here must not break the whole summary.
        try:
            from src.ml.feature_engineering import build_feature_vector
            features = build_feature_vector(str(aff.id), db)
        except Exception as exc:
            logger.warning(
                "build_feature_vector failed in get_affiliate_summary — "
                "continuing without feature evidence",
                extra={"affiliate_id": str(aff.id), "error": str(exc)},
            )
            features = None

        # Fast path: has_active_leak is recomputed from the full leaked_codes
        # table on every scan (src.scraping.leakage_scraper) — if it's False,
        # querying LeakedCode would necessarily return nothing (recommend()
        # treats None and [] identically), so skip the query. Only a safe
        # drop-in for the "no leak" case — when True, the actual rows are
        # still fetched below, since recommend()'s evidence text needs the
        # specific codes, not just a boolean.
        recent_leaks = None
        if aff.has_active_leak:
            try:
                recent_leaks = (
                    db.query(LeakedCode)
                    .filter(LeakedCode.affiliate_id == aff.id)
                    .order_by(LeakedCode.found_at.desc())
                    .limit(5)
                    .all()
                )
            except Exception as exc:
                logger.warning(
                    "Leak lookup failed in get_affiliate_summary — "
                    "continuing without leak evidence",
                    extra={"affiliate_id": str(aff.id), "error": str(exc)},
                )
                recent_leaks = None

        rec = recommend(aff, features, recent_leaks)

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

        # rec.reason_code is the churn/growth tier, optionally suffixed with
        # "_leak_detected". The tier alone explains the recommendation text
        # below; a leak is a separate fact riding along in the same code, not
        # something that changed the tier — so it's displayed as its own note
        # rather than folded into a single "reason" line implying causation.
        tier_reason = rec.reason_code.removesuffix("_leak_detected")
        leak_on_record = rec.reason_code != tier_reason

        lines += [
            "",
            "─── Churn Risk Drivers ──────────",
            f"  {', '.join(churn_factors) if churn_factors else 'Insufficient data'}",
            "",
            "─── Growth Drivers ──────────────",
            f"  {', '.join(growth_factors) if growth_factors else 'Insufficient data'}",
        ]
        if driver_staleness_note:
            lines.append(driver_staleness_note)
        lines += [
            "",
            "─── Recommended Action ──────────",
            f"  {rec.recommendation}",
            f"  (tier: {tier_reason}, based on churn/growth scores only)",
        ]
        if leak_on_record:
            lines.append(
                "  Note: a promo-code leak is also on record for this affiliate "
                "(see Evidence below) — unrelated to the tier above."
            )
        lines += [
            "",
            "─── SEO Signal ──────────────────",
            f"  Search trend: {aff.search_trend or 'stable'}"
            " (independent of the tier above — call get_seo_status for keyword/rank detail)",
        ]
        lines += [
            "",
            "─── Evidence ────────────────────",
        ]
        lines += [f"  • {e}" for e in rec.evidence]
        return "\n".join(lines)
    finally:
        db.close()


# ─── Tool 4: draft_email ──────────────────────────────────────────────────────

def _get_recent_leaks(db, aff) -> list:
    """Most recently recorded promo-code leak rows for one affiliate, or []
    if none. Fast-pathed on has_active_leak — see get_leakage_status for why
    this is safe. Shared by get_leakage_status and draft_email's fact
    gathering so both read the exact same query, not two copies of it."""
    if not (aff and aff.has_active_leak):
        return []
    return (
        db.query(LeakedCode)
        .filter(LeakedCode.affiliate_id == aff.id)
        .order_by(LeakedCode.found_at.desc())
        .limit(5)
        .all()
    )


def _get_recent_seo(db, aff) -> list:
    """Most recently recorded SEO rank-tracking rows for one affiliate, or
    [] if none. Fast-pathed on tracked_keyword — see get_seo_status for why
    this is safe. Shared by get_seo_status and draft_email's fact gathering
    so both read the exact same query, not two copies of it."""
    if not (aff and aff.tracked_keyword):
        return []
    return (
        db.query(SeoSignal)
        .filter(SeoSignal.affiliate_id == aff.id)
        .order_by(SeoSignal.checked_at.desc())
        .limit(5)
        .all()
    )


def _build_affiliate_facts(db, aff) -> dict:
    """Concrete, DB-sourced facts about one affiliate for email composition.
    This is what grounds draft_email in real data instead of a freeform
    'situation' string the agent would otherwise have to summarize itself —
    the agent no longer has to remember to mention a leak or SEO decline;
    draft_email looks them up directly, the same way get_leakage_status and
    get_seo_status would."""
    leaks = _get_recent_leaks(db, aff)
    seo = _get_recent_seo(db, aff)
    return {
        "health_score": aff.health_score,
        "churn_risk_score": aff.churn_risk_score,
        "growth_potential_score": aff.growth_potential_score,
        "days_since_contact": aff.days_since_contact,
        "leak_code": leaks[0].code if leaks else None,
        "leak_site": leaks[0].site if leaks else None,
        "seo_trend_direction": aff.search_trend if seo else None,
        "seo_keyword": seo[0].keyword if seo else None,
    }


def _format_facts_block(facts: dict) -> str:
    """Render an affiliate facts dict as a labelled bullet list for the
    composition prompt — only facts that are actually present are included,
    so the LLM has nothing to guess or pad around."""
    fact_lines = [
        f"Health score: {facts['health_score']:.1f}/100"
        if facts.get("health_score") is not None else None,
        f"Churn risk: {facts['churn_risk_score']:.0%}"
        if facts.get("churn_risk_score") is not None else None,
        f"Growth potential: {facts['growth_potential_score']:.0%}"
        if facts.get("growth_potential_score") is not None else None,
        f"Days since last contact: {facts['days_since_contact']}"
        if facts.get("days_since_contact") is not None else None,
        f'Active promo-code leak: code "{facts["leak_code"]}" found on {facts["leak_site"]}'
        if facts.get("leak_code") else None,
        f"SEO search trend: {facts['seo_trend_direction']} (keyword: {facts['seo_keyword']})"
        if facts.get("seo_trend_direction") else None,
    ]
    present = [f"- {line}" for line in fact_lines if line]
    return "\n".join(present) if present else "- No specific signals on record beyond standard profile data."


# Sender-signature placeholders are expected and fine — a human fills these
# in before sending. Everything else in brackets is treated as a content
# gap the model stubbed instead of omitting, which is the failure mode this
# validator exists to catch.
_SIGNATURE_PLACEHOLDER_ALLOWLIST = {
    "[your name]", "[your position]", "[your title]", "[your company]",
}
_BRACKET_PATTERN = re.compile(r"\[[^\[\]]+\]")
_THIRD_PERSON_PRONOUN_PATTERN = re.compile(r"\b(he|him|his|she|her|hers)\b", re.IGNORECASE)


def _validate_email_body(body: str, affiliate_name: str) -> list[str]:
    """Lightweight regex safety net — not a substitute for good prompting —
    catching the two failure modes actually observed in production:
    bracket placeholders standing in for facts the model didn't have (e.g.
    "[Recipient's Name]", "[specific topic]") and third-person
    self-reference to the very person the email is addressed to (e.g.
    "check out his recent insights" in an email TO that person). Returns a
    list of violation descriptions; empty means the body passed."""
    violations: list[str] = []

    for match in _BRACKET_PATTERN.findall(body):
        if match.lower() not in _SIGNATURE_PLACEHOLDER_ALLOWLIST:
            violations.append(f"bracket placeholder: {match}")

    if _THIRD_PERSON_PRONOUN_PATTERN.search(body):
        violations.append(
            "third-person pronoun (he/him/his/she/her/hers) — the affiliate "
            "should be addressed directly as 'you'"
        )

    first_name = affiliate_name.split()[0] if affiliate_name else ""
    if first_name and re.search(rf"\b{re.escape(first_name)}'s\b", body):
        violations.append(
            f"third-person possessive reference to the affiliate's own first "
            f"name ('{first_name}'s')"
        )
    if affiliate_name and re.search(rf"\b{re.escape(affiliate_name)}'(?!\w)", body):
        violations.append(
            f"third-person possessive reference to the affiliate's own full "
            f"name ('{affiliate_name}'')"
        )

    return violations


_MAX_COMPOSE_ATTEMPTS = 3


def _compose_email(
    affiliate_name: str, tone: str, facts: dict, situation_override: str = ""
) -> tuple[str, str]:
    """Compose (subject, body) for a re-engagement email, grounded in
    concrete DB-sourced facts (health/churn/growth scores, days since
    contact, active leak, SEO trend — see _build_affiliate_facts) rather
    than a freeform situation string the agent would otherwise have to
    compose and summarize itself. LLM-generated when available, template
    fallback otherwise. Pure composition — no DB writes, no side effects;
    draft_email() is the one that files the result for review."""
    fact_block = _format_facts_block(facts)
    first_name = affiliate_name.split()[0] if affiliate_name else "there"

    llm = _get_llm()
    if llm:
        try:
            from langchain_core.messages import HumanMessage

            base_prompt = (
                f"Write a professional affiliate marketing re-engagement email.\n\n"
                f"Affiliate: {affiliate_name}\n"
                f"Known facts about this affiliate (use ONLY these — do not invent "
                f"or assume anything beyond them):\n{fact_block}\n"
            )
            if situation_override:
                base_prompt += f"Additional context to emphasize: {situation_override}\n"
            base_prompt += (
                f"Tone: {tone}\n\n"
                f"Requirements:\n"
                f"- Under 150 words\n"
                f"- Start with Subject: on the first line\n"
                f"- Then Body: on the next line\n"
                f"- Sound human and specific to the facts above\n"
                f"- Address {first_name} directly, in second person ('you'/'your') "
                f"throughout — never refer to them in the third person (no "
                f"'he'/'him'/'his'/'she'/'her', and never use their own name as "
                f"if describing them to someone else, e.g. \"check out "
                f"{first_name}'s insights\")\n"
                f"- Do not use bracket placeholders (e.g. [Name], [specific topic]) "
                f"for anything — if a detail isn't in the facts above, leave it out "
                f"entirely rather than stubbing it with a placeholder. A "
                f"placeholder for the SENDER's own name/title/company (e.g. "
                f"[Your Name]) is fine — it is meant to be filled in by the human "
                f"who sends this\n"
                f"- Include one concrete next step"
            )

            # Retry on validation failure rather than shipping a draft with a
            # bracket placeholder or third-person self-reference — both were
            # observed in production (see draft_email's docstring). Each retry
            # tells the model exactly what it got wrong last time.
            correction_note = ""
            for attempt in range(1, _MAX_COMPOSE_ATTEMPTS + 1):
                prompt_text = base_prompt + (f"\n\n{correction_note}" if correction_note else "")
                response = llm.invoke([HumanMessage(content=prompt_text)])
                email_text = response.content
                if "Subject:" not in email_text:
                    email_text = f"Subject: Following up — {affiliate_name}\n\nBody:\n{email_text}"
                subject, body = _split_subject_body(email_text)

                violations = _validate_email_body(body, affiliate_name)
                if not violations:
                    return subject, body

                logger.warning(
                    "draft_email composition failed validation, retrying",
                    extra={"attempt": attempt, "violations": violations},
                )
                correction_note = (
                    "IMPORTANT: your previous attempt was rejected for: "
                    f"{'; '.join(violations)}. Rewrite it addressing {first_name} "
                    "directly as 'you' throughout, with no third-person reference "
                    "to them and no bracket placeholder other than the sender's "
                    "own name/title/company."
                )
            # All attempts still failed validation — fall through to the
            # deterministic template rather than ship a bad draft.
        except Exception:
            pass  # fall through to template

    # Template fallback when LLM unavailable, or when every LLM attempt above
    # still failed validation — built directly from facts, no situation
    # string to interpolate, and guaranteed to pass _validate_email_body on
    # its own (verified below; situation_override is the only free-text
    # component, so it's the only thing stripped if validation still fails).
    fact_lines = [line[2:] for line in fact_block.split("\n") if line.startswith("- ")]
    context_str = "; ".join(fact_lines) if fact_lines else "recent account activity"
    body = (
        f"Hi {first_name},\n\n"
        f"I wanted to reach out personally given the following: {context_str}.\n\n"
        f"I'd love to jump on a quick 20-minute call to discuss how we can best support you. "
        f"When works for you this week?\n\n"
        f"[Your Name]\nPartner Success Team"
    )
    if situation_override:
        body_with_override = body.replace(
            "When works for you this week?",
            f"When works for you this week? ({situation_override})",
        )
        # Only free text in the template path — validate before using it, in
        # case situation_override itself contains a placeholder or
        # third-person reference. Drop it rather than ship a bad draft.
        if not _validate_email_body(body_with_override, affiliate_name):
            body = body_with_override

    return f"Following up — {affiliate_name}", body


def _split_subject_body(email_text: str) -> tuple[str, str]:
    """Best-effort split of an LLM-composed 'Subject: ...\\n\\nBody:\\n...'
    block into (subject, body)."""
    lines = email_text.split("\n")
    subject = ""
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    if body.lower().startswith("body:"):
        body = body.split(":", 1)[1].strip()
    return subject or "Following up", body


@tool
def draft_email(
    affiliate_name: str,
    situation_override: str = "",
    tone: str = "warm",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Draft a personalised re-engagement or follow-up email for an affiliate.
    This tool automatically pulls the affiliate's current health/churn/growth
    scores, days since last contact, active promo-code leak status, and SEO
    rank trend directly from the database — you do NOT need to summarize
    these into a situation string yourself, and the email is grounded in
    this real data regardless of what you do or don't mention. Use
    situation_override only to add something the tool cannot already see
    (e.g. a specific detail from a recent call you want emphasized) — it
    augments, it does not replace, the DB-sourced facts.
    Use this as the final step after understanding an affiliate's situation.
    This does NOT send anything — it files the draft as a pending request in
    the approval queue (status: waiting_for_review). A human must approve it
    via POST /approvals/{id}/approve before it is ever sent.
    If a "please revise it" request follows an earlier draft for the same
    affiliate in this same conversation, calling this tool again UPDATES that
    same pending request in place instead of creating an unrelated new one —
    you do not need to do anything differently to get this behaviour.
    This tool returns the full composed subject AND body — describe only
    what is actually returned to you here, never content you were not given
    back by this tool.
    Example: draft_email(affiliate_name="Tom Bauer", tone="urgent but warm")"""
    # conversation_id is injected via RunnableConfig (see run_agent's call to
    # agent.invoke(..., config={"configurable": {"conversation_id": ...}})) —
    # it is never something the LLM supplies itself, so it can't be spoofed
    # or omitted by the agent's own reasoning. None for any caller that
    # doesn't go through run_agent (e.g. a direct test invocation).
    conversation_id = ((config or {}).get("configurable") or {}).get("conversation_id")

    db = _get_db()
    try:
        aff = (
            db.query(Affiliate)
            .filter(Affiliate.name.ilike(f"%{affiliate_name.strip()}%"))
            .first()
        )
        if not aff:
            return (
                f"Cannot create a draft: no affiliate found matching '{affiliate_name}'. "
                "Look up the affiliate first (e.g. via query_database or get_affiliate_summary) "
                "and retry with their exact name."
            )

        facts = _build_affiliate_facts(db, aff)
        subject, body = _compose_email(aff.name, tone, facts, situation_override)

        # No contact email is stored in this schema (see src.storage.models.Affiliate)
        # — flagged clearly rather than fabricating a plausible-looking address.
        to_placeholder = f"{aff.name} <no email on file>"
        new_payload = {
            "to": to_placeholder,
            "subject": subject,
            "body": body,
            "affiliate_id": str(aff.id),
        }

        # Revise-in-place vs. create-new. Deliberately never matches on
        # affiliate_id alone — without a real conversation_id there is no
        # way to tell "the same conversation asking for a revision" apart
        # from "an unrelated new conversation about the same affiliate",
        # and guessing wrong would silently overwrite someone else's
        # pending draft. See Phase 3 investigation: no session/conversation
        # tracking existed anywhere in this system before this field.
        existing = None
        if conversation_id:
            existing = (
                db.query(ApprovalRequest)
                .filter(
                    ApprovalRequest.affiliate_id == aff.id,
                    ApprovalRequest.session_id == conversation_id,
                    ApprovalRequest.kind == "email",
                    ApprovalRequest.status == "waiting_for_review",
                )
                .order_by(ApprovalRequest.created_at.desc())
                .first()
            )

        if existing:
            action = "revised"
            existing.payload = new_payload
            existing.updated_at = datetime.now(timezone.utc)
            approval = existing
        else:
            action = "created"
            approval = ApprovalRequest(
                kind="email",
                affiliate_id=aff.id,
                session_id=conversation_id,
                payload=new_payload,
                status="waiting_for_review",
            )
            db.add(approval)

        # Flush (not commit) so a brand-new row's server/default-generated id
        # is populated on the Python object before the audit entry below
        # needs to reference it — the revised case already has a real id,
        # this only matters for the "created" branch.
        db.flush()

        write_audit_entry(
            db,
            stage="agent",
            record_type="approval_request",
            record_id=approval.id,
            rule_or_tool="draft_email",
            input_snapshot={
                "affiliate_name": aff.name,
                "situation_override": situation_override,
                "tone": tone,
                "facts": facts,
                "conversation_id": conversation_id,
            },
            output_snapshot={"subject": subject, "body": body, "action": action},
        )

        db.commit()
        db.refresh(approval)

        logger.info(
            f"Draft email {action}",
            extra={"approval_id": str(approval.id), "affiliate_id": str(aff.id), "action": action},
        )

        action_verb = "Draft revised" if action == "revised" else "Draft created"
        return (
            f"{action_verb} for {aff.name} — pending approval as request {approval.id}.\n"
            f"Subject: {subject}\n\n"
            f"Body:\n{body}\n\n"
            f"It will not be sent until a human approves it via "
            f"POST /approvals/{approval.id}/approve."
        )
    finally:
        db.close()


# ─── Tool 5: get_portfolio_health ────────────────────────────────────────────

@tool
def get_portfolio_health(input_str: str = "") -> str:
    """Get a summary of the entire affiliate portfolio health including average
    health score, number of at-risk affiliates, high growth affiliates,
    churned affiliates, active promo-code leaks, and declining SEO trends.
    Use this for portfolio-level questions, including "which affiliates need
    attention" or "which affiliates have warning signs" — warning signs span
    all four signal types this system tracks (churn/growth tier, leaks, SEO
    trend, and low composite health score from weak growth), not just the
    rulebook tier — check the Combined Signal Groups section, not only the
    at-risk list.
    By default this returns only the top/worst 3 of each ranked section. If
    the user asks for the complete list (e.g. after seeing a capped summary,
    or "show me all of them"), call this tool AGAIN with input_str="full" —
    this returns every qualifying affiliate in the Worst/Top-by-health-score
    and Combined Signal Groups sections instead of just 3. Always re-call
    with input_str="full" for a full-list request — never invent additional
    affiliate names or numbers to pad out a list yourself; every name and
    every score you state must come from an actual tool result, never from
    memory or estimation."""
    db = _get_db()
    show_full = "full" in input_str.lower()
    try:
        affiliates = db.query(Affiliate).all()
        if not affiliates:
            return "No affiliates found. Run POST /ingest/full first."

        n = len(affiliates)
        avg_health = sum(a.health_score or 50.0 for a in affiliates) / n
        avg_churn = sum(a.churn_risk_score or 0.5 for a in affiliates) / n
        avg_growth = sum(a.growth_potential_score or 0.5 for a in affiliates) / n

        # Tiers are mutually exclusive (see src/rulebook/recommend.categorize) —
        # a churned affiliate counts only as "churned", not double-counted
        # under "at_risk" as well.
        tiers = {
            a.id: categorize(a.churn_risk_score or 0.0, a.growth_potential_score or 0.0)
            for a in affiliates
        }
        at_risk = [a for a in affiliates if tiers[a.id] == "at_risk"]
        high_growth = [a for a in affiliates if tiers[a.id] == "high_growth"]
        churned = [a for a in affiliates if tiers[a.id] == "churned"]

        # Leak and SEO signals are visibility-only here — deliberately never
        # folded into avg_churn/avg_health or the tier counts above (see
        # src.rulebook.recommend and Tier B / SEO task docs for why they stay
        # separate). This just gives the agent a way to see them exist at the
        # portfolio level without querying every affiliate individually.
        leaking = [a for a in affiliates if a.has_active_leak]
        declining_seo = [a for a in affiliates if a.search_trend == "declining"]

        score_history_count = db.query(ScoreHistory).count()

        # The health-score band (composite of churn + growth) and the rulebook
        # tier (churn alone — src.rulebook.recommend.categorize, the system's
        # single source of truth for risk classification) can disagree: a high
        # growth potential can pull an at_risk-tier affiliate's health score
        # above PERFORMING_WELL_THRESHOLD, and a low growth potential can pull
        # a churn-healthy affiliate's health score below NEEDS_ATTENTION_THRESHOLD.
        # The rulebook tier wins whenever the two conflict, so this tool never
        # calls an at_risk/churned-tier affiliate "performing well", and never
        # omits one from "needs attention" just because their composite score
        # looks fine.
        _at_risk_tiers = ("at_risk", "churned")
        performing_well = [
            a for a in affiliates
            if (a.health_score or 50.0) >= PERFORMING_WELL_THRESHOLD
            and tiers[a.id] not in _at_risk_tiers
        ]
        needs_attention = [
            a for a in affiliates
            if (a.health_score or 50.0) < NEEDS_ATTENTION_THRESHOLD
            or tiers[a.id] in _at_risk_tiers
        ]
        # The subset of needs_attention that the churn/growth tier signal alone
        # would miss — i.e. affiliates pulled below NEEDS_ATTENTION_THRESHOLD by
        # weak growth potential while their churn risk stays under the at_risk
        # cutoff (e.g. Marcus Williams: churn 45% < 50% threshold, growth 10%).
        # Tracked as its own signal source below so Combined Signal Groups can't
        # silently disagree with this section about who "needs attention" and why.
        low_health_only = [
            a for a in needs_attention if tiers[a.id] not in _at_risk_tiers
        ]
        best_full = sorted(performing_well, key=lambda a: a.health_score or 50.0, reverse=True)
        worst_full = sorted(needs_attention, key=lambda a: a.health_score or 50.0)
        best = best_full if show_full else best_full[:3]
        worst = worst_full if show_full else worst_full[:3]

        lines = [
            "═══ PORTFOLIO HEALTH SUMMARY ═══",
            f"Total affiliates:    {n}",
            f"Avg health score:    {avg_health:.1f} / 100",
            f"Avg churn risk:      {avg_churn:.1%}",
            f"Avg growth potential:{avg_growth:.1%}",
            f"Score history rows:  {score_history_count}",
            "",
            f"At-risk (50-80% churn):  {len(at_risk)} affiliate(s)",
            f"High-growth (>50%):      {len(high_growth)} affiliate(s)",
            f"Critical (churn > 80%):  {len(churned)} affiliate(s)",
            f"Active promo-code leaks: {len(leaking)} affiliate(s)  (separate signal — not a tier)",
            f"Declining SEO trend:     {len(declining_seo)} affiliate(s)  (separate signal — not a tier)",
            "",
            f"─── Worst {len(worst)} of {len(needs_attention)} by health score "
            f"(health < {NEEDS_ATTENTION_THRESHOLD:.0f}, or churn-tier at_risk/churned) ───",
            "(A narrower, health/tier-only slice — see Combined Signal Groups below "
            "for the full attention/warning-signs picture, which also includes "
            "leak- and SEO-only affiliates this section does not.)",
        ]
        if worst:
            for a in worst:
                lines.append(f"  • {a.name}: health={a.health_score:.1f}, churn={a.churn_risk_score:.1%}, silent={a.days_since_contact}d")
            if not show_full and len(needs_attention) > len(worst):
                lines += [
                    "",
                    f"(Showing {len(worst)} of {len(needs_attention)} — call this same tool "
                    "again with input_str='full' for the complete list; do not guess or "
                    "invent the remaining names.)",
                ]
        else:
            lines.append(
                f"No affiliates currently fall below the health/tier threshold (health < "
                f"{NEEDS_ATTENTION_THRESHOLD:.0f} and no churn-tier at_risk/churned) — "
                "portfolio is stable by this narrower measure "
                "(Combined Signal Groups below may still show leak/SEO-only affiliates)."
            )

        lines += [
            "",
            f"─── Top {len(best)} of {len(performing_well)} performing well "
            f"(health >= {PERFORMING_WELL_THRESHOLD:.0f}, excluding any churn-tier at_risk/churned) ───",
        ]
        if best:
            for a in best:
                lines.append(f"  • {a.name}: health={a.health_score:.1f}, growth={a.growth_potential_score:.1%}")
            if not show_full and len(performing_well) > len(best):
                lines += [
                    "",
                    f"(Showing {len(best)} of {len(performing_well)} — call this same tool "
                    "again with input_str='full' for the complete list; do not guess or "
                    "invent the remaining names.)",
                ]
        else:
            lines.append(
                f"No affiliates currently meet the performing-well threshold "
                f"(health >= {PERFORMING_WELL_THRESHOLD:.0f} with no churn-tier "
                "at_risk/churned)."
            )

        if at_risk:
            lines += ["", "─── At-Risk Names ───────────────"]
            lines.append("  " + ", ".join(a.name for a in at_risk))

        if leaking:
            lines += ["", "─── Active Leak Names ───────────"]
            lines.append("  " + ", ".join(a.name for a in leaking))

        if declining_seo:
            lines += ["", "─── Declining SEO Names ─────────"]
            lines.append("  " + ", ".join(a.name for a in declining_seo))

        # Which affiliates are flagged by 2+ of the four independent signal
        # types below vs. exactly 1 — computed here in code rather than left
        # for the agent to count itself. The LLM was observed to miscount this
        # split on its own (e.g. calling a single-signal at_risk affiliate
        # "multiple warning signs"), which is exactly the class of mechanical
        # judgment this codebase deliberately keeps out of the agent's own
        # reasoning (see SYSTEM_PROMPT's "business logic lives in tested code,
        # not in you" and src.rulebook.recommend). The fourth source
        # (low_health_only) exists so this section can't silently disagree
        # with the "Worst N needing attention" section above about who needs
        # attention and why — see that section's own comment for the Marcus
        # Williams case (low growth pulling health down without elevated churn).
        _signal_sources = [
            ("at-risk/churned churn-growth tier", at_risk + churned),
            ("active promo-code leak", leaking),
            ("declining SEO trend", declining_seo),
            ("low composite health score (weak growth, not elevated churn)", low_health_only),
        ]
        signal_labels: dict = {}
        for label, group in _signal_sources:
            for a in group:
                signal_labels.setdefault(a.id, []).append(label)

        flagged = [a for a in affiliates if a.id in signal_labels]

        # Signal COUNT (how many independent systems flagged someone) is not
        # the same thing as SEVERITY (how bad their underlying risk actually
        # is) — an affiliate tripping 3 mild signals should not outrank one
        # who is genuinely close to churning. Severity is the ONLY sort key:
        # churned tier is most urgent, then at_risk, then a low composite
        # health score from weak growth alone, then leak/SEO-only, health
        # ascending as the tiebreaker within each rank. Signal count is still
        # shown per line (breadth is useful context) but never determines
        # position in the list — a single-signal at_risk affiliate outranks a
        # multi-signal leak+SEO affiliate whose churn is otherwise healthy.
        _low_health_only_ids = {a.id for a in low_health_only}

        def _severity_rank(a) -> int:
            tier = tiers[a.id]
            if tier == "churned":
                return 0
            if tier == "at_risk":
                return 1
            if a.id in _low_health_only_ids:
                return 2
            return 3

        def _severity_sort_key(a):
            return (_severity_rank(a), a.health_score or 50.0)

        ordered = sorted(flagged, key=_severity_sort_key)

        lines += ["", "─── Combined Signal Groups (ordered by severity, most urgent first) ─────────"]
        if ordered:
            for a in ordered:
                count = len(signal_labels[a.id])
                suffix = f" [{count} signals]" if count >= 2 else ""
                lines.append(f"  • {a.name} — {', '.join(signal_labels[a.id])}{suffix}")

            # Cross-reference, not a re-ranking: leak/SEO-only names (severity
            # rank 3 — the mildest tier, structurally different from a churn/
            # growth-tier or low-health signal) can otherwise read as
            # afterthoughts despite still needing action. Deliberately scoped
            # to rank 3 only — an at_risk/churned-tier or low-health-only
            # affiliate with a single signal (e.g. Tom Bauer, James O'Brien)
            # is already unambiguously top priority by its position at the
            # top of the list; grouping them into this same reassurance note
            # would undersell them by association, exactly the confusion this
            # note exists to prevent for the milder cases.
            minor_single_signal_names = [
                a.name for a in ordered
                if len(signal_labels[a.id]) == 1 and _severity_rank(a) == 3
            ]
            if minor_single_signal_names:
                lines += [
                    "",
                    f"  Note: {', '.join(minor_single_signal_names)} show only a leak or "
                    "SEO signal (no elevated churn or low health) — still worth "
                    "addressing, not lower priority by default. This does not apply to "
                    "any at-risk/churned-tier or low-health affiliate above, which is "
                    "already top priority by its position in this list.",
                ]
        else:
            lines.append("  No affiliates currently flagged by any of these four signal types.")

        return "\n".join(lines)
    finally:
        db.close()


# ─── Tool 6: get_leakage_status ───────────────────────────────────────────────

@tool
def get_leakage_status(affiliate_id: str) -> str:
    """Report the most recently recorded promo-code leak findings for a
    specific affiliate. This is READ-ONLY — it does not run a new scan.
    It reflects the last completed scan, whether that was the nightly
    scheduled job (03:00 UTC) or an on-demand scan triggered via
    POST /leakage/scan; it cannot trigger a new scan itself. If the data
    might be stale and the user needs a fresh check, tell them to trigger
    POST /leakage/scan (or wait for the next scheduled run) — you cannot
    start one from this tool.
    Use this when a user asks whether an affiliate's code has been shared
    without authorisation, or to check before a retention call.
    Input must be the affiliate's UUID (not their name).
    To get the UUID first, use query_database:
      SELECT id, name FROM affiliates WHERE name ILIKE '%<name>%'"""
    try:
        aff_uuid = uuid.UUID(affiliate_id)
    except ValueError:
        return (
            f"'{affiliate_id}' is not a valid UUID. Look up the affiliate's UUID first "
            "via query_database: SELECT id, name FROM affiliates WHERE name ILIKE '%<name>%'"
        )

    db = _get_db()
    try:
        aff = db.query(Affiliate).filter(Affiliate.id == aff_uuid).first()
        recent = _get_recent_leaks(db, aff)
    finally:
        db.close()

    if not recent:
        return (
            "No promo-code leaks are on record for this affiliate as of the last scan. "
            "This reflects the most recent scheduled or on-demand scan, not a live check "
            "run just now."
        )

    lines = [f"⚠️  {len(recent)} recorded leak(s) for this affiliate (most recent first):\n"]
    for leak in recent:
        lines.append(
            f"  • Code: {leak.code}\n"
            f"    Site: {leak.site}\n"
            f"    URL:  {leak.source_url}\n"
            f"    Found at: {leak.found_at.isoformat() if leak.found_at else '?'} "
            f"(scan_type: {leak.scan_type})"
        )
    lines.append(
        "\nThis is the last recorded scan, not a live check run just now — "
        "trigger POST /leakage/scan for a fresh one."
    )
    return "\n".join(lines)


# ─── Tool 7: get_seo_status ────────────────────────────────────────────────────

@tool
def get_seo_status(affiliate_id: str) -> str:
    """Report the most recently recorded SEO rank-tracking signal for a
    specific affiliate. This is READ-ONLY — it does not run a new check.
    It reflects the last completed check, whether that was the weekly
    scheduled job (Monday 04:00 UTC) or an on-demand check triggered via
    POST /seo/scan; it cannot trigger a new check itself. If the data might
    be stale and the user needs a fresh check, tell them to trigger
    POST /seo/scan (or wait for the next scheduled run) — you cannot start
    one from this tool.
    Use this when a user asks about an affiliate's search visibility or
    organic ranking trend.
    Input must be the affiliate's UUID (not their name).
    To get the UUID first, use query_database:
      SELECT id, name FROM affiliates WHERE name ILIKE '%<name>%'"""
    try:
        aff_uuid = uuid.UUID(affiliate_id)
    except ValueError:
        return (
            f"'{affiliate_id}' is not a valid UUID. Look up the affiliate's UUID first "
            "via query_database: SELECT id, name FROM affiliates WHERE name ILIKE '%<name>%'"
        )

    db = _get_db()
    try:
        aff = db.query(Affiliate).filter(Affiliate.id == aff_uuid).first()
        recent = _get_recent_seo(db, aff)
    finally:
        db.close()

    if not recent:
        return (
            "No SEO rank-tracking data is on record for this affiliate. Either no "
            "keyword is tracked for them, or the last check found nothing recorded yet."
        )

    lines = [
        f"Search trend: {aff.search_trend} ({len(recent)} recorded check(s), most recent first):\n"
    ]
    for s in recent:
        change_str = f"{s.rank_change:+d}" if s.rank_change is not None else "n/a"
        lines.append(
            f"  • Keyword: {s.keyword}\n"
            f"    Rank: {s.rank} (change vs previous check: {change_str})\n"
            f"    Search volume: {s.search_volume}\n"
            f"    Checked at: {s.checked_at.isoformat() if s.checked_at else '?'}"
        )
    lines.append(
        "\nThis is the last recorded check, not a live check run just now — "
        "trigger POST /seo/scan for a fresh one."
    )
    return "\n".join(lines)


# Expose tools list for agent setup
TOOLS = [
    query_database,
    semantic_search,
    get_affiliate_summary,
    draft_email,
    get_portfolio_health,
    get_leakage_status,
    get_seo_status,
]