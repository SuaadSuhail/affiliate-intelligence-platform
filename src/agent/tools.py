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
from typing import Optional

from langchain.tools import tool
from sqlalchemy import text

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

def _compose_email(affiliate_name: str, situation: str, tone: str) -> tuple[str, str]:
    """Compose (subject, body) for a re-engagement email. LLM-generated when
    available, template fallback otherwise. Pure composition — no DB, no
    side effects; draft_email() is the one that files the result for review."""
    llm = _get_llm()
    if llm:
        try:
            from langchain_core.messages import HumanMessage
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
            return _split_subject_body(email_text)
        except Exception:
            pass  # fall through to template

    # Template fallback when LLM unavailable
    first_name = affiliate_name.split()[0] if affiliate_name else "there"
    body = (
        f"Hi {first_name},\n\n"
        f"I wanted to reach out personally given recent activity on your account. "
        f"Situation context: {situation}.\n\n"
        f"I'd love to jump on a quick 20-minute call to discuss how we can best support you. "
        f"When works for you this week?\n\n"
        f"Tone: {tone}\n\n"
        f"[Your Name]\nPartner Success Team"
    )
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
def draft_email(input_str: str) -> str:
    """Draft a personalised re-engagement or follow-up email for an affiliate.
    Input should be a string containing: affiliate name, their current situation
    (scores, recent behaviour), and the desired tone (urgent, warm, neutral).
    Use this as the final step after understanding an affiliate's situation.
    This does NOT send anything — it files the draft as a pending request in
    the approval queue (status: waiting_for_review). A human must approve it
    via POST /approvals/{id}/approve before it is ever sent.
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

        subject, body = _compose_email(aff.name, situation, tone)

        # No contact email is stored in this schema (see src.storage.models.Affiliate)
        # — flagged clearly rather than fabricating a plausible-looking address.
        to_placeholder = f"{aff.name} <no email on file>"

        approval = ApprovalRequest(
            kind="email",
            affiliate_id=aff.id,
            payload={
                "to": to_placeholder,
                "subject": subject,
                "body": body,
                "affiliate_id": str(aff.id),
            },
            status="waiting_for_review",
        )
        db.add(approval)
        db.commit()
        db.refresh(approval)

        logger.info(
            "Draft email filed for approval",
            extra={"approval_id": str(approval.id), "affiliate_id": str(aff.id)},
        )

        return (
            f"Draft created for {aff.name} — pending approval as request {approval.id}.\n"
            f"Subject: {subject}\n\n"
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
    all three signal types this system tracks (churn/growth tier, leaks, SEO
    trend), not just the rulebook tier, so check this tool's leak and SEO
    counts/names too, not only the at-risk list."""
    db = _get_db()
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
            f"At-risk (50-80% churn):  {len(at_risk)} affiliate(s)",
            f"High-growth (>50%):      {len(high_growth)} affiliate(s)",
            f"Critical (churn > 80%):  {len(churned)} affiliate(s)",
            f"Active promo-code leaks: {len(leaking)} affiliate(s)  (separate signal — not a tier)",
            f"Declining SEO trend:     {len(declining_seo)} affiliate(s)  (separate signal — not a tier)",
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

        if leaking:
            lines += ["", "─── Active Leak Names ───────────"]
            lines.append("  " + ", ".join(a.name for a in leaking))

        if declining_seo:
            lines += ["", "─── Declining SEO Names ─────────"]
            lines.append("  " + ", ".join(a.name for a in declining_seo))

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

        # Fast path: has_active_leak is recomputed from the full leaked_codes
        # table on every scan (src.scraping.leakage_scraper) — if it's False
        # (or the affiliate doesn't exist), querying LeakedCode would
        # necessarily return nothing, so skip it. Only a safe drop-in for the
        # "no leak" case — when True, the actual rows are still needed below
        # for site/code/url detail, which the flag alone cannot provide.
        recent = []
        if aff and aff.has_active_leak:
            recent = (
                db.query(LeakedCode)
                .filter(LeakedCode.affiliate_id == aff_uuid)
                .order_by(LeakedCode.found_at.desc())
                .limit(5)
                .all()
            )
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

        # Fast path: search_trend is recomputed from the full seo_signals
        # table on every check (src.seo.checker) — if the affiliate has no
        # tracked_keyword (or doesn't exist), querying SeoSignal would
        # necessarily return nothing, so skip it.
        recent = []
        if aff and aff.tracked_keyword:
            recent = (
                db.query(SeoSignal)
                .filter(SeoSignal.affiliate_id == aff_uuid)
                .order_by(SeoSignal.checked_at.desc())
                .limit(5)
                .all()
            )
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