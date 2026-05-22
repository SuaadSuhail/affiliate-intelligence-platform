"""
LangChain Tool Definitions
==========================
Six tools exposed to the ReAct agent:

  1. query_affiliates     — SQL query affiliate + score data
  2. search_communications — semantic search over ChromaDB
  3. summarise_affiliate   — narrative health summary
  4. draft_email           — personalised outreach draft
  5. flag_risk             — create urgent risk flag in DB
  6. run_scoring           — trigger re-scoring pipeline
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from langchain.tools import tool
from sqlalchemy.orm import Session

from src.storage.database import db_session
from src.storage.models import Affiliate, Communication, ScoreHistory
from src.ingestion.embedding_generator import get_generator
from src.ml.explainability import top_risk_drivers


# ─── Tool 1: Query affiliates ─────────────────────────────────────────────────

@tool
def query_affiliates(query: str) -> str:
    """
    Query the affiliate database. Accepts natural-language queries such as:
      - "list all gold tier affiliates"
      - "affiliates with churn_risk_score > 0.6"
      - "top 5 affiliates by monthly revenue"
      - "affiliates in finance niche"

    Returns a JSON string with matching affiliate summaries.
    """
    lower = query.lower()
    with db_session() as db:
        q = db.query(Affiliate)

        # Simple natural-language filter parsing
        if "churn" in lower and any(op in lower for op in [">", "above", "high"]):
            q = q.filter(Affiliate.churn_risk_score > 0.5).order_by(
                Affiliate.churn_risk_score.desc()
            )
        elif "growth" in lower and any(op in lower for op in [">", "above", "high"]):
            q = q.filter(Affiliate.growth_potential_score > 0.5).order_by(
                Affiliate.growth_potential_score.desc()
            )
        elif "platinum" in lower:
            q = q.filter(Affiliate.tier == "platinum")
        elif "gold" in lower:
            q = q.filter(Affiliate.tier == "gold")
        elif "silver" in lower:
            q = q.filter(Affiliate.tier == "silver")
        elif "bronze" in lower:
            q = q.filter(Affiliate.tier == "bronze")
        elif "revenue" in lower:
            q = q.order_by(Affiliate.monthly_revenue.desc())
        elif "health" in lower:
            q = q.order_by(Affiliate.health_score.desc())

        # Niche filter
        for niche in ["finance", "travel", "gaming", "saas", "e-commerce",
                       "health", "wellness", "education", "lifestyle"]:
            if niche in lower:
                q = q.filter(Affiliate.niche.ilike(f"%{niche}%"))
                break

        # Limit
        limit = 10
        for word, n in [("top 5", 5), ("top 3", 3), ("top 10", 10), ("all", 100)]:
            if word in lower:
                limit = n
                break
        affiliates = q.limit(limit).all()

        result = [
            {
                "id": str(a.id),
                "name": a.name,
                "email": a.email,
                "company": a.company,
                "tier": a.tier,
                "niche": a.niche,
                "monthly_revenue": a.monthly_revenue,
                "churn_risk_score": a.churn_risk_score,
                "growth_potential_score": a.growth_potential_score,
                "health_score": a.health_score,
                "last_contact_date": a.last_contact_date.isoformat() if a.last_contact_date else None,
            }
            for a in affiliates
        ]
    return json.dumps(result, indent=2)


# ─── Tool 2: Search communications ───────────────────────────────────────────

@tool
def search_communications(query: str) -> str:
    """
    Semantic search over all affiliate communications stored in ChromaDB.
    Accepts natural-language queries such as:
      - "affiliates complaining about payments"
      - "emails mentioning Black Friday campaigns"
      - "calls where affiliate threatened to leave"

    Returns top-5 matching communication excerpts with metadata.
    """
    gen = get_generator()
    results = gen.search_communications(query=query, n_results=5)
    if not results:
        return "No matching communications found."

    output = []
    for r in results:
        meta = r.get("metadata", {})
        snippet = r.get("document", "")[:300]
        output.append(
            f"[{meta.get('channel', '?').upper()} | {meta.get('direction', '?')} | "
            f"affiliate={meta.get('affiliate_id', '?')} | "
            f"sentiment={meta.get('sentiment_label', '?')} | "
            f"tags={meta.get('tags', '')}]\n"
            f"{snippet}…\n"
        )
    return "\n---\n".join(output)


# ─── Tool 3: Summarise affiliate ──────────────────────────────────────────────

@tool
def summarise_affiliate(affiliate_id: str) -> str:
    """
    Generate a structured narrative health summary for a single affiliate.
    Input: affiliate UUID string or email address.

    Returns a human-readable summary including:
    - Profile overview
    - Current health scores
    - Recent communication sentiment
    - Top risk / growth drivers (SHAP-based)
    - Recommended next action
    """
    with db_session() as db:
        # Accept UUID or email
        if "@" in affiliate_id:
            aff = db.query(Affiliate).filter_by(email=affiliate_id).first()
        else:
            aff = db.query(Affiliate).filter(
                Affiliate.id == affiliate_id
            ).first()

        if not aff:
            return f"Affiliate not found: {affiliate_id}"

        # Recent communications
        recent_comms = (
            db.query(Communication)
            .filter(Communication.affiliate_id == aff.id)
            .order_by(Communication.occurred_at.desc())
            .limit(5)
            .all()
        )

        # Latest score history
        latest_score = (
            db.query(ScoreHistory)
            .filter(ScoreHistory.affiliate_id == aff.id)
            .order_by(ScoreHistory.scored_at.desc())
            .first()
        )

        comm_summary = []
        for c in recent_comms:
            tags_str = ", ".join(c.tags) if c.tags else "none"
            comm_summary.append(
                f"  • [{c.channel.upper()}] {c.occurred_at.strftime('%Y-%m-%d') if c.occurred_at else '?'} "
                f"— sentiment: {c.sentiment_label} | tags: {tags_str}"
            )

    # SHAP drivers
    try:
        churn_drivers = top_risk_drivers(str(aff.id), model_type="churn")[:3]
        growth_drivers = top_risk_drivers(str(aff.id), model_type="growth")[:3]
    except Exception:
        churn_drivers = []
        growth_drivers = []

    # Recommended action
    if aff.churn_risk_score >= 0.65:
        action = "⚠️  URGENT: Schedule retention call within 48 hours."
    elif aff.growth_potential_score >= 0.70:
        action = "🚀  Growth opportunity: Propose scale-up plan and commission uplift."
    elif aff.churn_risk_score >= 0.45:
        action = "📞  Monitor closely: Follow up within 7 days to check satisfaction."
    else:
        action = "✅  Healthy affiliate: Maintain regular check-in cadence."

    lines = [
        f"═══ AFFILIATE HEALTH SUMMARY ═══",
        f"Name:     {aff.name}",
        f"Company:  {aff.company or 'N/A'}",
        f"Email:    {aff.email}",
        f"Tier:     {aff.tier.upper()}  |  Niche: {aff.niche or 'N/A'}  |  Country: {aff.country or 'N/A'}",
        f"Revenue:  ${aff.monthly_revenue:,.2f}/month",
        f"",
        f"─── Scores ───────────────────",
        f"Health Score:          {aff.health_score:.1f}/100",
        f"Churn Risk:            {aff.churn_risk_score:.2%}",
        f"Growth Potential:      {aff.growth_potential_score:.2%}",
        f"Last Contact:          {aff.last_contact_date.strftime('%Y-%m-%d') if aff.last_contact_date else 'Never'}",
        f"",
        f"─── Recent Communications ────",
    ] + (comm_summary or ["  No communications on record."]) + [
        f"",
        f"─── Risk Drivers (Churn) ─────",
        f"  {', '.join(churn_drivers) if churn_drivers else 'Insufficient data'}",
        f"",
        f"─── Growth Drivers ───────────",
        f"  {', '.join(growth_drivers) if growth_drivers else 'Insufficient data'}",
        f"",
        f"─── Recommended Action ───────",
        f"  {action}",
    ]
    return "\n".join(lines)


# ─── Tool 4: Draft email ──────────────────────────────────────────────────────

@tool
def draft_email(input_json: str) -> str:
    """
    Draft a personalised outreach email for an affiliate.

    Input must be a JSON string with keys:
      - affiliate_id : UUID or email of the affiliate
      - goal         : one of "retention", "growth", "check_in", "payment_follow_up",
                       "feature_announcement", "bfcm_campaign"

    Returns a ready-to-send email draft (Subject + Body).
    """
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError:
        return "Error: input must be valid JSON with keys affiliate_id and goal."

    affiliate_id = params.get("affiliate_id", "")
    goal = params.get("goal", "check_in")

    with db_session() as db:
        if "@" in affiliate_id:
            aff = db.query(Affiliate).filter_by(email=affiliate_id).first()
        else:
            aff = db.query(Affiliate).filter(Affiliate.id == affiliate_id).first()
        if not aff:
            return f"Affiliate not found: {affiliate_id}"
        name = aff.name.split()[0]  # first name
        company = aff.company or "your company"
        tier = aff.tier
        revenue = aff.monthly_revenue
        niche = aff.niche or "your niche"

    templates = {
        "retention": f"""Subject: We value your partnership — let's talk

Hi {name},

I wanted to reach out personally because we noticed some recent activity on your account that made me want to check in directly.

You've been a valued {tier.upper()} partner since joining us, and the work you do in the {niche} space — currently driving ${revenue:,.0f}/month — is genuinely impressive. I'd hate to think there's anything we could be doing better that we haven't addressed.

Would you be open to a 20-minute call this week? I'd love to hear what's on your mind and see how we can better support {company}'s goals.

Looking forward to hearing from you.

Warm regards,
[Your Name]
Partner Success Team""",

        "growth": f"""Subject: Let's unlock your next revenue milestone

Hi {name},

Your recent performance in the {niche} space has been outstanding — and I think there's a real opportunity for us to grow together even further.

Based on what we're seeing from your campaigns, I believe we could get {company} from ${revenue:,.0f}/month to significantly higher with a few targeted optimisations. I'd like to explore:

• Co-branded landing pages tailored to your top traffic sources
• An enhanced commission structure for new campaign launches
• Early access to our upcoming seasonal promotions toolkit

Can we set up a quick 30-minute strategy call? I'll come prepared with specific data and ideas.

Excited about what we can build together!

Best,
[Your Name]
Partner Growth Team""",

        "check_in": f"""Subject: Quick check-in — how are things going?

Hi {name},

Just wanted to touch base and see how things are going on your end.

It's been a little while since we last spoke, and I always like to make sure our {tier.upper()} partners have everything they need to succeed.

Is there anything on your radar — upcoming campaigns, technical questions, or ideas you'd like to explore? I'm here to help.

Hope all is well at {company}!

Cheers,
[Your Name]
Partner Success Team""",

        "payment_follow_up": f"""Subject: Payment update for your account

Hi {name},

I'm following up regarding your recent payment query. I want to make sure this is resolved as quickly as possible for you.

I've escalated your case to our Finance team and you should receive an update within 1 business day. If you have a specific invoice reference or date range in question, please reply with those details and I'll fast-track it.

Apologies for any inconvenience — we take payment accuracy very seriously.

Kind regards,
[Your Name]
Partner Support Team""",

        "bfcm_campaign": f"""Subject: 🛍️ Your exclusive BFCM toolkit is ready

Hi {name},

Black Friday / Cyber Monday is just around the corner — and we've built something special for our {tier.upper()} partners.

As one of our top performers in the {niche} space, you get early access to:

✅ Co-branded BFCM landing pages (3 designs included)
✅ Boosted commission rate: 30% on all BFCM conversions
✅ Dedicated tracking dashboard for the holiday season
✅ Priority support throughout November & December

Last year our top affiliates saw 3x their normal monthly revenue during BFCM. With ${revenue:,.0f}/month as your baseline, the potential here is huge.

Want to lock this in? Just reply to this email and I'll get everything set up for {company} within 24 hours.

Let's make this your best Q4 yet!

[Your Name]
Partner Growth Team""",
    }

    draft = templates.get(goal, templates["check_in"])
    return f"=== EMAIL DRAFT ({goal.upper()}) ===\n\n{draft}"


# ─── Tool 5: Flag risk ────────────────────────────────────────────────────────

@tool
def flag_risk(input_json: str) -> str:
    """
    Create an urgent risk flag for an affiliate by updating their churn_risk_score
    and logging a note in the score_history.

    Input must be a JSON string with keys:
      - affiliate_id : UUID or email
      - reason       : short description of the risk (stored in features.flag_reason)
      - urgency      : "low" | "medium" | "high" (default: "high")

    Returns confirmation string.
    """
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError:
        return "Error: input must be valid JSON."

    affiliate_id = params.get("affiliate_id", "")
    reason = params.get("reason", "Manual risk flag")
    urgency = params.get("urgency", "high")

    urgency_boost = {"low": 0.1, "medium": 0.2, "high": 0.35}.get(urgency, 0.35)

    with db_session() as db:
        if "@" in affiliate_id:
            aff = db.query(Affiliate).filter_by(email=affiliate_id).first()
        else:
            aff = db.query(Affiliate).filter(Affiliate.id == affiliate_id).first()

        if not aff:
            return f"Affiliate not found: {affiliate_id}"

        # Boost churn risk score
        old_score = aff.churn_risk_score
        new_score = min(1.0, old_score + urgency_boost)
        aff.churn_risk_score = round(new_score, 4)
        aff.health_score = round(
            ((1 - aff.churn_risk_score) * 0.6 + aff.growth_potential_score * 0.4) * 100, 1
        )

        # Log in score_history
        entry = ScoreHistory(
            affiliate_id=aff.id,
            churn_risk_score=new_score,
            growth_potential_score=aff.growth_potential_score,
            health_score=aff.health_score,
            features={"flag_reason": reason, "flagged_by": "agent", "urgency": urgency},
            shap_values={},
            model_version="manual_flag",
        )
        db.add(entry)
        name = aff.name

    return (
        f"⚠️  Risk flag created for {name}.\n"
        f"Churn risk: {old_score:.2%} → {new_score:.2%}\n"
        f"Health score updated to {aff.health_score:.1f}/100\n"
        f"Reason: {reason}\nUrgency: {urgency.upper()}"
    )


# ─── Tool 6: Run scoring ──────────────────────────────────────────────────────

@tool
def run_scoring(affiliate_id: str = "") -> str:
    """
    Trigger the full scoring pipeline (churn + growth + SHAP) for one affiliate
    or all affiliates.

    Input: affiliate UUID string, email address, or empty string for all affiliates.
    Returns a summary of scores computed.
    """
    from src.ml.churn_model import predict as churn_predict
    from src.ml.growth_model import predict as growth_predict
    from src.ml.explainability import explain_affiliate

    affiliate_ids = [affiliate_id.strip()] if affiliate_id.strip() else None

    churn_df = churn_predict(affiliate_ids=affiliate_ids)
    growth_df = growth_predict(affiliate_ids=affiliate_ids)

    if churn_df.empty:
        return "No affiliates found to score."

    results = []
    with db_session() as db:
        for _, crow in churn_df.iterrows():
            aid = crow["affiliate_id"]
            grow_row = growth_df[growth_df["affiliate_id"] == aid]
            g_score = float(grow_row["growth_potential_score"].iloc[0]) if not grow_row.empty else 0.5
            c_score = float(crow["churn_risk_score"])
            h_score = round(((1 - c_score) * 0.6 + g_score * 0.4) * 100, 1)

            # SHAP
            try:
                churn_shap = explain_affiliate(aid, model_type="churn")
                growth_shap = explain_affiliate(aid, model_type="growth")
            except Exception:
                churn_shap = {}
                growth_shap = {}

            # Update affiliate row
            aff = db.query(Affiliate).filter(Affiliate.id == aid).first()
            if aff:
                aff.churn_risk_score = c_score
                aff.growth_potential_score = g_score
                aff.health_score = h_score

                # Log score history
                entry = ScoreHistory(
                    affiliate_id=aff.id,
                    churn_risk_score=c_score,
                    growth_potential_score=g_score,
                    health_score=h_score,
                    features=crow["features"] if isinstance(crow["features"], dict) else {},
                    shap_values={"churn": churn_shap, "growth": growth_shap},
                )
                db.add(entry)
                results.append(f"  {aff.name}: churn={c_score:.2%} growth={g_score:.2%} health={h_score}")

    summary = f"✅ Scored {len(results)} affiliate(s):\n" + "\n".join(results)
    return summary
