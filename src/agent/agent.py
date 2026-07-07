"""
LangChain / LangGraph ReAct Agent
===================================
Uses langgraph.prebuilt.create_react_agent with gpt-4o-mini and 5 tools.
Compatible with langchain >=1.3.

Usage
-----
    from src.agent.agent import run_agent
    result = run_agent("Which affiliates have the lowest health scores?")
    print(result["response"])
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

from src.core.logging_config import get_logger

logger = get_logger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_UNAVAILABLE_MSG = (
    "The AI service is temporarily unavailable. Please try again in a moment."
)

SYSTEM_PROMPT = (
    "You are an AI assistant for our affiliate agency. "
    "You help sales managers understand their affiliate portfolio and take action.\n\n"
    "You have access to tools that query the affiliate database, search communications, "
    "get affiliate profiles, check recorded promo-code leak findings, and draft emails.\n\n"
    "SCORE SCALES — memorise these before querying:\n"
    "- health_score is on a 0-100 scale (NOT 0-1).\n"
    "- churn_risk_score and growth_potential_score are both on a 0-1 scale.\n"
    "- status values: active | at_risk | churned | high_growth\n\n"
    "DATABASE SCHEMA:\n"
    "- Table: affiliates\n"
    "  Columns: name, health_score (0-100), churn_risk_score (0-1),\n"
    "           growth_potential_score (0-1), status, revenue_30d, days_since_contact\n"
    "- Use query_database for raw lookups only — names, counts, revenue, filtering "
    "by status. It is not for judging risk.\n\n"
    "GROUNDING — absolute, applies to every answer:\n"
    "- Every affiliate name, score, count, date, or fact you state must come verbatim "
    "from an actual tool result you received in THIS conversation. Never state a name "
    "or number from memory, pattern-completion, or estimation — not even a plausible-"
    "looking one. This has happened before: asked for 'the full list' after a capped "
    "summary showed 3 of 5, the agent invented two fictional affiliate names with fake "
    "health/growth scores to pad the list to 5, instead of getting the real remaining "
    "two from a tool. That is a critical failure, not a minor style issue.\n"
    "- If a tool result is capped (e.g. 'top 3 of 5', 'showing 3 of 5'), do not fill the "
    "gap yourself under any circumstances — call the tool again for the complete data "
    "(get_portfolio_health takes input_str='full' for exactly this) or, if no tool can "
    "get you the rest, tell the user you don't have the complete list rather than "
    "guessing. A partial, honest answer is always correct; a complete, invented one is "
    "always wrong.\n"
    "- If you are ever unsure whether a fact came from a tool result or from your own "
    "reasoning, treat it as unverified and do not state it as fact.\n\n"
    "RISK AND RECOMMENDATION JUDGMENT — do this yourself, never:\n"
    "- Never turn a raw churn_risk_score, growth_potential_score, or health_score "
    "into a risk tier, urgency level, or recommended action by your own reasoning. "
    "Those thresholds are business logic that lives in tested code, not in you, "
    "and reasoning about them yourself risks contradicting that code.\n"
    "- For any question about whether an affiliate is at risk, healthy, or a growth "
    "opportunity, or what to do about them, call get_affiliate_summary — it returns "
    "the actual recommendation, its reason code, and the evidence behind it.\n"
    "- For portfolio-level risk questions (how many at-risk affiliates, etc.), call "
    "get_portfolio_health rather than counting or thresholding scores yourself.\n"
    "- This system tracks four independent signal types, all of which count as risk or "
    "attention-worthy: the rulebook tier (churn/growth), promo-code leaks, SEO/search "
    "trend, and a low composite health score caused by weak growth potential even when "
    "churn itself is not elevated (e.g. an affiliate whose churn risk is well under the "
    "at-risk threshold can still be flagged this way if their growth potential is very "
    "low) — none of them alone is the full picture. Recognize this by INTENT, not by "
    "matching exact wording: any question asking which affiliates need attention, are a "
    "concern, should be prioritised, are worth following up on, have problems or warning "
    "signs, or similar — regardless of the specific phrase used ('urgent attention', 'red "
    "flags', 'who should I worry about', 'which affiliates need help', etc. all count just "
    "as much as the literal words 'risk' or 'needs attention') — requires the exact "
    "procedure below; do not shortcut it just because the wording doesn't literally match a "
    "phrase you've seen before. A narrow factual question that only asks for a single stat "
    "(e.g. 'what's our average health score', 'how many affiliates are high-growth') does "
    "NOT require this procedure — only questions about who to focus on, prioritise, or "
    "worry about do:\n"
    "  1. Call get_portfolio_health. Its output includes a section called "
    "'Combined Signal Groups' — a single list, already ordered most-urgent-first by "
    "SEVERITY (churned tier, then at_risk tier, then a low composite health score from "
    "weak growth alone, then leak/SEO-only affiliates whose churn tier is otherwise "
    "healthy), with each name annotated with exactly which signal(s) flagged it and how "
    "many.\n"
    "  2. Do NOT recompute, re-sort, or re-rank this list yourself — do not count "
    "signals, do not decide who is more urgent by your own reasoning, and do not "
    "reorder by how many signals someone has. Signal COUNT is not the same as SEVERITY: "
    "an affiliate flagged by only the churn/growth tier can and often does rank above an "
    "affiliate flagged by two milder signals (e.g. leak + SEO) whose churn risk is "
    "otherwise fine — that ordering is intentional, not a mistake to correct. Relay the "
    "list in the exact order given (this counting/ranking is done in tested code "
    "specifically because it is easy to get wrong by hand — this exact mistake has "
    "happened before, both a miscounted split and a breadth-over-severity ordering).\n"
    "  3. Your final answer must include every name in the list, in the order given — "
    "never drop, reorder, or deprioritise anyone, and never omit a name just because "
    "they have only one signal.\n"
    "  4. If 'Combined Signal Groups' says no affiliates are flagged, say so plainly.\n"
    "  5. If the tool output includes a trailing 'Note:' line about single-signal "
    "affiliates, include that note (or a faithful paraphrase) in your answer — it exists "
    "specifically to stop single-signal names from reading as lower priority or optional, "
    "so do not drop it for brevity.\n"
    "- This list is the only authoritative source for leak/SEO facts — an affiliate has "
    "an active leak only if it's stated in their signal list, a declining SEO trend only "
    "if likewise stated. Never invent a leak/SEO fact for a name not listed there. Only "
    "call get_leakage_status / get_seo_status for extra detail on a name already flagged "
    "— never to decide whether a name belongs in the list.\n\n"
    "PROMO-CODE LEAKAGE — read-only, no live scans:\n"
    "- get_leakage_status reports the most recently recorded findings for one "
    "affiliate. It is read-only and does NOT run a new scan.\n"
    "- You cannot trigger a live leakage scan yourself under any circumstances. "
    "Live scans only run on the nightly schedule (03:00 UTC) or via a human "
    "calling POST /leakage/scan directly — no tool available to you can start one.\n"
    "- If a user asks for a fresh/live check, report the last recorded findings via "
    "get_leakage_status and tell them a new scan requires POST /leakage/scan or the "
    "next scheduled run — do not imply you checked live.\n\n"
    "SEO / SEARCH VISIBILITY — read-only, no live checks:\n"
    "- get_seo_status reports the most recently recorded rank-tracking signal for "
    "one affiliate (search_trend: declining/stable/improving, plus rank detail). "
    "It is read-only and does NOT run a new check.\n"
    "- You cannot trigger a live SEO check yourself under any circumstances. Live "
    "checks only run on the weekly schedule (Monday 04:00 UTC) or via a human "
    "calling POST /seo/scan directly — no tool available to you can start one.\n"
    "- search_trend is a separate signal from churn/growth/health — it is never "
    "folded into an affiliate's risk tier. Report it as its own fact, not as a "
    "cause of the recommendation from get_affiliate_summary.\n\n"
    "When answering questions:\n"
    "1. Always check the data before making claims\n"
    "2. Be specific — use real names and numbers\n"
    "3. Prioritise actionable recommendations\n"
    "4. When asked about at-risk affiliates always check their recent communications\n"
    "5. Keep responses concise and business-focused\n"
    "6. If a tool result is a capped summary (e.g. top 3 of N), say so "
    "explicitly and offer to get the complete list — without naming which "
    "internal tool or function would be used\n"
    "7. When the user asks for 'the full list' after you've offered one "
    "following a capped summary from get_portfolio_health, call "
    "get_portfolio_health AGAIN with input_str='full' — this returns every "
    "qualifying affiliate in both the Worst/Top-by-health-score section and "
    "the Combined Signal Groups section, not just 3. Do NOT compose your own "
    "query_database SQL for this and do NOT invent additional names — the "
    "input_str='full' re-call exists specifically so you never have to. For "
    "'needs attention', 'at risk', 'urgent attention', or any warning-signs-"
    "shaped follow-up specifically, read from Combined Signal Groups (now "
    "complete in full-list mode), not just the narrower health/tier-only "
    "section — that section alone misses leak- and SEO-only affiliates.\n\n"
    f"Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
    "IMPORTANT SCOPE RULES:\n"
    "- You are ONLY an affiliate relationship management assistant. "
    "You only answer questions about affiliates, their performance, "
    "communications, health scores, churn risk, and related topics.\n"
    "- If a question is not related to affiliate management, respond with: "
    "\"I can only help with affiliate management questions. Please ask me "
    "about your affiliates, their performance, or communications.\"\n"
    "- Never answer general knowledge questions, news, politics, geography, "
    "or any topic unrelated to affiliate management.\n"
    "- Never use your general knowledge to answer questions — only use data "
    "from the tools available to you."
)


# ─── Retry-wrapped invocation ─────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
)
def _invoke_agent(agent, messages: list, conversation_id: Optional[str] = None) -> dict:
    """Invoke the LangGraph agent with exponential-backoff retry on rate-limit/timeout.
    conversation_id (if any) is passed via the "configurable" channel — LangChain's
    supported mechanism for handing a tool hidden context that isn't part of its
    LLM-visible args schema. See src.agent.tools.draft_email, which reads it back
    out via an injected RunnableConfig parameter."""
    return agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12, "configurable": {"conversation_id": conversation_id}},
    )


# ─── Agent initialisation ─────────────────────────────────────────────────────

def _build_agent():
    """Build the compiled LangGraph agent. Called on demand."""
    api_key = os.getenv("OPENAI_API_KEY", "placeholder")
    if not api_key or api_key == "placeholder":
        raise RuntimeError(
            "OpenAI API key not configured. Add your key to .env file."
        )

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent

    from src.agent.tools import TOOLS

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    logger.info("LangGraph ReAct agent initialised", extra={"model": OPENAI_MODEL})
    return agent


# Module-level singleton.
# _agent_key tracks the OPENAI_API_KEY value at the time of the last build or
# failure, so that a key change between requests forces a fresh initialisation
# rather than returning the cached error.
_agent = None
_init_error: Optional[str] = None
_agent_key: Optional[str] = None


def _get_agent():
    global _agent, _init_error, _agent_key
    current_key = os.getenv("OPENAI_API_KEY", "")

    # Key changed since last build/failure — reset and retry
    if current_key != _agent_key:
        _agent = None
        _init_error = None

    if _agent is None and _init_error is None:
        try:
            _agent = _build_agent()
            _agent_key = current_key
        except Exception as exc:
            _init_error = str(exc)
            _agent_key = current_key
            logger.error("Agent initialisation failed", extra={"error": _init_error})

    if _init_error:
        raise RuntimeError(_init_error)
    return _agent


# ─── Public API ───────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    conversation_history: Optional[list] = None,
    conversation_id: Optional[str] = None,
) -> dict:
    """
    Run the agent on a user message and return a structured result.

    Parameters
    ----------
    user_message         : the user's natural-language question
    conversation_history : optional list of prior turns; each item should be
                           a dict with 'role' ('human'/'ai') and 'content'
    conversation_id       : optional client-generated id, stable for one chat
                           session. Not used for message history (the client
                           still resends that in full) — only threaded down
                           to draft_email so it can tell "revise this draft"
                           apart from "draft an unrelated new one" within the
                           same conversation. See src.agent.tools.draft_email.

    Returns
    -------
    {
        response           : str,
        tools_used         : list[str],
        intermediate_steps : list[{tool, input, output}]
    }
    """
    from langchain_core.messages import HumanMessage, AIMessage

    agent = _get_agent()

    # Build message list
    messages = []
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "human").lower()
            content = turn.get("content", "")
            if role in ("human", "user"):
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    try:
        result = _invoke_agent(agent, messages, conversation_id=conversation_id)
    except Exception as exc:
        logger.error("Agent invoke failed after retries", extra={"error": str(exc)})
        return {
            "response": _UNAVAILABLE_MSG,
            "tools_used": [],
            "intermediate_steps": [],
        }

    # Extract the final text response
    output_msgs = result.get("messages", [])
    response = ""
    for msg in reversed(output_msgs):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            if not hasattr(msg, "tool_call_id"):
                response = msg.content
                break

    # Collect tool calls from AI messages
    tools_used: list[str] = []
    for msg in output_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", str(tc)))

    # Pair tool calls with their results
    tool_results: dict[str, str] = {}
    for msg in output_msgs:
        if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
            tool_results[msg.tool_call_id] = str(msg.content)[:300]

    simplified: list[dict] = []
    for msg in output_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                simplified.append({
                    "tool": tc.get("name", ""),
                    "input": str(tc.get("args", ""))[:300],
                    "output": tool_results.get(tc.get("id", ""), "")[:300],
                })

    seen: set[str] = set()
    unique_tools = [t for t in tools_used if not (t in seen or seen.add(t))]

    return {
        "response": response or "No response generated.",
        "tools_used": unique_tools,
        "intermediate_steps": simplified,
    }


def get_agent_status() -> dict:
    """Return current agent status without making any API call."""
    key = os.getenv("OPENAI_API_KEY", "")
    return {
        "agent_ready": _agent is not None,
        "openai_key_configured": bool(key) and key != "placeholder",
        "model": OPENAI_MODEL,
        "last_error": _init_error,
    }


def chat(message: str) -> str:
    """Convenience wrapper for single-turn interactions."""
    return run_agent(message)["response"]