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
    "- \"Warning signs\", \"risk\", and \"needs attention\" are broader than the churn/growth "
    "tier alone — this system tracks three independent signal types: the rulebook tier "
    "(churn/growth), promo-code leaks, and SEO/search trend. For any question asking which "
    "affiliates have warning signs, risk factors, or need attention, follow this exact "
    "procedure — do not shortcut it:\n"
    "  1. Call get_portfolio_health. Its output ends with three labelled lines: "
    "'At-Risk Names', 'Active Leak Names', 'Declining SEO Names' — each a literal "
    "comma-separated list of affiliate names.\n"
    "  2. Copy all three lists out verbatim, name by name.\n"
    "  3. Build ALL_NAMES = every name that appears on AT LEAST ONE of the three lists "
    "(a plain union — a name appearing on only one list still goes into ALL_NAMES).\n"
    "  4. Your final answer must cover every single name in ALL_NAMES — not a subset of "
    "it, not just the names that also happen to have a low health score. Split them into "
    "two labelled groups: 'Multiple warning signs' (names on 2 or more of the three "
    "lists) and 'Single warning sign — for awareness' (names on exactly 1 list). Even if "
    "the user asked specifically about 'multiple' signs, always include BOTH groups in "
    "full — never silently drop the single-signal group just because the question said "
    "'multiple'; a name with only one flag (e.g. only on Active Leak Names) must still be "
    "named and its one signal stated, just under the single-signal group rather than the "
    "multiple one.\n"
    "- The three lists are fully independent of each other and of health/growth scores: a "
    "name can be in 'Active Leak Names' while also appearing in get_portfolio_health's "
    "'Top 3 (performing well)' list — a high health score or good growth number is NEVER a "
    "valid reason to leave a name out of ALL_NAMES in step 3-4 above.\n"
    "- These name lists are the only authoritative source for leak/SEO facts — an "
    "affiliate has an active leak only if their exact name is on 'Active Leak Names', a "
    "declining SEO trend only if their exact name is on 'Declining SEO Names'. Never "
    "invent a leak/SEO fact for a name that isn't literally printed there. Only call "
    "get_leakage_status / get_seo_status for extra detail on a name already in ALL_NAMES "
    "— never to decide whether a name belongs in ALL_NAMES.\n\n"
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
    "5. Keep responses concise and business-focused\n\n"
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
def _invoke_agent(agent, messages: list) -> dict:
    """Invoke the LangGraph agent with exponential-backoff retry on rate-limit/timeout."""
    return agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12},
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
) -> dict:
    """
    Run the agent on a user message and return a structured result.

    Parameters
    ----------
    user_message         : the user's natural-language question
    conversation_history : optional list of prior turns; each item should be
                           a dict with 'role' ('human'/'ai') and 'content'

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
        result = _invoke_agent(agent, messages)
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