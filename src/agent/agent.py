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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "placeholder")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are an AI assistant for our affiliate agency. "
    "You help sales managers understand their affiliate portfolio and take action.\n\n"
    "You have access to tools that query the affiliate database, search communications, "
    "get affiliate profiles, and draft emails.\n\n"
    "SCORE SCALES — memorise these before querying:\n"
    "- health_score is on a 0-100 scale (NOT 0-1).\n"
    "  Below 40 = needs urgent attention.\n"
    "  40-60 = monitor closely.\n"
    "  Above 60 = performing well.\n"
    "- churn_risk_score is on a 0-1 scale.\n"
    "  Above 0.6 = high churn risk.\n"
    "  Above 0.8 = critical / likely churned.\n"
    "- growth_potential_score is on a 0-1 scale.\n"
    "  Above 0.6 = strong growth opportunity.\n"
    "- status values: active | at_risk | churned | high_growth\n\n"
    "DATABASE SCHEMA:\n"
    "- Table: affiliates\n"
    "  Columns: name, health_score (0-100), churn_risk_score (0-1),\n"
    "           growth_potential_score (0-1), status, revenue_30d, days_since_contact\n"
    "- Always query the affiliates table directly to get current scores.\n"
    "- To find urgent affiliates: SELECT ... FROM affiliates "
    "WHERE health_score < 40 ORDER BY health_score ASC\n\n"
    "When answering questions:\n"
    "1. Always check the data before making claims\n"
    "2. Be specific — use real names and numbers\n"
    "3. Prioritise actionable recommendations\n"
    "4. When asked about at-risk affiliates always check their recent communications\n"
    "5. Keep responses concise and business-focused\n\n"
    f"Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
)


# ─── Agent initialisation ─────────────────────────────────────────────────────

def _build_agent():
    """Build the compiled LangGraph agent. Called once at module level."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "placeholder":
        raise RuntimeError(
            "OpenAI API key not configured. Add your key to .env file."
        )

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent

    from src.agent.tools import TOOLS

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


# Module-level singleton — lazy-initialised so startup doesn't fail
# when OPENAI_API_KEY is not configured.
# Reset both fields whenever the system prompt changes.
_agent = None
_init_error: Optional[str] = None


def _get_agent():
    global _agent, _init_error
    if _agent is None and _init_error is None:
        try:
            _agent = _build_agent()
        except Exception as exc:
            _init_error = str(exc)
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

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12},  # ~6 tool-call iterations
    )

    # Extract the final text response
    output_msgs = result.get("messages", [])
    response = ""
    for msg in reversed(output_msgs):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            # Skip tool messages
            if not hasattr(msg, "tool_call_id"):
                response = msg.content
                break

    # Collect tool calls from AI messages
    tools_used: list[str] = []
    simplified: list[dict] = []
    for msg in output_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", str(tc))
                tools_used.append(name)

    # Pair tool calls with their results
    tool_results: dict[str, str] = {}
    for msg in output_msgs:
        if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
            tool_results[msg.tool_call_id] = str(msg.content)[:300]

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


def chat(message: str) -> str:
    """Convenience wrapper for single-turn interactions."""
    return run_agent(message)["response"]