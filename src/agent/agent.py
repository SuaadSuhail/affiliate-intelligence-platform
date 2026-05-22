"""
LangChain ReAct Agent
=====================
Combines all six tools into a ReAct (Reason + Act) agent powered by GPT-4o.

The agent can answer questions, analyse affiliate health, draft emails,
flag risks, and trigger re-scoring — all through natural language.

Usage
-----
    from src.agent.agent import get_agent
    agent = get_agent()
    response = agent.invoke({"input": "Who are my highest churn risk affiliates?"})
    print(response["output"])
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.agent.tools import (
    query_affiliates,
    search_communications,
    summarise_affiliate,
    draft_email,
    flag_risk,
    run_scoring,
)

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

TOOLS = [
    query_affiliates,
    search_communications,
    summarise_affiliate,
    draft_email,
    flag_risk,
    run_scoring,
]

SYSTEM_PROMPT = """You are the Affiliate Intelligence Agent — an expert AI assistant for
an affiliate marketing partner success team.

You have access to the following tools:
{tools}

Your role is to help the team:
1. Identify at-risk affiliates (high churn probability) and recommend retention actions
2. Spot high-growth opportunities and suggest how to capitalise on them
3. Analyse communication patterns and sentiment across the affiliate portfolio
4. Draft personalised, empathetic outreach emails
5. Flag urgent risks and trigger re-scoring when needed

Health Score formula: ((1 - churn_risk) × 0.6 + growth_potential × 0.4) × 100

Always be data-driven. When discussing scores, explain what the key drivers are.
Be concise but thorough. Format tables and lists clearly.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def get_agent(
    model: str = OPENAI_MODEL,
    verbose: bool = True,
    memory_window: int = 10,
) -> AgentExecutor:
    """
    Build and return a LangChain ReAct AgentExecutor.

    Parameters
    ----------
    model         : OpenAI model ID (default: gpt-4o)
    verbose       : whether to print agent reasoning steps
    memory_window : number of conversation turns to retain in memory

    Returns
    -------
    AgentExecutor ready to call with .invoke({"input": "..."})
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.1,   # low temp for consistent analysis
        streaming=False,
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )

    memory = ConversationBufferWindowMemory(
        k=memory_window,
        return_messages=True,
        memory_key="chat_history",
    )

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )
    return executor


# ─── Module-level singleton ───────────────────────────────────────────────────

_agent_instance: Optional[AgentExecutor] = None


def get_agent_singleton() -> AgentExecutor:
    """Return (or lazily create) the module-level agent. Thread-unsafe — use per-request in prod."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = get_agent()
    return _agent_instance


def chat(message: str) -> str:
    """
    Convenience wrapper for single-turn agent interaction.

    Parameters
    ----------
    message : user message / question

    Returns
    -------
    Agent's final answer as a string
    """
    agent = get_agent_singleton()
    result = agent.invoke({"input": message})
    return result.get("output", "No response generated.")


# ─── Interactive CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Affiliate Intelligence Agent  (type 'exit') ║")
    print("╚══════════════════════════════════════════════╝\n")
    agent = get_agent(verbose=True)
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            response = agent.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
