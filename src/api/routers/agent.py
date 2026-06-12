"""
Agent Router
============
LangChain ReAct agent endpoints.

POST /agent/chat   — full conversation with history support
POST /agent/quick  — single-turn query, no history
GET  /agent/demo   — runs 3 pre-set demo questions
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.auth import get_api_key

router = APIRouter()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[list] = None


class QuickRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str]
    message_count: int


class DemoResult(BaseModel):
    question: str
    response: str
    tools_used: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
def agent_chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the LangChain ReAct agent with optional conversation history.

    Conversation history is a list of {'role': 'human'/'ai', 'content': '...'} dicts.
    """
    from src.agent.agent import run_agent

    try:
        result = run_agent(request.message, request.conversation_history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    history_len = len(request.conversation_history or [])
    return ChatResponse(
        response=result["response"],
        tools_used=result["tools_used"],
        message_count=history_len + 1,
    )


@router.post("/quick", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
def agent_quick(request: QuickRequest) -> ChatResponse:
    """
    Single-turn agent query with no conversation history.
    Simpler than /agent/chat — use for testing or one-off questions.
    """
    from src.agent.agent import run_agent

    try:
        result = run_agent(request.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(
        response=result["response"],
        tools_used=result["tools_used"],
        message_count=1,
    )


@router.get("/demo", response_model=list[DemoResult])
def agent_demo() -> list[DemoResult]:
    """
    Run 3 pre-set demo questions through the agent automatically.
    Returns all 3 responses with the tools used for each.
    Use this to demonstrate the agent's capabilities end-to-end.
    """
    from src.agent.agent import run_agent

    demo_questions = [
        "Which affiliates have the lowest health scores right now?",
        "What is happening with Tom Bauer and what should I do?",
        "Draft a re-engagement email for the most at-risk affiliate",
    ]

    results: list[DemoResult] = []
    for question in demo_questions:
        try:
            r = run_agent(question)
            results.append(DemoResult(
                question=question,
                response=r["response"],
                tools_used=r["tools_used"],
            ))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            results.append(DemoResult(
                question=question,
                response=f"Error: {exc}",
                tools_used=[],
            ))

    return results