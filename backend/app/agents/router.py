"""Router Agent — classifies user intent."""

import re

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.services.llm import get_llm

ROUTER_PROMPT = """You are an intent router for a paper reading assistant.
Given the user's message, classify it into one of these categories:

- "rag": The user is asking a question about the paper's content that requires retrieval from the document (specific facts, methods, results, equations, etc.)
- "summarize": The user wants a summary of the paper, key contributions, or an overview.
- "chat": The user is making casual conversation, asking a general knowledge question, or something not directly about the paper content.

Respond with ONLY one word: rag, summarize, or chat."""

SUMMARY_PATTERNS = (
    r"\bsummary\b",
    r"\boverview\b",
    r"\bkey contributions?\b",
    r"\bsummarize\b",
    r"总结",
    r"概述",
    r"摘要",
    r"主要贡献",
)


def _fallback_route(question: str) -> str:
    normalized = question.strip().lower()
    if any(re.search(pattern, normalized) for pattern in SUMMARY_PATTERNS):
        return "summarize"
    return "rag"


async def router_node(state: AgentState) -> dict:
    """Classify user intent."""
    try:
        llm = get_llm(temperature=0)
        response = await llm.ainvoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"User message: {state['question']}"),
        ])

        route_text = response.content.strip().lower()
        if "rag" in route_text:
            route = "rag"
        elif "summar" in route_text:
            route = "summarize"
        else:
            route = "chat"
        return {"route": route}
    except Exception:
        return {"route": _fallback_route(state["question"])}


def route_decision(state: AgentState) -> str:
    """Conditional edge: return the route string for graph branching."""
    return state.get("route", "chat")
