"""Router Agent — classifies user intent."""

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.services.llm import get_llm

ROUTER_PROMPT = """You are an intent router for a paper reading assistant.
Given the user's message, classify it into one of these categories:

- "rag": The user is asking a question about the paper's content that requires retrieval from the document (specific facts, methods, results, equations, etc.)
- "summarize": The user wants a summary of the paper, key contributions, or an overview.
- "chat": The user is making casual conversation, asking a general knowledge question, or something not directly about the paper content.

Respond with ONLY one word: rag, summarize, or chat."""


async def router_node(state: AgentState) -> dict:
    """Classify user intent."""
    llm = get_llm(temperature=0)
    response = await llm.ainvoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"User message: {state['question']}"),
    ])

    route_text = response.content.strip().lower()
    # Normalize
    if "rag" in route_text:
        route = "rag"
    elif "summar" in route_text:
        route = "summarize"
    else:
        route = "chat"

    return {"route": route}


def route_decision(state: AgentState) -> str:
    """Conditional edge: return the route string for graph branching."""
    return state.get("route", "chat")