"""Retriever Agent — fetches evidence blocks from the hybrid store."""

from app.agents.state import AgentState
from app.services.vector_store import vector_store


async def retriever_node(state: AgentState) -> dict:
    """Retrieve relevant document evidence for the user's question."""
    paper_id = state["paper_id"]
    question = state["question"]
    sources = vector_store.search(paper_id, question, top_k=6)
    return {"context": sources, "sources": sources}


async def retriever_summary_node(state: AgentState) -> dict:
    """Retrieve broad coverage context for summarization prompts."""
    paper_id = state["paper_id"]
    sources = vector_store.get_overview_context(paper_id, max_items=10)
    return {"context": sources, "sources": sources}
