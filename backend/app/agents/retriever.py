"""Retriever Agent — fetches relevant chunks from the vector store."""

from app.agents.state import AgentState
from app.services.vector_store import vector_store


async def retriever_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks for the user's question."""
    paper_id = state["paper_id"]
    question = state["question"]

    chunks = vector_store.search(paper_id, question, top_k=5)

    return {"context": chunks}