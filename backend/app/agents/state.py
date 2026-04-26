from __future__ import annotations

"""LangGraph state definition for the multi-agent RAG pipeline."""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State that flows through the LangGraph pipeline."""

    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Current user question
    question: str

    # Paper context
    paper_id: str

    # Router decision
    route: Optional[Literal["rag", "chat", "summarize"]]

    # Retrieved evidence blocks from hybrid retriever
    context: list[dict]

    # Final answer
    answer: str

    # Evidence passed back to the client
    sources: list[dict]
