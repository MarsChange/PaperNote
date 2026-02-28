"""LangGraph state definition for the multi-agent RAG pipeline."""

from typing import Annotated, Literal
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
    route: Literal["rag", "chat", "summarize"] | None

    # Retrieved chunks from vector store
    context: list[str]

    # Final answer
    answer: str