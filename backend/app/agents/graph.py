"""LangGraph entrypoint for PaperNote's superMew-style Agentic RAG."""

from app.agents.agentic_rag import build_agentic_graph

rag_graph = build_agentic_graph()
