"""LangGraph pipeline — wires Router → Retriever → Generator into a StateGraph."""

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.router import router_node, route_decision
from app.agents.retriever import retriever_node
from app.agents.generator import answer_rag_node, answer_chat_node, answer_summarize_node


def build_graph() -> StateGraph:
    """
    Build the multi-agent RAG graph:

        ┌─────────┐
        │  Router  │
        └────┬─────┘
             │ route_decision
       ┌─────┼──────────┐
       ▼     ▼          ▼
     [rag] [summarize] [chat]
       │     │          │
       ▼     ▼          ▼
    Retriever Retriever  │
       │     │          │
       ▼     ▼          ▼
    RAG_Gen  Sum_Gen  Chat_Gen
       │     │          │
       └─────┴──────────┘
              ▼
             END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answer_rag", answer_rag_node)
    graph.add_node("answer_chat", answer_chat_node)
    graph.add_node("retriever_summarize", retriever_node)
    graph.add_node("answer_summarize", answer_summarize_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "retriever",
            "summarize": "retriever_summarize",
            "chat": "answer_chat",
        },
    )

    # RAG path: retriever → answer_rag → END
    graph.add_edge("retriever", "answer_rag")
    graph.add_edge("answer_rag", END)

    # Summarize path: retriever → answer_summarize → END
    graph.add_edge("retriever_summarize", "answer_summarize")
    graph.add_edge("answer_summarize", END)

    # Chat path: answer_chat → END
    graph.add_edge("answer_chat", END)

    return graph.compile()


# Singleton compiled graph
rag_graph = build_graph()