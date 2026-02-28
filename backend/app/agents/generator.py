"""Answer Generator Agent — produces the final response."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.state import AgentState
from app.services.llm import get_llm

RAG_PROMPT = """You are a helpful research paper reading assistant. Answer the user's question based on the provided paper context.

Rules:
- Answer in the same language as the user's question.
- Use the provided context to give accurate, specific answers.
- If the context doesn't contain enough information, say so honestly.
- Reference specific sections, equations, or figures when relevant.
- Support LaTeX formatting for math expressions using $...$ or $$...$$.
- Keep answers concise but thorough.

Paper context:
{context}"""

CHAT_PROMPT = """You are a friendly research paper reading assistant. The user is having a general conversation.
Answer in the same language as the user's question. Be helpful and concise."""

SUMMARIZE_PROMPT = """You are a research paper summarization assistant. Provide a structured summary based on the paper context.

Include:
1. Main objective/research question
2. Key methodology
3. Core findings/contributions
4. Limitations (if mentioned)

Answer in the same language as the user's question. Support LaTeX for math.

Paper context:
{context}"""


async def answer_rag_node(state: AgentState) -> dict:
    """Generate answer using retrieved context."""
    context_text = "\n\n---\n\n".join(state.get("context", []))
    system = RAG_PROMPT.format(context=context_text)

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        *state["messages"],
        HumanMessage(content=state["question"]),
    ])

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ],
    }


async def answer_chat_node(state: AgentState) -> dict:
    """Generate answer for general chat (no retrieval needed)."""
    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=CHAT_PROMPT),
        *state["messages"],
        HumanMessage(content=state["question"]),
    ])

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ],
    }


async def answer_summarize_node(state: AgentState) -> dict:
    """Generate a structured summary using retrieved context."""
    context_text = "\n\n---\n\n".join(state.get("context", []))
    system = SUMMARIZE_PROMPT.format(context=context_text)

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        *state["messages"],
        HumanMessage(content=state["question"]),
    ])

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ],
    }