"""superMew-style Agentic RAG graph adapted to PaperNote papers."""

from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents.generator import (
    RAG_PROMPT,
    SUMMARIZE_PROMPT,
    _invoke_with_fallback,
    _sanitize_sources,
    answer_chat_node,
)
from app.agents.router import route_decision, router_node
from app.agents.state import AgentState
from app.core.config import settings
from app.services.agentic_runtime import emit_rag_step
from app.services.llm import get_llm
from app.services.vector_store import vector_store


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if evidence is relevant, otherwise 'no'")


class RewriteStrategy(BaseModel):
    strategy: Literal["step_back", "hyde", "complex"]


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved paper evidence to a user question.\n"
    "Retrieved evidence:\n\n{context}\n\n"
    "User question: {question}\n"
    "Return yes if the evidence can answer or materially help answer the question; otherwise return no."
)


async def retrieve_initial_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    paper_id = state["paper_id"]
    emit_rag_step("🔍", "正在检索论文知识库", f"查询: {question[:80]}")
    retrieved = vector_store.agentic_retrieve(paper_id, question, top_k=6)
    sources = retrieved.get("sources", [])
    meta = retrieved.get("meta", {})
    emit_rag_step(
        "🧱",
        "Hybrid + 三级分块检索",
        (
            f"模式: {meta.get('retrieval_mode', 'unknown')}，"
            f"L{meta.get('leaf_retrieve_level', 3)} 召回，候选 {meta.get('candidate_k', 0)}"
        ),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 父块合并",
        (
            f"启用: {bool(meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(meta.get('auto_merge_applied'))}，"
            f"替换: {meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    emit_rag_step("✅", f"初次检索完成，找到 {len(sources)} 个证据片段")
    rag_trace = {
        "tool_used": True,
        "tool_name": "paper_agentic_rag",
        "query": question,
        "retrieval_stage": "initial",
        "initial_retrieved_chunks": sources,
        "retrieved_chunks": sources,
        **meta,
    }
    return {
        "context": sources,
        "sources": sources,
        "docs": retrieved.get("docs", []),
        "rag_trace": rag_trace,
        "rewrite_count": int(state.get("rewrite_count") or 0),
    }


async def grade_documents_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    sources = state.get("context", [])
    context = vector_store.render_sources_for_prompt(sources)
    emit_rag_step("📊", "正在评估证据相关性")

    score = "yes" if sources else "no"
    model_name = settings.rag_grade_model or settings.llm_model
    try:
        llm = get_llm(model=model_name, streaming=False, temperature=0)
        grader = llm.with_structured_output(GradeDocuments)
        response = await grader.ainvoke(
            [
                {
                    "role": "user",
                    "content": GRADE_PROMPT.format(question=question, context=context),
                }
            ]
        )
        score = (response.binary_score or "").strip().lower()
    except Exception:
        score = "yes" if sources else "no"

    rewrite_count = int(state.get("rewrite_count") or 0)
    route = (
        "rewrite_question"
        if score != "yes" and rewrite_count < settings.rag_max_rewrites
        else "generate_answer"
    )
    if route == "generate_answer":
        emit_rag_step("✅", "证据相关性通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "证据不足，进入查询重写", f"评分: {score}")

    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update(
        {
            "grade_score": score,
            "grade_route": route,
            "rewrite_needed": route == "rewrite_question",
        }
    )
    return {"route": route, "rag_trace": rag_trace}


async def rewrite_question_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    emit_rag_step("✏️", "正在选择查询重写策略")
    strategy = await _choose_rewrite_strategy(question)
    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用 {strategy} 策略", "生成退步问题与背景答案")
        step_back_question = await _generate_text(
            "请将用户的具体论文问题抽象成更高层次的退步问题。只输出一句话。\n"
            f"用户问题：{question}",
            temperature=0.2,
        )
        if step_back_question:
            step_back_answer = await _generate_text(
                "请简要回答以下退步问题，提供通用原则或背景，控制在120字以内。只输出答案。\n"
                f"退步问题：{step_back_question}",
                temperature=0.2,
            )
        expanded_query = (
            f"{question}\n退步问题：{step_back_question}\n退步问题答案：{step_back_answer}"
        ).strip()

    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "正在生成 HyDE 假设性文档")
        hypothetical_doc = await _generate_text(
            "请基于用户问题生成一段像论文片段的假设性文档，用于扩展检索。"
            "可以包含合理语义推测，但不要编造具体实验数值。只输出正文。\n"
            f"用户问题：{question}",
            temperature=0.3,
        )

    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update(
        {
            "rewrite_strategy": strategy,
            "rewrite_query": expanded_query,
            "step_back_question": step_back_question,
            "step_back_answer": step_back_answer,
            "hypothetical_doc": hypothetical_doc,
        }
    )
    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rewrite_count": int(state.get("rewrite_count") or 0) + 1,
        "rag_trace": rag_trace,
    }


async def retrieve_expanded_node(state: AgentState) -> dict[str, Any]:
    paper_id = state["paper_id"]
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "正在使用扩展查询二次检索", f"策略: {strategy}")
    retrievals: list[dict[str, Any]] = []

    if strategy in ("hyde", "complex") and state.get("hypothetical_doc"):
        retrievals.append(
            vector_store.agentic_retrieve(paper_id, state["hypothetical_doc"], top_k=6)
        )
    if strategy in ("step_back", "complex"):
        retrievals.append(
            vector_store.agentic_retrieve(
                paper_id,
                state.get("expanded_query") or state["question"],
                top_k=6,
            )
        )
    if not retrievals:
        retrievals.append(vector_store.agentic_retrieve(paper_id, state["question"], top_k=6))

    sources = _dedupe_sources(
        [source for result in retrievals for source in result.get("sources", [])]
    )[:6]
    docs = [doc for result in retrievals for doc in result.get("docs", [])]
    meta = _merge_retrieval_meta([result.get("meta", {}) for result in retrievals])
    emit_rag_step("✅", f"扩展检索完成，共 {len(sources)} 个证据片段")

    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update(
        {
            "retrieval_stage": "expanded",
            "expanded_retrieved_chunks": sources,
            "retrieved_chunks": sources,
            **meta,
        }
    )
    return {"context": sources, "sources": sources, "docs": docs, "rag_trace": rag_trace}


async def retrieve_summary_node(state: AgentState) -> dict[str, Any]:
    emit_rag_step("📚", "正在收集论文概览证据")
    sources = vector_store.get_overview_context(state["paper_id"], max_items=12)
    rag_trace = {
        "tool_used": True,
        "tool_name": "paper_agentic_summary",
        "query": state["question"],
        "retrieval_stage": "summary",
        "retrieved_chunks": sources,
    }
    return {"context": sources, "sources": sources, "rag_trace": rag_trace}


async def answer_agentic_node(state: AgentState) -> dict[str, Any]:
    answer = await _invoke_with_fallback(
        RAG_PROMPT,
        state["question"],
        state.get("messages", []),
        state.get("context", []),
    )
    sources = _sanitize_sources(state.get("context", []))
    return {
        "answer": answer,
        "sources": sources,
        "rag_trace": state.get("rag_trace", {}),
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
    }


async def answer_summary_node(state: AgentState) -> dict[str, Any]:
    answer = await _invoke_with_fallback(
        SUMMARIZE_PROMPT,
        state["question"],
        state.get("messages", []),
        state.get("context", []),
    )
    sources = _sanitize_sources(state.get("context", []))
    return {
        "answer": answer,
        "sources": sources,
        "rag_trace": state.get("rag_trace", {}),
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
    }


def grade_route(state: AgentState) -> str:
    return state.get("route") or "generate_answer"


def build_agentic_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve_initial", retrieve_initial_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded_node)
    graph.add_node("answer_agentic", answer_agentic_node)
    graph.add_node("retrieve_summary", retrieve_summary_node)
    graph.add_node("answer_summary_agentic", answer_summary_node)
    graph.add_node("answer_chat", answer_chat_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "retrieve_initial",
            "summarize": "retrieve_summary",
            "chat": "answer_chat",
        },
    )
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        grade_route,
        {
            "generate_answer": "answer_agentic",
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", "answer_agentic")
    graph.add_edge("answer_agentic", END)
    graph.add_edge("retrieve_summary", "answer_summary_agentic")
    graph.add_edge("answer_summary_agentic", END)
    graph.add_edge("answer_chat", END)
    return graph.compile()


async def _choose_rewrite_strategy(question: str) -> str:
    fallback = "complex" if any(token in question for token in ("比较", "关系", "为什么", "如何")) else "step_back"
    try:
        llm = get_llm(streaming=False, temperature=0)
        router = llm.with_structured_output(RewriteStrategy)
        response = await router.ainvoke(
            [
                {
                    "role": "user",
                    "content": (
                        "请为论文RAG检索选择策略，仅返回结构化 strategy。\n"
                        "step_back：具体问题需要先抽象概念。\n"
                        "hyde：问题模糊、概念性或术语不明确。\n"
                        "complex：需要多步骤综合、比较或组合多个信息。\n"
                        f"用户问题：{question}"
                    ),
                }
            ]
        )
        return response.strategy or fallback
    except Exception:
        return fallback


async def _generate_text(prompt: str, temperature: float = 0.2) -> str:
    try:
        llm = get_llm(streaming=False, temperature=temperature)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return str(response.content or "").strip()
    except Exception:
        return ""


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        key = source.get("chunk_id") or source.get("id") or source.get("content")
        if key in seen:
            continue
        seen.add(str(key))
        deduped.append(source)
    return deduped


def _merge_retrieval_meta(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if key not in merged or not merged[key]:
                merged[key] = value
    merged["rerank_applied"] = any(bool(item.get("rerank_applied")) for item in items)
    merged["auto_merge_applied"] = any(bool(item.get("auto_merge_applied")) for item in items)
    merged["auto_merge_replaced_chunks"] = sum(
        int(item.get("auto_merge_replaced_chunks") or 0) for item in items
    )
    return merged
