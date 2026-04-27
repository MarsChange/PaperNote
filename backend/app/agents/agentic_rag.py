"""superMew-style Agentic RAG graph adapted to PaperNote papers."""

from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents.generator import (
    RAG_PROMPT,
    SUMMARIZE_PROMPT,
    _missing_model_message,
    _invoke_with_fallback,
    _sanitize_sources,
    _to_data_url,
    answer_chat_node,
)
from app.agents.router import router_node
from app.agents.state import AgentState
from app.core.config import settings
from app.services.agentic_runtime import emit_rag_step
from app.services.context_builder import ContextBuilder, ContextConfig
from app.services.harness_tools import (
    note_memory_tool,
    paper_rag_tool,
    tavily_research_tool,
)
from app.services.llm import get_llm
from app.services.vector_store import vector_store


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if evidence is relevant, otherwise 'no'")


class RewriteStrategy(BaseModel):
    strategy: Literal["step_back", "hyde", "complex"]


class ToolPlan(BaseModel):
    use_rag: bool = Field(
        default=False,
        description="Call paper_rag when the question needs evidence from the uploaded paper.",
    )
    use_research: bool = Field(
        default=False,
        description="Call tavily_research when external, current, or broader research evidence is needed.",
    )
    use_notes: bool = Field(
        default=True,
        description="Search persistent notes for cross-turn or cross-session memory.",
    )
    write_note: bool = Field(
        default=False,
        description="Persist a structured note when the user asks to remember or the answer is a durable conclusion/action.",
    )
    rag_query: str = Field(default="", description="Query for the paper_rag tool.")
    research_query: str = Field(default="", description="Task for Tavily Research.")
    note_query: str = Field(default="", description="Query for NoteTool search.")
    answer_mode: Literal["paper_qa", "summary", "research_augmented", "chat"] = "paper_qa"
    rationale: str = Field(default="", description="Short reason for tool selection.")


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved paper evidence to a user question.\n"
    "Retrieved evidence:\n\n{context}\n\n"
    "User question: {question}\n"
    "Return yes if the evidence can answer or materially help answer the question; otherwise return no."
)

TOOL_PLANNER_PROMPT = """You are a tool planner for a paper-reading harness agent.

The LLM router already selected route: {route}.
Available tools:
- paper_rag: search the uploaded paper through PaperNote's Agentic RAG and Milvus/BM25 retriever.
- tavily_research: create a Tavily Research task when the answer needs external, current, or broader information beyond the paper.
- note_tool: search or write persistent Markdown notes for cross-turn memory.

Decide which tools are needed for the next answer.
Rules:
- Use paper_rag for questions about the uploaded paper, figures, tables, methods, datasets, experiments, equations, or paper summary.
- Use tavily_research only when the user asks for latest/current information, asks to compare with outside work, asks for URLs/background not guaranteed to be in the paper, or the paper evidence alone is insufficient by nature.
- Use note_tool by default for continuity.
- Set write_note only when the user asks to remember/save/note something, or when the turn produces a durable research conclusion or action item worth persisting.
- Do not call Tavily Research for ordinary paper-content questions unless outside evidence is explicitly needed.

User question:
{question}
"""

HARNESS_SYSTEM_PROMPT = """You are PaperNote's harness-engineering agent.
You receive a structured GSSC context with role policies, task, state, evidence, memory, and output constraints.

Answer rules:
- Answer in the same language as the user's question.
- Ground paper claims in paper evidence and cite [S1], [S2] when available.
- Ground external research claims in Tavily Research evidence and cite [R1], [R2] when available.
- If persistent notes are used, cite [N1], [N2].
- Do not invent missing evidence. If evidence is insufficient, say what is missing and what tool result would be needed.
- Keep the answer direct and useful for paper reading.
"""


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


async def tool_planner_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    route = state.get("route") or "rag"
    emit_rag_step("🧭", "LLM Router 已完成，正在规划工具", f"route={route}")
    plan = _fallback_tool_plan(question, route)
    try:
        llm = get_llm(streaming=False, temperature=0)
        planner = llm.with_structured_output(ToolPlan)
        response = await planner.ainvoke(
            [
                SystemMessage(
                    content=TOOL_PLANNER_PROMPT.format(route=route, question=question)
                )
            ]
        )
        plan = response.model_dump()
    except Exception:
        pass

    plan = _normalize_tool_plan(plan, question, route)
    emit_rag_step(
        "🧰",
        "工具规划完成",
        (
            f"RAG={plan['use_rag']}，Research={plan['use_research']}，"
            f"Notes={plan['use_notes']}"
        ),
    )
    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update({"route": route, "tool_plan": plan})
    return {"tool_plan": plan, "rag_trace": rag_trace}


async def run_tools_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    paper_id = state.get("paper_id", "")
    conversation_id = state.get("conversation_id", "")
    plan = _normalize_tool_plan(state.get("tool_plan") or {}, question, state.get("route") or "rag")
    tool_results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    research_results: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []

    if plan.get("use_notes"):
        emit_rag_step("🗒️", "检索持久化笔记", (plan.get("note_query") or question)[:80])
        note_result = note_memory_tool.search(
            query=plan.get("note_query") or question,
            paper_id=paper_id,
            conversation_id=conversation_id,
            limit=5,
        )
        notes = note_result.get("notes", [])
        tool_results.append(note_result)

    if plan.get("use_rag"):
        rag_query = plan.get("rag_query") or question
        mode = "summary" if plan.get("answer_mode") == "summary" else "qa"
        emit_rag_step("🔍", "调用论文 RAG 工具", rag_query[:80])
        rag_result = paper_rag_tool.run(
            paper_id=paper_id,
            query=rag_query,
            top_k=6,
            mode=mode,
        )
        tool_results.append(rag_result)
        sources = rag_result.get("sources", []) or []
        docs = rag_result.get("docs", []) or []
        meta = rag_result.get("meta", {}) or {}
        emit_rag_step(
            "🧱",
            "论文 RAG 工具完成",
            (
                f"模式: {meta.get('retrieval_mode', 'unknown')}，"
                f"证据: {len(sources)}，候选: {meta.get('candidate_k', len(sources))}"
            ),
        )

    if plan.get("use_research"):
        research_query = plan.get("research_query") or question
        emit_rag_step("🌐", "调用 Tavily Research 工具", research_query[:80])
        research_result = tavily_research_tool.run(query=research_query)
        research_results = research_result.get("sources", []) or []
        tool_results.append(research_result)
        emit_rag_step(
            "🌐",
            "Research 工具完成",
            f"状态: {research_result.get('status')}，来源: {len(research_results)}",
        )

    if not tool_results:
        emit_rag_step("💬", "无需外部工具", "直接基于对话上下文回答")

    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update(
        {
            "tool_results": _tool_result_summary(tool_results),
            "retrieved_chunks": sources,
            "research_results": research_results,
            "notes": notes,
        }
    )
    return {
        "tool_results": tool_results,
        "context": sources,
        "sources": sources,
        "docs": docs,
        "research_results": research_results,
        "notes": notes,
        "rag_trace": rag_trace,
    }


async def build_context_node(state: AgentState) -> dict[str, Any]:
    emit_rag_step("🧩", "构建 GSSC 上下文", "Gather -> Select -> Structure -> Compress")
    builder = ContextBuilder(
        ContextConfig(
            max_tokens=settings.context_max_tokens,
            reserve_ratio=settings.context_reserve_ratio,
            min_relevance=settings.context_min_relevance,
            enable_compression=settings.context_enable_compression,
            recency_weight=settings.context_recency_weight,
            relevance_weight=settings.context_relevance_weight,
        )
    )
    built = builder.build(
        user_query=state["question"],
        conversation_history=state.get("messages", []),
        system_instructions=_harness_role_policy(),
        state_summary=_harness_state_summary(state),
        tool_results=state.get("tool_results", []),
        output_instructions=_harness_output_instructions(state),
    )
    emit_rag_step(
        "📦",
        "上下文构建完成",
        (
            f"选择 {built.stats['selected_packets']}/{built.stats['gathered_packets']} 个信息包，"
            f"{built.stats['token_count']}/{built.stats['max_tokens']} tokens"
        ),
    )
    rag_trace = dict(state.get("rag_trace") or {})
    rag_trace.update({"context_builder": built.stats})
    return {
        "built_context": built.context,
        "context_stats": built.stats,
        "rag_trace": rag_trace,
    }


async def answer_agentic_node(state: AgentState) -> dict[str, Any]:
    if not state.get("built_context"):
        answer = await _invoke_with_fallback(
            RAG_PROMPT,
            state["question"],
            state.get("messages", []),
            state.get("context", []),
        )
    else:
        answer = await _invoke_harness_answer(state)
        _maybe_write_interaction_note(state, answer)

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
    graph.add_node("tool_planner", tool_planner_node)
    graph.add_node("run_tools", run_tools_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("answer_agentic", answer_agentic_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "tool_planner")
    graph.add_edge("tool_planner", "run_tools")
    graph.add_edge("run_tools", "build_context")
    graph.add_edge("build_context", "answer_agentic")
    graph.add_edge("answer_agentic", END)
    return graph.compile()


def _fallback_tool_plan(question: str, route: str) -> dict[str, Any]:
    needs_research = _question_needs_research(question)
    asks_note = any(
        token in question.lower()
        for token in ("记住", "记录", "做笔记", "保存", "note", "remember", "save")
    )
    answer_mode = "summary" if route == "summarize" else "paper_qa"
    if route == "chat" and needs_research:
        answer_mode = "research_augmented"
    elif route == "chat":
        answer_mode = "chat"
    return {
        "use_rag": route in {"rag", "summarize"},
        "use_research": needs_research,
        "use_notes": True,
        "write_note": asks_note,
        "rag_query": question,
        "research_query": question,
        "note_query": question,
        "answer_mode": answer_mode,
        "rationale": "fallback heuristic",
    }


def _normalize_tool_plan(plan: dict[str, Any], question: str, route: str) -> dict[str, Any]:
    fallback = _fallback_tool_plan(question, route)
    normalized = {**fallback, **(plan or {})}
    normalized["use_rag"] = bool(normalized.get("use_rag")) or route in {
        "rag",
        "summarize",
    }
    if "use_web_search" in normalized and "use_research" not in plan:
        normalized["use_research"] = bool(normalized.get("use_web_search"))
    normalized["use_research"] = bool(normalized.get("use_research"))
    normalized["use_notes"] = bool(normalized.get("use_notes", True))
    normalized["write_note"] = bool(normalized.get("write_note"))
    normalized["rag_query"] = str(normalized.get("rag_query") or question)
    if "web_query" in normalized and "research_query" not in plan:
        normalized["research_query"] = normalized.get("web_query")
    normalized["research_query"] = str(normalized.get("research_query") or question)
    normalized["note_query"] = str(normalized.get("note_query") or question)
    if normalized.get("answer_mode") not in {
        "paper_qa",
        "summary",
        "research_augmented",
        "chat",
    }:
        normalized["answer_mode"] = fallback["answer_mode"]
    if route == "summarize":
        normalized["answer_mode"] = "summary"
    return normalized


def _question_needs_research(question: str) -> bool:
    normalized = question.lower()
    research_markers = (
        "联网",
        "搜索",
        "最新",
        "最近",
        "今天",
        "现在",
        "新闻",
        "官网",
        "链接",
        "url",
        "web",
        "search",
        "latest",
        "recent",
        "current",
        "today",
        "compare with",
        "outside",
    )
    return any(marker in normalized for marker in research_markers)


def _tool_result_summary(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for result in tool_results:
        summary.append(
            {
                "tool": result.get("tool"),
                "status": result.get("status"),
                "query": result.get("query"),
                "meta": result.get("meta", {}),
                "source_count": len(result.get("sources", []) or []),
                "result_count": len(result.get("results", []) or []),
                "note_count": len(result.get("notes", []) or []),
            }
        )
    return summary


def _harness_role_policy() -> str:
    return (
        "You are PaperNote's paper-reading harness agent. Use the available tool "
        "results rather than guessing. Paper evidence uses [S#], Tavily Research "
        "evidence uses [R#], persistent notes use [N#]. Keep claims faithful to the cited source."
    )


def _harness_state_summary(state: AgentState) -> str:
    plan = state.get("tool_plan") or {}
    return "\n".join(
        [
            f"paper_id: {state.get('paper_id', '')}",
            f"conversation_id: {state.get('conversation_id', '')}",
            f"router_route: {state.get('route', '')}",
            f"answer_mode: {plan.get('answer_mode', '')}",
            f"tool_plan: {plan}",
        ]
    )


def _harness_output_instructions(state: AgentState) -> str:
    plan = state.get("tool_plan") or {}
    if plan.get("answer_mode") == "summary":
        return (
            "Produce a structured paper summary covering objective, method, findings, "
            "limitations, and useful reading notes. Cite [S#] where possible."
        )
    return (
        "Answer the user's question directly. If paper evidence was used, cite [S#]. "
        "If Tavily Research evidence was used, cite [R#]. If persistent notes were used, cite [N#]. "
        "If evidence is insufficient, state that explicitly and avoid speculation."
    )


async def _invoke_harness_answer(state: AgentState) -> str:
    try:
        llm = get_llm(streaming=True)
    except Exception:
        return _missing_model_message(state["question"])

    user_message = _build_harness_user_message(
        state.get("built_context", ""),
        state.get("context", []),
    )
    messages = [SystemMessage(content=HARNESS_SYSTEM_PROMPT), user_message]
    try:
        response = await llm.ainvoke(messages)
        return str(response.content)
    except Exception:
        fallback_response = await llm.ainvoke(
            [
                SystemMessage(content=HARNESS_SYSTEM_PROMPT),
                HumanMessage(content=state.get("built_context", "")),
            ]
        )
        return str(fallback_response.content)


def _build_harness_user_message(context: str, sources: list[dict[str, Any]]) -> HumanMessage:
    content: Any = context
    if settings.enable_multimodal_answers:
        parts: list[dict[str, Any]] = [{"type": "text", "text": context}]
        image_limit = max(settings.multimodal_answer_image_limit, 0)
        image_count = 0
        for source_index, source in enumerate(sources, start=1):
            if image_count >= image_limit:
                break
            if source.get("type") != "image":
                continue
            data_url = _to_data_url(str(source.get("asset_path", "")))
            if not data_url:
                continue
            parts.append(
                {
                    "type": "text",
                    "text": f"Visual paper evidence [S{source_index}] "
                    f"{source.get('title') or source.get('id') or 'image'}",
                }
            )
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
            image_count += 1
        if len(parts) > 1:
            content = parts
    return HumanMessage(content=content)


def _maybe_write_interaction_note(state: AgentState, answer: str):
    plan = state.get("tool_plan") or {}
    if not plan.get("write_note") or not answer.strip():
        return
    try:
        note_memory_tool.create_interaction_note(
            question=state["question"],
            answer=answer,
            paper_id=state.get("paper_id", ""),
            conversation_id=state.get("conversation_id", ""),
            tags=["auto", str(plan.get("answer_mode") or "conversation")],
        )
    except Exception:
        pass


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
