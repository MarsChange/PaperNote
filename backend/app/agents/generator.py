"""Answer generation nodes for multimodal paper QA."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.config import settings
from app.services.llm import get_llm
from app.services.vector_store import vector_store

RAG_PROMPT = """You are an expert paper reading assistant.

Answer the user's question using only the provided evidence.

Rules:
- Answer in the same language as the user's question.
- Be precise and grounded in the evidence.
- If the evidence is insufficient, say so directly.
- When you make a claim from the paper, cite evidence with markers like [S1], [S2].
- Mention figures, tables, equations, and pages when relevant.
- Keep terminology faithful to the paper.
"""

CHAT_PROMPT = """You are a helpful paper reading assistant.
Answer in the same language as the user's question. Keep the reply concise."""

SUMMARIZE_PROMPT = """You are a paper summarization assistant.

Based on the provided evidence, produce a structured summary that includes:
1. Problem and objective
2. Method or system design
3. Key results or findings
4. Limitations or open questions

Use the same language as the user's question and cite evidence with [S1], [S2] when possible.
"""


def _sanitize_sources(sources: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for source in sources:
        cleaned.append(
            {
                "id": source.get("id", ""),
                "type": source.get("type", "text"),
                "page_number": source.get("page_number", 1),
                "title": source.get("title", ""),
                "section": source.get("section", ""),
                "content": source.get("content", ""),
                "score": source.get("score", 0),
                "asset_url": source.get("asset_url", ""),
            }
        )
    return cleaned


def _missing_model_message(question: str) -> str:
    if any(token in question.lower() for token in ("summary", "summarize", "总结", "概述", "摘要")):
        return "当前还没有配置可用的大模型 API Key。请先在右上角“模型设置”中配置提供商、模型和 Key，然后再请求论文总结。"
    return "当前还没有配置可用的大模型 API Key。请先在右上角“模型设置”中完成配置，然后再进行问答或翻译。"


def _to_data_url(asset_path: str) -> Optional[str]:
    path = Path(asset_path)
    if not asset_path or not path.exists() or path.stat().st_size > 5 * 1024 * 1024:
        return None

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_user_message(question: str, sources: list[dict]) -> HumanMessage:
    context_text = vector_store.render_sources_for_prompt(sources)
    prompt_text = f"User question:\n{question}\n\nEvidence:\n{context_text or 'No evidence was retrieved.'}"
    content: Union[str, list[dict]] = prompt_text

    if settings.enable_multimodal_answers:
        multimodal_parts: list[dict] = [{"type": "text", "text": prompt_text}]
        for source in sources[:2]:
            data_url = _to_data_url(source.get("asset_path", ""))
            if data_url:
                multimodal_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
        if len(multimodal_parts) > 1:
            content = multimodal_parts

    return HumanMessage(content=content)


async def _invoke_with_fallback(
    system_prompt: str,
    question: str,
    history: list,
    sources: list[dict],
) -> str:
    try:
        llm = get_llm(streaming=True)
    except Exception:
        return _missing_model_message(question)

    user_message = _build_user_message(question, sources)
    messages = [SystemMessage(content=system_prompt), *history, user_message]

    try:
        response = await llm.ainvoke(messages)
        return str(response.content)
    except Exception:
        fallback_prompt = f"User question:\n{question}\n\nEvidence:\n{vector_store.render_sources_for_prompt(sources)}"
        fallback_response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                *history,
                HumanMessage(content=fallback_prompt),
            ]
        )
        return str(fallback_response.content)


async def answer_rag_node(state: AgentState) -> dict:
    sources = state.get("context", [])
    answer = await _invoke_with_fallback(
        RAG_PROMPT,
        state["question"],
        state["messages"],
        sources,
    )
    sanitized_sources = _sanitize_sources(sources)
    return {
        "answer": answer,
        "sources": sanitized_sources,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }


async def answer_chat_node(state: AgentState) -> dict:
    try:
        llm = get_llm(streaming=True)
        response = await llm.ainvoke(
            [
                SystemMessage(content=CHAT_PROMPT),
                *state["messages"],
                HumanMessage(content=state["question"]),
            ]
        )
        answer = str(response.content)
    except Exception:
        answer = _missing_model_message(state["question"])
    return {
        "answer": answer,
        "sources": [],
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }


async def answer_summarize_node(state: AgentState) -> dict:
    sources = state.get("context", [])
    answer = await _invoke_with_fallback(
        SUMMARIZE_PROMPT,
        state["question"],
        state["messages"],
        sources,
    )
    sanitized_sources = _sanitize_sources(sources)
    return {
        "answer": answer,
        "sources": sanitized_sources,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }
