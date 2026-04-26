"""High-level paper intelligence utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.services.llm import get_llm
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def _fallback_keywords(context: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-]{4,}", context)
    deduped: list[str] = []
    for word in candidates:
        lowered = word.lower()
        if lowered in {"paper", "results", "method", "approach", "model"}:
            continue
        if word not in deduped:
            deduped.append(word)
        if len(deduped) >= 6:
            break
    return deduped


class PaperAIService:
    async def generate_overview(self, paper_id: str, filename: str) -> dict[str, Any]:
        sources = vector_store.get_overview_context(paper_id, max_items=10)
        context_text = vector_store.render_sources_for_prompt(sources)
        fallback_summary = context_text[:500].strip() or f"{filename} 已完成解析。"
        fallback = {
            "summary": fallback_summary,
            "keywords": _fallback_keywords(context_text),
        }

        if not context_text:
            return fallback

        try:
            llm = get_llm(streaming=False, temperature=0.2)
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是论文阅读助手。请根据提供的论文证据，输出一段简洁中文概述和 4 到 6 个关键词。"
                            "严格返回 JSON，格式为 {\"summary\": string, \"keywords\": string[]}。"
                        )
                    ),
                    HumanMessage(
                        content=f"论文文件：{filename}\n\n证据：\n{context_text}"
                    ),
                ]
            )
            payload = json.loads(_strip_code_fence(str(response.content)))
            summary = str(payload.get("summary", "")).strip()
            keywords = [
                str(item).strip()
                for item in payload.get("keywords", [])
                if str(item).strip()
            ]
            return {
                "summary": summary or fallback["summary"],
                "keywords": keywords or fallback["keywords"],
            }
        except Exception as exc:
            logger.warning("Overview generation failed for %s: %s", paper_id, exc)
            return fallback

    async def translate_selection(
        self,
        paper_id: str,
        text: str,
        target_language: Optional[str] = None,
    ) -> str:
        selection = text.strip()
        if not selection:
            return ""

        target_language = target_language or settings.translation_target_language
        context_sources = vector_store.search(paper_id, selection, top_k=2)
        context_text = vector_store.render_sources_for_prompt(context_sources)

        try:
            llm = get_llm(streaming=False, temperature=0.1)
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是学术翻译助手。请将用户选中的论文片段翻译成目标语言，保留术语、数学符号和原文语气。"
                            "如果提供了论文上下文，请优先保持术语一致。只返回译文。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"目标语言：{target_language}\n\n"
                            f"选中文本：\n{selection}\n\n"
                            f"相关上下文：\n{context_text or '无'}"
                        )
                    ),
                ]
            )
            return str(response.content).strip()
        except Exception:
            return "请先在模型设置中配置可用的大模型 API Key，随后即可使用自动翻译。"


paper_ai_service = PaperAIService()
