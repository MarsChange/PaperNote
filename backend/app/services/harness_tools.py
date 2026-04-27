"""Tool layer used by the PaperNote harness agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.services.note_tool import note_tool
from app.services.vector_store import vector_store


class PaperRAGTool:
    """Wrap PaperNote's existing Agentic RAG retriever as an agent tool."""

    name = "paper_rag"

    def run(
        self,
        *,
        paper_id: str,
        query: str,
        top_k: int = 6,
        mode: str = "qa",
    ) -> dict[str, Any]:
        if mode == "summary":
            sources = vector_store.get_overview_context(paper_id, max_items=max(top_k, 10))
            meta = {
                "retrieval_mode": "overview_context",
                "candidate_k": len(sources),
                "top_k": len(sources),
            }
            docs: list[dict[str, Any]] = []
        else:
            retrieved = vector_store.agentic_retrieve(paper_id, query, top_k=top_k)
            sources = retrieved.get("sources", [])
            docs = retrieved.get("docs", [])
            meta = retrieved.get("meta", {})

        return {
            "tool": self.name,
            "status": "ok",
            "timestamp": _now(),
            "query": query,
            "mode": mode,
            "content": vector_store.render_sources_for_prompt(sources)
            or "No paper evidence was retrieved.",
            "sources": sources,
            "docs": docs,
            "meta": meta,
        }


class TavilyResearchTool:
    """Research tool using Tavily's official /research task API."""

    name = "tavily_research"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def run(
        self,
        *,
        query: str,
        model: Optional[str] = None,
        citation_format: Optional[str] = None,
    ) -> dict[str, Any]:
        if not settings.enable_research_tool:
            return self._skipped(query, "research tool is disabled")
        api_key = settings.tavily_api_key if self.api_key is None else self.api_key
        endpoint = self.endpoint or settings.tavily_research_endpoint
        if not api_key:
            return self._skipped(query, "TAVILY_API_KEY is not configured")

        payload = {
            "input": query,
            "model": model or settings.tavily_research_model,
            "stream": False,
            "citation_format": citation_format or settings.tavily_research_citation_format,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                create_response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                create_response.raise_for_status()
                create_data = create_response.json()
                data = self._poll_research_task(
                    client=client,
                    endpoint=endpoint,
                    api_key=api_key,
                    create_data=create_data,
                )
        except Exception as exc:
            return {
                "tool": self.name,
                "status": "error",
                "timestamp": _now(),
                "query": query,
                "content": f"Tavily research failed: {exc}",
                "sources": [],
                "report": "",
                "error": str(exc),
            }

        sources = data.get("sources") if isinstance(data, dict) else []
        if not isinstance(sources, list):
            sources = []
        normalized_sources = [self._normalize_source(item) for item in sources]
        report = data.get("content", "") if isinstance(data, dict) else ""
        status = data.get("status", "unknown") if isinstance(data, dict) else "unknown"
        return {
            "tool": self.name,
            "status": "ok" if status == "completed" else status,
            "timestamp": _now(),
            "query": query,
            "content": self._format_research_report(report, normalized_sources, status, data),
            "sources": normalized_sources,
            "report": report,
            "meta": {
                "request_id": data.get("request_id"),
                "created_at": data.get("created_at"),
                "status": status,
                "model": data.get("model") or payload["model"],
                "response_time": data.get("response_time"),
                "source_count": len(normalized_sources),
            },
        }

    def _skipped(self, query: str, reason: str) -> dict[str, Any]:
        return {
            "tool": self.name,
            "status": "skipped",
            "timestamp": _now(),
            "query": query,
            "content": f"Tavily research skipped: {reason}.",
            "sources": [],
            "report": "",
            "error": reason,
        }

    def _poll_research_task(
        self,
        *,
        client: httpx.Client,
        endpoint: str,
        api_key: str,
        create_data: dict[str, Any],
    ) -> dict[str, Any]:
        import time

        request_id = str(create_data.get("request_id") or "")
        if not request_id or create_data.get("status") == "completed":
            return create_data

        status_endpoint = f"{endpoint.rstrip('/')}/{request_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        last_data = create_data
        for _ in range(max(settings.tavily_research_max_polls, 0)):
            time.sleep(max(settings.tavily_research_poll_interval_seconds, 0.1))
            response = client.get(status_endpoint, headers=headers)
            if response.status_code == 202:
                try:
                    last_data = response.json()
                except Exception:
                    last_data = create_data
                continue
            response.raise_for_status()
            last_data = response.json()
            if last_data.get("status") in {"completed", "failed"}:
                return last_data
        return last_data

    def _normalize_source(self, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"title": "", "url": "", "favicon": "", "content": str(item)}
        return {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "favicon": str(item.get("favicon") or ""),
            "content": str(item.get("content") or item.get("snippet") or ""),
        }

    def _format_research_report(
        self,
        report: Any,
        sources: list[dict[str, Any]],
        status: str,
        data: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        if status != "completed":
            parts.append(
                f"Tavily research task status: {status}. "
                f"request_id={data.get('request_id', '')}"
            )
        if report:
            parts.append(f"Tavily research report:\n{report}")
        for index, source in enumerate(sources, start=1):
            snippet = str(source.get("content") or "").strip()
            if len(snippet) > 900:
                snippet = snippet[:900].rstrip() + "..."
            parts.append(
                "\n".join(
                    [
                        f"[R{index}] {source.get('title') or 'Untitled'}",
                        f"URL: {source.get('url', '')}",
                        f"Snippet: {snippet}",
                    ]
                ).strip()
            )
        return "\n\n".join(parts).strip() or "No Tavily research report was returned."


class NoteMemoryTool:
    """Adapter that exposes NoteTool search/create as harness tool results."""

    name = "note_tool"

    def search(
        self,
        *,
        query: str,
        paper_id: str = "",
        conversation_id: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        notes = note_tool.search(
            query=query,
            paper_id=paper_id,
            conversation_id=conversation_id,
            limit=limit,
        )
        return {
            "tool": self.name,
            "status": "ok",
            "timestamp": _now(),
            "query": query,
            "content": self._format_notes(notes),
            "notes": notes,
            "meta": {"result_count": len(notes)},
        }

    def create_interaction_note(
        self,
        *,
        question: str,
        answer: str,
        paper_id: str = "",
        conversation_id: str = "",
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        content = (
            f"## User question\n{question.strip()}\n\n"
            f"## Agent answer\n{answer.strip()[:3000]}"
        ).strip()
        return note_tool.create(
            title=question.strip()[:80] or "Conversation note",
            content=content,
            note_type="conversation",
            tags=tags or ["auto", "conversation"],
            paper_id=paper_id,
            conversation_id=conversation_id,
            source="harness_agent",
        )

    def _format_notes(self, notes: list[dict[str, Any]]) -> str:
        if not notes:
            return "No relevant persistent notes were found."
        parts = []
        for index, note in enumerate(notes, start=1):
            content = str(note.get("content") or "").strip()
            if len(content) > 800:
                content = content[:800].rstrip() + "..."
            parts.append(
                "\n".join(
                    [
                        f"[N{index}] {note.get('title') or 'Untitled'}",
                        f"Type: {note.get('type', 'general')}",
                        f"Tags: {', '.join(note.get('tags', []))}",
                        content,
                    ]
                ).strip()
            )
        return "\n\n".join(parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


paper_rag_tool = PaperRAGTool()
tavily_research_tool = TavilyResearchTool()
note_memory_tool = NoteMemoryTool()
