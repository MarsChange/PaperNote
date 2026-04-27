from __future__ import annotations

import tempfile
from pathlib import Path

from app.services.harness_tools import TavilyResearchTool
from app.services.note_tool import NoteTool


def test_note_tool_create_search_read_update_summary_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes = NoteTool(Path(tmpdir))
        created = notes.run(
            {
                "action": "create",
                "title": "Dataset finding",
                "content": "The paper uses the MATR open-source dataset.",
                "note_type": "conclusion",
                "tags": ["dataset", "paper"],
                "paper_id": "paper-1",
            }
        )
        note_id = created["note_id"]

        results = notes.run(
            {
                "action": "search",
                "query": "MATR dataset",
                "paper_id": "paper-1",
            }
        )
        assert results[0]["note_id"] == note_id
        assert "MATR" in notes.run({"action": "read", "note_id": note_id})["content"]

        notes.run(
            {
                "action": "update",
                "note_id": note_id,
                "content": "Updated note about MATR.",
                "tags": ["updated"],
            }
        )
        summary = notes.run({"action": "summary", "paper_id": "paper-1"})
        assert summary["total_notes"] == 1

        deleted = notes.run({"action": "delete", "note_id": note_id})
        assert deleted["deleted"] is True


def test_tavily_research_tool_skips_without_key():
    tool = TavilyResearchTool(api_key="")
    result = tool.run(query="latest retrieval augmented generation papers")

    assert result["tool"] == "tavily_research"
    assert result["status"] == "skipped"
    assert "TAVILY_API_KEY" in result["content"]


def test_tavily_research_tool_formats_completed_report(monkeypatch=None):
    tool = TavilyResearchTool(api_key="tvly-test", endpoint="https://example.test/research")
    data = {
        "request_id": "req-1",
        "status": "completed",
        "content": "Research report",
        "sources": [{"title": "Source", "url": "https://example.test/source"}],
    }

    formatted = tool._format_research_report(
        data["content"],
        [tool._normalize_source(data["sources"][0])],
        data["status"],
        data,
    )

    assert "Research report" in formatted
    assert "[R1] Source" in formatted
