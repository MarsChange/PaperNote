"""Structured Markdown notes for long-running PaperNote agent memory."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings

NOTE_TYPES = {
    "task_state",
    "conclusion",
    "blocker",
    "action",
    "reference",
    "conversation",
    "general",
}


class NoteTool:
    """Markdown + frontmatter note store with a JSON search index."""

    def __init__(self, workspace: Optional[Path | str] = None):
        self.workspace = Path(workspace or settings.note_workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.index_path = self.workspace / "notes_index.json"
        self.index = self._load_index()

    def run(self, payload: dict[str, Any]) -> Any:
        action = str(payload.get("action") or "search").lower()
        if action == "create":
            return self.create(
                title=str(payload.get("title") or "Untitled"),
                content=str(payload.get("content") or ""),
                note_type=str(payload.get("note_type") or payload.get("type") or "general"),
                tags=self._as_list(payload.get("tags")),
                paper_id=payload.get("paper_id"),
                conversation_id=payload.get("conversation_id"),
                source=payload.get("source") or "agent",
            )
        if action == "read":
            return self.read(str(payload.get("note_id") or payload.get("id") or ""))
        if action == "update":
            return self.update(
                str(payload.get("note_id") or payload.get("id") or ""),
                title=payload.get("title"),
                content=payload.get("content"),
                note_type=payload.get("note_type") or payload.get("type"),
                tags=payload.get("tags"),
            )
        if action == "search":
            return self.search(
                query=str(payload.get("query") or ""),
                limit=int(payload.get("limit") or 10),
                note_type=payload.get("note_type") or payload.get("type"),
                tags=self._as_list(payload.get("tags")),
                paper_id=payload.get("paper_id"),
                conversation_id=payload.get("conversation_id"),
            )
        if action == "list":
            return self.list_notes(
                limit=int(payload.get("limit") or 20),
                note_type=payload.get("note_type") or payload.get("type"),
                tags=self._as_list(payload.get("tags")),
                paper_id=payload.get("paper_id"),
                conversation_id=payload.get("conversation_id"),
            )
        if action == "summary":
            return self.summary(paper_id=payload.get("paper_id"))
        if action == "delete":
            return self.delete(str(payload.get("note_id") or payload.get("id") or ""))
        raise ValueError(f"Unsupported note action: {action}")

    def create(
        self,
        *,
        title: str,
        content: str,
        note_type: str = "general",
        tags: Optional[list[str]] = None,
        paper_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        source: str = "agent",
    ) -> dict[str, Any]:
        now = self._now()
        note_type = note_type if note_type in NOTE_TYPES else "general"
        note_id = f"note_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        safe_paper = self._safe_name(paper_id or "global")
        note_dir = self.workspace / safe_paper
        note_dir.mkdir(parents=True, exist_ok=True)
        file_path = note_dir / f"{note_id}.md"

        metadata = {
            "id": note_id,
            "title": title.strip() or "Untitled",
            "type": note_type,
            "tags": tags or [],
            "paper_id": paper_id or "",
            "conversation_id": conversation_id or "",
            "source": source,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "file_path": str(file_path),
        }
        file_path.write_text(self._build_markdown(metadata, content), encoding="utf-8")
        self.index[note_id] = metadata
        self._save_index()
        return {"note_id": note_id, "metadata": metadata}

    def read(self, note_id: str) -> dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"Note does not exist: {note_id}")
        metadata = dict(self.index[note_id])
        file_path = Path(metadata["file_path"])
        raw = file_path.read_text(encoding="utf-8")
        content = self._strip_frontmatter(raw)
        return {"metadata": metadata, "content": content}

    def update(
        self,
        note_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        note = self.read(note_id)
        metadata = note["metadata"]
        body = note["content"] if content is None else str(content)
        if title:
            metadata["title"] = str(title)
        if note_type:
            metadata["type"] = str(note_type) if str(note_type) in NOTE_TYPES else "general"
        if tags is not None:
            metadata["tags"] = self._as_list(tags)
        metadata["updated_at"] = self._now().isoformat()
        file_path = Path(metadata["file_path"])
        file_path.write_text(self._build_markdown(metadata, body), encoding="utf-8")
        self.index[note_id] = metadata
        self._save_index()
        return {"note_id": note_id, "metadata": metadata}

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        note_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        paper_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_terms = set(self._terms(query))
        results: list[dict[str, Any]] = []
        for note_id, metadata in self.index.items():
            if not self._metadata_matches(metadata, note_type, tags, paper_id, conversation_id):
                continue
            try:
                note = self.read(note_id)
            except Exception:
                continue
            haystack = " ".join(
                [
                    str(metadata.get("title", "")),
                    " ".join(metadata.get("tags", [])),
                    str(note.get("content", "")),
                ]
            )
            score = self._score(query_terms, haystack)
            if query_terms and score <= 0:
                continue
            item = {
                "note_id": note_id,
                "title": metadata.get("title", ""),
                "type": metadata.get("type", "general"),
                "tags": metadata.get("tags", []),
                "paper_id": metadata.get("paper_id", ""),
                "conversation_id": metadata.get("conversation_id", ""),
                "updated_at": metadata.get("updated_at", ""),
                "content": note.get("content", ""),
                "score": score,
            }
            results.append(item)

        results.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
        return results[: max(1, limit)]

    def list_notes(
        self,
        *,
        limit: int = 20,
        note_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        paper_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        notes = [
            dict(metadata)
            for metadata in self.index.values()
            if self._metadata_matches(metadata, note_type, tags, paper_id, conversation_id)
        ]
        notes.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return notes[: max(1, limit)]

    def summary(self, *, paper_id: Optional[str] = None) -> dict[str, Any]:
        notes = self.list_notes(limit=1000, paper_id=paper_id)
        type_counts: dict[str, int] = {}
        for note in notes:
            note_type = note.get("type", "general")
            type_counts[note_type] = type_counts.get(note_type, 0) + 1
        return {
            "total_notes": len(notes),
            "type_distribution": type_counts,
            "recent_notes": notes[:5],
        }

    def delete(self, note_id: str) -> dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"Note does not exist: {note_id}")
        metadata = self.index.pop(note_id)
        file_path = Path(metadata["file_path"])
        if file_path.exists():
            file_path.unlink()
        self._save_index()
        return {"deleted": True, "note_id": note_id, "title": metadata.get("title", "")}

    def _metadata_matches(
        self,
        metadata: dict[str, Any],
        note_type: Optional[str],
        tags: Optional[list[str]],
        paper_id: Optional[str],
        conversation_id: Optional[str],
    ) -> bool:
        if note_type and metadata.get("type") != note_type:
            return False
        if paper_id and metadata.get("paper_id") not in {paper_id, ""}:
            return False
        if conversation_id and metadata.get("conversation_id") not in {conversation_id, ""}:
            return False
        if tags:
            note_tags = set(metadata.get("tags", []))
            if not note_tags.intersection(tags):
                return False
        return True

    def _build_markdown(self, metadata: dict[str, Any], content: str) -> str:
        lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = json.dumps(str(value), ensure_ascii=False)
            lines.append(f"{key}: {rendered}")
        lines.append("---")
        body = str(content or "").strip()
        if not body.startswith("#"):
            body = f"# {metadata.get('title', 'Untitled')}\n\n{body}".strip()
        return "\n".join(lines) + "\n\n" + body + "\n"

    def _strip_frontmatter(self, raw: str) -> str:
        if raw.startswith("---\n"):
            parts = raw.split("---\n", 2)
            if len(parts) == 3:
                return parts[2].strip()
        return raw.strip()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_index(self):
        self.index_path.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.1
        text_terms = set(self._terms(text))
        if not text_terms:
            return 0.0
        overlap = query_terms & text_terms
        return len(overlap) / max(len(query_terms), 1)

    def _terms(self, text: str) -> list[str]:
        normalized = str(text or "").lower()
        words = re.findall(r"[a-z0-9_]{2,}", normalized)
        cjk = re.findall(r"[\u3400-\u9fff]", normalized)
        return words + cjk + [a + b for a, b in zip(cjk, cjk[1:])]

    def _as_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(",") if part.strip()]

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "global"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)


note_tool = NoteTool()
