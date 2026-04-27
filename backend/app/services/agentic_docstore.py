"""Local parent-chunk DocStore for PaperNote Agentic RAG.

superMew stores L1/L2 parent chunks in PostgreSQL + Redis. PaperNote does not
need that account/cache stack, so the same parent-document idea is persisted as
JSON next to each parsed paper artifact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import settings

LEVEL_SIZES = {1: 2600, 2: 1300, 3: 650}
LEVEL_OVERLAPS = {1: 320, 2: 180, 3: 90}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class AgenticChunkBuilder:
    """Create L1/L2/L3 sliding-window chunks from PaperNote parsed blocks."""

    def build(self, paper_id: str, blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parents: list[dict[str, Any]] = []
        leaves: list[dict[str, Any]] = []
        chunk_idx = 0
        for block in blocks:
            block_text = normalize_text(block.get("search_text") or block.get("content") or "")
            if not block_text:
                continue
            for l1_index, l1_text in enumerate(self._split(block_text, 1)):
                l1_id = self._chunk_id(block["id"], 1, l1_index)
                l1 = self._chunk_payload(paper_id, block, l1_text, l1_id, "", l1_id, 1, chunk_idx)
                chunk_idx += 1
                parents.append(l1)
                for l2_index, l2_text in enumerate(self._split(l1_text, 2)):
                    l2_id = self._chunk_id(block["id"], 2, l1_index, l2_index)
                    l2 = self._chunk_payload(paper_id, block, l2_text, l2_id, l1_id, l1_id, 2, chunk_idx)
                    chunk_idx += 1
                    parents.append(l2)
                    for l3_index, l3_text in enumerate(self._split(l2_text, 3)):
                        l3_id = self._chunk_id(block["id"], 3, l1_index, l2_index, l3_index)
                        leaves.append(
                            self._chunk_payload(
                                paper_id,
                                block,
                                l3_text,
                                l3_id,
                                l2_id,
                                l1_id,
                                3,
                                chunk_idx,
                            )
                        )
                        chunk_idx += 1
        return parents, leaves

    def _split(self, text: str, level: int) -> list[str]:
        text = normalize_text(text)
        size = LEVEL_SIZES[level]
        overlap = LEVEL_OVERLAPS[level]
        if len(text) <= size:
            return [text] if text else []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start + size // 2, end),
                    text.rfind("。", start + size // 2, end),
                    text.rfind(".", start + size // 2, end),
                    text.rfind(" ", start + size // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - overlap, 0)
        return chunks

    def _chunk_payload(
        self,
        paper_id: str,
        block: dict[str, Any],
        text: str,
        chunk_id: str,
        parent_id: str,
        root_id: str,
        level: int,
        chunk_idx: int,
    ) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "paper_id": paper_id,
            "block_id": block.get("id", ""),
            "type": block.get("type", "text"),
            "filename": paper_id,
            "file_type": "PaperNotePDF",
            "file_path": block.get("asset_path", ""),
            "page_number": int(block.get("page_number") or 1),
            "title": block.get("title") or block.get("section") or "",
            "section": block.get("section", ""),
            "text": text,
            "content": text,
            "search_text": text,
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_id,
            "root_chunk_id": root_id,
            "chunk_level": level,
            "chunk_idx": chunk_idx,
            "order": int(block.get("order") or 0),
            "asset_path": block.get("asset_path", ""),
            "asset_relpath": block.get("asset_relpath", ""),
            "semantic_summary": block.get("semantic_summary", ""),
            "semantic_metadata": block.get("semantic_metadata", {}),
        }

    def _chunk_id(self, block_id: str, level: int, *indices: int) -> str:
        suffix = "-".join(str(index) for index in indices)
        return f"{block_id}:l{level}:{suffix}"


class AgenticDocStore:
    def paper_dir(self, paper_id: str) -> Path:
        return settings.upload_dir / paper_id / "parsed"

    def parents_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "agentic_parent_chunks.json"

    def leaves_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "agentic_leaf_chunks.json"

    def write(self, paper_id: str, parents: list[dict[str, Any]], leaves: list[dict[str, Any]]):
        directory = self.paper_dir(paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.parents_path(paper_id).write_text(
            json.dumps(parents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.leaves_path(paper_id).write_text(
            json.dumps(leaves, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_parents(self, paper_id: str) -> list[dict[str, Any]]:
        return self._load_json_list(self.parents_path(paper_id))

    def load_leaves(self, paper_id: str) -> list[dict[str, Any]]:
        return self._load_json_list(self.leaves_path(paper_id))

    def get_parents_by_ids(self, paper_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        wanted = {chunk_id for chunk_id in chunk_ids if chunk_id}
        by_id = {
            item.get("chunk_id"): item
            for item in self.load_parents(paper_id)
            if item.get("chunk_id") in wanted
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def remove_existing_leaf_texts(self, paper_id: str) -> list[str]:
        leaves = self.load_leaves(paper_id)
        return [str(item.get("text", "")) for item in leaves if item.get("text")]

    def _load_json_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return payload if isinstance(payload, list) else []


agentic_chunk_builder = AgenticChunkBuilder()
agentic_docstore = AgenticDocStore()
