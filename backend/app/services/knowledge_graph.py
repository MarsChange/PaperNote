"""Multimodal paper knowledge graph indexing.

The graph is intentionally lightweight: it persists as JSON next to the parsed
paper artifacts and augments Milvus retrieval with paper structure, modality,
and concept relationships.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.mineru import ParseResult

logger = logging.getLogger(__name__)

GRAPH_VERSION = "papernote-kg-v1"
MAX_CONCEPTS_PER_BLOCK = 8

STOPWORDS = {
    "about",
    "after",
    "also",
    "analysis",
    "based",
    "between",
    "data",
    "different",
    "during",
    "figure",
    "from",
    "method",
    "model",
    "paper",
    "result",
    "results",
    "study",
    "table",
    "that",
    "their",
    "these",
    "this",
    "using",
    "with",
}


class PaperKnowledgeGraphIndexer:
    """Builds and loads a deterministic multimodal KG for each parsed paper."""

    def graph_path(self, paper_id: str) -> Path:
        return settings.upload_dir / paper_id / "parsed" / "knowledge_graph.json"

    def index_paper(
        self,
        paper_id: str,
        parsed: ParseResult,
        blocks: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        graph = self._build_graph(paper_id, parsed, blocks, metadata)
        path = self.graph_path(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "path": str(path),
            "version": GRAPH_VERSION,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "concept_count": graph["metadata"].get("concept_count", 0),
        }

    def load_graph(self, paper_id: str) -> dict[str, Any]:
        path = self.graph_path(paper_id)
        if not path.exists():
            return {"paper_id": paper_id, "nodes": [], "edges": [], "metadata": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read knowledge graph for %s: %s", paper_id, exc)
            return {"paper_id": paper_id, "nodes": [], "edges": [], "metadata": {}}

    def expand_scores(
        self,
        paper_id: str,
        base_scores: dict[str, float],
        query: str,
    ) -> dict[str, float]:
        """Return small score bonuses from KG neighborhoods and query concepts."""
        if not base_scores:
            return {}

        graph = self.load_graph(paper_id)
        if not graph.get("nodes"):
            return {}

        block_page, block_section, block_concepts, concept_blocks = self._graph_indexes(
            graph
        )
        bonuses: dict[str, float] = defaultdict(float)
        query_concepts = {concept.lower() for concept in self.extract_concepts(query, 12)}

        for concept, block_ids in concept_blocks.items():
            if concept.lower() in query_concepts:
                for block_id in block_ids:
                    bonuses[block_id] += 0.08

        top_blocks = sorted(base_scores.items(), key=lambda item: item[1], reverse=True)[:4]
        for seed_id, seed_score in top_blocks:
            seed_page = block_page.get(seed_id)
            seed_section = block_section.get(seed_id)
            seed_concepts = block_concepts.get(seed_id, set())

            for block_id, page in block_page.items():
                if block_id == seed_id:
                    continue
                if seed_page is not None and page == seed_page:
                    bonuses[block_id] += 0.02 * seed_score
                if seed_section and block_section.get(block_id) == seed_section:
                    bonuses[block_id] += 0.04 * seed_score

            for concept in seed_concepts:
                for block_id in concept_blocks.get(concept, set()):
                    if block_id != seed_id:
                        bonuses[block_id] += 0.05 * seed_score

        return dict(bonuses)

    def context_for_blocks(
        self, paper_id: str, block_ids: list[str], max_items: int = 6
    ) -> dict[str, str]:
        graph = self.load_graph(paper_id)
        if not graph.get("nodes"):
            return {}

        nodes = {node["id"]: node for node in graph.get("nodes", [])}
        block_page, block_section, block_concepts, _ = self._graph_indexes(graph)
        page_modal_blocks: dict[int, list[str]] = defaultdict(list)

        for node in graph.get("nodes", []):
            if node.get("type") == "block" and node.get("properties", {}).get("is_modal"):
                page = node.get("properties", {}).get("page_number")
                if isinstance(page, int):
                    page_modal_blocks[page].append(node["id"])

        contexts: dict[str, str] = {}
        for block_id in block_ids:
            page = block_page.get(block_id)
            section = block_section.get(block_id)
            concepts = sorted(block_concepts.get(block_id, set()))[:max_items]
            related_modal = [
                nodes[item]["label"]
                for item in page_modal_blocks.get(page or -1, [])
                if item != block_id and item in nodes
            ][:3]

            parts = []
            if section:
                parts.append(f"KG section: {section}")
            if concepts:
                parts.append(f"KG concepts: {', '.join(concepts)}")
            if related_modal:
                parts.append(f"Related multimodal evidence: {', '.join(related_modal)}")
            contexts[block_id] = "\n".join(parts)

        return contexts

    def _build_graph(
        self,
        paper_id: str,
        parsed: ParseResult,
        blocks: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        concept_seen: set[str] = set()

        def add_node(
            node_id: str,
            node_type: str,
            label: str,
            properties: dict[str, Any] | None = None,
        ):
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "label": label,
                    "properties": properties or {},
                }

        def add_edge(
            source: str,
            target: str,
            edge_type: str,
            weight: float = 1.0,
            properties: dict[str, Any] | None = None,
        ):
            edge_id = self._edge_id(source, target, edge_type)
            if edge_id not in edges:
                edges[edge_id] = {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "type": edge_type,
                    "weight": round(weight, 4),
                    "properties": properties or {},
                }

        paper_node_id = f"paper:{paper_id}"
        add_node(
            paper_node_id,
            "paper",
            metadata.get("summary_preview", "")[:120] or paper_id,
            {
                "paper_id": paper_id,
                "page_count": parsed.page_count,
                "stats": metadata.get("stats", {}),
            },
        )

        previous_block_id = ""
        for block in blocks:
            page_number = int(block.get("page_number") or 1)
            block_id = block["id"]
            block_type = block.get("type", "text")
            section = block.get("section") or f"Page {page_number}"
            title = block.get("title") or section
            semantic_metadata = block.get("semantic_metadata") or {}
            semantic_entity = semantic_metadata.get("entity") or {}

            page_node_id = f"page:{paper_id}:{page_number}"
            section_node_id = f"section:{paper_id}:{page_number}:{self._slug(section)}"
            add_node(page_node_id, "page", f"Page {page_number}", {"page_number": page_number})
            add_node(section_node_id, "section", section, {"page_number": page_number})
            add_edge(paper_node_id, page_node_id, "HAS_PAGE")
            add_edge(page_node_id, section_node_id, "HAS_SECTION")

            add_node(
                block_id,
                "block",
                title,
                {
                    "paper_id": paper_id,
                    "block_type": block_type,
                    "page_number": page_number,
                    "section": section,
                    "order": block.get("order", 0),
                    "is_modal": block_type != "text",
                    "asset_relpath": block.get("asset_relpath", ""),
                    "excerpt": self._normalize(block.get("content", ""))[:500],
                    "semantic_summary": block.get("semantic_summary", ""),
                    "semantic_entity": semantic_entity,
                },
            )
            add_edge(section_node_id, block_id, "HAS_BLOCK")
            add_edge(page_node_id, block_id, "HAS_BLOCK")

            if previous_block_id:
                add_edge(previous_block_id, block_id, "NEXT_BLOCK", 0.35)
                add_edge(block_id, previous_block_id, "PREVIOUS_BLOCK", 0.35)
            previous_block_id = block_id

            if block_type != "text":
                modal_node_id = f"modal:{block_id}"
                add_node(
                    modal_node_id,
                    block_type,
                    title,
                    {
                        "page_number": page_number,
                        "asset_relpath": block.get("asset_relpath", ""),
                        "summary": self._normalize(
                            block.get("semantic_summary")
                            or semantic_metadata.get("summary")
                            or block.get("content", "")
                        )[:700],
                        "semantic_metadata": semantic_metadata,
                    },
                )
                add_edge(block_id, modal_node_id, "HAS_MODAL_ASSET")
                add_edge(modal_node_id, section_node_id, "APPEARS_IN_SECTION", 0.6)
                if semantic_entity.get("name"):
                    entity_node_id = f"entity:{paper_id}:{self._stable_id(str(semantic_entity['name']))}"
                    add_node(
                        entity_node_id,
                        "semantic_entity",
                        str(semantic_entity["name"]),
                        {
                            "entity_type": semantic_entity.get("type", block_type),
                            "summary": semantic_entity.get("summary", ""),
                        },
                    )
                    add_edge(modal_node_id, entity_node_id, "DESCRIBED_AS", 0.8)

            concept_source = " ".join(
                [
                    str(block.get("title", "")),
                    str(block.get("section", "")),
                    str(block.get("content", "")),
                    str(block.get("semantic_summary", "")),
                    " ".join(str(item) for item in semantic_metadata.get("keywords", [])),
                    str(semantic_entity.get("summary", "")),
                ]
            )
            for concept in self.extract_concepts(concept_source, MAX_CONCEPTS_PER_BLOCK):
                concept_id = f"concept:{paper_id}:{self._stable_id(concept)}"
                concept_seen.add(concept_id)
                add_node(
                    concept_id,
                    "concept",
                    concept,
                    {"paper_id": paper_id, "label_norm": concept.lower()},
                )
                add_edge(block_id, concept_id, "MENTIONS", 0.75)
                add_edge(concept_id, page_node_id, "APPEARS_ON_PAGE", 0.35)

        return {
            "version": GRAPH_VERSION,
            "paper_id": paper_id,
            "metadata": {
                "source": "mineru_content_list",
                "patterns": [
                    "document-page-section-block hierarchy",
                    "multimodal asset nodes",
                    "concept mention edges",
                    "local graph reranking",
                ],
                "node_count": len(nodes),
                "edge_count": len(edges),
                "concept_count": len(concept_seen),
            },
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }

    def _graph_indexes(
        self, graph: dict[str, Any]
    ) -> tuple[dict[str, int], dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
        nodes = {node["id"]: node for node in graph.get("nodes", [])}
        block_page: dict[str, int] = {}
        block_section: dict[str, str] = {}
        block_concepts: dict[str, set[str]] = defaultdict(set)
        concept_blocks: dict[str, set[str]] = defaultdict(set)

        for node_id, node in nodes.items():
            if node.get("type") != "block":
                continue
            props = node.get("properties", {})
            page = props.get("page_number")
            if isinstance(page, int):
                block_page[node_id] = page
            if props.get("section"):
                block_section[node_id] = props["section"]

        for edge in graph.get("edges", []):
            if edge.get("type") != "MENTIONS":
                continue
            source = edge.get("source")
            target = edge.get("target")
            concept_node = nodes.get(target or "")
            if not source or not concept_node:
                continue
            concept = str(concept_node.get("label", "")).strip()
            if not concept:
                continue
            block_concepts[source].add(concept)
            concept_blocks[concept].add(source)

        return block_page, block_section, block_concepts, concept_blocks

    def extract_concepts(self, text: str, limit: int = MAX_CONCEPTS_PER_BLOCK) -> list[str]:
        clean_text = self._normalize(text)
        if not clean_text:
            return []

        candidates: list[str] = []
        candidates.extend(re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", clean_text))
        candidates.extend(
            match.strip()
            for match in re.findall(
                r"\b(?:[A-Z][a-z0-9+-]{2,}|[a-z][a-z0-9+-]{4,})"
                r"(?:\s+(?:[A-Z]?[a-z0-9+-]{3,})){0,3}\b",
                clean_text,
            )
        )
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", clean_text))

        scored: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            concept = self._clean_concept(candidate)
            if not concept:
                continue
            scored[concept] += 1

        return [
            item[0]
            for item in sorted(scored.items(), key=lambda entry: (-entry[1], entry[0]))
            [:limit]
        ]

    def _clean_concept(self, value: str) -> str:
        concept = re.sub(r"\s+", " ", value).strip(" .,:;()[]{}")
        if len(concept) < 3 or len(concept) > 80:
            return ""
        if concept.lower() in STOPWORDS:
            return ""
        words = concept.split()
        if words and all(word.lower() in STOPWORDS for word in words):
            return ""
        return concept

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug[:64] or self._stable_id(value)

    def _stable_id(self, value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

    def _edge_id(self, source: str, target: str, edge_type: str) -> str:
        return f"edge:{self._stable_id(f'{source}|{target}|{edge_type}')}"


knowledge_graph_indexer = PaperKnowledgeGraphIndexer()
