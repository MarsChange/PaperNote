"""Semantic enrichment for MinerU multimodal paper blocks.

This module follows the RAG-Anything pattern at a smaller scale: multimodal
content gets context-aware descriptions and entity metadata before retrieval
indexing. LLM/VLM calls are optional; deterministic caption/context heuristics
always run first and remain the fallback path.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Optional

import openai

from app.core.config import settings
from app.services.llm import PROVIDER_CONFIG
from app.services.mineru import ParseResult

logger = logging.getLogger(__name__)

ENRICHMENT_VERSION = "papernote-multimodal-enrichment-v1"
MODAL_TYPES = {"image", "table", "equation"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class MultimodalEnricher:
    """Adds structured semantic metadata to image/table/equation blocks."""

    def __init__(self):
        self._client: Optional[openai.OpenAI] = None

    def enrich_blocks(
        self,
        paper_id: str,
        parsed: ParseResult,
        blocks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not settings.enable_multimodal_enrichment:
            return blocks, {"enabled": False, "version": ENRICHMENT_VERSION}

        content_items = parsed.content_list
        source_items = {
            index: item
            for index, item in enumerate(content_items)
            if isinstance(item, dict)
        }
        stats = {"heuristic": 0, "llm": 0, "failed_llm": 0, "total": 0}
        llm_budget = max(settings.multimodal_enrichment_llm_limit, 0)
        can_call_llm = llm_budget > 0 and self._can_call_llm()
        llm_block_ids = (
            self._llm_priority_block_ids(blocks, llm_budget)
            if can_call_llm
            else set()
        )

        for block in blocks:
            block_type = str(block.get("type", "")).lower()
            if block_type not in MODAL_TYPES:
                continue

            source_index = int(block.get("source_index", -1))
            source_item = source_items.get(source_index, {})
            context_text = self._context_for_item(content_items, source_index)
            metadata = self._heuristic_metadata(block, source_item, context_text)
            stats["heuristic"] += 1
            stats["total"] += 1

            if str(block.get("id", "")) in llm_block_ids:
                llm_metadata = self._llm_metadata(block, source_item, context_text)
                if llm_metadata:
                    metadata = self._merge_metadata(metadata, llm_metadata)
                    stats["llm"] += 1
                else:
                    stats["failed_llm"] += 1

            self._apply_metadata(block, metadata)

        return blocks, {
            "enabled": True,
            "version": ENRICHMENT_VERSION,
            "stats": stats,
            "llm_limit": settings.multimodal_enrichment_llm_limit,
            "llm_priority": "image_first",
        }

    def _llm_priority_block_ids(
        self, blocks: list[dict[str, Any]], llm_budget: int
    ) -> set[str]:
        if llm_budget <= 0:
            return set()
        modal_blocks = [
            block
            for block in blocks
            if str(block.get("type", "")).lower() in MODAL_TYPES
        ]
        priority = {"image": 0, "table": 1, "equation": 2}
        ordered_blocks = sorted(
            modal_blocks,
            key=lambda block: (
                priority.get(str(block.get("type", "")).lower(), 99),
                int(block.get("order", 0)),
            ),
        )
        return {
            str(block.get("id", ""))
            for block in ordered_blocks[:llm_budget]
            if block.get("id")
        }

    def _can_call_llm(self) -> bool:
        provider = settings.llm_provider
        config = PROVIDER_CONFIG.get(provider)
        if not config:
            return False
        _, api_key_attr, _ = config
        return bool(getattr(settings, api_key_attr, ""))

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            provider = settings.llm_provider
            base_url_attr, api_key_attr, _ = PROVIDER_CONFIG[provider]
            self._client = openai.OpenAI(
                api_key=getattr(settings, api_key_attr, ""),
                base_url=getattr(settings, base_url_attr, ""),
            )
        return self._client

    def _heuristic_metadata(
        self,
        block: dict[str, Any],
        source_item: dict[str, Any],
        context_text: str,
    ) -> dict[str, Any]:
        block_type = str(block.get("type", "generic"))
        captions = self._list_texts(
            source_item.get("image_caption")
            or source_item.get("img_caption")
            or source_item.get("table_caption")
            or source_item.get("equation_caption")
            or []
        )
        footnotes = self._list_texts(
            source_item.get("image_footnote")
            or source_item.get("img_footnote")
            or source_item.get("table_footnote")
            or []
        )
        body = self._body_text(block_type, source_item, block)
        section = str(block.get("section") or block.get("title") or "").strip()
        page_number = int(block.get("page_number") or 1)

        details = [f"{block_type.title()} on page {page_number}."]
        if captions:
            details.append(f"Caption: {'; '.join(captions)}.")
        if body:
            details.append(f"Content: {body[:700]}.")
        if context_text:
            details.append(f"Nearby context: {context_text[:700]}.")

        keywords = self._keywords(" ".join([*captions, body, context_text, section]))
        entity_name = self._entity_name(block_type, captions, section, page_number)
        summary = self._compact_summary(details, block_type, page_number)
        modality_specific = self._modality_specific(block_type, source_item, body)

        return {
            "version": ENRICHMENT_VERSION,
            "source": "heuristic",
            "modality": block_type,
            "summary": summary,
            "detailed_description": " ".join(details).strip(),
            "entity": {
                "name": entity_name,
                "type": block_type,
                "summary": summary,
            },
            "keywords": keywords,
            "claims": [],
            "relations": self._relations(block, keywords),
            "caption": captions,
            "footnote": footnotes,
            "nearby_context": context_text,
            "modality_specific": modality_specific,
        }

    def _llm_metadata(
        self,
        block: dict[str, Any],
        source_item: dict[str, Any],
        context_text: str,
    ) -> Optional[dict[str, Any]]:
        try:
            provider = settings.llm_provider
            _, _, default_model = PROVIDER_CONFIG[provider]
            model = settings.llm_model or default_model
            prompt = self._build_prompt(block, source_item, context_text)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You enrich multimodal scientific paper content for RAG. "
                        "Return compact JSON only. Do not invent facts beyond the "
                        "caption, content, image, and nearby context."
                    ),
                },
                {"role": "user", "content": self._message_content(block, prompt)},
            ]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                )
            raw_content = response.choices[0].message.content or ""
            parsed = self._parse_json(raw_content)
            if not parsed:
                return None
            return self._normalize_llm_payload(parsed, block)
        except Exception as exc:
            logger.warning("Multimodal LLM enrichment failed for %s: %s", block.get("id"), exc)
            return None

    def _build_prompt(
        self,
        block: dict[str, Any],
        source_item: dict[str, Any],
        context_text: str,
    ) -> str:
        block_type = block.get("type", "generic")
        payload = {
            "block_id": block.get("id"),
            "modality": block_type,
            "page_number": block.get("page_number"),
            "section": block.get("section"),
            "title": block.get("title"),
            "caption": self._list_texts(
                source_item.get("image_caption")
                or source_item.get("img_caption")
                or source_item.get("table_caption")
                or source_item.get("equation_caption")
                or []
            ),
            "footnote": self._list_texts(
                source_item.get("image_footnote")
                or source_item.get("img_footnote")
                or source_item.get("table_footnote")
                or []
            ),
            "content": self._body_text(str(block_type), source_item, block)[:2500],
            "nearby_context": context_text[:2500],
            "figure_id": source_item.get("figure_id"),
            "figure_source": source_item.get("figure_source"),
            "subfigure_count": source_item.get("subfigure_count"),
        }
        return (
            "Create semantic metadata for this paper block. Return JSON with keys: "
            "summary, detailed_description, entity {name,type,summary}, "
            "keywords array, claims array, relations array of {type,target,description}. "
            f"\n\nInput:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _message_content(self, block: dict[str, Any], prompt: str) -> Any:
        if block.get("type") != "image" or not settings.enable_multimodal_answers:
            return prompt

        data_url = self._image_data_url(str(block.get("asset_path", "")))
        if not data_url:
            return prompt
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    def _normalize_llm_payload(
        self, payload: dict[str, Any], block: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        block_type = str(block.get("type", "generic"))
        raw_entity = (
            payload.get("entity")
            or payload.get("entity_info")
            or payload.get("entities")
            or {}
        )
        entity = self._normalize_entity_payload(raw_entity, block, block_type)
        summary = self._pick_text(
            payload.get("summary"),
            payload.get("semantic_summary"),
            entity.get("summary"),
            payload.get("description"),
            payload.get("analysis"),
            block.get("title"),
        )
        detailed = self._pick_text(
            payload.get("detailed_description"),
            payload.get("detailed"),
            payload.get("description"),
            payload.get("analysis"),
            summary,
        )
        entity["summary"] = self._pick_text(entity.get("summary"), summary)[:900]
        return {
            "version": ENRICHMENT_VERSION,
            "source": "llm",
            "modality": block_type,
            "summary": summary[:900],
            "detailed_description": detailed[:2500],
            "entity": entity,
            "keywords": self._dedupe_texts(payload.get("keywords", []), 12),
            "claims": self._dedupe_texts(payload.get("claims", []), 8),
            "relations": self._normalize_relations(payload.get("relations", []), block),
        }

    def _normalize_entity_payload(
        self, raw_entity: Any, block: dict[str, Any], block_type: str
    ) -> dict[str, str]:
        entity: dict[str, Any] = {}
        if isinstance(raw_entity, dict):
            entity = raw_entity
        elif isinstance(raw_entity, list):
            entity = next(
                (item for item in raw_entity if isinstance(item, dict)),
                {},
            )
            if not entity:
                entity = {"name": self._pick_text(raw_entity)}
        elif raw_entity:
            entity = {"name": self._pick_text(raw_entity)}

        return {
            "name": self._pick_text(
                entity.get("name"),
                entity.get("entity_name"),
                entity.get("label"),
                entity.get("title"),
                block.get("title"),
                block.get("id"),
            )[:160],
            "type": self._pick_text(
                entity.get("type"),
                entity.get("entity_type"),
                entity.get("modality"),
                block_type,
            )[:80],
            "summary": self._pick_text(
                entity.get("summary"),
                entity.get("description"),
                entity.get("text"),
            )[:900],
        }

    def _merge_metadata(
        self, heuristic: dict[str, Any], llm_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        merged = {**heuristic, **llm_metadata}
        merged["caption"] = heuristic.get("caption", [])
        merged["footnote"] = heuristic.get("footnote", [])
        merged["nearby_context"] = heuristic.get("nearby_context", "")
        merged["modality_specific"] = heuristic.get("modality_specific", {})
        merged["keywords"] = self._dedupe_texts(
            [*llm_metadata.get("keywords", []), *heuristic.get("keywords", [])],
            12,
        )
        merged["relations"] = [
            *llm_metadata.get("relations", []),
            *heuristic.get("relations", []),
        ][:12]
        return merged

    def _apply_metadata(self, block: dict[str, Any], metadata: dict[str, Any]):
        semantic_text = self.semantic_text(metadata)
        block["semantic_metadata"] = metadata
        block["semantic_summary"] = metadata.get("summary", "")
        if semantic_text and semantic_text not in block.get("content", ""):
            block["content"] = f"{block.get('content', '').strip()}\n\nSemantic metadata:\n{semantic_text}".strip()
        if semantic_text and semantic_text not in block.get("search_text", ""):
            block["search_text"] = f"{block.get('search_text', '').strip()}\n{semantic_text}".strip()

    def semantic_text(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("summary", ""),
            metadata.get("detailed_description", ""),
        ]
        entity = metadata.get("entity") or {}
        if entity:
            if isinstance(entity, dict):
                entity_values = [
                    str(entity.get("name", "")),
                    str(entity.get("type", "")),
                    str(entity.get("summary", "")),
                ]
            else:
                entity_values = self._dedupe_texts(entity, 3)
            parts.append(
                "Entity: "
                + " | ".join(
                    value
                    for value in entity_values
                    if value
                )
            )
        keywords = self._dedupe_texts(metadata.get("keywords", []), 12)
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        claims = self._dedupe_texts(metadata.get("claims", []), 8)
        if claims:
            parts.append("Claims: " + "; ".join(claims))
        return "\n".join(part for part in parts if part).strip()

    def _context_for_item(
        self, content_list: list[dict[str, Any]], source_index: int, window: int = 1
    ) -> str:
        if source_index < 0 or source_index >= len(content_list):
            return ""
        current_item = content_list[source_index]
        current_page = int(current_item.get("page_idx", 0))
        texts: list[str] = []
        figure_context = self._normalize(str(current_item.get("figure_context", "")))
        if figure_context:
            texts.append(figure_context)
        for index, item in enumerate(content_list):
            if index == source_index or not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            page_idx = int(item.get("page_idx", 0))
            if abs(page_idx - current_page) > window:
                continue
            text = self._normalize(str(item.get("text", "")))
            if text:
                texts.append(text)
        return "\n".join(texts[:5])[:2500]

    def _body_text(
        self, block_type: str, source_item: dict[str, Any], block: dict[str, Any]
    ) -> str:
        if block_type == "table":
            return str(source_item.get("table_body") or block.get("content") or "").strip()
        if block_type == "equation":
            return self._normalize(
                " ".join(
                    str(value)
                    for value in [
                        source_item.get("latex"),
                        source_item.get("text"),
                        block.get("content"),
                    ]
                    if value
                )
            )
        return self._normalize(str(block.get("content", "")))

    def _modality_specific(
        self, block_type: str, source_item: dict[str, Any], body: str
    ) -> dict[str, Any]:
        if block_type == "table":
            rows = [line for line in body.splitlines() if "|" in line]
            header = rows[0] if rows else ""
            columns = [
                item.strip()
                for item in header.strip("|").split("|")
                if item.strip()
            ]
            data_rows = max(len(rows) - 2, 0) if len(rows) > 1 else len(rows)
            return {"columns": columns[:12], "row_count_estimate": data_rows}
        if block_type == "equation":
            latex = str(source_item.get("latex") or body)
            variables = self._dedupe_texts(
                re.findall(r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9_]*)", latex),
                12,
            )
            return {"latex": latex[:1200], "variables": variables}
        if block_type == "image":
            image_path = str(source_item.get("img_path") or source_item.get("image_path") or "")
            return {
                "asset_available": bool(image_path and Path(image_path).exists()),
                "asset_path": image_path,
                "figure_id": source_item.get("figure_id", ""),
                "figure_source": source_item.get("figure_source", ""),
                "is_composite_figure": bool(source_item.get("is_composite_figure")),
                "subfigure_count": source_item.get("subfigure_count", 0),
                "figure_bbox": source_item.get("figure_bbox", []),
            }
        return {}

    def _relations(self, block: dict[str, Any], keywords: list[str]) -> list[dict[str, str]]:
        relations: list[dict[str, str]] = []
        if block.get("section"):
            relations.append(
                {
                    "type": "appears_in_section",
                    "target": str(block["section"]),
                    "description": "Multimodal evidence appears in this section.",
                }
            )
        for keyword in keywords[:4]:
            relations.append(
                {
                    "type": "mentions_concept",
                    "target": keyword,
                    "description": "Concept inferred from caption, content, or context.",
                }
            )
        return relations

    def _normalize_relations(
        self, relations: Any, block: dict[str, Any]
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        if not isinstance(relations, list):
            return normalized
        for relation in relations[:8]:
            if isinstance(relation, str):
                normalized.append(
                    {
                        "type": "related_to",
                        "target": relation[:120],
                        "description": relation[:240],
                    }
                )
                continue
            if not isinstance(relation, dict):
                continue
            normalized.append(
                {
                    "type": str(relation.get("type") or "related_to")[:80],
                    "target": str(relation.get("target") or block.get("section") or "")[:160],
                    "description": str(relation.get("description") or "")[:300],
                }
            )
        return normalized

    def _compact_summary(self, details: list[str], block_type: str, page_number: int) -> str:
        text = " ".join(details).strip()
        if not text:
            return f"{block_type.title()} evidence on page {page_number}."
        return text[:700]

    def _entity_name(
        self,
        block_type: str,
        captions: list[str],
        section: str,
        page_number: int,
    ) -> str:
        if captions:
            return captions[0][:120]
        if section:
            return f"{block_type.title()} in {section[:80]}"
        return f"{block_type.title()} evidence on page {page_number}"

    def _keywords(self, text: str, limit: int = 12) -> list[str]:
        candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9+\-]{3,}\b", text)
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", text))
        stopwords = {
            "caption",
            "content",
            "context",
            "figure",
            "image",
            "method",
            "model",
            "page",
            "section",
            "table",
            "this",
            "with",
        }
        return self._dedupe_texts(
            [
                value.strip(".,:;()[]{}")
                for value in candidates
                if value.lower() not in stopwords
            ],
            limit,
        )

    def _dedupe_texts(self, values: Any, limit: int) -> list[str]:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = self._pick_text(value)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            deduped.append(text[:180])
            if len(deduped) >= limit:
                break
        return deduped

    def _list_texts(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _pick_text(self, *values: Any) -> str:
        for value in values:
            text = self._coerce_text(value)
            if text:
                return text
        return ""

    def _coerce_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return self._pick_text(*value)
        if isinstance(value, dict):
            return self._pick_text(
                value.get("summary"),
                value.get("detailed_description"),
                value.get("description"),
                value.get("name"),
                value.get("entity_name"),
                value.get("label"),
                value.get("title"),
                value.get("text"),
                value.get("target"),
                value.get("keyword"),
                value.get("term"),
                value.get("claim"),
            )
        return self._normalize(str(value))

    def _parse_json(self, text: str) -> Optional[dict[str, Any]]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
            stripped = re.sub(r"\n```$", "", stripped)
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None

    def _image_data_url(self, asset_path: str) -> str:
        path = Path(asset_path)
        if not asset_path or not path.exists() or path.stat().st_size > MAX_IMAGE_BYTES:
            return ""
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()


multimodal_enricher = MultimodalEnricher()
