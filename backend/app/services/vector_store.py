"""Hybrid retrieval and multimodal indexing for parsed papers."""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import openai
from pymilvus import DataType, MilvusClient

from app.core.config import settings
from app.services.knowledge_graph import knowledge_graph_indexer
from app.services.mineru import ParseResult
from app.services.multimodal_enricher import multimodal_enricher

logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

BLOCK_CHAR_LIMIT = 1200
BLOCK_CHAR_OVERLAP = 180

EMBEDDING_PROVIDERS = {
    "openai": ("openai_api_key", "openai_base_url"),
    "qwen": ("qwen_api_key", "qwen_base_url"),
    "kimi": ("kimi_api_key", "kimi_base_url"),
    "gemini": ("gemini_api_key", "gemini_base_url"),
}


class VectorStore:
    """Manages multimodal paper indexing and hybrid retrieval."""

    def __init__(self):
        self._client: Optional[MilvusClient] = None
        self._embedding_client: Optional[openai.OpenAI] = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            uri = settings.milvus_uri
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", uri):
                Path(uri).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(
                uri=uri,
                grpc_options={
                    "grpc.keepalive_time_ms": settings.milvus_grpc_keepalive_time_ms,
                    "grpc.keepalive_timeout_ms": settings.milvus_grpc_keepalive_timeout_ms,
                    "grpc.keepalive_permit_without_calls": (
                        settings.milvus_grpc_keepalive_permit_without_calls
                    ),
                },
            )
        return self._client

    def _embedding_credentials(self) -> Optional[tuple[str, str]]:
        provider = settings.embedding_provider.lower().strip()
        provider_fields = EMBEDDING_PROVIDERS.get(provider)
        if not provider_fields:
            logger.warning("Unsupported embedding provider: %s", provider)
            return None

        key_attr, base_url_attr = provider_fields
        api_key = settings.embedding_api_key or str(getattr(settings, key_attr, ""))
        base_url = settings.embedding_base_url or str(getattr(settings, base_url_attr, ""))
        if not api_key:
            return None
        return api_key, base_url

    @property
    def embedding_client(self) -> Optional[openai.OpenAI]:
        if self._embedding_client is not None:
            return self._embedding_client

        credentials = self._embedding_credentials()
        if not credentials:
            return None

        api_key, base_url = credentials
        self._embedding_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._embedding_client

    def _collection_name(self, dimension: int) -> str:
        base = re.sub(r"[^a-zA-Z0-9_]", "_", settings.milvus_collection).strip("_")
        return f"{base or 'papernote_blocks'}_{dimension}"

    def _ensure_collection(self, dimension: int) -> str:
        collection_name = self._collection_name(dimension)
        if self.client.has_collection(collection_name):
            return collection_name

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="block_type", datatype=DataType.VARCHAR, max_length=32
        )
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="order", datatype=DataType.INT64)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        return collection_name

    def _manifest_path(self, paper_id: str) -> Path:
        return settings.upload_dir / paper_id / "parsed" / "index_manifest.json"

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _chunk_text(self, text: str) -> list[str]:
        clean_text = self._normalize_text(text)
        if len(clean_text) <= BLOCK_CHAR_LIMIT:
            return [clean_text] if clean_text else []

        chunks: list[str] = []
        start = 0
        text_length = len(clean_text)
        while start < text_length:
            end = min(text_length, start + BLOCK_CHAR_LIMIT)
            if end < text_length:
                boundary = clean_text.rfind(" ", start + BLOCK_CHAR_LIMIT // 2, end)
                if boundary > start:
                    end = boundary
            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(end - BLOCK_CHAR_OVERLAP, 0)
        return chunks

    def _extract_context(
        self, content_list: list[dict[str, Any]], index: int, window: int = 1
    ) -> str:
        texts: list[str] = []
        current_page = int(content_list[index].get("page_idx", 0))
        start = max(index - window, 0)
        end = min(index + window + 1, len(content_list))

        for pos in range(start, end):
            if pos == index:
                continue
            item = content_list[pos]
            page_idx = int(item.get("page_idx", 0))
            if abs(page_idx - current_page) > window:
                continue
            if item.get("type") != "text":
                continue
            text = self._normalize_text(str(item.get("text", "")))
            if text:
                texts.append(text)

        return "\n".join(texts[:4]).strip()

    def _resolve_section_title(
        self, item: dict[str, Any], current_section: str, page_number: int
    ) -> str:
        item_type = item.get("type", "text")
        if item_type == "text":
            text = self._normalize_text(str(item.get("text", "")))
            if int(item.get("text_level") or 0) > 0 and text:
                return text
        captions = (
            item.get("image_caption")
            or item.get("table_caption")
            or item.get("equation_caption")
            or []
        )
        if isinstance(captions, list) and captions:
            return self._normalize_text(str(captions[0]))
        if current_section:
            return current_section
        return f"Page {page_number}"

    def _compose_modal_content(
        self, item: dict[str, Any], context_text: str, section: str, page_number: int
    ) -> str:
        item_type = item.get("type", "generic")
        parts = [f"Section: {section}", f"Page: {page_number}", f"Modality: {item_type}"]

        if item_type == "image":
            caption = ", ".join(item.get("image_caption", []) or [])
            footnote = ", ".join(item.get("image_footnote", []) or [])
            figure_id = self._normalize_text(str(item.get("figure_id", "")))
            figure_context = self._normalize_text(str(item.get("figure_context", "")))
            subfigure_count = item.get("subfigure_count")
            if figure_id:
                parts.append(f"Figure ID: {figure_id}")
            if caption:
                parts.append(f"Caption: {caption}")
            if footnote:
                parts.append(f"Footnote: {footnote}")
            if subfigure_count:
                parts.append(f"Composite crop merged {subfigure_count} visual regions")
            if figure_context:
                parts.append(f"Figure reference context: {figure_context}")
        elif item_type == "table":
            caption = ", ".join(item.get("table_caption", []) or [])
            footnote = ", ".join(item.get("table_footnote", []) or [])
            table_body = self._normalize_text(str(item.get("table_body", "")))
            if caption:
                parts.append(f"Caption: {caption}")
            if table_body:
                parts.append(f"Table body: {table_body}")
            if footnote:
                parts.append(f"Footnote: {footnote}")
        elif item_type == "equation":
            equation_caption = ", ".join(item.get("equation_caption", []) or [])
            latex = self._normalize_text(str(item.get("latex", "")))
            description = self._normalize_text(str(item.get("text", "")))
            if equation_caption:
                parts.append(f"Caption: {equation_caption}")
            if latex:
                parts.append(f"LaTeX: {latex}")
            if description:
                parts.append(f"Description: {description}")
        else:
            raw_text = self._normalize_text(json.dumps(item, ensure_ascii=False))
            if raw_text:
                parts.append(raw_text)

        if context_text:
            parts.append(f"Nearby context: {context_text}")
        return "\n".join(parts).strip()

    def _build_blocks(self, paper_id: str, parsed: ParseResult) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        stats = {"text": 0, "image": 0, "table": 0, "equation": 0, "generic": 0}
        current_section = ""
        order = 0
        paper_root = settings.upload_dir / paper_id

        for index, item in enumerate(parsed.content_list):
            item_type = str(item.get("type", "generic")).lower()
            page_number = int(item.get("page_idx", 0)) + 1
            title = self._resolve_section_title(item, current_section, page_number)

            if item_type == "text":
                text = self._normalize_text(str(item.get("text", "")))
                text_level = int(item.get("text_level") or 0)
                if text_level > 0 and text:
                    current_section = text
                    title = text

                for chunk_index, chunk in enumerate(self._chunk_text(text)):
                    block_id = f"{paper_id}-text-{index}-{chunk_index}"
                    searchable_text = "\n".join(
                        filter(None, [f"Section: {current_section}", chunk])
                    ).strip()
                    blocks.append(
                        {
                            "id": block_id,
                            "paper_id": paper_id,
                            "type": "text",
                            "page_number": page_number,
                            "section": current_section,
                            "title": title,
                            "content": chunk,
                            "search_text": searchable_text,
                            "asset_path": "",
                            "asset_relpath": "",
                            "order": order,
                            "source_index": index,
                            "source_chunk_index": chunk_index,
                        }
                    )
                    order += 1
                    stats["text"] += 1
                continue

            context_text = self._extract_context(parsed.content_list, index)
            content = self._compose_modal_content(
                item, context_text=context_text, section=current_section or title, page_number=page_number
            )
            asset_path = (
                str(item.get("img_path", ""))
                or str(item.get("table_img_path", ""))
                or str(item.get("equation_img_path", ""))
            )
            asset_relpath = ""
            if asset_path:
                resolved_asset = Path(asset_path)
                try:
                    asset_relpath = str(resolved_asset.relative_to(paper_root))
                except ValueError:
                    asset_relpath = ""

            block_id = f"{paper_id}-{item_type}-{index}"
            blocks.append(
                {
                    "id": block_id,
                    "paper_id": paper_id,
                    "type": item_type if item_type in stats else "generic",
                    "page_number": page_number,
                    "section": current_section,
                    "title": title,
                    "content": content,
                    "search_text": content,
                    "asset_path": asset_path,
                    "asset_relpath": asset_relpath,
                    "order": order,
                    "source_index": index,
                }
            )
            order += 1
            stats[item_type if item_type in stats else "generic"] += 1

        preview_blocks = [block["content"] for block in blocks[:6] if block["content"]]
        metadata = {
            "page_count": parsed.page_count,
            "stats": stats,
            "summary_preview": " ".join(preview_blocks)[:900],
        }
        return blocks, metadata

    def _write_manifest(
        self, paper_id: str, blocks: list[dict[str, Any]], metadata: dict[str, Any]
    ):
        manifest_path = self._manifest_path(paper_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "paper_id": paper_id,
            "metadata": metadata,
            "blocks": blocks,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self, paper_id: str) -> dict[str, Any]:
        manifest_path = self._manifest_path(paper_id)
        if not manifest_path.exists():
            return {"paper_id": paper_id, "metadata": {}, "blocks": []}

        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read index manifest for %s: %s", paper_id, exc)
            return {"paper_id": paper_id, "metadata": {}, "blocks": []}

    def can_embed(self) -> bool:
        return self.embedding_client is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self.embedding_client
        if not client or not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 6
        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            resp = client.embeddings.create(model=settings.embedding_model, input=batch)
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    def index_paper(self, paper_id: str, parsed: ParseResult) -> dict[str, Any]:
        blocks, metadata = self._build_blocks(paper_id, parsed)
        blocks, enrichment_metadata = multimodal_enricher.enrich_blocks(
            paper_id, parsed, blocks
        )
        metadata["semantic_enrichment"] = enrichment_metadata
        graph_metadata = knowledge_graph_indexer.index_paper(
            paper_id, parsed, blocks, metadata
        )
        metadata["knowledge_graph"] = graph_metadata
        self._write_manifest(paper_id, blocks, metadata)

        if not blocks:
            logger.warning("No blocks generated for paper %s", paper_id)
            return metadata

        if not self.can_embed():
            logger.warning(
                "Embedding credentials are not configured. Falling back to lexical retrieval for %s",
                paper_id,
            )
            return metadata

        try:
            self.delete_paper(paper_id)
            search_texts = [block["search_text"] for block in blocks]
            embeddings = self.embed_texts(search_texts)
            if not embeddings:
                return metadata
            dimension = len(embeddings[0])
            collection_name = self._ensure_collection(dimension)
            metadata["vector_store"] = {
                "backend": "milvus",
                "uri": settings.milvus_uri,
                "collection": collection_name,
                "dimension": dimension,
            }
            self._write_manifest(paper_id, blocks, metadata)

            self.client.insert(
                collection_name=collection_name,
                data=[
                    {
                        "id": block["id"],
                        "paper_id": paper_id,
                        "block_type": block["type"],
                        "page_number": block["page_number"],
                        "order": block["order"],
                        "embedding": embedding,
                        "search_text": block["search_text"][:8000],
                        "section": block["section"],
                        "title": block["title"],
                        "content": block["content"][:4000],
                        "asset_relpath": block["asset_relpath"],
                        "semantic_summary": block.get("semantic_summary", "")[:1200],
                        "semantic_metadata": json.dumps(
                            block.get("semantic_metadata", {}),
                            ensure_ascii=False,
                        )[:8000],
                    }
                    for block, embedding in zip(blocks, embeddings)
                ],
            )
        except Exception as exc:
            logger.error("Vector indexing failed for %s: %s", paper_id, exc)

        return metadata

    def _tokenize_query(self, text: str) -> list[str]:
        normalized = self._normalize_text(text).lower()
        latin_terms = re.findall(r"[a-z0-9][a-z0-9\-_]{1,}", normalized)
        cjk_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
        cjk_bigrams: list[str] = []
        for phrase in cjk_phrases:
            if len(phrase) == 2:
                cjk_bigrams.append(phrase)
            else:
                cjk_bigrams.extend(phrase[i : i + 2] for i in range(len(phrase) - 1))
        return list(dict.fromkeys(latin_terms + cjk_phrases + cjk_bigrams))

    def _lexical_score(self, query: str, search_text: str) -> float:
        normalized_query = self._normalize_text(query).lower()
        normalized_block = self._normalize_text(search_text).lower()
        if not normalized_query or not normalized_block:
            return 0.0

        query_terms = self._tokenize_query(normalized_query)
        if not query_terms:
            return 1.0 if normalized_query in normalized_block else 0.0

        overlap = 0
        for term in query_terms:
            if term in normalized_block:
                overlap += 1

        coverage = overlap / max(len(query_terms), 1)
        phrase_bonus = 0.2 if normalized_query in normalized_block else 0.0
        return min(coverage + phrase_bonus, 1.25)

    def _modality_bias(self, query: str, block_type: str) -> float:
        query_lower = query.lower()
        hints = {
            "image": ["figure", "image", "diagram", "visual", "图片", "图", "示意图"],
            "table": ["table", "tabular", "表格", "表", "数据"],
            "equation": ["equation", "formula", "latex", "公式", "方程"],
        }
        for modality, keywords in hints.items():
            if any(keyword in query_lower for keyword in keywords):
                return 0.18 if block_type == modality else 0.0
        return 0.0

    def _vector_scores(self, paper_id: str, query: str) -> dict[str, float]:
        if not self.can_embed():
            return {}

        try:
            manifest = self._load_manifest(paper_id)
            vector_meta = manifest.get("metadata", {}).get("vector_store", {})
            collection_name = vector_meta.get("collection")
            if not collection_name or not self.client.has_collection(collection_name):
                return {}

            query_embeddings = self.embed_texts([query])
            if not query_embeddings:
                return {}
            results = self.client.search(
                collection_name=collection_name,
                data=query_embeddings,
                anns_field="embedding",
                filter=f'paper_id == "{paper_id}"',
                limit=10,
                output_fields=["id"],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:
            logger.warning("Vector query failed for %s: %s", paper_id, exc)
            return {}

        scores: dict[str, float] = {}
        for hit in results[0] if results else []:
            block_id = hit.get("id") or hit.get("entity", {}).get("id")
            if not block_id:
                continue
            scores[block_id] = float(hit.get("distance") or hit.get("score") or 0.0)
        return scores

    def _serialize_result(self, paper_id: str, block: dict[str, Any], score: float) -> dict[str, Any]:
        asset_url = (
            f"/api/papers/{paper_id}/assets/{block['asset_relpath']}"
            if block.get("asset_relpath")
            else ""
        )
        excerpt = block.get("content", "")
        return {
            "id": block["id"],
            "type": block.get("type", "text"),
            "page_number": block.get("page_number", 1),
            "title": block.get("title") or block.get("section") or f"Page {block.get('page_number', 1)}",
            "section": block.get("section", ""),
            "content": excerpt,
            "score": round(score, 4),
            "asset_url": asset_url,
            "asset_relpath": block.get("asset_relpath", ""),
            "asset_path": block.get("asset_path", ""),
            "semantic_metadata": block.get("semantic_metadata", {}),
            "semantic_summary": block.get("semantic_summary", ""),
        }

    def search(self, paper_id: str, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        manifest = self._load_manifest(paper_id)
        blocks = manifest.get("blocks", [])
        if not blocks:
            return []

        vector_scores = self._vector_scores(paper_id, query)
        scored_results: list[tuple[float, dict[str, Any]]] = []
        for block in blocks:
            lexical_score = self._lexical_score(query, block.get("search_text", ""))
            vector_score = vector_scores.get(block["id"], 0.0)
            modality_bias = self._modality_bias(query, block.get("type", "text"))
            order_bonus = max(
                0.08 - (block.get("order", 0) / max(len(blocks), 1)) * 0.04, 0
            )
            score = (
                (vector_score * 0.68)
                + (lexical_score * 0.28)
                + modality_bias
                + order_bonus
            )
            if score > 0:
                scored_results.append((score, block))

        base_scores = {block["id"]: score for score, block in scored_results}
        graph_scores = knowledge_graph_indexer.expand_scores(paper_id, base_scores, query)
        if graph_scores:
            scored_results = [
                (score + graph_scores.get(block["id"], 0.0), block)
                for score, block in scored_results
            ]

        scored_results.sort(key=lambda item: item[0], reverse=True)

        unique_results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for score, block in scored_results:
            if block["id"] in seen_ids:
                continue
            seen_ids.add(block["id"])
            unique_results.append(self._serialize_result(paper_id, block, score))
            if len(unique_results) >= top_k:
                break

        graph_context = knowledge_graph_indexer.context_for_blocks(
            paper_id, [item["id"] for item in unique_results]
        )
        for item in unique_results:
            item["graph_context"] = graph_context.get(item["id"], "")

        return unique_results

    def get_overview_context(self, paper_id: str, max_items: int = 10) -> list[dict[str, Any]]:
        manifest = self._load_manifest(paper_id)
        blocks = manifest.get("blocks", [])
        if not blocks:
            return []

        ordered_blocks = sorted(blocks, key=lambda block: block.get("order", 0))
        selected: list[dict[str, Any]] = []

        text_blocks = [block for block in ordered_blocks if block.get("type") == "text"][
            :max_items
        ]
        modal_blocks = [block for block in ordered_blocks if block.get("type") != "text"][
            : max(2, max_items // 3)
        ]

        for block in text_blocks + modal_blocks:
            if block not in selected:
                selected.append(block)
            if len(selected) >= max_items:
                break

        results = [
            self._serialize_result(paper_id, block, score=1 - (idx * 0.05))
            for idx, block in enumerate(selected)
        ]
        graph_context = knowledge_graph_indexer.context_for_blocks(
            paper_id, [item["id"] for item in results]
        )
        for item in results:
            item["graph_context"] = graph_context.get(item["id"], "")
        return results

    def render_sources_for_prompt(self, sources: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for index, source in enumerate(sources, start=1):
            title = source.get("title") or source.get("section") or "Untitled"
            section = source.get("section") or "General"
            page_number = source.get("page_number") or 1
            modality = source.get("type") or "text"
            content = self._normalize_text(str(source.get("content", "")))
            semantic_summary = self._normalize_text(
                str(source.get("semantic_summary", ""))
            )
            graph_context = self._normalize_text(str(source.get("graph_context", "")))
            if semantic_summary:
                content = f"{content}\nSemantic summary: {semantic_summary}"
            if graph_context:
                content = f"{content}\nKG context: {graph_context}"
            parts.append(
                f"[S{index}] page {page_number} | {modality} | {title} | section: {section}\n{content}"
            )
        return "\n\n".join(parts).strip()

    def delete_paper(self, paper_id: str):
        manifest = self._load_manifest(paper_id)
        collection_name = (
            manifest.get("metadata", {}).get("vector_store", {}).get("collection")
        )
        if not collection_name:
            return
        try:
            if self.client.has_collection(collection_name):
                self.client.delete(
                    collection_name=collection_name,
                    filter=f'paper_id == "{paper_id}"',
                )
        except Exception as exc:
            logger.warning("Could not delete Milvus vectors for %s: %s", paper_id, exc)


vector_store = VectorStore()
