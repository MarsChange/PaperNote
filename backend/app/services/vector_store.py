"""Hybrid retrieval and multimodal indexing for parsed papers."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import httpx
import openai
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from app.core.config import settings
from app.services.agentic_bm25 import bm25_encoder
from app.services.agentic_docstore import agentic_chunk_builder, agentic_docstore
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
QUERY_MAX_LIMIT = 16384
RERANK_IMAGE_MAX_BYTES = 4 * 1024 * 1024

EMBEDDING_PROVIDERS = {
    "openai": ("openai_api_key", "openai_base_url"),
    "qwen": ("qwen_api_key", "qwen_base_url"),
    "kimi": ("kimi_api_key", "kimi_base_url"),
    "gemini": ("gemini_api_key", "gemini_base_url"),
}

FIGURE_REF_RE = re.compile(
    r"(?:fig(?:ure)?\.?|figure|图|圖)\s*[_\-.:：]?\s*0*(\d+[a-zA-Z]?)",
    flags=re.IGNORECASE,
)
SUBFIGURE_QUERY_HINTS = ("subfigure", "panel", "subplot", "子图", "分图", "图中", "图里")
SUBFIGURE_LABEL_RE = re.compile(
    r"(?:^|[\s,，;；(（])([a-h])(?=[\s,，、;；.．:：)）])",
    flags=re.IGNORECASE,
)


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

    def _agentic_collection_name(self, dimension: int) -> str:
        base = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            settings.agentic_collection_prefix,
        ).strip("_")
        return f"{base or 'papernote_agentic_blocks'}_{dimension}"

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

    def _ensure_agentic_collection(self, dimension: int) -> str:
        collection_name = self._agentic_collection_name(dimension)
        if self.client.has_collection(collection_name):
            return collection_name

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="block_id", datatype=DataType.VARCHAR, max_length=160)
        schema.add_field(field_name="block_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="order", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(
            field_name="parent_chunk_id",
            datatype=DataType.VARCHAR,
            max_length=256,
        )
        schema.add_field(field_name="root_chunk_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="chunk_level", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_idx", datatype=DataType.INT64)
        schema.add_field(field_name="dense_embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8000)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1200)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=1200)
        schema.add_field(field_name="asset_relpath", datatype=DataType.VARCHAR, max_length=1200)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
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

        parent_chunks, leaf_chunks = agentic_chunk_builder.build(paper_id, blocks)
        removed_leaf_texts = agentic_docstore.remove_existing_leaf_texts(paper_id)
        if removed_leaf_texts:
            bm25_encoder.increment_remove_documents(removed_leaf_texts)
        agentic_docstore.write(paper_id, parent_chunks, leaf_chunks)
        metadata["agentic_rag"] = {
            "enabled": settings.agentic_rag_enabled,
            "source": "superMew_adapted",
            "chunking": "three_level_sliding_window",
            "leaf_only_vectors": True,
            "parent_docstore": str(agentic_docstore.parents_path(paper_id)),
            "leaf_docstore": str(agentic_docstore.leaves_path(paper_id)),
            "parent_chunk_count": len(parent_chunks),
            "leaf_chunk_count": len(leaf_chunks),
            "bm25_state_path": str(settings.bm25_state_path),
        }
        self._write_manifest(paper_id, blocks, metadata)

        if not blocks:
            logger.warning("No blocks generated for paper %s", paper_id)
            return metadata

        if settings.agentic_rag_enabled and leaf_chunks and self.can_embed():
            agentic_meta = self._index_agentic_chunks(paper_id, leaf_chunks)
            metadata["agentic_rag"].update(agentic_meta)
            self._write_manifest(paper_id, blocks, metadata)

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

    def _index_agentic_chunks(self, paper_id: str, leaf_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [chunk["text"] for chunk in leaf_chunks if chunk.get("text")]
        if not texts:
            return {"indexed": False, "reason": "no_leaf_texts"}

        try:
            self._delete_agentic_vectors(paper_id)
            bm25_encoder.increment_add_documents(texts)
            dense_embeddings = self.embed_texts(texts)
            sparse_embeddings = bm25_encoder.encode_many(texts)
            if not dense_embeddings:
                return {"indexed": False, "reason": "no_dense_embeddings"}

            dimension = len(dense_embeddings[0])
            collection_name = self._ensure_agentic_collection(dimension)
            self.client.insert(
                collection_name=collection_name,
                data=[
                    self._agentic_insert_payload(chunk, dense, sparse)
                    for chunk, dense, sparse in zip(
                        leaf_chunks,
                        dense_embeddings,
                        sparse_embeddings,
                    )
                ],
            )
            return {
                "indexed": True,
                "vector_backend": "milvus_hybrid",
                "collection": collection_name,
                "dimension": dimension,
                "dense_field": "dense_embedding",
                "sparse_field": "sparse_embedding",
                "bm25_total_docs": bm25_encoder.total_docs,
            }
        except Exception as exc:
            logger.error("Agentic hybrid indexing failed for %s: %s", paper_id, exc)
            return {"indexed": False, "error": str(exc)}

    def _agentic_insert_payload(
        self,
        chunk: dict[str, Any],
        dense_embedding: list[float],
        sparse_embedding: dict[int, float],
    ) -> dict[str, Any]:
        return {
            "id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "block_id": chunk.get("block_id", ""),
            "block_type": chunk.get("type", "text"),
            "page_number": int(chunk.get("page_number") or 1),
            "order": int(chunk.get("order") or 0),
            "chunk_id": chunk.get("chunk_id", ""),
            "parent_chunk_id": chunk.get("parent_chunk_id", ""),
            "root_chunk_id": chunk.get("root_chunk_id", ""),
            "chunk_level": int(chunk.get("chunk_level") or 3),
            "chunk_idx": int(chunk.get("chunk_idx") or 0),
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
            "text": str(chunk.get("text", ""))[:8000],
            "title": str(chunk.get("title", ""))[:1200],
            "section": str(chunk.get("section", ""))[:1200],
            "asset_relpath": str(chunk.get("asset_relpath", ""))[:1200],
            "asset_path": chunk.get("asset_path", ""),
            "semantic_summary": chunk.get("semantic_summary", ""),
            "semantic_metadata": json.dumps(
                chunk.get("semantic_metadata", {}),
                ensure_ascii=False,
            )[:8000],
        }

    def _delete_agentic_vectors(self, paper_id: str):
        manifest = self._load_manifest(paper_id)
        collection_name = (
            manifest.get("metadata", {})
            .get("agentic_rag", {})
            .get("collection")
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
            logger.warning("Could not delete agentic vectors for %s: %s", paper_id, exc)

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

    def _normalize_figure_refs(self, raw_ref: str) -> set[str]:
        match = re.match(r"0*(\d+)([a-zA-Z]?)", str(raw_ref).strip())
        if not match:
            return set()
        number = str(int(match.group(1)))
        suffix = match.group(2).lower()
        refs = {number}
        if suffix:
            refs.add(f"{number}{suffix}")
        return refs

    def _extract_figure_refs(self, text: str) -> set[str]:
        refs: set[str] = set()
        for match in FIGURE_REF_RE.finditer(text or ""):
            refs.update(self._normalize_figure_refs(match.group(1)))
        return refs

    def _block_figure_refs(self, block: dict[str, Any]) -> set[str]:
        semantic_metadata = block.get("semantic_metadata") or {}
        if not isinstance(semantic_metadata, dict):
            semantic_metadata = {}
        modality_specific = semantic_metadata.get("modality_specific") or {}
        if not isinstance(modality_specific, dict):
            modality_specific = {}
        entity = semantic_metadata.get("entity") or {}
        if not isinstance(entity, dict):
            entity = {}
        content_header = str(block.get("content", "")).split("Nearby context:", 1)[0]
        search_header = str(block.get("search_text", "")).split("Nearby context:", 1)[0]

        text = " ".join(
            str(value)
            for value in [
                block.get("id", ""),
                block.get("title", ""),
                content_header,
                search_header,
                block.get("asset_relpath", ""),
                modality_specific.get("figure_id", ""),
                entity.get("name", ""),
                entity.get("summary", ""),
            ]
            if value
        )
        return self._extract_figure_refs(text)

    def _extract_subfigure_labels(self, query: str) -> set[str]:
        normalized = self._normalize_text(query).lower()
        if not any(hint in normalized for hint in SUBFIGURE_QUERY_HINTS):
            return set()
        return {match.group(1).lower() for match in SUBFIGURE_LABEL_RE.finditer(normalized)}

    def _block_subfigure_labels(self, block: dict[str, Any]) -> set[str]:
        text = " ".join(
            str(value)
            for value in [
                block.get("title", ""),
                block.get("content", ""),
                block.get("search_text", ""),
                block.get("semantic_summary", ""),
            ]
            if value
        )
        return {match.group(1).lower() for match in SUBFIGURE_LABEL_RE.finditer(text)}

    def _figure_ref_bonus(self, query: str, block: dict[str, Any]) -> float:
        query_refs = self._extract_figure_refs(query)
        if not query_refs:
            return 0.0
        if query_refs.isdisjoint(self._block_figure_refs(block)):
            return 0.0
        return 1.15 if block.get("type") == "image" else 0.35

    def _subfigure_bonus(self, query: str, block: dict[str, Any]) -> float:
        if block.get("type") != "image":
            return 0.0
        query_labels = self._extract_subfigure_labels(query)
        if not query_labels:
            return 0.0
        overlap = query_labels & self._block_subfigure_labels(block)
        return min(0.12 * len(overlap), 0.36)

    def _rerank_endpoint(self) -> str:
        host = settings.rerank_binding_host.strip().rstrip("/")
        if not host:
            return ""
        if "/services/rerank/" in host or host.endswith(("/v1/rerank", "/v1/reranks")):
            return host
        return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"

    def _is_dashscope_vl_rerank(self) -> bool:
        return settings.rerank_model.strip().lower() == "qwen3-vl-rerank"

    def _rerank_doc_text(self, doc: dict[str, Any]) -> str:
        parts = [
            str(doc.get("title") or ""),
            str(doc.get("section") or ""),
            str(doc.get("semantic_summary") or ""),
            str(doc.get("text") or doc.get("content") or doc.get("search_text") or ""),
        ]
        seen: set[str] = set()
        compact: list[str] = []
        for part in parts:
            normalized = part.strip()
            if normalized and normalized not in seen:
                compact.append(normalized)
                seen.add(normalized)
        return "\n".join(compact)[:12000]

    def _image_data_url_for_rerank(self, asset_path: str) -> str:
        if not asset_path:
            return ""
        path = Path(asset_path).expanduser()
        if (
            not path.exists()
            or not path.is_file()
            or path.stat().st_size > RERANK_IMAGE_MAX_BYTES
        ):
            return ""
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _dashscope_vl_rerank_document(self, doc: dict[str, Any]) -> dict[str, str]:
        if (doc.get("type") or doc.get("block_type")) == "image":
            data_url = self._image_data_url_for_rerank(str(doc.get("asset_path") or ""))
            if data_url:
                return {"image": data_url}
        return {"text": self._rerank_doc_text(doc)}

    def _rerank_payload(
        self, query: str, ranked: list[dict[str, Any]], top_k: int
    ) -> dict[str, Any]:
        top_n = min(top_k, len(ranked))
        if self._is_dashscope_vl_rerank():
            return {
                "model": settings.rerank_model,
                "input": {
                    "query": {"text": query},
                    "documents": [
                        self._dashscope_vl_rerank_document(doc) for doc in ranked
                    ],
                },
                "parameters": {
                    "top_n": top_n,
                    "return_documents": False,
                    "instruct": (
                        "Given a scientific paper reading query, retrieve passages, "
                        "figures, or tables that answer the query."
                    ),
                },
            }
        return {
            "model": settings.rerank_model,
            "query": query,
            "documents": [self._rerank_doc_text(doc) for doc in ranked],
            "top_n": top_n,
            "return_documents": False,
        }

    def _rerank_results_from_response(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        results = payload.get("results")
        if isinstance(results, list):
            return results
        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            return output["results"]
        return []

    def _agentic_output_fields(self) -> list[str]:
        return [
            "paper_id",
            "block_id",
            "block_type",
            "page_number",
            "order",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
            "text",
            "title",
            "section",
            "asset_relpath",
            "asset_path",
            "semantic_summary",
            "semantic_metadata",
        ]

    def _hit_entity(self, hit: Any) -> dict[str, Any]:
        entity = hit.get("entity", {}) if hasattr(hit, "get") else {}
        if not isinstance(entity, dict):
            entity = {}
        out = dict(entity)
        for field in self._agentic_output_fields():
            if field not in out and hasattr(hit, "get"):
                value = hit.get(field)
                if value is not None:
                    out[field] = value
        if hasattr(hit, "get"):
            out["score"] = float(hit.get("distance") or hit.get("score") or 0.0)
            out["id"] = hit.get("id") or out.get("chunk_id")
        return out

    def _hybrid_retrieve_agentic(
        self,
        collection_name: str,
        paper_id: str,
        query: str,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        dense_embedding = self.embed_texts([query])[0]
        sparse_embedding = bm25_encoder.encode(query)
        filter_expr = f'paper_id == "{paper_id}" and chunk_level == {settings.leaf_retrieve_level}'
        dense_request = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidate_k * 2,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=candidate_k * 2,
            expr=filter_expr,
        )
        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(k=60),
            limit=candidate_k,
            output_fields=self._agentic_output_fields(),
        )
        return [self._hit_entity(hit) for hits in results for hit in hits]

    def _dense_retrieve_agentic(
        self,
        collection_name: str,
        paper_id: str,
        query: str,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        dense_embedding = self.embed_texts([query])[0]
        results = self.client.search(
            collection_name=collection_name,
            data=[dense_embedding],
            anns_field="dense_embedding",
            search_params={"metric_type": "COSINE", "params": {}},
            filter=f'paper_id == "{paper_id}" and chunk_level == {settings.leaf_retrieve_level}',
            limit=candidate_k,
            output_fields=self._agentic_output_fields(),
        )
        return [self._hit_entity(hit) for hits in results for hit in hits]

    def _lexical_retrieve_agentic(
        self, paper_id: str, query: str, candidate_k: int
    ) -> list[dict[str, Any]]:
        leaves = agentic_docstore.load_leaves(paper_id)
        if not leaves:
            manifest = self._load_manifest(paper_id)
            leaves = manifest.get("blocks", [])
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in leaves:
            text = doc.get("text") or doc.get("search_text") or doc.get("content") or ""
            score = self._lexical_score(query, str(text))
            if doc.get("type") == "image":
                score += self._modality_bias(query, "image")
                score += self._figure_ref_bonus(query, doc)
                score += self._subfigure_bonus(query, doc)
            if score > 0:
                scored.append((score, {**doc, "score": score}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:candidate_k]]

    def _rerank_agentic_docs(
        self, query: str, docs: list[dict[str, Any]], top_k: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        meta = {
            "rerank_enabled": bool(
                settings.rerank_model
                and settings.rerank_binding_host
                and settings.rerank_api_key
            ),
            "rerank_applied": False,
            "rerank_model": settings.rerank_model,
            "rerank_endpoint": self._rerank_endpoint(),
            "rerank_error": None,
            "candidate_count": len(docs),
        }
        ranked = [{**doc, "rrf_rank": index} for index, doc in enumerate(docs, 1)]
        if not ranked or not meta["rerank_enabled"]:
            return ranked[:top_k], meta

        payload = self._rerank_payload(query, ranked, top_k)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.rerank_api_key}",
        }
        try:
            meta["rerank_applied"] = True
            with httpx.Client(timeout=15) as client:
                response = client.post(meta["rerank_endpoint"], headers=headers, json=payload)
            if response.status_code >= 400:
                meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
                return ranked[:top_k], meta
            reranked: list[dict[str, Any]] = []
            for item in self._rerank_results_from_response(response.json()):
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(ranked):
                    doc = dict(ranked[index])
                    if item.get("relevance_score") is not None:
                        doc["rerank_score"] = item["relevance_score"]
                    reranked.append(doc)
            return (reranked or ranked)[:top_k], meta
        except Exception as exc:
            meta["rerank_error"] = str(exc)
            return ranked[:top_k], meta

    def _merge_to_parent_level(
        self,
        paper_id: str,
        docs: list[dict[str, Any]],
        threshold: int,
    ) -> tuple[list[dict[str, Any]], int]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for doc in docs:
            parent_id = str(doc.get("parent_chunk_id") or "").strip()
            if parent_id:
                groups.setdefault(parent_id, []).append(doc)
        merge_parent_ids = [
            parent_id for parent_id, children in groups.items() if len(children) >= threshold
        ]
        if not merge_parent_ids:
            return docs, 0

        parent_docs = agentic_docstore.get_parents_by_ids(paper_id, merge_parent_ids)
        parent_map = {
            doc.get("chunk_id"): doc
            for doc in parent_docs
            if doc.get("chunk_id")
        }
        merged: list[dict[str, Any]] = []
        replaced = 0
        for doc in docs:
            parent_id = str(doc.get("parent_chunk_id") or "").strip()
            parent_doc = parent_map.get(parent_id)
            if not parent_doc:
                merged.append(doc)
                continue
            merged_doc = dict(parent_doc)
            merged_doc["score"] = max(
                float(parent_doc.get("score") or 0.0),
                float(doc.get("score") or 0.0),
            )
            merged_doc["merged_from_children"] = True
            merged_doc["merged_child_count"] = len(groups[parent_id])
            merged.append(merged_doc)
            replaced += 1

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in merged:
            key = item.get("chunk_id") or item.get("id") or item.get("text")
            if key in seen:
                continue
            seen.add(str(key))
            deduped.append(item)
        return deduped, replaced

    def _auto_merge_agentic_docs(
        self,
        paper_id: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        meta = {
            "auto_merge_enabled": settings.auto_merge_enabled,
            "auto_merge_applied": False,
            "auto_merge_threshold": settings.auto_merge_threshold,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }
        if not settings.auto_merge_enabled or not docs:
            return docs[:top_k], meta

        merged, replaced_l3_l2 = self._merge_to_parent_level(
            paper_id,
            docs,
            settings.auto_merge_threshold,
        )
        merged, replaced_l2_l1 = self._merge_to_parent_level(
            paper_id,
            merged,
            settings.auto_merge_threshold,
        )
        merged.sort(key=lambda item: float(item.get("rerank_score") or item.get("score") or 0.0), reverse=True)
        replaced = replaced_l3_l2 + replaced_l2_l1
        meta.update(
            {
                "auto_merge_applied": replaced > 0,
                "auto_merge_replaced_chunks": replaced,
                "auto_merge_steps": int(replaced_l3_l2 > 0) + int(replaced_l2_l1 > 0),
            }
        )
        return merged[:top_k], meta

    def agentic_retrieve(
        self,
        paper_id: str,
        query: str,
        top_k: int = 6,
    ) -> dict[str, Any]:
        candidate_k = max(top_k * max(settings.agentic_candidate_multiplier, 1), top_k)
        manifest = self._load_manifest(paper_id)
        agentic_meta = manifest.get("metadata", {}).get("agentic_rag", {})
        collection_name = agentic_meta.get("collection")
        retrieval_mode = "lexical_fallback"
        retrieval_error = None
        docs: list[dict[str, Any]] = []

        if collection_name and self.can_embed():
            try:
                docs = self._hybrid_retrieve_agentic(collection_name, paper_id, query, candidate_k)
                retrieval_mode = "hybrid"
            except Exception as exc:
                retrieval_error = str(exc)
                try:
                    docs = self._dense_retrieve_agentic(collection_name, paper_id, query, candidate_k)
                    retrieval_mode = "dense_fallback"
                except Exception as dense_exc:
                    retrieval_error = f"{retrieval_error}; dense_fallback={dense_exc}"
                    docs = []

        if not docs:
            docs = self._lexical_retrieve_agentic(paper_id, query, candidate_k)

        reranked, rerank_meta = self._rerank_agentic_docs(query, docs, top_k=top_k)
        merged, merge_meta = self._auto_merge_agentic_docs(paper_id, reranked, top_k=top_k)
        sources = [self._serialize_agentic_doc(paper_id, doc, index) for index, doc in enumerate(merged, 1)]
        graph_context = knowledge_graph_indexer.context_for_blocks(
            paper_id,
            [source.get("block_id") or source["id"] for source in sources],
        )
        for source in sources:
            source["graph_context"] = graph_context.get(
                source.get("block_id") or source["id"],
                "",
            )

        meta = {
            **rerank_meta,
            **merge_meta,
            "retrieval_mode": retrieval_mode,
            "retrieval_error": retrieval_error,
            "candidate_k": candidate_k,
            "leaf_retrieve_level": settings.leaf_retrieve_level,
            "source": "agentic_rag_supermew_adapted",
            "hybrid_search": retrieval_mode == "hybrid",
            "dense_sparse_rrf": retrieval_mode == "hybrid",
        }
        return {"sources": sources, "docs": merged, "meta": meta}

    def _serialize_agentic_doc(
        self,
        paper_id: str,
        doc: dict[str, Any],
        rank: int,
    ) -> dict[str, Any]:
        block_id = doc.get("block_id") or doc.get("id") or doc.get("chunk_id")
        semantic_metadata = doc.get("semantic_metadata", {})
        if isinstance(semantic_metadata, str):
            try:
                semantic_metadata = json.loads(semantic_metadata)
            except Exception:
                semantic_metadata = {}
        asset_url = (
            f"/api/papers/{paper_id}/assets/{doc['asset_relpath']}"
            if doc.get("asset_relpath")
            else ""
        )
        score = doc.get("rerank_score")
        if score is None:
            score = doc.get("score", 0.0)
        return {
            "id": doc.get("chunk_id") or block_id,
            "block_id": block_id,
            "chunk_id": doc.get("chunk_id", ""),
            "parent_chunk_id": doc.get("parent_chunk_id", ""),
            "root_chunk_id": doc.get("root_chunk_id", ""),
            "chunk_level": int(doc.get("chunk_level") or 0),
            "type": doc.get("type") or doc.get("block_type") or "text",
            "page_number": int(doc.get("page_number") or 1),
            "title": doc.get("title") or doc.get("section") or f"Page {doc.get('page_number', 1)}",
            "section": doc.get("section", ""),
            "content": doc.get("text") or doc.get("content") or "",
            "score": round(float(score or 0.0), 4),
            "rerank_score": doc.get("rerank_score"),
            "rrf_rank": doc.get("rrf_rank", rank),
            "asset_url": asset_url,
            "asset_relpath": doc.get("asset_relpath", ""),
            "asset_path": doc.get("asset_path", ""),
            "semantic_metadata": semantic_metadata,
            "semantic_summary": doc.get("semantic_summary", ""),
            "merged_from_children": bool(doc.get("merged_from_children")),
            "merged_child_count": int(doc.get("merged_child_count") or 0),
        }

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
            "image": [
                "figure",
                "fig.",
                "image",
                "diagram",
                "visual",
                "subfigure",
                "panel",
                "图片",
                "插图",
                "图像",
                "图中",
                "子图",
                "图",
                "示意图",
            ],
            "table": ["table", "tabular", "表格", "表", "数据"],
            "equation": ["equation", "formula", "latex", "公式", "方程"],
        }
        for modality, keywords in hints.items():
            if any(keyword in query_lower for keyword in keywords):
                if block_type != modality:
                    return 0.0
                return 0.32 if modality == "image" else 0.18
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
        if settings.agentic_rag_enabled:
            agentic = self.agentic_retrieve(paper_id, query, top_k=top_k)
            sources = agentic.get("sources", [])
            if sources:
                return sources

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
            figure_ref_bonus = self._figure_ref_bonus(query, block)
            subfigure_bonus = self._subfigure_bonus(query, block)
            order_bonus = max(
                0.08 - (block.get("order", 0) / max(len(blocks), 1)) * 0.04, 0
            )
            score = (
                (vector_score * 0.68)
                + (lexical_score * 0.28)
                + modality_bias
                + figure_ref_bonus
                + subfigure_bonus
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
        removed_leaf_texts = agentic_docstore.remove_existing_leaf_texts(paper_id)
        if removed_leaf_texts:
            bm25_encoder.increment_remove_documents(removed_leaf_texts)
        self._delete_agentic_vectors(paper_id)
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
