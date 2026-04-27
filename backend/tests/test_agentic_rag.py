from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.core.config import settings
from app.services.agentic_bm25 import BM25SparseEncoder
from app.services.agentic_docstore import agentic_chunk_builder, agentic_docstore
from app.services.vector_store import VectorStore


def test_bm25_state_persists_incremental_statistics():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25_state.json"
        encoder = BM25SparseEncoder(path)
        encoder.increment_add_documents(["battery aging model", "battery figure"])
        sparse = encoder.encode("battery")

        reloaded = BM25SparseEncoder(path)

        assert encoder.total_docs == 2
        assert reloaded.total_docs == 2
        assert sparse


def test_agentic_chunk_builder_creates_leaf_only_vector_payloads():
    parents, leaves = agentic_chunk_builder.build(
        "paper",
        [
            {
                "id": "paper-image-2",
                "type": "image",
                "page_number": 2,
                "title": "Fig. 2 | Overview",
                "section": "Method",
                "search_text": "Figure ID: Figure_2. Caption: Fig. 2 | Overview. " * 80,
                "content": "",
                "order": 2,
                "asset_relpath": "parsed/assets/figures/Figure_2.png",
            }
        ],
    )

    assert parents
    assert leaves
    assert all(item["chunk_level"] == 3 for item in leaves)
    assert leaves[0]["parent_chunk_id"]


def test_agentic_retrieve_uses_papernote_docstore_without_milvus():
    original_upload_dir = settings.upload_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings.upload_dir = Path(tmpdir)
            paper_id = "paper"
            parsed_dir = settings.upload_dir / paper_id / "parsed"
            parsed_dir.mkdir(parents=True)
            (parsed_dir / "index_manifest.json").write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "metadata": {"agentic_rag": {"enabled": True}},
                        "blocks": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            parents, leaves = agentic_chunk_builder.build(
                paper_id,
                [
                    {
                        "id": "paper-image-3",
                        "type": "image",
                        "page_number": 3,
                        "title": "Fig. 3 | Other",
                        "section": "Results",
                        "search_text": "Figure ID: Figure_3. Caption: Fig. 3 | Other.",
                        "order": 1,
                    },
                    {
                        "id": "paper-image-2",
                        "type": "image",
                        "page_number": 2,
                        "title": "Fig. 2 | Overview",
                        "section": "Method",
                        "search_text": "Figure ID: Figure_2. Caption: Fig. 2 | Overview.",
                        "order": 2,
                    },
                ],
            )
            agentic_docstore.write(paper_id, parents, leaves)

            result = VectorStore().agentic_retrieve(paper_id, "图2展示了什么？", top_k=2)

            assert result["meta"]["retrieval_mode"] == "lexical_fallback"
            assert result["sources"][0]["block_id"] == "paper-image-2"
    finally:
        settings.upload_dir = original_upload_dir


def test_dashscope_vl_rerank_endpoint_and_payload():
    original_model = settings.rerank_model
    original_host = settings.rerank_binding_host
    try:
        settings.rerank_model = "qwen3-vl-rerank"
        settings.rerank_binding_host = (
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
        store = VectorStore()
        payload = store._rerank_payload(
            "图2展示了什么？",
            [
                {
                    "type": "text",
                    "title": "Fig. 2",
                    "text": "Fig. 2 presents the overview of the proposed framework.",
                }
            ],
            top_k=1,
        )

        assert store._rerank_endpoint() == settings.rerank_binding_host
        assert payload["model"] == "qwen3-vl-rerank"
        assert payload["input"]["query"] == {"text": "图2展示了什么？"}
        assert payload["input"]["documents"] == [
            {"text": "Fig. 2\nFig. 2 presents the overview of the proposed framework."}
        ]
        assert payload["parameters"]["top_n"] == 1
    finally:
        settings.rerank_model = original_model
        settings.rerank_binding_host = original_host


def test_dashscope_vl_rerank_serializes_image_document_as_data_uri():
    original_model = settings.rerank_model
    try:
        settings.rerank_model = "qwen3-vl-rerank"
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "figure.png"
            image_path.write_bytes(b"figure")

            payload = VectorStore()._rerank_payload(
                "图中曲线有什么趋势？",
                [{"type": "image", "asset_path": str(image_path), "text": "fallback"}],
                top_k=1,
            )

        assert payload["input"]["documents"][0]["image"].startswith(
            "data:image/png;base64,"
        )
    finally:
        settings.rerank_model = original_model


def test_empty_agentic_collection_rebuilds_from_local_leaves():
    original_upload_dir = settings.upload_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings.upload_dir = Path(tmpdir)
            paper_id = "paper"
            parsed_dir = settings.upload_dir / paper_id / "parsed"
            parsed_dir.mkdir(parents=True)
            manifest_path = parsed_dir / "index_manifest.json"
            manifest = {
                "paper_id": paper_id,
                "metadata": {"agentic_rag": {"collection": "empty_collection"}},
                "blocks": [],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            agentic_docstore.write(
                paper_id,
                [],
                [{"chunk_id": "leaf-1", "text": "Data availability mentions MATR."}],
            )

            store = VectorStore()
            store._agentic_collection_row_count = lambda _collection: 0
            store._index_agentic_chunks = lambda _paper_id, _leaves: {
                "indexed": True,
                "collection": "rebuilt_collection",
                "dimension": 1024,
            }

            collection = store._ensure_agentic_vectors_available(
                paper_id,
                manifest,
                "empty_collection",
            )

            reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert collection == "rebuilt_collection"
            assert reloaded["metadata"]["agentic_rag"]["collection"] == "rebuilt_collection"
            assert reloaded["metadata"]["agentic_rag"]["indexed"] is True
    finally:
        settings.upload_dir = original_upload_dir


def test_agentic_retrieve_marks_lexical_fallback_when_hybrid_returns_empty():
    original_upload_dir = settings.upload_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings.upload_dir = Path(tmpdir)
            paper_id = "paper"
            parsed_dir = settings.upload_dir / paper_id / "parsed"
            parsed_dir.mkdir(parents=True)
            (parsed_dir / "index_manifest.json").write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "metadata": {"agentic_rag": {"collection": "collection"}},
                        "blocks": [],
                    }
                ),
                encoding="utf-8",
            )

            store = VectorStore()
            store.can_embed = lambda: True
            store._ensure_agentic_vectors_available = (
                lambda _paper_id, _manifest, collection: collection
            )
            store._hybrid_retrieve_agentic = lambda *_args, **_kwargs: []
            store._lexical_retrieve_agentic = lambda *_args, **_kwargs: [
                {
                    "chunk_id": "leaf-1",
                    "block_id": "block-1",
                    "type": "text",
                    "page_number": 1,
                    "title": "Data availability",
                    "section": "Data availability",
                    "text": "All datasets are publicly accessible.",
                    "score": 1.0,
                }
            ]
            store._rerank_agentic_docs = lambda _query, docs, top_k: (
                docs[:top_k],
                {
                    "rerank_enabled": False,
                    "rerank_applied": False,
                    "rerank_model": "",
                    "rerank_endpoint": "",
                    "rerank_error": None,
                    "candidate_count": len(docs),
                },
            )
            store._auto_merge_agentic_docs = lambda _paper_id, docs, top_k: (
                docs[:top_k],
                {
                    "auto_merge_enabled": False,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": 2,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                },
            )

            result = store.agentic_retrieve(paper_id, "这个论文用了什么开源数据集", top_k=1)

            assert result["meta"]["retrieval_mode"] == "lexical_fallback"
            assert result["sources"][0]["title"] == "Data availability"
    finally:
        settings.upload_dir = original_upload_dir


if __name__ == "__main__":
    test_bm25_state_persists_incremental_statistics()
    test_agentic_chunk_builder_creates_leaf_only_vector_payloads()
    test_agentic_retrieve_uses_papernote_docstore_without_milvus()
    test_dashscope_vl_rerank_endpoint_and_payload()
    test_dashscope_vl_rerank_serializes_image_document_as_data_uri()
    test_empty_agentic_collection_rebuilds_from_local_leaves()
    test_agentic_retrieve_marks_lexical_fallback_when_hybrid_returns_empty()
    print(f"{Path(__file__).name}: ok")
