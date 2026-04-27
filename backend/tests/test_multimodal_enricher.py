from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.services.mineru import ParseResult
from app.services.multimodal_enricher import MultimodalEnricher


def test_heuristic_multimodal_enrichment_without_llm_key():
    original_limit = settings.multimodal_enrichment_llm_limit
    original_enabled = settings.enable_multimodal_enrichment
    settings.multimodal_enrichment_llm_limit = 0
    settings.enable_multimodal_enrichment = True
    try:
        content_list = [
            {
                "type": "text",
                "text": "The proposed iMOE model predicts battery degradation from capacity voltage curves.",
                "text_level": 0,
                "page_idx": 0,
            },
            {
                "type": "image",
                "img_path": "",
                "image_caption": ["Figure 1: iMOE architecture"],
                "image_footnote": [],
                "page_idx": 0,
            },
            {
                "type": "table",
                "table_body": "| Method | MAE |\n|---|---|\n| iMOE | 0.03 |",
                "table_caption": ["Table 1: prediction performance"],
                "table_footnote": [],
                "page_idx": 0,
            },
            {
                "type": "equation",
                "latex": "SOH_t = f(x_t)",
                "text": "State of health prediction.",
                "page_idx": 0,
            },
        ]
        parsed = ParseResult(
            markdown_path="",
            markdown_content="",
            content_list_path="",
            content_list=content_list,
            output_dir="",
            assets_dir="",
            page_count=1,
        )
        blocks = [
            {
                "id": "paper-image-1",
                "type": "image",
                "page_number": 1,
                "section": "Method",
                "title": "Figure 1: iMOE architecture",
                "content": "Caption: Figure 1: iMOE architecture",
                "search_text": "Caption: Figure 1: iMOE architecture",
                "asset_path": "",
                "asset_relpath": "",
                "order": 1,
                "source_index": 1,
            },
            {
                "id": "paper-table-2",
                "type": "table",
                "page_number": 1,
                "section": "Results",
                "title": "Table 1: prediction performance",
                "content": "Table body: | Method | MAE |",
                "search_text": "Table body: | Method | MAE |",
                "asset_path": "",
                "asset_relpath": "",
                "order": 2,
                "source_index": 2,
            },
            {
                "id": "paper-equation-3",
                "type": "equation",
                "page_number": 1,
                "section": "Method",
                "title": "Page 1",
                "content": "LaTeX: SOH_t = f(x_t)",
                "search_text": "LaTeX: SOH_t = f(x_t)",
                "asset_path": "",
                "asset_relpath": "",
                "order": 3,
                "source_index": 3,
            },
        ]

        enriched_blocks, metadata = MultimodalEnricher().enrich_blocks(
            "paper", parsed, blocks
        )

        assert metadata["enabled"] is True
        assert metadata["stats"]["total"] == 3
        assert metadata["stats"]["llm"] == 0

        for block in enriched_blocks:
            semantic = block["semantic_metadata"]
            assert semantic["source"] == "heuristic"
            assert semantic["summary"]
            assert semantic["entity"]["name"]
            assert block["semantic_summary"] in block["search_text"]

        table_semantic = enriched_blocks[1]["semantic_metadata"]
        assert table_semantic["modality_specific"]["columns"] == ["Method", "MAE"]

        equation_semantic = enriched_blocks[2]["semantic_metadata"]
        assert "SOH_t" in equation_semantic["modality_specific"]["latex"]
    finally:
        settings.multimodal_enrichment_llm_limit = original_limit
        settings.enable_multimodal_enrichment = original_enabled


def test_llm_payload_normalizes_list_entity():
    enricher = MultimodalEnricher()
    block = {
        "id": "paper-image-28",
        "type": "image",
        "title": "Figure 2: degradation workflow",
    }
    payload = {
        "summary": "Figure 2 summarizes the degradation prediction workflow.",
        "detailed_description": "The figure links raw curves, feature extraction, and model output.",
        "entity": [
            {
                "name": "Figure 2",
                "type": "figure",
                "summary": "A workflow diagram for degradation prediction.",
            }
        ],
        "keywords": ["degradation", {"term": "workflow"}],
        "claims": [{"claim": "The model uses curve-derived features."}],
        "relations": [{"type": "shows", "target": "feature extraction"}],
    }

    metadata = enricher._normalize_llm_payload(payload, block)

    assert metadata["source"] == "llm"
    assert metadata["entity"]["name"] == "Figure 2"
    assert metadata["entity"]["type"] == "figure"
    assert metadata["keywords"] == ["degradation", "workflow"]
    assert metadata["claims"] == ["The model uses curve-derived features."]
    assert metadata["relations"][0]["target"] == "feature extraction"


def test_semantic_text_accepts_legacy_list_entity():
    enricher = MultimodalEnricher()
    semantic_text = enricher.semantic_text(
        {
            "summary": "A semantic summary.",
            "detailed_description": "",
            "entity": [{"name": "Figure 3"}],
            "keywords": ["battery"],
            "claims": [],
        }
    )

    assert "Figure 3" in semantic_text
    assert "Keywords: battery" in semantic_text


def test_llm_budget_prioritizes_images_before_tables_and_equations():
    blocks = [
        {"id": "table-1", "type": "table", "order": 1},
        {"id": "equation-1", "type": "equation", "order": 2},
        {"id": "image-1", "type": "image", "order": 3},
        {"id": "image-2", "type": "image", "order": 4},
    ]

    selected = MultimodalEnricher()._llm_priority_block_ids(blocks, llm_budget=2)

    assert selected == {"image-1", "image-2"}


if __name__ == "__main__":
    test_heuristic_multimodal_enrichment_without_llm_key()
    test_llm_payload_normalizes_list_entity()
    test_semantic_text_accepts_legacy_list_entity()
    test_llm_budget_prioritizes_images_before_tables_and_equations()
    print(f"{Path(__file__).name}: ok")
