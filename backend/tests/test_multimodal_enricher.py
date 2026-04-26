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


if __name__ == "__main__":
    test_heuristic_multimodal_enrichment_without_llm_key()
    print(f"{Path(__file__).name}: ok")
