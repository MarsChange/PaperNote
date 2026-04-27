from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.core.config import settings
from app.services.vector_store import VectorStore


def test_chinese_figure_reference_prioritizes_matching_image():
    original_upload_dir = settings.upload_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings.upload_dir = Path(tmpdir)
            paper_id = "paper"
            parsed_dir = settings.upload_dir / paper_id / "parsed"
            parsed_dir.mkdir(parents=True)
            manifest = {
                "paper_id": paper_id,
                "metadata": {},
                "blocks": [
                    {
                        "id": "text-1",
                        "type": "text",
                        "page_number": 1,
                        "title": "The framework",
                        "section": "Method",
                        "content": "The framework is discussed in this section.",
                        "search_text": "The framework is discussed in this section.",
                        "asset_path": "",
                        "asset_relpath": "",
                        "order": 1,
                    },
                    {
                        "id": "image-3",
                        "type": "image",
                        "page_number": 3,
                        "title": "Fig. 3 | Performance comparison",
                        "section": "Results",
                        "content": "Figure ID: Figure_3\nCaption: Fig. 3 | Performance comparison",
                        "search_text": "Figure ID: Figure_3\nCaption: Fig. 3 | Performance comparison",
                        "asset_path": "",
                        "asset_relpath": "parsed/assets/figures/Figure_3.png",
                        "order": 2,
                    },
                    {
                        "id": "image-2",
                        "type": "image",
                        "page_number": 2,
                        "title": "Fig. 2 | An overview of the framework",
                        "section": "Method",
                        "content": "Figure ID: Figure_2\nCaption: Fig. 2 | An overview of the framework",
                        "search_text": "Figure ID: Figure_2\nCaption: Fig. 2 | An overview of the framework",
                        "asset_path": "",
                        "asset_relpath": "parsed/assets/figures/Figure_2.png",
                        "order": 3,
                    },
                ],
            }
            (parsed_dir / "index_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )

            store = VectorStore()
            store._vector_scores = lambda _paper_id, _query: {}
            results = store.search(paper_id, "图2展示了什么？", top_k=3)

            assert results[0]["id"] == "image-2"
    finally:
        settings.upload_dir = original_upload_dir


if __name__ == "__main__":
    test_chinese_figure_reference_prioritizes_matching_image()
    print(f"{Path(__file__).name}: ok")
