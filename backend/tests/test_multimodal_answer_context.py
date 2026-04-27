from __future__ import annotations

import tempfile
from pathlib import Path

from app.agents.generator import _build_user_message
from app.core.config import settings


def test_multimodal_answer_includes_nonleading_image_sources():
    original_enabled = settings.enable_multimodal_answers
    original_limit = settings.multimodal_answer_image_limit
    settings.enable_multimodal_answers = True
    settings.multimodal_answer_image_limit = 3
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for index in range(4):
                path = Path(tmpdir) / f"figure_{index}.png"
                path.write_bytes(b"tiny-image")
                image_paths.append(str(path))

            sources = [
                {"id": "text-1", "type": "text", "content": "text evidence"},
                {"id": "image-1", "type": "image", "title": "Fig. 1", "asset_path": image_paths[0]},
                {"id": "text-2", "type": "text", "content": "more text"},
                {"id": "image-2", "type": "image", "title": "Fig. 2", "asset_path": image_paths[1]},
                {"id": "image-3", "type": "image", "title": "Fig. 3", "asset_path": image_paths[2]},
                {"id": "image-4", "type": "image", "title": "Fig. 4", "asset_path": image_paths[3]},
            ]

            message = _build_user_message("请解释图2", sources)

            assert isinstance(message.content, list)
            image_parts = [
                part for part in message.content if part.get("type") == "image_url"
            ]
            text_parts = [
                part.get("text", "")
                for part in message.content
                if part.get("type") == "text"
            ]
            assert len(image_parts) == 3
            assert any("Visual evidence [S4] Fig. 2" in part for part in text_parts)
    finally:
        settings.enable_multimodal_answers = original_enabled
        settings.multimodal_answer_image_limit = original_limit


if __name__ == "__main__":
    test_multimodal_answer_includes_nonleading_image_sources()
    print(f"{Path(__file__).name}: ok")
