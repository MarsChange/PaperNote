import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from app.services.figure_extractor import mineru_figure_extractor


def _write_synthetic_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.draw_rect(fitz.Rect(90, 90, 270, 260), color=(0, 0, 0), fill=(0.9, 0.95, 1))
    page.draw_rect(fitz.Rect(330, 90, 510, 260), color=(0, 0, 0), fill=(0.95, 0.9, 1))
    page.insert_text((95, 285), "Fig. 1. Overall framework with two subfigures.", fontsize=12)
    doc.save(path)
    doc.close()


def test_mineru_model_bbox_crops_composite_figure():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        pdf_path = root / "paper.pdf"
        output_dir = root / "parsed"
        output_dir.mkdir()
        raw_images_dir = output_dir / "images"
        raw_images_dir.mkdir()
        raw_fragment = raw_images_dir / "fragment.png"
        raw_fragment.write_bytes(b"raw")
        _write_synthetic_pdf(pdf_path)

        model_json = [
            [
                {"type": "image", "bbox": [0.15, 0.11, 0.45, 0.33], "content": ""},
                {"type": "image", "bbox": [0.55, 0.11, 0.85, 0.33], "content": ""},
                {
                    "type": "image_caption",
                    "bbox": [0.15, 0.35, 0.85, 0.39],
                    "content": None,
                },
            ]
        ]
        (output_dir / "paper_model.json").write_text(json.dumps(model_json), encoding="utf-8")

        content_list = [
            {
                "type": "text",
                "text": "As shown in Fig. 1, the method combines two visual branches.",
                "page_idx": 0,
            },
            {
                "type": "image",
                "img_path": str(raw_fragment),
                "image_caption": ["Fig. 1. Overall framework with two subfigures."],
                "page_idx": 0,
            },
        ]

        enhanced, metadata = mineru_figure_extractor.enhance_content_list(
            pdf_path=str(pdf_path),
            output_dir=output_dir,
            content_list=content_list,
            markdown_content=content_list[0]["text"],
        )

        images = [item for item in enhanced if item.get("type") == "image"]
        assert metadata["extracted"] == 1
        assert len(images) == 1
        assert images[0]["figure_id"] == "Figure_1"
        assert images[0]["subfigure_count"] == 2
        assert "Overall framework" in images[0]["image_caption"][0]
        assert "Reference context" in images[0]["figure_context"]
        assert Path(images[0]["img_path"]).exists()
        assert metadata["raw_images_deleted"] is True
        assert metadata["raw_images_deleted_files"] == 1
        assert not raw_images_dir.exists()


if __name__ == "__main__":
    test_mineru_model_bbox_crops_composite_figure()
    print("test_figure_extractor.py: ok")
