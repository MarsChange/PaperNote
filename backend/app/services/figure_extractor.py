"""Composite figure extraction from MinerU layout metadata.

MinerU can emit one image item for each sub-panel inside a scientific figure.
For VLM analysis we want the full Figure region plus its caption, so this module
uses MinerU model-layout bboxes to crop whole figure regions from the source PDF.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

FIGURE_CONTEXT_CHARS = 900
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:(?:Fig(?:ure)?\.?)\s*([0-9]+[A-Za-z]?)|图\s*([0-9]+[A-Za-z]?))",
    re.IGNORECASE,
)
FIGURE_CAPTION_ANYWHERE_RE = re.compile(
    r"(?:Fig(?:ure)?\.?)\s*([0-9]+[A-Za-z]?)|图\s*([0-9]+[A-Za-z]?)",
    re.IGNORECASE,
)
FIGURE_REFERENCE_TEMPLATE = r"(?:Figure|Fig\.?|figure)\s*{num}(?:[A-Za-z]|\b|[^0-9])|图\s*{num}"
RAW_IMAGE_PATH_FIELDS = ("img_path", "image_path", "table_img_path", "equation_img_path")


@dataclass
class LayoutItem:
    item_type: str
    bbox: list[float]
    text: str
    page_idx: int


@dataclass
class ExtractedFigure:
    figure_id: str
    figure_number: str
    page_idx: int
    image_path: str
    caption: str
    bbox: list[float]
    subfigure_count: int
    context_text: str

    def to_content_item(self) -> dict[str, Any]:
        return {
            "type": "image",
            "img_path": self.image_path,
            "image_caption": [self.caption] if self.caption else [],
            "img_caption": [self.caption] if self.caption else [],
            "image_footnote": [],
            "img_footnote": [],
            "page_idx": self.page_idx,
            "figure_id": self.figure_id,
            "figure_number": self.figure_number,
            "figure_bbox": self.bbox,
            "figure_context": self.context_text,
            "figure_source": "mineru_model_bbox_crop",
            "is_composite_figure": True,
            "subfigure_count": self.subfigure_count,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "figure_number": self.figure_number,
            "page_idx": self.page_idx,
            "image_path": self.image_path,
            "caption": self.caption,
            "bbox": self.bbox,
            "subfigure_count": self.subfigure_count,
            "context_text": self.context_text,
        }


class MinerUFigureExtractor:
    """Extracts whole scientific figures from MinerU model JSON + source PDF."""

    def enhance_content_list(
        self,
        *,
        pdf_path: str,
        output_dir: str | Path,
        content_list: list[dict[str, Any]],
        markdown_content: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not settings.enable_mineru_figure_crops:
            return content_list, {"enabled": False, "reason": "disabled"}

        output_path = Path(output_dir)
        model_json = self._find_model_json(output_path)
        if not model_json:
            return content_list, {"enabled": True, "extracted": 0, "reason": "model_json_missing"}

        source_pdf = self._find_source_pdf(output_path, pdf_path)
        if not source_pdf or not source_pdf.exists():
            return content_list, {"enabled": True, "extracted": 0, "reason": "pdf_missing"}

        try:
            figures = self.extract_figures(
                model_json_path=model_json,
                pdf_path=source_pdf,
                output_dir=output_path,
                content_list=content_list,
                markdown_content=markdown_content,
            )
        except Exception as exc:
            logger.warning("Composite figure extraction failed: %s", exc)
            return content_list, {"enabled": True, "extracted": 0, "reason": "failed"}

        if not figures:
            return content_list, {"enabled": True, "extracted": 0, "reason": "no_figures"}

        enhanced = self._replace_image_fragments(content_list, figures)
        self._write_figures_metadata(output_path, figures)
        raw_images_deleted = False
        raw_images_deleted_files = 0
        if settings.delete_mineru_raw_images_after_figure_crops:
            enhanced = self._drop_raw_image_references(enhanced, output_path / "images")
            raw_images_deleted_files = self._delete_raw_images_dir(output_path)
            raw_images_deleted = raw_images_deleted_files > 0
        return enhanced, {
            "enabled": True,
            "extracted": len(figures),
            "source": "mineru_model_bbox_crop",
            "raw_images_deleted": raw_images_deleted,
            "raw_images_deleted_files": raw_images_deleted_files,
            "figures": [figure.to_metadata() for figure in figures],
        }

    def extract_figures(
        self,
        *,
        model_json_path: Path,
        pdf_path: Path,
        output_dir: Path,
        content_list: list[dict[str, Any]],
        markdown_content: str = "",
    ) -> list[ExtractedFigure]:
        import fitz  # type: ignore

        raw_model = json.loads(model_json_path.read_text(encoding="utf-8"))
        pages = self._iter_model_pages(raw_model)
        caption_lookup = self._caption_lookup(content_list)
        content_figures_by_page = self._content_figure_captions(content_list)
        full_text = markdown_content or self._content_list_text(content_list)
        figures_dir = output_dir / "assets" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        extracted: list[ExtractedFigure] = []
        with fitz.open(pdf_path) as doc:
            for page_idx, page_data in pages:
                if page_idx < 0 or page_idx >= len(doc):
                    continue

                items = self._layout_items(page_data, page_idx)
                image_items = [
                    item
                    for item in items
                    if self._is_image_type(item.item_type) and self._valid_bbox(item.bbox)
                ]
                caption_items = [
                    item
                    for item in items
                    if self._is_caption_type(item.item_type)
                    and self._valid_bbox(item.bbox)
                ]
                if not image_items or not caption_items:
                    continue

                caption_specs = self._caption_specs_for_page(
                    page_idx=page_idx,
                    caption_items=caption_items,
                    content_figures=content_figures_by_page.get(page_idx, []),
                )
                if not caption_specs:
                    continue

                unassigned = image_items[:]

                for spec_index, caption_spec in enumerate(caption_specs):
                    figure_number = caption_spec["figure_number"]
                    caption_bbox = caption_spec["bbox"]
                    previous_bottom = (
                        caption_specs[spec_index - 1]["bbox"][3] if spec_index > 0 else 0.0
                    )
                    related = self._related_images(caption_bbox, unassigned, previous_bottom)
                    if not related:
                        related = self._related_images_below_caption(caption_bbox, unassigned)
                    if not related:
                        continue

                    for image_item in related:
                        if image_item in unassigned:
                            unassigned.remove(image_item)

                    all_bboxes = [item.bbox for item in related] + [caption_bbox]
                    merged_bbox = self._merge_bboxes(all_bboxes, padding=settings.figure_crop_padding)
                    caption = self._best_caption(
                        figure_number,
                        caption_spec["caption"],
                        caption_lookup,
                    )
                    context_text = self._figure_context(full_text, figure_number, caption)
                    image_path = self._crop_pdf_region(
                        doc[page_idx],
                        merged_bbox,
                        figures_dir / f"Figure_{figure_number}.png",
                    )
                    if not image_path:
                        continue

                    extracted.append(
                        ExtractedFigure(
                            figure_id=f"Figure_{figure_number}",
                            figure_number=figure_number,
                            page_idx=page_idx,
                            image_path=str(image_path.resolve()),
                            caption=caption,
                            bbox=merged_bbox,
                            subfigure_count=len(related),
                            context_text=context_text,
                        )
                    )

        extracted.sort(key=lambda figure: (figure.page_idx, self._figure_sort_key(figure.figure_number)))
        return self._dedupe_figures(extracted)

    def _find_model_json(self, output_dir: Path) -> Optional[Path]:
        candidates = [
            *output_dir.rglob("*_model.json"),
            *output_dir.rglob("model.json"),
        ]
        candidates = [path for path in candidates if path.is_file()]
        candidates.sort(key=lambda path: (len(path.parts), len(str(path))))
        return candidates[0] if candidates else None

    def _find_source_pdf(self, output_dir: Path, fallback_pdf_path: str) -> Optional[Path]:
        candidates = [
            *output_dir.rglob("*_origin.pdf"),
            *output_dir.rglob("origin.pdf"),
            *output_dir.rglob("*.pdf"),
        ]
        for path in candidates:
            if path.is_file():
                return path
        fallback = Path(fallback_pdf_path)
        return fallback if fallback.exists() else None

    def _iter_model_pages(self, raw_model: Any) -> list[tuple[int, Any]]:
        if isinstance(raw_model, list):
            return [(index, page) for index, page in enumerate(raw_model)]
        if not isinstance(raw_model, dict):
            return []

        page_candidates = (
            raw_model.get("pages")
            or raw_model.get("pdf_info")
            or raw_model.get("page_info")
            or raw_model.get("model")
            or []
        )
        if isinstance(page_candidates, list):
            pages: list[tuple[int, Any]] = []
            for index, page in enumerate(page_candidates):
                page_idx = self._safe_int(
                    page.get("page_idx") if isinstance(page, dict) else None,
                    index,
                )
                pages.append((page_idx, page))
            return pages
        return []

    def _layout_items(self, page_data: Any, page_idx: int) -> list[LayoutItem]:
        page_width, page_height = self._page_dimensions(page_data)
        raw_items: Iterable[Any]
        if isinstance(page_data, list):
            raw_items = page_data
        elif isinstance(page_data, dict):
            raw_items = (
                page_data.get("items")
                or page_data.get("layout_dets")
                or page_data.get("layout")
                or page_data.get("blocks")
                or []
            )
        else:
            raw_items = []

        items: list[LayoutItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            bbox = self._item_bbox(raw, page_width, page_height)
            if not bbox:
                continue
            item_type = str(
                raw.get("type")
                or raw.get("category_type")
                or raw.get("label")
                or raw.get("category")
                or ""
            ).lower()
            text = self._item_text(raw)
            items.append(LayoutItem(item_type=item_type, bbox=bbox, text=text, page_idx=page_idx))
        return items

    def _page_dimensions(self, page_data: Any) -> tuple[float | None, float | None]:
        if not isinstance(page_data, dict):
            return None, None
        info = page_data.get("page_info") if isinstance(page_data.get("page_info"), dict) else {}
        width = (
            page_data.get("width")
            or page_data.get("page_width")
            or info.get("width")
            or info.get("page_width")
        )
        height = (
            page_data.get("height")
            or page_data.get("page_height")
            or info.get("height")
            or info.get("page_height")
        )
        return self._safe_float(width), self._safe_float(height)

    def _item_bbox(
        self,
        raw: dict[str, Any],
        page_width: float | None,
        page_height: float | None,
    ) -> list[float]:
        bbox_value = raw.get("bbox") or raw.get("box") or raw.get("rect")
        if not bbox_value and raw.get("poly"):
            poly = raw.get("poly") or []
            if isinstance(poly, list) and len(poly) >= 4:
                if all(isinstance(point, list) and len(point) >= 2 for point in poly):
                    xs = [self._safe_float(point[0]) for point in poly]
                    ys = [self._safe_float(point[1]) for point in poly]
                else:
                    coords = [self._safe_float(value) for value in poly]
                    coords = [value for value in coords if value is not None]
                    xs = coords[0::2]
                    ys = coords[1::2]
                xs = [value for value in xs if value is not None]
                ys = [value for value in ys if value is not None]
                if xs and ys:
                    bbox_value = [min(xs), min(ys), max(xs), max(ys)]

        if not isinstance(bbox_value, list) or len(bbox_value) < 4:
            return []

        values = [self._safe_float(value) for value in bbox_value[:4]]
        if any(value is None for value in values):
            return []
        x0, y0, x1, y1 = [float(value) for value in values if value is not None]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 2:
            return [self._clamp(x0), self._clamp(y0), self._clamp(x1), self._clamp(y1)]

        if page_width and page_height and page_width > 0 and page_height > 0:
            return [
                self._clamp(x0 / page_width),
                self._clamp(y0 / page_height),
                self._clamp(x1 / page_width),
                self._clamp(y1 / page_height),
            ]

        return []

    def _item_text(self, raw: dict[str, Any]) -> str:
        parts = [
            raw.get("content"),
            raw.get("text"),
            raw.get("value"),
        ]
        for span_key in ("spans", "lines"):
            spans = raw.get(span_key)
            if isinstance(spans, list):
                for span in spans:
                    if isinstance(span, dict):
                        parts.append(span.get("content") or span.get("text"))
                    elif isinstance(span, str):
                        parts.append(span)
        return self._normalize(" ".join(str(part) for part in parts if part))

    def _is_image_type(self, item_type: str) -> bool:
        lowered = item_type.lower()
        if "caption" in lowered or "footnote" in lowered:
            return False
        return any(token in lowered for token in ("image", "figure", "fig"))

    def _is_caption_type(self, item_type: str) -> bool:
        lowered = item_type.lower()
        return "caption" in lowered

    def _figure_number(self, caption: str) -> str:
        match = FIGURE_CAPTION_RE.search(caption or "")
        if not match:
            return ""
        number = match.group(1) or match.group(2) or ""
        return re.sub(r"[^0-9A-Za-z]+", "", number)

    def _figure_number_anywhere(self, caption: str) -> str:
        match = FIGURE_CAPTION_ANYWHERE_RE.search(caption or "")
        if not match:
            return ""
        number = match.group(1) or match.group(2) or ""
        return re.sub(r"[^0-9A-Za-z]+", "", number)

    def _caption_specs_for_page(
        self,
        *,
        page_idx: int,
        caption_items: list[LayoutItem],
        content_figures: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        explicit_specs: list[dict[str, Any]] = []
        for caption_item in caption_items:
            figure_number = self._figure_number_anywhere(caption_item.text)
            if figure_number:
                explicit_specs.append(
                    {
                        "figure_number": figure_number,
                        "caption": caption_item.text,
                        "bbox": caption_item.bbox,
                    }
                )
        if explicit_specs:
            return sorted(explicit_specs, key=lambda spec: (spec["bbox"][1], spec["bbox"][0]))

        if not content_figures:
            return []

        caption_groups = self._caption_bbox_groups(caption_items)
        specs: list[dict[str, Any]] = []
        for index, content_figure in enumerate(content_figures):
            if index >= len(caption_groups):
                break
            specs.append(
                {
                    "figure_number": content_figure["figure_number"],
                    "caption": content_figure["caption"],
                    "bbox": caption_groups[index],
                }
            )
        if not specs and len(content_figures) == 1 and caption_items:
            specs.append(
                {
                    "figure_number": content_figures[0]["figure_number"],
                    "caption": content_figures[0]["caption"],
                    "bbox": self._merge_bboxes([item.bbox for item in caption_items], padding=0.0),
                }
            )
        return specs

    def _caption_bbox_groups(self, caption_items: list[LayoutItem]) -> list[list[float]]:
        candidates = [
            item
            for item in caption_items
            if self._bbox_width(item.bbox) >= 0.12 or self._bbox_height(item.bbox) >= 0.025
        ]
        if not candidates:
            candidates = caption_items[:]

        candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        rows: list[list[LayoutItem]] = []
        for item in candidates:
            if not rows:
                rows.append([item])
                continue
            row_y0 = min(existing.bbox[1] for existing in rows[-1])
            row_y1 = max(existing.bbox[3] for existing in rows[-1])
            if item.bbox[1] <= row_y1 + 0.025 and abs(item.bbox[1] - row_y0) <= 0.08:
                rows[-1].append(item)
            else:
                rows.append([item])

        return [
            self._merge_bboxes([item.bbox for item in row], padding=0.0)
            for row in rows
        ]

    def _related_images(
        self,
        caption_bbox: list[float],
        images: list[LayoutItem],
        previous_caption_bottom: float = 0.0,
    ) -> list[LayoutItem]:
        caption_y0 = caption_bbox[1]
        candidates = [
            image
            for image in images
            if image.bbox[3] <= caption_y0 + 0.025
            and image.bbox[1] >= max(previous_caption_bottom - 0.01, 0.0)
            and self._x_overlap_ratio(image.bbox, caption_bbox) > 0.08
        ]
        if not candidates:
            candidates = [
                image
                for image in images
                if image.bbox[3] <= caption_y0 + 0.025
                and image.bbox[1] >= max(previous_caption_bottom - 0.01, 0.0)
            ]
        return sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _related_images_below_caption(self, caption_bbox: list[float], images: list[LayoutItem]) -> list[LayoutItem]:
        caption_y1 = caption_bbox[3]
        candidates = [
            image
            for image in images
            if image.bbox[1] >= caption_y1 - 0.02
            and self._x_overlap_ratio(image.bbox, caption_bbox) > 0.08
        ]
        return sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _merge_bboxes(self, bboxes: list[list[float]], padding: float) -> list[float]:
        return [
            self._clamp(min(bbox[0] for bbox in bboxes) - padding),
            self._clamp(min(bbox[1] for bbox in bboxes) - padding),
            self._clamp(max(bbox[2] for bbox in bboxes) + padding),
            self._clamp(max(bbox[3] for bbox in bboxes) + padding),
        ]

    def _crop_pdf_region(self, page: Any, bbox: list[float], output_path: Path) -> Optional[Path]:
        import fitz  # type: ignore

        page_rect = page.rect
        clip = fitz.Rect(
            bbox[0] * page_rect.width,
            bbox[1] * page_rect.height,
            bbox[2] * page_rect.width,
            bbox[3] * page_rect.height,
        ) & page_rect
        if clip.is_empty or clip.width < 8 or clip.height < 8:
            return None

        zoom = max(settings.figure_crop_dpi, 72) / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        pix.save(str(output_path))
        return output_path

    def _caption_lookup(self, content_list: list[dict[str, Any]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for item in content_list:
            if not isinstance(item, dict):
                continue
            captions = item.get("image_caption") or item.get("img_caption") or []
            if isinstance(captions, str):
                captions = [captions]
            if not isinstance(captions, list):
                continue
            for caption in captions:
                text = self._normalize(str(caption))
                number = self._figure_number_anywhere(text)
                if not number:
                    continue
                if len(text) > len(lookup.get(number, "")):
                    lookup[number] = text
        return lookup

    def _content_figure_captions(
        self, content_list: list[dict[str, Any]]
    ) -> dict[int, list[dict[str, str]]]:
        by_page: dict[int, list[dict[str, str]]] = {}
        seen: set[tuple[int, str]] = set()
        for item in content_list:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            captions = item.get("image_caption") or item.get("img_caption") or []
            if isinstance(captions, str):
                captions = [captions]
            if not isinstance(captions, list):
                continue
            caption = self._main_caption_text(captions)
            figure_number = self._figure_number_anywhere(caption)
            if not figure_number:
                continue
            page_idx = self._safe_int(item.get("page_idx"), 0)
            key = (page_idx, figure_number)
            if key in seen:
                continue
            seen.add(key)
            by_page.setdefault(page_idx, []).append(
                {"figure_number": figure_number, "caption": caption}
            )
        return by_page

    def _main_caption_text(self, captions: list[Any]) -> str:
        texts = [self._normalize(str(value)) for value in captions if str(value).strip()]
        if not texts:
            return ""
        for index, text in enumerate(texts):
            if self._figure_number_anywhere(text):
                return self._normalize(" ".join(texts[index:]))
        return self._normalize(" ".join(texts))

    def _best_caption(
        self,
        figure_number: str,
        layout_caption: str,
        caption_lookup: dict[str, str],
    ) -> str:
        content_caption = caption_lookup.get(figure_number, "")
        layout_caption = self._normalize(layout_caption)
        return content_caption if len(content_caption) >= len(layout_caption) else layout_caption

    def _figure_context(self, full_text: str, figure_number: str, caption: str) -> str:
        contexts: list[str] = []
        if caption:
            contexts.append(f"Caption: {caption}")
        if not full_text or not figure_number:
            return "\n\n".join(contexts)

        pattern = FIGURE_REFERENCE_TEMPLATE.format(num=re.escape(figure_number))
        seen: set[str] = set()
        for match in re.finditer(pattern, full_text, flags=re.IGNORECASE):
            context = self._window_text(full_text, match.start(), FIGURE_CONTEXT_CHARS // 2)
            key = context[:120].lower()
            if context and key not in seen:
                seen.add(key)
                contexts.append(f"Reference context: {context}")
            if len(contexts) >= 4:
                break
        return "\n\n".join(contexts)[:2400]

    def _window_text(self, text: str, position: int, window: int) -> str:
        start = max(0, position - window)
        end = min(len(text), position + window)
        if start > 0:
            previous = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
            if previous > 0 and start - previous < window // 2:
                start = previous + 1
        if end < len(text):
            next_period = text.find(".", end)
            next_newline = text.find("\n", end)
            boundaries = [value for value in [next_period, next_newline] if value > 0]
            if boundaries:
                boundary = min(boundaries)
                if boundary - end < window // 2:
                    end = boundary + 1
        return self._normalize(text[start:end])

    def _replace_image_fragments(
        self,
        content_list: list[dict[str, Any]],
        figures: list[ExtractedFigure],
    ) -> list[dict[str, Any]]:
        figures_by_page: dict[int, list[ExtractedFigure]] = {}
        for figure in figures:
            figures_by_page.setdefault(figure.page_idx, []).append(figure)

        enhanced: list[dict[str, Any]] = []
        inserted_pages: set[int] = set()
        for item in content_list:
            if not isinstance(item, dict):
                continue
            page_idx = self._safe_int(item.get("page_idx"), 0)
            if item.get("type") == "image" and page_idx in figures_by_page:
                if page_idx not in inserted_pages:
                    enhanced.extend(figure.to_content_item() for figure in figures_by_page[page_idx])
                    inserted_pages.add(page_idx)
                continue
            enhanced.append(item)

        for page_idx, page_figures in figures_by_page.items():
            if page_idx not in inserted_pages:
                enhanced.extend(figure.to_content_item() for figure in page_figures)
        return enhanced

    def _write_figures_metadata(self, output_dir: Path, figures: list[ExtractedFigure]):
        figures_dir = output_dir / "assets" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / "figures.json").write_text(
            json.dumps([figure.to_metadata() for figure in figures], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _drop_raw_image_references(
        self,
        content_list: list[dict[str, Any]],
        raw_images_dir: Path,
    ) -> list[dict[str, Any]]:
        if not raw_images_dir.exists():
            return content_list

        raw_root = raw_images_dir.resolve()
        cleaned: list[dict[str, Any]] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue

            item_copy = dict(item)
            has_raw_image_path = any(
                self._path_is_under(item_copy.get(field), raw_root)
                for field in RAW_IMAGE_PATH_FIELDS
            )
            if item_copy.get("type") == "image" and has_raw_image_path:
                continue

            for field in RAW_IMAGE_PATH_FIELDS:
                if self._path_is_under(item_copy.get(field), raw_root):
                    item_copy[field] = ""
            cleaned.append(item_copy)
        return cleaned

    def _delete_raw_images_dir(self, output_dir: Path) -> int:
        raw_images_dir = output_dir / "images"
        if not raw_images_dir.exists() or not raw_images_dir.is_dir():
            return 0

        file_count = sum(1 for path in raw_images_dir.rglob("*") if path.is_file())
        shutil.rmtree(raw_images_dir)
        logger.info("Deleted MinerU raw images directory %s (%s files)", raw_images_dir, file_count)
        return file_count

    def _path_is_under(self, value: Any, root: Path) -> bool:
        if not value:
            return False
        try:
            path = Path(str(value)).resolve()
            return path.is_relative_to(root)
        except Exception:
            return False

    def _dedupe_figures(self, figures: list[ExtractedFigure]) -> list[ExtractedFigure]:
        deduped: list[ExtractedFigure] = []
        seen: set[tuple[int, str]] = set()
        for figure in figures:
            key = (figure.page_idx, figure.figure_number)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(figure)
        return deduped

    def _content_list_text(self, content_list: list[dict[str, Any]]) -> str:
        texts = [
            str(item.get("text", ""))
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n\n".join(texts)

    def _figure_sort_key(self, figure_number: str) -> tuple[int, str]:
        match = re.match(r"(\d+)([A-Za-z]?)", figure_number)
        if not match:
            return 9999, figure_number
        suffix = match.group(2) or ""
        return int(match.group(1)), suffix

    def _x_overlap_ratio(self, bbox_a: list[float], bbox_b: list[float]) -> float:
        overlap = max(0.0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
        width = max(min(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0]), 1e-6)
        return overlap / width

    def _valid_bbox(self, bbox: list[float]) -> bool:
        return len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]

    def _bbox_width(self, bbox: list[float]) -> float:
        return max(bbox[2] - bbox[0], 0.0) if len(bbox) == 4 else 0.0

    def _bbox_height(self, bbox: list[float]) -> float:
        return max(bbox[3] - bbox[1], 0.0) if len(bbox) == 4 else 0.0

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    def _safe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default


mineru_figure_extractor = MinerUFigureExtractor()
