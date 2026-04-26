"""MinerU-backed parsing with structured content extraction."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

MINERU_BASE = "https://mineru.net/api/v4"


@dataclass
class ParseResult:
    markdown_path: str
    markdown_content: str
    content_list_path: str
    content_list: list[dict[str, Any]]
    output_dir: str
    assets_dir: str
    page_count: int


@dataclass
class MinerUBatchUpload:
    batch_id: str
    upload_url: str


class MinerUService:
    """Calls MinerU API v4 and normalizes output into a content list."""

    def __init__(self):
        self.api_key = settings.mineru_api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def parse_pdf(self, pdf_path: str, output_dir: str) -> Optional[ParseResult]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("MinerU API key not configured, using local fallback parser")
            return await self._fallback_extract(pdf_path, output_dir)

        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                upload = await self._create_batch_upload(client, pdf_path)
                if not upload:
                    return await self._fallback_extract(pdf_path, output_dir)

                uploaded = await self._upload_file(client, upload.upload_url, pdf_path)
                if not uploaded:
                    return await self._fallback_extract(pdf_path, output_dir)

                zip_url = await self._poll_batch(
                    client, upload.batch_id, Path(pdf_path).name
                )
                if not zip_url:
                    return await self._fallback_extract(pdf_path, output_dir)

                parsed = await self._download_and_extract(
                    client, zip_url, output_dir, pdf_path
                )
                if parsed:
                    return parsed

                logger.warning(
                    "MinerU result ZIP could not be extracted, using local fallback parser"
                )
                return await self._fallback_extract(pdf_path, output_dir)
        except Exception as exc:
            logger.error("MinerU pipeline error: %s", exc)
            return await self._fallback_extract(pdf_path, output_dir)

    async def _create_batch_upload(
        self, client: httpx.AsyncClient, pdf_path: str
    ) -> Optional[MinerUBatchUpload]:
        filename = Path(pdf_path).name
        try:
            data_id = f"{Path(pdf_path).stem[:64]}-{uuid4().hex[:12]}"
            payload = {
                "files": [{"name": filename, "data_id": data_id}],
                "model_version": settings.mineru_model_version,
                "language": settings.mineru_language,
                "enable_formula": True,
                "enable_table": True,
            }
            resp = await client.post(
                f"{MINERU_BASE}/file-urls/batch",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            response_payload = resp.json()
            if response_payload.get("code") != 0:
                logger.error(
                    "MinerU upload URL request failed: code=%s msg=%s trace_id=%s",
                    response_payload.get("code"),
                    response_payload.get("msg"),
                    response_payload.get("trace_id"),
                )
                return None

            data = response_payload.get("data") or {}
            batch_id = str(data.get("batch_id") or "")
            file_urls = data.get("file_urls") or []
            if not file_urls:
                logger.error(
                    "No upload URL returned from MinerU: msg=%s trace_id=%s",
                    response_payload.get("msg"),
                    response_payload.get("trace_id"),
                )
                return None

            item = file_urls[0]
            upload_url = (
                item
                if isinstance(item, str)
                else item.get("upload_url") or item.get("put_url")
            )
            if not batch_id or not upload_url:
                logger.error(
                    "Missing batch_id or upload URL in MinerU response: msg=%s trace_id=%s",
                    response_payload.get("msg"),
                    response_payload.get("trace_id"),
                )
                return None

            return MinerUBatchUpload(batch_id=batch_id, upload_url=upload_url)
        except Exception as exc:
            logger.error("MinerU upload URL request failed: %s", exc)
            return None

    async def _upload_file(
        self, client: httpx.AsyncClient, upload_url: str, pdf_path: str
    ) -> bool:
        try:
            with open(pdf_path, "rb") as handle:
                resp = await client.put(upload_url, content=handle.read())
            if resp.status_code >= 400:
                logger.error(
                    "MinerU file PUT upload failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:300],
                )
                return False
            return True
        except Exception as exc:
            logger.error("MinerU file PUT upload failed: %s", exc)
            return False

    async def _poll_batch(
        self, client: httpx.AsyncClient, batch_id: str, filename: str
    ) -> Optional[str]:
        active_states = {"waiting-file", "pending", "running", "converting"}
        for attempt in range(120):
            await asyncio.sleep(3)
            try:
                resp = await client.get(
                    f"{MINERU_BASE}/extract-results/batch/{batch_id}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                response_payload = resp.json()
                if response_payload.get("code") != 0:
                    logger.error(
                        "MinerU batch polling failed: code=%s msg=%s trace_id=%s",
                        response_payload.get("code"),
                        response_payload.get("msg"),
                        response_payload.get("trace_id"),
                    )
                    return None

                data = response_payload.get("data") or {}
                results = data.get("extract_result") or data.get("extract_results") or []
                if isinstance(results, dict):
                    results = [results]
                if not results:
                    continue

                result = next(
                    (
                        item
                        for item in results
                        if isinstance(item, dict) and item.get("file_name") == filename
                    ),
                    results[0],
                )
                if not isinstance(result, dict):
                    continue

                state = str(result.get("state") or "").lower()

                if state == "done":
                    zip_url = result.get("full_zip_url")
                    if not zip_url:
                        logger.error(
                            "MinerU batch %s finished without full_zip_url", batch_id
                        )
                    return zip_url
                if state in {"failed", "error"}:
                    logger.error(
                        "MinerU batch %s failed for %s: %s",
                        batch_id,
                        filename,
                        result.get("err_msg") or result,
                    )
                    return None
                if state and state not in active_states:
                    logger.debug(
                        "MinerU batch %s state for %s: %s", batch_id, filename, state
                    )
            except Exception as exc:
                logger.warning(
                    "MinerU batch polling error on attempt %s: %s", attempt, exc
                )

        logger.error("MinerU batch %s timed out", batch_id)
        return None

    async def _download_and_extract(
        self,
        client: httpx.AsyncClient,
        zip_url: str,
        output_dir: str,
        pdf_path: str,
    ) -> Optional[ParseResult]:
        try:
            resp: Optional[httpx.Response] = None
            for attempt in range(3):
                try:
                    resp = await client.get(zip_url, follow_redirects=True)
                    break
                except httpx.RequestError as exc:
                    logger.warning(
                        "MinerU result ZIP download failed on attempt %s: type=%s repr=%r url=%s",
                        attempt + 1,
                        type(exc).__name__,
                        exc,
                        zip_url,
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
            if resp is None:
                return None

            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            content_length = len(resp.content)
            if not zipfile.is_zipfile(io.BytesIO(resp.content)):
                logger.error(
                    "MinerU result is not a ZIP: status=%s content_type=%s length=%s url=%s preview=%r",
                    resp.status_code,
                    content_type,
                    content_length,
                    resp.url,
                    resp.content[:120],
                )
                return None

            output_path = Path(output_dir)
            self._safe_extract_zip(resp.content, output_path)

            stem = Path(pdf_path).stem
            markdown_path, content_list_path = self._locate_output_files(output_path, stem)
            markdown_content = (
                markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
            )
            content_list = self._load_content_list(content_list_path)

            if not content_list and markdown_content:
                content_list = self._build_content_list_from_markdown(
                    markdown_content,
                    content_list_path.parent if content_list_path.exists() else output_path,
                )

            if not markdown_content and content_list:
                markdown_content = self._content_list_to_markdown(content_list)

            if not markdown_path.exists() and markdown_content:
                markdown_path = output_path / f"{stem}.md"
                markdown_path.write_text(markdown_content, encoding="utf-8")

            if not content_list_path.exists():
                content_list_path = output_path / f"{stem}_content_list.json"
                content_list_path.write_text(
                    json.dumps(content_list, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            return ParseResult(
                markdown_path=str(markdown_path),
                markdown_content=markdown_content,
                content_list_path=str(content_list_path),
                content_list=content_list,
                output_dir=str(output_path),
                assets_dir=str(content_list_path.parent),
                page_count=self._infer_page_count(content_list),
            )
        except Exception as exc:
            logger.exception(
                "Download/extract failed: type=%s repr=%r url=%s",
                type(exc).__name__,
                exc,
                zip_url,
            )
            return None

    async def _fallback_extract(self, pdf_path: str, output_dir: str) -> Optional[ParseResult]:
        try:
            import fitz  # PyMuPDF

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            assets_dir = output_path / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            doc = fitz.open(pdf_path)
            content_list: list[dict[str, Any]] = []
            markdown_pages: list[str] = []
            page_count = len(doc)

            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                text = (page.get_text("text") or "").strip()
                if text:
                    content_list.append(
                        {
                            "type": "text",
                            "text": text,
                            "text_level": 0,
                            "page_idx": page_index,
                        }
                    )
                    markdown_pages.append(f"# Page {page_number}\n\n{text}")

                seen_xrefs: set[int] = set()
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue

                    ext = extracted.get("ext", "png")
                    image_path = assets_dir / f"page_{page_number}_image_{image_index}.{ext}"
                    image_path.write_bytes(image_bytes)
                    content_list.append(
                        {
                            "type": "image",
                            "img_path": str(image_path.resolve()),
                            "image_caption": [f"Page {page_number} image {image_index}"],
                            "image_footnote": [],
                            "page_idx": page_index,
                        }
                    )

            doc.close()

            markdown_content = "\n\n---\n\n".join(markdown_pages).strip()
            if not markdown_content:
                markdown_content = Path(pdf_path).name

            stem = Path(pdf_path).stem
            markdown_path = output_path / f"{stem}.md"
            content_list_path = output_path / f"{stem}_content_list.json"
            markdown_path.write_text(markdown_content, encoding="utf-8")
            content_list_path.write_text(
                json.dumps(content_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return ParseResult(
                markdown_path=str(markdown_path),
                markdown_content=markdown_content,
                content_list_path=str(content_list_path),
                content_list=content_list,
                output_dir=str(output_path),
                assets_dir=str(assets_dir),
                page_count=page_count,
            )
        except ImportError:
            logger.error("Neither MinerU API nor PyMuPDF are available for parsing")
            return None
        except Exception as exc:
            logger.error("Local fallback parsing failed: %s", exc)
            return None

    def _locate_output_files(self, output_dir: Path, stem: str) -> tuple[Path, Path]:
        markdown_candidates = self._unique_paths(
            [
                *output_dir.rglob(f"{stem}.md"),
                *output_dir.rglob("full.md"),
                *output_dir.rglob("*.md"),
            ]
        )
        markdown_candidates.sort(
            key=lambda path: (
                0 if path.name == f"{stem}.md" else 1 if path.name == "full.md" else 2,
                len(path.parts),
                len(str(path)),
            )
        )
        content_list_candidates = self._unique_paths(
            [
                *output_dir.rglob(f"{stem}_content_list.json"),
                *output_dir.rglob("*_content_list.json"),
                *output_dir.rglob("content_list.json"),
            ]
        )
        content_list_candidates.sort(
            key=lambda path: (
                0
                if path.name == f"{stem}_content_list.json"
                else 1
                if path.name.endswith("_content_list.json")
                else 2,
                len(path.parts),
                len(str(path)),
            )
        )

        markdown_path = (
            markdown_candidates[0] if markdown_candidates else output_dir / f"{stem}.md"
        )
        content_list_path = (
            content_list_candidates[0]
            if content_list_candidates
            else output_dir / f"{stem}_content_list.json"
        )
        return markdown_path, content_list_path

    def _unique_paths(self, paths: list[Path]) -> list[Path]:
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique

    def _load_content_list(self, json_path: Path) -> list[dict[str, Any]]:
        if not json_path.exists():
            return []

        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                raw_content = json.load(handle)
        except Exception as exc:
            logger.warning("Could not read MinerU content list %s: %s", json_path, exc)
            return []

        if not isinstance(raw_content, list):
            return []

        normalized: list[dict[str, Any]] = []
        base_dir = json_path.parent.resolve()
        field_aliases = {
            "img_caption": "image_caption",
            "img_footnote": "image_footnote",
        }

        for item in raw_content:
            if not isinstance(item, dict):
                continue

            normalized_item = dict(item)
            for old_name, new_name in field_aliases.items():
                if old_name in normalized_item and new_name not in normalized_item:
                    normalized_item[new_name] = normalized_item[old_name]
                if new_name in normalized_item and old_name not in normalized_item:
                    normalized_item[old_name] = normalized_item[new_name]

            for list_field in [
                "image_caption",
                "image_footnote",
                "table_caption",
                "table_footnote",
            ]:
                value = normalized_item.get(list_field)
                if isinstance(value, str):
                    normalized_item[list_field] = [value] if value else []

            for path_field in ["img_path", "table_img_path", "equation_img_path"]:
                relative_path = normalized_item.get(path_field)
                if relative_path:
                    resolved = (base_dir / relative_path).resolve()
                    if resolved.is_relative_to(base_dir):
                        normalized_item[path_field] = str(resolved)
                    else:
                        normalized_item[path_field] = ""

            normalized.append(normalized_item)

        return normalized

    def _build_content_list_from_markdown(
        self, markdown_content: str, base_dir: Path
    ) -> list[dict[str, Any]]:
        content_list: list[dict[str, Any]] = []
        text_buffer: list[str] = []
        current_page = 0
        current_heading_level = 0
        lines = markdown_content.splitlines()
        index = 0

        def flush_text():
            if not text_buffer:
                return
            text = "\n".join(text_buffer).strip()
            text_buffer.clear()
            if text:
                content_list.append(
                    {
                        "type": "text",
                        "text": text,
                        "text_level": current_heading_level,
                        "page_idx": current_page,
                    }
                )

        while index < len(lines):
            line = lines[index].rstrip()
            if re.match(r"^#\s+Page\s+\d+", line, re.IGNORECASE):
                flush_text()
                page_match = re.search(r"(\d+)", line)
                if page_match:
                    current_page = max(int(page_match.group(1)) - 1, 0)
                current_heading_level = 1
                content_list.append(
                    {
                        "type": "text",
                        "text": line.lstrip("#").strip(),
                        "text_level": 1,
                        "page_idx": current_page,
                    }
                )
                index += 1
                continue

            if re.match(r"^#{1,6}\s+", line):
                flush_text()
                current_heading_level = len(line) - len(line.lstrip("#"))
                content_list.append(
                    {
                        "type": "text",
                        "text": line[current_heading_level:].strip(),
                        "text_level": current_heading_level,
                        "page_idx": current_page,
                    }
                )
                index += 1
                continue

            image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if image_match:
                flush_text()
                alt_text, path_value = image_match.groups()
                resolved = (base_dir / path_value).resolve()
                if resolved.exists():
                    content_list.append(
                        {
                            "type": "image",
                            "img_path": str(resolved),
                            "image_caption": [alt_text] if alt_text else [],
                            "image_footnote": [],
                            "page_idx": current_page,
                        }
                    )
                index += 1
                continue

            if "|" in line and index + 1 < len(lines) and re.match(
                r"^\s*\|?[-:\s|]+\|?\s*$", lines[index + 1]
            ):
                flush_text()
                table_lines = [line]
                index += 1
                while index < len(lines):
                    table_line = lines[index].rstrip()
                    if not table_line or "|" not in table_line:
                        break
                    table_lines.append(table_line)
                    index += 1
                content_list.append(
                    {
                        "type": "table",
                        "table_body": "\n".join(table_lines).strip(),
                        "table_caption": [],
                        "table_footnote": [],
                        "page_idx": current_page,
                    }
                )
                continue

            if not line.strip():
                flush_text()
                current_heading_level = 0
            else:
                text_buffer.append(line)

            index += 1

        flush_text()
        return content_list

    def _content_list_to_markdown(self, content_list: list[dict[str, Any]]) -> str:
        markdown_parts: list[str] = []
        for item in content_list:
            item_type = item.get("type")
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                text_level = int(item.get("text_level") or 0)
                if text_level > 0:
                    markdown_parts.append(f"{'#' * min(text_level, 6)} {text}")
                else:
                    markdown_parts.append(text)
            elif item_type == "image":
                captions = item.get("image_caption", []) or []
                caption = captions[0] if captions else "Image"
                image_path = item.get("img_path", "")
                markdown_parts.append(f"![{caption}]({image_path})")
            elif item_type == "table":
                table_body = str(item.get("table_body", "")).strip()
                if table_body:
                    markdown_parts.append(table_body)
            elif item_type == "equation":
                latex = str(item.get("latex", "")).strip()
                if latex:
                    markdown_parts.append(f"$$\n{latex}\n$$")

        return "\n\n".join(markdown_parts).strip()

    def _infer_page_count(self, content_list: list[dict[str, Any]]) -> int:
        page_indexes = [
            int(item.get("page_idx", 0))
            for item in content_list
            if isinstance(item.get("page_idx", 0), int)
        ]
        return (max(page_indexes) + 1) if page_indexes else 0

    def _safe_extract_zip(self, zip_bytes: bytes, output_dir: Path):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for member in archive.infolist():
                member_path = (output_dir / member.filename).resolve()
                if not member_path.is_relative_to(output_dir.resolve()):
                    raise ValueError(f"Unsafe path detected in zip file: {member.filename}")

            archive.extractall(output_dir)


mineru_service = MinerUService()
