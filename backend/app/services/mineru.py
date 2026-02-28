"""MinerU API v4 client for PDF parsing."""

import asyncio
import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

MINERU_BASE = "https://mineru.net/api/v4"


class MinerUService:
    """Calls MinerU API v4 to convert PDF → Markdown."""

    def __init__(self):
        self.api_key = settings.mineru_api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def parse_pdf(self, pdf_path: str, output_dir: str) -> Optional[str]:
        """
        Full pipeline: upload → create task → poll → download zip → extract markdown.

        Args:
            pdf_path: Absolute path to the uploaded PDF file.
            output_dir: Directory to store the output markdown.

        Returns:
            Path to the generated markdown file, or None on failure.
        """
        if not self.api_key:
            logger.warning("MinerU API key not configured, using fallback extraction")
            return await self._fallback_extract(pdf_path, output_dir)

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                # Step 1: Get upload URL
                file_url = await self._upload_file(client, pdf_path)
                if not file_url:
                    return await self._fallback_extract(pdf_path, output_dir)

                # Step 2: Create extraction task
                task_id = await self._create_task(client, file_url)
                if not task_id:
                    return await self._fallback_extract(pdf_path, output_dir)

                # Step 3: Poll for result
                zip_url = await self._poll_task(client, task_id)
                if not zip_url:
                    return await self._fallback_extract(pdf_path, output_dir)

                # Step 4: Download and extract markdown from ZIP
                return await self._download_and_extract(client, zip_url, output_dir, pdf_path)

        except Exception as e:
            logger.error(f"MinerU pipeline error: {e}")
            return await self._fallback_extract(pdf_path, output_dir)

    async def _upload_file(self, client: httpx.AsyncClient, pdf_path: str) -> Optional[str]:
        """Upload local PDF file via file-urls/batch endpoint."""
        filename = Path(pdf_path).name
        try:
            # Get presigned upload URL
            resp = await client.post(
                f"{MINERU_BASE}/file-urls/batch",
                headers=self._headers(),
                json={"file_names": [filename]},
            )
            resp.raise_for_status()
            data = resp.json()

            batch_data = data.get("data", {})
            file_urls = batch_data.get("file_urls", [])
            if not file_urls:
                logger.error(f"No upload URL returned: {data}")
                return None

            item = file_urls[0]
            upload_url = item.get("upload_url") or item.get("put_url")
            file_url = item.get("url") or item.get("file_url")

            if not upload_url or not file_url:
                logger.error(f"Missing upload/file URL: {item}")
                return None

            # Upload the actual file via PUT
            with open(pdf_path, "rb") as f:
                put_resp = await client.put(
                    upload_url,
                    content=f.read(),
                    headers={"Content-Type": "application/pdf"},
                )
                put_resp.raise_for_status()

            logger.info(f"Uploaded {filename} → {file_url}")
            return file_url

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return None

    async def _create_task(self, client: httpx.AsyncClient, file_url: str) -> Optional[str]:
        """Submit extraction task to MinerU."""
        try:
            resp = await client.post(
                f"{MINERU_BASE}/extract/task",
                headers=self._headers(),
                json={
                    "url": file_url,
                    "model_version": "vlm",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("data", {}).get("task_id")
            if not task_id:
                logger.error(f"No task_id in response: {data}")
                return None

            logger.info(f"Created MinerU task: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            return None

    async def _poll_task(self, client: httpx.AsyncClient, task_id: str) -> Optional[str]:
        """Poll task status until done or timeout."""
        for attempt in range(120):  # max ~6 minutes
            await asyncio.sleep(3)
            try:
                resp = await client.get(
                    f"{MINERU_BASE}/extract/task/{task_id}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                state = data.get("state", "")

                if state == "done":
                    zip_url = data.get("full_zip_url")
                    if zip_url:
                        logger.info(f"Task {task_id} done, zip: {zip_url}")
                        return zip_url
                    logger.error(f"Task done but no zip_url: {data}")
                    return None

                if state in ("failed", "error"):
                    logger.error(f"Task {task_id} failed: {data}")
                    return None

                # Still running — log progress if available
                progress = data.get("extract_progress", {})
                extracted = progress.get("extracted_pages", "?")
                total = progress.get("total_pages", "?")
                logger.debug(f"Task {task_id}: {state} ({extracted}/{total} pages)")

            except Exception as e:
                logger.warning(f"Poll error (attempt {attempt}): {e}")

        logger.error(f"Task {task_id} timed out after polling")
        return None

    async def _download_and_extract(
        self, client: httpx.AsyncClient, zip_url: str, output_dir: str, pdf_path: str
    ) -> Optional[str]:
        """Download result ZIP and extract markdown file."""
        try:
            resp = await client.get(zip_url)
            resp.raise_for_status()

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(pdf_path).stem

            # Extract ZIP in memory
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                md_files = [n for n in zf.namelist() if n.endswith(".md")]
                if not md_files:
                    logger.error(f"No .md file in ZIP: {zf.namelist()}")
                    return None

                md_content = zf.read(md_files[0]).decode("utf-8")

            md_path = out / f"{stem}.md"
            md_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Extracted markdown to {md_path}")
            return str(md_path)

        except Exception as e:
            logger.error(f"Download/extract failed: {e}")
            return None

    async def _fallback_extract(self, pdf_path: str, output_dir: str) -> Optional[str]:
        """Simple text extraction fallback when MinerU is not available."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()

            content = "\n\n---\n\n".join(pages)
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            md_path = out / f"{Path(pdf_path).stem}.md"
            md_path.write_text(content, encoding="utf-8")
            return str(md_path)
        except ImportError:
            logger.error("Neither MinerU API nor PyMuPDF available for PDF parsing")
            return None


mineru_service = MinerUService()
