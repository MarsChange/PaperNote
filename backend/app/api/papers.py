"""Paper upload, parsing, indexing, and asset APIs."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.database import get_db
from app.services.knowledge_graph import knowledge_graph_indexer
from app.services.mineru import mineru_service
from app.services.paper_ai import paper_ai_service
from app.services.vector_store import vector_store

router = APIRouter(tags=["papers"])


def _parse_keywords(raw_keywords: Optional[str]) -> list[str]:
    if not raw_keywords:
        return []
    try:
        parsed = json.loads(raw_keywords)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in raw_keywords.split(",") if item.strip()]


def _parse_metadata(raw_metadata: Optional[str]) -> dict:
    if not raw_metadata:
        return {}
    try:
        payload = json.loads(raw_metadata)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


async def _process_paper(paper_id: str, filepath: str):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE papers SET status = 'parsing', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (paper_id,),
        )
        await db.commit()

        output_dir = str(Path(filepath).parent / "parsed")
        parsed = await mineru_service.parse_pdf(filepath, output_dir)
        if not parsed:
            await db.execute(
                "UPDATE papers SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (paper_id,),
            )
            await db.commit()
            return

        await db.execute(
            """UPDATE papers
               SET status = 'indexing',
                   markdown_path = ?,
                   content_list_path = ?,
                   assets_dir = ?,
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (parsed.markdown_path, parsed.content_list_path, parsed.assets_dir, paper_id),
        )
        await db.commit()

        index_metadata = vector_store.index_paper(paper_id, parsed)
        overview = await paper_ai_service.generate_overview(
            paper_id=paper_id,
            filename=Path(filepath).name,
        )
        metadata_payload = {
            "page_count": parsed.page_count,
            "stats": index_metadata.get("stats", {}),
            "summary_preview": index_metadata.get("summary_preview", ""),
        }

        await db.execute(
            """UPDATE papers
               SET status = 'ready',
                   summary = ?,
                   keywords = ?,
                   page_count = ?,
                   metadata_json = ?,
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (
                overview.get("summary", ""),
                json.dumps(overview.get("keywords", []), ensure_ascii=False),
                parsed.page_count,
                json.dumps(metadata_payload, ensure_ascii=False),
                paper_id,
            ),
        )
        await db.commit()
    except Exception:
        await db.execute(
            "UPDATE papers SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (paper_id,),
        )
        await db.commit()
        raise
    finally:
        await db.close()


@router.post("/papers/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    paper_id = str(uuid.uuid4())
    paper_dir = Path(settings.upload_dir) / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    filepath = paper_dir / safe_filename
    with open(filepath, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO papers (id, filename, filepath, status) VALUES (?, ?, ?, 'uploading')",
            (paper_id, safe_filename, str(filepath)),
        )
        await db.commit()
    finally:
        await db.close()

    background_tasks.add_task(_process_paper, paper_id, str(filepath))
    return {"id": paper_id, "filename": safe_filename, "status": "uploading"}


@router.get("/papers")
async def list_papers():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, filename, status, summary, created_at FROM papers ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper = dict(row)
        paper["keywords"] = _parse_keywords(paper.get("keywords"))
        paper["metadata"] = _parse_metadata(paper.get("metadata_json"))

        cursor = await db.execute(
            """SELECT id, title, created_at
               FROM conversations
               WHERE paper_id = ?
               ORDER BY created_at DESC""",
            (paper_id,),
        )
        paper["conversations"] = [dict(item) for item in await cursor.fetchall()]
        return paper
    finally:
        await db.close()


@router.get("/papers/{paper_id}/pdf")
async def get_paper_pdf(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT filepath, filename FROM papers WHERE id = ?",
            (paper_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")

        filepath = Path(row["filepath"])
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")

        return FileResponse(
            path=str(filepath),
            media_type="application/pdf",
            filename=row["filename"],
        )
    finally:
        await db.close()


@router.get("/papers/{paper_id}/assets/{asset_path:path}")
async def get_paper_asset(paper_id: str, asset_path: str):
    paper_dir = (Path(settings.upload_dir) / paper_id).resolve()
    resolved_asset = (paper_dir / asset_path).resolve()
    if not resolved_asset.is_relative_to(paper_dir) or not resolved_asset.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path=str(resolved_asset))


@router.get("/papers/{paper_id}/knowledge-graph")
async def get_paper_knowledge_graph(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
    finally:
        await db.close()

    graph = knowledge_graph_indexer.load_graph(paper_id)
    if not graph.get("nodes"):
        raise HTTPException(status_code=404, detail="Knowledge graph not found")
    return graph


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id, filepath FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")

        await db.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        await db.commit()

        vector_store.delete_paper(paper_id)

        paper_dir = Path(row["filepath"]).parent
        if paper_dir.exists():
            shutil.rmtree(paper_dir)

        return {"detail": "Paper deleted"}
    finally:
        await db.close()


@router.get("/papers/{paper_id}/status")
async def get_paper_status(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT id, status, summary, keywords, page_count, metadata_json
               FROM papers
               WHERE id = ?""",
            (paper_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")

        payload = dict(row)
        payload["keywords"] = _parse_keywords(payload.get("keywords"))
        payload["metadata"] = _parse_metadata(payload.get("metadata_json"))
        return payload
    finally:
        await db.close()
