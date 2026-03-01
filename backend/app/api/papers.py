"""Paper upload and management API."""

import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.database import get_db
from app.services.mineru import mineru_service
from app.services.vector_store import vector_store

router = APIRouter(tags=["papers"])


async def _process_paper(paper_id: str, filepath: str):
    """Background task: parse PDF with MinerU, chunk, embed, store vectors."""
    db = await get_db()
    try:
        # Update status to parsing
        await db.execute(
            "UPDATE papers SET status = 'parsing', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (paper_id,),
        )
        await db.commit()

        # Parse with MinerU
        output_dir = str(Path(filepath).parent / "parsed")
        md_path = await mineru_service.parse_pdf(filepath, output_dir)

        if not md_path:
            await db.execute(
                "UPDATE papers SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (paper_id,),
            )
            await db.commit()
            return

        # Read markdown content
        md_content = Path(md_path).read_text(encoding="utf-8")

        # Chunk and embed into vector store
        vector_store.index_paper(paper_id, md_content)

        # Generate a quick summary (first 500 chars)
        summary = md_content[:500].strip()

        # Update paper record
        await db.execute(
            """UPDATE papers
               SET status = 'ready', markdown_path = ?, summary = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (md_path, summary, paper_id),
        )
        await db.commit()

    except Exception as e:
        await db.execute(
            "UPDATE papers SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (paper_id,),
        )
        await db.commit()
        raise e
    finally:
        await db.close()


@router.post("/papers/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    paper_id = str(uuid.uuid4())

    # Save file
    paper_dir = Path(settings.upload_dir) / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename).name
    filepath = paper_dir / safe_filename
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Insert DB record
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO papers (id, filename, filepath, status) VALUES (?, ?, ?, 'uploading')",
            (paper_id, safe_filename, str(filepath)),
        )
        await db.commit()
    finally:
        await db.close()

    # Kick off background processing
    background_tasks.add_task(_process_paper, paper_id, str(filepath))

    return {"id": paper_id, "filename": safe_filename, "status": "uploading"}

@router.get("/papers")
async def list_papers():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, filename, status, created_at FROM papers ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
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

        cursor = await db.execute(
            "SELECT id, title, created_at FROM conversations WHERE paper_id = ? ORDER BY created_at DESC",
            (paper_id,),
        )
        conversations = await cursor.fetchall()
        paper["conversations"] = [dict(c) for c in conversations]
        return paper
    finally:
        await db.close()


@router.get("/papers/{paper_id}/pdf")
async def get_paper_pdf(paper_id: str):
    """Serve the original uploaded PDF file."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT filepath, filename FROM papers WHERE id = ?", (paper_id,))
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


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id, filepath FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")

        # Cascade deletes handle conversations, messages, and annotations via FK
        await db.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        await db.commit()

        # Remove vectors
        vector_store.delete_paper(paper_id)

        # Remove uploaded files
        filepath = row["filepath"]
        paper_dir = Path(filepath).parent
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
            "SELECT id, status, summary, keywords FROM papers WHERE id = ?",
            (paper_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        return dict(row)
    finally:
        await db.close()
