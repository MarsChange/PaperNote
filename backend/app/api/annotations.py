"""Annotation and reader action APIs."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.database import get_db
from app.services.paper_ai import paper_ai_service

router = APIRouter(tags=["annotations"])


class AnnotationCreate(BaseModel):
    page_number: int = Field(ge=1)
    text_content: str
    color: str = "#fef08a"
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: str):
        if not re.match(r"^#[0-9a-fA-F]{6}$", value):
            raise ValueError("color must be hex like #fef08a")
        return value


class SelectionTranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    target_language: Optional[str] = None


async def _ensure_paper_exists(paper_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM papers WHERE id = ?", (paper_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Paper not found")
    finally:
        await db.close()


@router.post("/papers/{paper_id}/annotations")
async def create_annotation(paper_id: str, body: AnnotationCreate):
    await _ensure_paper_exists(paper_id)

    db = await get_db()
    try:
        annotation_id = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO annotations
               (id, paper_id, page_number, text_content, color, start_offset, end_offset, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotation_id,
                paper_id,
                body.page_number,
                body.text_content,
                body.color,
                body.start_offset,
                body.end_offset,
                body.note,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
        row = await cursor.fetchone()
        return dict(row)
    finally:
        await db.close()


@router.get("/papers/{paper_id}/annotations")
async def list_annotations(paper_id: str):
    await _ensure_paper_exists(paper_id)

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM annotations WHERE paper_id = ? ORDER BY page_number, created_at",
            (paper_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@router.delete("/papers/{paper_id}/annotations/{annotation_id}")
async def delete_annotation(paper_id: str, annotation_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM annotations WHERE id = ? AND paper_id = ?",
            (annotation_id, paper_id),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Annotation not found")

        await db.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        await db.commit()
        return {"detail": "Annotation deleted"}
    finally:
        await db.close()


@router.post("/papers/{paper_id}/selection/translate")
async def translate_selection(paper_id: str, body: SelectionTranslateRequest):
    await _ensure_paper_exists(paper_id)
    translation = await paper_ai_service.translate_selection(
        paper_id=paper_id,
        text=body.text,
        target_language=body.target_language,
    )
    return {"translation": translation}
