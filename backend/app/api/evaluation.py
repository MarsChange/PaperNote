"""Internal evaluation endpoints for retrieval experiments."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.vector_store import vector_store

router = APIRouter(tags=["evaluation"])


class RetrievalEvalRequest(BaseModel):
    paper_id: str
    query: str
    top_k: int = Field(default=6, ge=1, le=50)
    retrieval_mode: Literal["agentic", "dense_only"] = "agentic"


@router.post("/evaluation/retrieve")
async def retrieve_for_evaluation(req: RetrievalEvalRequest):
    """Run the same single-paper Agentic RAG retriever used by chat."""

    if req.retrieval_mode == "dense_only":
        return vector_store.agentic_dense_only_retrieve(
            req.paper_id,
            req.query,
            top_k=req.top_k,
        )
    return vector_store.agentic_retrieve(req.paper_id, req.query, top_k=req.top_k)
