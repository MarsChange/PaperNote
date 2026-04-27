"""Chat API with SSE streaming backed by LangGraph agents."""

from __future__ import annotations

import json
import logging
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import rag_graph
from app.core.database import get_db
from app.services.agentic_runtime import set_rag_step_queue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _chunk_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


class CreateConversationRequest(BaseModel):
    paper_id: str


class SendMessageRequest(BaseModel):
    conversation_id: str
    content: str
    paper_id: str


class UpdateTitleRequest(BaseModel):
    title: str


@router.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    conv_id = str(uuid.uuid4())
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id FROM papers WHERE id = ?", (req.paper_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Paper not found")

        await db.execute(
            "INSERT INTO conversations (id, paper_id) VALUES (?, ?)",
            (conv_id, req.paper_id),
        )
        await db.commit()
    finally:
        await db.close()

    return {"id": conv_id, "paper_id": req.paper_id}


@router.put("/papers/{paper_id}/conversations/{conversation_id}/title")
async def update_conversation_title(paper_id: str, conversation_id: str, req: UpdateTitleRequest):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND paper_id = ?",
            (conversation_id, paper_id),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (req.title, conversation_id),
        )
        await db.commit()
        return {"id": conversation_id, "title": req.title}
    finally:
        await db.close()


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT id, role, content, metadata_json, created_at
               FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at""",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def _load_chat_history(conversation_id: str) -> list:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            if row["role"] == "user":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "assistant":
                messages.append(AIMessage(content=row["content"]))
        return messages
    finally:
        await db.close()


async def _save_assistant_message(
    conversation_id: str,
    answer: str,
    sources: list[dict],
    rag_trace: dict | None = None,
):
    db = await get_db()
    try:
        assistant_msg_id = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO messages (id, conversation_id, role, content, metadata_json)
               VALUES (?, ?, 'assistant', ?, ?)""",
            (
                assistant_msg_id,
                conversation_id,
                answer,
                json.dumps(
                    {"sources": sources, "rag_trace": rag_trace or {}},
                    ensure_ascii=False,
                ),
            ),
        )
        await db.commit()
    finally:
        await db.close()


@router.post("/chat/stream")
async def chat_stream(req: SendMessageRequest, request: Request):
    history = await _load_chat_history(req.conversation_id)

    db = await get_db()
    user_msg_id = str(uuid.uuid4())
    try:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'user', ?)",
            (user_msg_id, req.conversation_id, req.content),
        )
        await db.commit()
    finally:
        await db.close()

    async def event_generator():
        full_answer = ""
        final_sources: list[dict] = []
        final_trace: dict = {}
        output_queue: asyncio.Queue = asyncio.Queue()

        class _RagStepProxy:
            def put_nowait(self, step):
                output_queue.put_nowait({"event": "rag_step", "data": {"rag_step": step}})

        set_rag_step_queue(_RagStepProxy())

        state = {
            "messages": history,
            "question": req.content,
            "paper_id": req.paper_id,
            "route": None,
            "context": [],
            "docs": [],
            "rag_trace": {},
            "rewrite_count": 0,
            "answer": "",
            "sources": [],
        }

        async def _graph_worker():
            nonlocal full_answer, final_sources, final_trace
            try:
                async for event in rag_graph.astream_events(state, version="v2"):
                    kind = event.get("event", "")

                    if kind == "on_chain_end" and event.get("name") == "router":
                        route = event.get("data", {}).get("output", {}).get("route", "")
                        if route:
                            await output_queue.put(
                                {"event": "route", "data": {"route": route}}
                            )

                    if kind == "on_chain_end" and event.get("name") in {
                        "answer_agentic",
                        "answer_chat",
                        "answer_summary_agentic",
                    }:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            final_sources = output.get("sources", []) or []
                            final_trace = output.get("rag_trace", {}) or final_trace

                    if kind == "on_chain_end" and event.get("name") in {
                        "retrieve_initial",
                        "retrieve_expanded",
                        "retrieve_summary",
                        "grade_documents",
                        "rewrite_question",
                    }:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and output.get("rag_trace"):
                            final_trace = output["rag_trace"]

                    if kind == "on_chat_model_stream":
                        node = event.get("metadata", {}).get("langgraph_node", "")
                        if node not in {"answer_agentic", "answer_chat", "answer_summary_agentic"}:
                            continue

                        chunk = event.get("data", {}).get("chunk")
                        if chunk and getattr(chunk, "content", None):
                            text = _chunk_text(chunk.content)
                            if not text:
                                continue
                            full_answer += text
                            await output_queue.put(
                                {"event": "token", "data": {"content": text}}
                            )
            except Exception as exc:
                logger.error("Stream worker error: %s", exc)
                await output_queue.put(
                    {"event": "error", "data": {"error": "An internal error occurred"}}
                )
            finally:
                await output_queue.put(None)

        graph_task = asyncio.create_task(_graph_worker())

        try:
            while True:
                item = await output_queue.get()
                if item is None:
                    break
                yield {
                    "event": item["event"],
                    "data": json.dumps(item["data"], ensure_ascii=False),
                }
                if await request.is_disconnected():
                    graph_task.cancel()
                    break

            if not graph_task.done():
                await graph_task

            yield {
                "event": "trace",
                "data": json.dumps({"rag_trace": final_trace}, ensure_ascii=False),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "answer": full_answer,
                        "sources": final_sources,
                        "rag_trace": final_trace,
                    },
                    ensure_ascii=False,
                ),
            }
            await _save_assistant_message(
                req.conversation_id,
                full_answer,
                final_sources,
                final_trace,
            )
        except asyncio.CancelledError:
            graph_task.cancel()
            raise
        except GeneratorExit:
            graph_task.cancel()
            raise
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"error": "An internal error occurred"}),
            }
        finally:
            set_rag_step_queue(None)

    return EventSourceResponse(event_generator())


@router.post("/chat")
async def chat_non_stream(req: SendMessageRequest):
    history = await _load_chat_history(req.conversation_id)

    db = await get_db()
    user_msg_id = str(uuid.uuid4())
    try:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'user', ?)",
            (user_msg_id, req.conversation_id, req.content),
        )
        await db.commit()
    finally:
        await db.close()
    state = {
        "messages": history,
        "question": req.content,
        "paper_id": req.paper_id,
        "route": None,
        "context": [],
        "docs": [],
        "rag_trace": {},
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    }

    result = await rag_graph.ainvoke(state)
    answer = result.get("answer", "")
    sources = result.get("sources", []) or []
    rag_trace = result.get("rag_trace", {}) or {}
    await _save_assistant_message(req.conversation_id, answer, sources, rag_trace)

    return {
        "message_id": user_msg_id,
        "response": answer,
        "sources": sources,
        "rag_trace": rag_trace,
    }
