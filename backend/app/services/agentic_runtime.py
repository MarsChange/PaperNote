"""Runtime hooks for Agentic RAG observability.

The pattern mirrors superMew's cross-thread RAG step streaming: synchronous
retrieval nodes can emit progress into the async SSE loop without blocking it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

_RAG_STEP_QUEUE: Any = None
_RAG_STEP_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_rag_step_queue(queue: Any):
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue is None:
        _RAG_STEP_LOOP = None
        return
    try:
        _RAG_STEP_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        _RAG_STEP_LOOP = asyncio.get_event_loop()


def emit_rag_step(icon: str, label: str, detail: str = ""):
    if _RAG_STEP_QUEUE is None or _RAG_STEP_LOOP is None:
        return
    step = {"icon": icon, "label": label, "detail": detail}
    try:
        if not _RAG_STEP_LOOP.is_closed():
            _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
    except Exception:
        pass
