"""Settings API for managing LLM provider configuration."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.core.config import settings

router = APIRouter(tags=["settings"])

PROVIDERS = [
    {
        "id": "openai",
        "name": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.5", "gpt-5.4", "gpt-5.3"],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "default_base_url": "https://api.anthropic.com",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"],
    },
    {
        "id": "qwen",
        "name": "Qwen",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.6-plus", "qwen3.6-flash"],
    },
    {
        "id": "kimi",
        "name": "Kimi",
        "default_base_url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k2.6", "kimi-k2.5"],
    },
    {
        "id": "gemini",
        "name": "Gemini",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.0-flash", "gemini-2.0-pro"],
    },
]


class UpdateSettingsRequest(BaseModel):
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


@router.get("/settings/providers")
async def get_providers():
    return {"providers": PROVIDERS}


@router.get("/settings")
async def get_settings():
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "milvus_uri": settings.milvus_uri,
        "milvus_collection": settings.milvus_collection,
    }


@router.put("/settings")
async def update_settings(req: UpdateSettingsRequest):
    """Update runtime settings (in-memory only, not persisted to .env)."""
    if req.llm_provider:
        settings.llm_provider = req.llm_provider
    if req.model:
        settings.llm_model = req.model
    if req.api_key:
        # Set the appropriate API key based on provider
        provider = req.llm_provider or settings.llm_provider
        key_map = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "qwen": "qwen_api_key",
            "kimi": "kimi_api_key",
            "gemini": "gemini_api_key",
        }
        attr = key_map.get(provider)
        if attr:
            setattr(settings, attr, req.api_key)
    if req.base_url:
        provider = req.llm_provider or settings.llm_provider
        url_map = {
            "openai": "openai_base_url",
            "qwen": "qwen_base_url",
            "kimi": "kimi_base_url",
            "gemini": "gemini_base_url",
        }
        attr = url_map.get(provider)
        if attr:
            setattr(settings, attr, req.base_url)

    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }
