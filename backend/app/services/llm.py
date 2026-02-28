"""Multi-provider LLM factory."""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.core.config import settings

# Provider → (base_url_setting, api_key_setting, default_model)
PROVIDER_CONFIG = {
    "openai": ("openai_base_url", "openai_api_key", "gpt-4o"),
    "anthropic": ("openai_base_url", "anthropic_api_key", "claude-sonnet-4-20250514"),
    "qwen": ("qwen_base_url", "qwen_api_key", "qwen-max"),
    "kimi": ("kimi_base_url", "kimi_api_key", "moonshot-v1-128k"),
    "minimax": ("minimax_base_url", "minimax_api_key", "abab6.5s-chat"),
    "gemini": ("gemini_base_url", "gemini_api_key", "gemini-2.0-flash"),
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Create a LangChain chat model for the given provider.

    All providers use OpenAI-compatible API format (ChatOpenAI),
    since Qwen/Kimi/MiniMax/Gemini all expose OpenAI-compatible endpoints.
    """
    provider = provider or settings.llm_provider
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        raise ValueError(f"Unknown LLM provider: {provider}")

    base_url_attr, api_key_attr, default_model = config

    final_api_key = api_key or getattr(settings, api_key_attr, "")
    final_base_url = base_url or getattr(settings, base_url_attr, "")
    final_model = model or default_model

    if not final_api_key:
        raise ValueError(f"API key not configured for provider: {provider}")

    return ChatOpenAI(
        model=final_model,
        api_key=final_api_key,
        base_url=final_base_url,
        temperature=temperature,
        streaming=streaming,
    )
