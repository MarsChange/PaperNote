import os
from pathlib import Path

import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(BACKEND_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAPERNOTE_", extra="ignore")

    # App
    app_name: str = "PaperNote"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    db_path: Path = base_dir / "data" / "papernote.db"
    milvus_dir: Path = base_dir / "data" / "milvus"
    milvus_uri: str = os.getenv("MILVUS_URI") or str(
        base_dir / "data" / "milvus" / "papernote.db"
    )
    milvus_collection: str = os.getenv("MILVUS_COLLECTION") or "papernote_blocks"

    # Frontend
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # LLM provider: qwen / kimi / openai / anthropic / gemini
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "")
    translation_target_language: str = os.getenv(
        "TRANSLATION_TARGET_LANGUAGE", "简体中文"
    )
    enable_multimodal_answers: bool = (
        os.getenv("ENABLE_MULTIMODAL_ANSWERS", "true").lower() == "true"
    )
    enable_multimodal_enrichment: bool = (
        os.getenv("ENABLE_MULTIMODAL_ENRICHMENT", "true").lower() == "true"
    )
    multimodal_enrichment_llm_limit: int = int(
        os.getenv("MULTIMODAL_ENRICHMENT_LLM_LIMIT", "8")
    )

    # API Keys (set via environment variables)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    kimi_api_key: str = os.getenv("KIMI_API_KEY", "")
    kimi_base_url: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Embeddings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "qwen")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")

    # MinerU API
    mineru_api_key: str = os.getenv("MINERU_API_KEY", "")
    mineru_model_version: str = os.getenv("MINERU_MODEL_VERSION", "vlm")
    mineru_language: str = os.getenv("MINERU_LANGUAGE", "ch")


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.milvus_dir.mkdir(parents=True, exist_ok=True)
