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
    milvus_grpc_keepalive_time_ms: int = int(
        os.getenv("MILVUS_GRPC_KEEPALIVE_TIME_MS", "120000")
    )
    milvus_grpc_keepalive_timeout_ms: int = int(
        os.getenv("MILVUS_GRPC_KEEPALIVE_TIMEOUT_MS", "20000")
    )
    milvus_grpc_keepalive_permit_without_calls: bool = (
        os.getenv("MILVUS_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "false").lower()
        == "true"
    )

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
    multimodal_answer_image_limit: int = int(
        os.getenv("MULTIMODAL_ANSWER_IMAGE_LIMIT", "4")
    )
    enable_multimodal_enrichment: bool = (
        os.getenv("ENABLE_MULTIMODAL_ENRICHMENT", "true").lower() == "true"
    )
    multimodal_enrichment_llm_limit: int = int(
        os.getenv("MULTIMODAL_ENRICHMENT_LLM_LIMIT", "8")
    )

    # Agentic RAG (adapted from superMew, without auth/RBAC/frontend stack)
    agentic_rag_enabled: bool = (
        os.getenv("AGENTIC_RAG_ENABLED", "true").lower() == "true"
    )
    bm25_state_path: Path = Path(
        os.getenv("BM25_STATE_PATH") or base_dir / "data" / "bm25_state.json"
    )
    agentic_collection_prefix: str = os.getenv(
        "AGENTIC_MILVUS_COLLECTION_PREFIX", "papernote_agentic_blocks"
    )
    dense_embedding_dim: int = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
    rerank_model: str = os.getenv("RERANK_MODEL", "")
    rerank_binding_host: str = os.getenv("RERANK_BINDING_HOST", "")
    rerank_api_key: str = os.getenv("RERANK_API_KEY", "")
    auto_merge_enabled: bool = (
        os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
    )
    auto_merge_threshold: int = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
    leaf_retrieve_level: int = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))
    agentic_candidate_multiplier: int = int(os.getenv("AGENTIC_CANDIDATE_MULTIPLIER", "3"))
    rag_grade_model: str = os.getenv("RAG_GRADE_MODEL", "")
    rag_max_rewrites: int = int(os.getenv("RAG_MAX_REWRITES", "1"))
    conversation_summary_trigger_messages: int = int(
        os.getenv("CONVERSATION_SUMMARY_TRIGGER_MESSAGES", "12")
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
    enable_mineru_figure_crops: bool = (
        os.getenv("ENABLE_MINERU_FIGURE_CROPS", "true").lower() == "true"
    )
    delete_mineru_raw_images_after_figure_crops: bool = (
        os.getenv("DELETE_MINERU_RAW_IMAGES_AFTER_FIGURE_CROPS", "true").lower()
        == "true"
    )
    figure_crop_dpi: int = int(os.getenv("FIGURE_CROP_DPI", "220"))
    figure_crop_padding: float = float(os.getenv("FIGURE_CROP_PADDING", "0.012"))


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.milvus_dir.mkdir(parents=True, exist_ok=True)
