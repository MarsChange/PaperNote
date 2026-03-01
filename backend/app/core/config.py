from pydantic_settings import BaseSettings
from pathlib import Path
import os
import dotenv

dotenv.load_dotenv()

class Settings(BaseSettings):
    # App
    app_name: str = "PaperNote"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    db_path: Path = base_dir / "data" / "papernote.db"
    chroma_dir: Path = base_dir / "data" / "chroma"

    # LLM provider: qwen / kimi / minimax / openai / anthropic / gemini
    llm_provider: str = "openai"

    # API Keys (set via environment variables)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    qwen_base_url: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    kimi_api_key: str = os.getenv("KIMI_API_KEY", "")
    kimi_base_url: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    minimax_base_url: str = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    # Embedding (Qwen text-embedding-v4)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    # MinerU API
    mineru_api_key: str = os.getenv("MINERU_API_KEY", "")

settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)