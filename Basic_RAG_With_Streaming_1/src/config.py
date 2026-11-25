from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr = SecretStr("")       # optional at startup: /ask answers 503 if missing, /search still works
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: float = 30.0
    gemini_max_retries: int = 3
    gemini_thinking_budget: int | None = 0

    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 10
    min_relevance_score: float = 0.25               # below this best-chunk cosine score we refuse without calling the LLM
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    return Settings()
