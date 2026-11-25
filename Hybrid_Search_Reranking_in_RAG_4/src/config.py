


"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    """Day 11: Pinecone index = llama-text-embed-v2 / 1024 dims / cosine (validated on load)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------
    # Gemini configuration
    # -------------------------------------------------

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta"
    )

    # -------------------------------------------------
    # Pinecone configuration
    # -------------------------------------------------

    pinecone_api_key: str
    pinecone_index_name: str = "advanced-rag-document-qa"

    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    pinecone_namespace: str = "documents"

    # -------------------------------------------------
    # Embedding configuration
    # -------------------------------------------------

        # -------------------------------------------------
    # Embedding configuration
    # -------------------------------------------------

    embedding_provider: Literal["pinecone"] = "pinecone"

    embedding_model: str = "llama-text-embed-v2"

    embedding_dimension: int = Field(default=1024, ge=1)

    embedding_metric: Literal["cosine"] = "cosine"

    embedding_input_type_passage: Literal["passage"] = "passage"

    embedding_input_type_query: Literal["query"] = "query"

    embedding_truncate: Literal["END", "NONE"] = "END"

    embedding_batch_size: int = Field(default=64, ge=1, le=96)

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        allowed_dimensions = {1024}  # Day 11 requirement

        if value not in allowed_dimensions:
            raise ValueError(
                f"embedding_dimension must be one of {sorted(allowed_dimensions)}"
            )

        return value


    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if value != "llama-text-embed-v2":
            raise ValueError("embedding_model must be 'llama-text-embed-v2'")
        return value

    # -------------------------------------------------
    # Retrieval configuration (Day 11)
    # -------------------------------------------------

    top_k: int = 5                      # legacy Day 10 name = final top-k
    dense_top_k: int = Field(default=20, ge=1, le=100)
    keyword_top_k: int = Field(default=20, ge=1, le=100)
    hybrid_top_k: int = Field(default=20, ge=1, le=100)
    final_top_k: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1)
    rerank_model: str = "bge-reranker-v2-m3"   # Pinecone hosted reranker
    # chunks scoring below this are treated as irrelevant (0 disables); tune with the Phase 7 evaluation
    rerank_min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    chunking_strategy: Literal["fixed", "recursive", "heading", "parent_child"] = "recursive"

    # -------------------------------------------------
    # Document processing
    # -------------------------------------------------

    chunk_size: int = 800

    chunk_overlap: int = 150

    max_upload_mb: int = 20


    # -------------------------------------------------
    # Day 10: Celery / Redis / PostgreSQL / uploads
    # -------------------------------------------------

    redis_url: str = "redis://localhost:6379/0"

    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_db"
    )

    # Must be a volume shared by the api and worker containers.
    upload_dir: str = "./uploads"

    # Documents stuck in PENDING/PROCESSING longer than this are re-queued.
    stuck_after_minutes: int = 15


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
