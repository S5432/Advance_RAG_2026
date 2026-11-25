


"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):

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
    gemini_model: str = "gemini-3.5-flash-lite"

    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta"
    )

    # -------------------------------------------------
    # Pinecone configuration
    # -------------------------------------------------

    pinecone_api_key: str
    pinecone_index_name: str = "rag-document-qa"

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
        allowed_dimensions = {1024, 2048, 3072}

        if value not in allowed_dimensions:
            raise ValueError(
                f"embedding_dimension must be one of {sorted(allowed_dimensions)}"
            )

        return value


    # -------------------------------------------------
    # Retrieval configuration
    # -------------------------------------------------

    top_k: int = 5

    # -------------------------------------------------
    # Document processing
    # -------------------------------------------------

    chunk_size: int = 800

    chunk_overlap: int = 150

    max_upload_mb: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
