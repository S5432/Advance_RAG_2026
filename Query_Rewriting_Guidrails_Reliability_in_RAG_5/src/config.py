


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
    gemini_model: str = "gemini-3.5-flash-lite"

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
    # Query rewriting + multi-query retrieval
    # -------------------------------------------------
    # Gemini rewrites the question into a few search queries; each is searched (dense + BM25) and all
    # ranked lists are fused with RRF. Any failure falls back to the plain single-query pipeline.
    query_rewrite_enabled: bool = True
    multi_query_count: int = Field(default=3, ge=1, le=5)          # rewritten queries (the original is always searched too)
    query_rewrite_model: str | None = None                         # None -> use gemini_model
    query_rewrite_timeout_s: float = Field(default=10.0, gt=0, le=60)
    original_query_weight: float = Field(default=1.5, ge=1.0, le=5.0)  # RRF weight of the user's own wording

    # -------------------------------------------------
    # LLM resilience: retries, timeouts, fallback model, usage logging
    # -------------------------------------------------
    # Fallback model(s) tried, in order, when the primary model still fails after its retries.
    # Comma-separated for a chain; empty = no fallback.
    gemini_fallback_model: str | None = "gemini-3.1-flash-lite"
    llm_max_retries: int = Field(default=2, ge=0, le=5)             # retries per model (after the first try)
    llm_retry_backoff_s: float = Field(default=1.0, ge=0, le=30)    # delay before retry n = base * 2**n (+ jitter)
    llm_retry_backoff_max_s: float = Field(default=10.0, ge=0, le=60)
    llm_timeout_s: float = Field(default=30.0, gt=0, le=300)        # one blocking Gemini attempt
    llm_stream_timeout_s: float = Field(default=60.0, gt=0, le=600) # one streaming attempt
    llm_total_timeout_s: float = Field(default=90.0, gt=0, le=900)  # all attempts + backoff + fallbacks together
    llm_usage_persist: bool = True                                  # also write usage rows to the llm_usage table

    # -------------------------------------------------
    # Guardrails: PII redaction, prompt-injection checks, output validation (see src/guardrails.py)
    # -------------------------------------------------
    pii_redaction_enabled: bool = True          # redact PII in the question (and the answer) before it goes anywhere
    redact_context: bool = False                # also redact PII inside retrieved chunks sent to Gemini
    pii_types: str = ""                         # comma-separated subset (email,phone,...); empty = all types
    injection_check_enabled: bool = True
    injection_action: Literal["block", "flag", "off"] = "block"   # what to do with a suspicious question
    injection_threshold: int = Field(default=5, ge=1, le=20)
    context_injection_action: Literal["drop", "neutralize", "off"] = "drop"  # poisoned retrieved chunks
    output_validation_enabled: bool = True
    max_answer_chars: int = Field(default=6000, ge=100, le=50000)

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
