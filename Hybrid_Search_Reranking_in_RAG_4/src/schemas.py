"""Pydantic request / response models."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

DocStatus = Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]


class UrlIngestRequest(BaseModel):
    url: HttpUrl
    department: str | None = Field(default=None, max_length=128)
    access_level: Literal["public", "internal", "confidential"] = "internal"


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocStatus
    task_id: str | None = None
    duplicate: bool = False
    message: str = ""


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    source_type: str
    status: DocStatus
    progress: int = 0
    total_chunks: int = 0
    task_id: str | None = None
    retries: int = 0
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    source_type: str
    source_url: str | None = None
    uploaded_by: str
    department: str | None = None
    access_level: str = "internal"
    status: DocStatus

    total_chunks: int = 0

    task_id: str | None = None
    progress: int = 0
    retries: int = 0
    error_message: str | None = None

    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentListItem]

    

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    # whitelisted keys only (see src/filters.py): source_type, department, access_level, date_from, date_to, ...
    filters: dict[str, str | list[str]] | None = None


class Citation(BaseModel):
    source_id: int
    document_name: str
    page_number: int | None = None
    section: str | None = None
    chunk_id: str
    score: float  # rerank score (or RRF score when the reranker fell back)
    rerank_score: float | None = None
    rrf_score: float | None = None
    dense_score: float | None = None
    keyword_score: float | None = None
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieved_chunks: int
    retrieval: dict[str, Any] | None = None  # stage counts + latencies (dense / keyword / fused / rerank)


class SearchRequest(AskRequest):
    # hybrid_rerank = production pipeline; the others expose a single stage (used by the Phase 7 evaluation)
    mode: Literal["dense", "keyword", "hybrid", "hybrid_rerank"] = "hybrid_rerank"


class SearchHit(BaseModel):
    source_id: int
    chunk_id: str
    score: float  # the score that produced the ordering for the chosen mode
    document_id: str | None = None
    document_name: str
    page_number: int | None = None
    section: str | None = None
    text: str
    rerank_score: float | None = None
    rrf_score: float | None = None
    dense_score: float | None = None
    keyword_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str = "hybrid_rerank"
    results: list[SearchHit]
    stats: dict[str, Any] | None = None


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool = True
    vectors_removed: bool
    purge_queued: bool = False  # inline removal failed -> retried in the background
    chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    database: bool
    redis: bool
    pinecone: bool
    model: str
