"""Pydantic request / response models."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

DocStatus = Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]


class UrlIngestRequest(BaseModel):
    url: HttpUrl


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


class Citation(BaseModel):
    source_id: int
    document_name: str
    page_number: int | None = None
    section: str | None = None
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieved_chunks: int


class SearchRequest(AskRequest):
    pass


class SearchHit(BaseModel):
    source_id: int
    chunk_id: str
    score: float
    document_name: str
    page_number: int | None = None
    section: str | None = None
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    database: bool
    redis: bool
    pinecone: bool
    model: str
