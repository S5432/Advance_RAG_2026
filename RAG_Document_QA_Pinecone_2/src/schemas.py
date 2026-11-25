# """Pydantic request / response models."""
# from __future__ import annotations

# from typing import Literal

# from pydantic import BaseModel, Field, HttpUrl

# Status = Literal["processing", "completed", "failed", "duplicate"]


# class UrlIngestRequest(BaseModel):
#     url: HttpUrl


# class UploadResponse(BaseModel):
#     document_id: str
#     document_name: str
#     status: Status
#     message: str = ""


# class DocumentStatus(BaseModel):
#     document_id: str
#     document_name: str
#     source_type: str
#     status: Status
#     chunks: int = 0
#     pages: int = 0
#     ocr_used: bool = False
#     low_ocr_quality_pages: int = 0
#     error: str | None = None
#     created_at: str


# class AskRequest(BaseModel):
#     question: str = Field(min_length=3, max_length=1000)
#     top_k: int = Field(default=5, ge=1, le=10)
#     document_ids: list[str] | None = Field(default=None, max_length=20)


# class Citation(BaseModel):
#     source_id: int
#     document_name: str
#     page_number: int | None = None
#     chunk_id: str
#     score: float
#     text: str


# class AskResponse(BaseModel):
#     question: str
#     answer: str
#     citations: list[Citation]
#     model: str
#     retrieved_chunks: int


# class SearchRequest(AskRequest):
#     pass


# class SearchHit(BaseModel):
#     chunk_id: str
#     score: float
#     document_name: str
#     page_number: int | None = None
#     section: str | None = None
#     text: str


# class SearchResponse(BaseModel):
#     query: str
#     results: list[SearchHit]


# class HealthResponse(BaseModel):
#     status: str
#     pinecone: bool
#     model: str



"""Pydantic request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


Status = Literal[
    "processing",
    "completed",
    "failed",
    "duplicate",
]


class UrlIngestRequest(BaseModel):
    url: HttpUrl


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    status: Status
    message: str = ""


class DocumentStatus(BaseModel):
    document_id: str
    document_name: str
    source_type: str
    status: Status

    chunks: int = 0
    pages: int = 0

    ocr_used: bool = False
    low_ocr_quality_pages: int = 0

    error: str | None = None
    created_at: str


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    document_ids: list[str] | None = Field(
        default=None,
        max_length=20,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:

        if value is None:
            return None

        cleaned = [
            document_id.strip()
            for document_id in value
            if document_id.strip()
        ]

        if not cleaned:
            raise ValueError(
                "document_ids cannot contain only empty values."
            )

        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                "Duplicate document IDs are not allowed."
            )

        return cleaned


class Citation(BaseModel):
    source_id: int
    document_name: str

    page_number: int | None = None

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
    pinecone: bool
    model: str
