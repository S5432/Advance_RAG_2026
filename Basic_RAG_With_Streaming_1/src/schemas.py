from pydantic import BaseModel, Field, field_validator

DOC_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20, description="defaults to TOP_K from settings")
    document_id: str | None = Field(None, pattern=DOC_ID_PATTERN)

    @field_validator("query")
    @classmethod
    def _q(cls, v: str) -> str:
        if not v.strip(): raise ValueError("query must not be empty")
        return v.strip()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)
    document_id: str | None = Field(None, pattern=DOC_ID_PATTERN)

    @field_validator("question")
    @classmethod
    def _q(cls, v: str) -> str:
        if not v.strip(): raise ValueError("question must not be empty")
        return v.strip()

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page: int | None
    chunk_index: int | None = None
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]

class Citation(BaseModel):
    document_id: str
    document_name: str
    page: int | None
    chunk_id: str

class RAGAnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    retrieved_chunks: int
    confidence: str = "low"

class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    page_count: int
    chunk_count: int
    pages_without_text: list[int] = []
    status: str

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    page_count: int | None = None
    chunk_count: int
    uploaded_at: str | None = None

class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    status: str = "deleted"
