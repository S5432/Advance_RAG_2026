"""FastAPI Document Q&A service (RAG + Pinecone + Gemini Flash).

Run:  uvicorn src.main:app --reload
Tenant/user identity comes from X-Tenant-ID / X-User-ID headers (in production, derive
them from a verified JWT instead of trusting client headers).
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from .config import get_settings
from .pinecone_service import PineconeService
from google import genai

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile

from .chunking import chunk_documents
from .loaders import LoaderError, load_document
from .pinecone_service import upsert_chunks_to_pinecone, PineconeService
from .rag_chain import answer_question, retrieve_chunks
from .schemas import (
    AskRequest, AskResponse, DocumentStatus, HealthResponse, SearchHit,
    SearchRequest, SearchResponse, UploadResponse, UrlIngestRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docqa")

EXT_TO_TYPE = {
    ".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".markdown": "markdown",
    ".csv": "csv", ".html": "html", ".htm": "html",
}
MAGIC = {"pdf": b"%PDF", "docx": b"PK"}  # DOCX is a zip container


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize external clients once during application startup."""

    settings = get_settings()

    app.state.settings = settings

    # Initialize Pinecone vector database
    app.state.pinecone = PineconeService.from_settings(settings)

    # Initialize Gemini client
    app.state.gemini = genai.Client(
        api_key=settings.gemini_api_key
    )

    # Temporary in-memory document registry
    app.state.registry = {}

    logger.info("Application startup completed successfully.")

    yield

    logger.info("Application shutdown completed.")

app = FastAPI(title="Document Q&A Assistant", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers
def identity(
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    if not x_tenant_id or not x_user_id:
        raise HTTPException(401, "X-Tenant-ID and X-User-ID headers are required")
    return {"tenant_id": x_tenant_id.strip(), "user_id": x_user_id.strip()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _doc_id(tenant_id: str, payload: bytes) -> str:
    """Deterministic and tenant-scoped: same file + same tenant => same ID (dedup),
    different tenants never collide."""
    return "doc_" + hashlib.sha256(tenant_id.encode() + b"\0" + payload).hexdigest()[:12]


# def _tenant_filter(ident: dict, document_ids: list[str] | None) -> dict:
#     f: dict = {}
#     if document_ids:
#         f["document_id"] = {"$in": document_ids}
#     f["tenant_id"] = ident["tenant_id"]  # set LAST: callers can never override it
#     return f


def _tenant_filter(
    ident: dict,
    document_ids: list[str] | None,
) -> dict:
    """
    Always enforce tenant isolation.

    A caller cannot override the tenant_id filter.
    """

    filters = {
        "tenant_id": {
            "$eq": ident["tenant_id"]
        }
    }

    if document_ids:
        filters["document_id"] = {
            "$in": document_ids
        }

    return filters


def _register(request: Request, doc_id: str, name: str, source_type: str, ident: dict) -> dict | None:
    """Create a registry entry; return the existing one if this is a duplicate."""
    registry = request.app.state.registry
    existing = registry.get(doc_id)
    if existing and existing["status"] in ("processing", "completed"):
        return existing
    registry[doc_id] = {
        "document_id": doc_id, "document_name": name, "source_type": source_type,
        "status": "processing", "chunks": 0, "pages": 0, "ocr_used": False,
        "low_ocr_quality_pages": 0, "error": None, "created_at": _now(),
        "tenant_id": ident["tenant_id"],
    }
    return None


# def ingest_document(state, doc_id: str, path_or_url: str, source_type: str, name: str,
#                     ident: dict, cleanup_path: str | None = None) -> None:
#     """Background job: load -> chunk -> embed -> upsert. Never raises; records status."""
#     entry = state.registry[doc_id]
#     upserted: list[str] = []
#     try:
#         docs = load_document(path_or_url, source_type, doc_id)
#         if not any(d["text"].strip() for d in docs):
#             raise LoaderError("No extractable text found (empty or unreadable document)")
#         for d in docs:  # keep the user-facing name (not the temp file name)
#             d["metadata"]["source"] = name

#         s = state.settings
#         chunks = chunk_documents(
#             docs, s.chunk_size, s.chunk_overlap,
#             extra_metadata={
#                 "document_name": name, "uploaded_by": ident["user_id"],
#                 "tenant_id": ident["tenant_id"], "created_at": entry["created_at"],
#             },
#         )
#         # embeddings = embed_chunks(chunks, state.embedding_model)
#         embeddings = embed_chunks(chunks)
#         upserted = upsert_chunks_to_pinecone(chunks, embeddings, state.pinecone, s.pinecone_namespace)

#         entry.update(
#             status="completed", chunks=len(chunks), pages=len({d["metadata"]["page_number"] for d in docs}),
#             ocr_used=any(d["metadata"].get("ocr") for d in docs),
#             low_ocr_quality_pages=sum(1 for d in docs if d["metadata"].get("low_ocr_quality")),
#         )
#         logger.info("Ingested %s (%d chunks)", doc_id, len(chunks))
#     except Exception as exc:
#         logger.exception("Ingestion failed for %s", doc_id)
#         if upserted:  # roll back partial writes
#             try:
#                 state.pinecone.delete_ids(upserted, namespace=state.settings.pinecone_namespace)
#             except Exception:
#                 logger.exception("Rollback failed for %s", doc_id)
#         msg = str(exc) if isinstance(exc, LoaderError) else "Ingestion failed; see server logs"
#         entry.update(status="failed", error=msg)
#     finally:
#         if cleanup_path and os.path.exists(cleanup_path):
#             os.remove(cleanup_path)


def ingest_document(
    state,
    doc_id: str,
    path_or_url: str,
    source_type: str,
    name: str,
    ident: dict,
    cleanup_path: str | None = None,
) -> None:
    """
    Background ingestion pipeline:

    Load -> Chunk -> Pinecone Documents API upsert.

    Pinecone generates document embeddings internally.
    """

    entry = state.registry[doc_id]
    upserted: list[str] = []

    try:
        # 1. Load document
        docs = load_document(
            path_or_url,
            source_type,
            doc_id,
        )

        if not any(
            document["text"].strip()
            for document in docs
        ):
            raise LoaderError(
                "No extractable text found "
                "(empty or unreadable document)"
            )

        # 2. Preserve the original user-facing document name
        for document in docs:
            document["metadata"]["source"] = name

        settings = state.settings

        # 3. Create chunks and attach application metadata
        chunks = chunk_documents(
            documents=docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            extra_metadata={
                "document_name": name,
                "uploaded_by": ident["user_id"],
                "tenant_id": ident["tenant_id"],
                "created_at": entry["created_at"],
            },
        )

        if not chunks:
            raise LoaderError(
                "Document did not produce any valid text chunks."
            )

        # 4. Upsert documents directly.
        # No embed_chunks() call is needed.
        upserted = upsert_chunks_to_pinecone(
            chunks=chunks,
            pinecone_service=state.pinecone,
            namespace=settings.pinecone_namespace,
        )

        # 5. Update document status
        page_numbers = {
            document["metadata"].get("page_number")
            for document in docs
            if document["metadata"].get("page_number") is not None
        }

        entry.update(
            status="completed",
            chunks=len(upserted),
            pages=len(page_numbers),
            ocr_used=any(
                document["metadata"].get("ocr")
                for document in docs
            ),
            low_ocr_quality_pages=sum(
                1
                for document in docs
                if document["metadata"].get("low_ocr_quality")
            ),
        )

        logger.info(
            "Successfully ingested document=%s chunks=%d",
            doc_id,
            len(upserted),
        )

    except Exception as exc:
        logger.exception(
            "Ingestion failed for document=%s",
            doc_id,
        )

        # Roll back any documents already written.
        if upserted:
            try:
                state.pinecone.delete_ids(
                    ids=upserted,
                    namespace=state.settings.pinecone_namespace,
                )
            except Exception:
                logger.exception(
                    "Pinecone rollback failed for document=%s",
                    doc_id,
                )

        message = (
            str(exc)
            if isinstance(exc, LoaderError)
            else "Ingestion failed; see server logs"
        )

        entry.update(
            status="failed",
            error=message,
        )

    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.remove(cleanup_path)



# ---------------------------------------------------------------- endpoints
@app.post("/documents/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    ident: dict = Depends(identity),
):
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    source_type = EXT_TO_TYPE.get(ext)
    if not source_type:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(EXT_TO_TYPE)}")

    max_bytes = request.app.state.settings.max_upload_mb * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"File exceeds {request.app.state.settings.max_upload_mb} MB")
    if not payload:
        raise HTTPException(400, "Empty file")
    if source_type in MAGIC and not payload.startswith(MAGIC[source_type]):
        raise HTTPException(400, f"File content does not look like a valid {source_type.upper()}")

    doc_id = _doc_id(ident["tenant_id"], payload)
    existing = _register(request, doc_id, name, source_type, ident)
    if existing:
        return UploadResponse(document_id=doc_id, document_name=name, status="duplicate",
                              message="This document was already uploaded")

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
    background.add_task(ingest_document, request.app.state, doc_id, tmp_path, source_type, name, ident, tmp_path)
    return UploadResponse(document_id=doc_id, document_name=name, status="processing",
                          message="Ingestion started")


@app.post("/documents/url", response_model=UploadResponse, status_code=202)
def ingest_url(
    body: UrlIngestRequest, request: Request, background: BackgroundTasks,
    ident: dict = Depends(identity),
):
    url = str(body.url)
    doc_id = _doc_id(ident["tenant_id"], url.encode())
    existing = _register(request, doc_id, url, "url", ident)
    if existing:
        return UploadResponse(document_id=doc_id, document_name=url, status="duplicate",
                              message="This URL was already ingested")
    background.add_task(ingest_document, request.app.state, doc_id, url, "url", url, ident, None)
    return UploadResponse(document_id=doc_id, document_name=url, status="processing",
                          message="Ingestion started")


@app.get("/documents/{document_id}/status", response_model=DocumentStatus)
def document_status(document_id: str, request: Request, ident: dict = Depends(identity)):
    entry = request.app.state.registry.get(document_id)
    # 404 (not 403) for other tenants' documents: don't reveal that the ID exists
    if not entry or entry["tenant_id"] != ident["tenant_id"]:
        raise HTTPException(404, "Document not found")
    return DocumentStatus(**{k: v for k, v in entry.items() if k != "tenant_id"})



@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    ident: dict = Depends(identity),
):
    st = request.app.state

    try:
        return answer_question(
            query=body.question,
            pinecone_service=st.pinecone,
            gemini_client=st.gemini,
            top_k=body.top_k,
            model=st.settings.gemini_model,
            metadata_filter=_tenant_filter(
                ident,
                body.document_ids,
            ),
        )

    except Exception:
        logger.exception("Ask endpoint failed")

        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable",
        )



@app.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    request: Request,
    ident: dict = Depends(identity),
):
    st = request.app.state

    try:
        chunks = retrieve_chunks(
            query=body.question,
            pinecone_service=st.pinecone,
            top_k=body.top_k,
            metadata_filter=_tenant_filter(
                ident,
                body.document_ids,
            ),
        )

    except Exception:
        logger.exception("Search endpoint failed")

        raise HTTPException(
            status_code=502,
            detail="Search is temporarily unavailable",
        )

    return SearchResponse(
        query=body.question,
        results=[
            SearchHit(
                **{
                    k: chunk[k]
                    for k in (
                        "chunk_id",
                        "score",
                        "document_name",
                        "page_number",
                        "section",
                        "text",
                    )
                }
            )
            for chunk in chunks
        ],
    )



@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    st = request.app.state
    try:
        ok = bool(st.pinecone.ping())
    except Exception:
        ok = False
    return HealthResponse(status="ok" if ok else "degraded", pinecone=ok, model=st.settings.gemini_model)
