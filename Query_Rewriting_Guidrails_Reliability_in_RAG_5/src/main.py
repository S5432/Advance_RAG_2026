"""Async Document Q&A API: Celery ingestion + Pinecone retrieval + Gemini streaming over SSE.

Run:  uvicorn src.main:app --reload      (plus a Celery worker, see README)
Identity comes from X-Tenant-ID / X-User-ID headers; in production derive them from a verified JWT.
"""
from __future__ import annotations

import hashlib
import dataclasses
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from . import database
from .filters import ACCESS_LEVELS, filters_from_request
from .gemini_client import LLMConfig
from .guardrails import BLOCKED_MESSAGE, GuardrailBlocked, GuardrailConfig, InputCheck, check_question
from .loaders import LoaderError, _assert_public_url
from .models import Document, DocumentChunk, IngestionJob, Status, utcnow
from .rag_chain import RetrievalConfig, answer_question, retrieve_context, stream_rag_answer
from .schemas import (
    AskRequest, AskResponse, DeleteResponse, DocumentStatus, HealthResponse, SearchHit, SearchRequest,
    SearchResponse, UploadResponse, UrlIngestRequest,DocumentListItem, DocumentListResponse
)
from .usage import record_calls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docqa")

EXT_TO_TYPE = {
    ".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".markdown": "markdown",
    ".csv": "csv", ".html": "html", ".htm": "html",
}
MAGIC = {"pdf": b"%PDF", "docx": b"PK"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create heavy clients ONCE at startup (Pinecone service, Gemini client, DB)."""
    from .config import get_settings
    from .gemini_client import build_gemini_client
    from .pinecone_service import PineconeService

    s = get_settings()
    app.state.settings = s
    database.configure(s.database_url)
    database.init_db()
    os.makedirs(s.upload_dir, exist_ok=True)
    # Schema-based Pinecone index with integrated embedding (llama-text-embed-v2): validated at startup,
    # no embedding model is loaded in the API or in the worker.
    app.state.pinecone = PineconeService.from_settings(s)
    app.state.gemini = build_gemini_client(s)
    # Pinecone Inference client for the hosted reranker (no network call until the first rerank)
    from pinecone import Pinecone

    app.state.reranker = Pinecone(api_key=s.pinecone_api_key).inference
    yield


app = FastAPI(title="Async Document Q&A Assistant", version="2.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers
def identity(
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    if not x_tenant_id or not x_user_id:
        raise HTTPException(401, "X-Tenant-ID and X-User-ID headers are required")
    return {"tenant_id": x_tenant_id.strip(), "user_id": x_user_id.strip()}


def _doc_id(tenant_id: str, payload: bytes) -> str:
    """Deterministic + tenant-scoped: same tenant & content => same ID (duplicate detection)."""
    return "doc_" + hashlib.sha256(tenant_id.encode() + b"\0" + payload).hexdigest()[:12]


def _search_filter(body, ident: dict) -> dict:
    """Tenant isolation + whitelisted client filters, applied at Pinecone query level.
    `tenant_id` always comes from the authenticated identity and cannot be overridden by the caller."""
    try:
        return filters_from_request(ident["tenant_id"], body.filters, body.document_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _retrieval_deps(st) -> dict:
    """Retrieval settings + reranker client from app state (same pipeline for /search, /ask, /ask/stream)."""
    return {"config": RetrievalConfig.from_settings(st.settings), "rerank_client": getattr(st, "reranker", None)}


def _llm_deps(st, ident: dict, endpoint: str) -> dict:
    """Retries / timeouts / fallback model (from settings) + the context that tags usage rows with request, tenant, user."""
    return {
        "llm_config": LLMConfig.from_settings(st.settings),
        "usage_context": {"request_id": uuid.uuid4().hex, "tenant_id": ident["tenant_id"],
                          "user_id": ident["user_id"], "endpoint": endpoint,
                          "persist": bool(getattr(st.settings, "llm_usage_persist", True))},
    }


def _check_access_level(level: str) -> str:
    if level not in ACCESS_LEVELS:
        raise HTTPException(400, f"access_level must be one of {list(ACCESS_LEVELS)}")
    return level


def enqueue_ingestion(document_id: str, file_path: str | None, source_url: str | None, task_id: str) -> None:
    """Send the job to the Celery broker. Returns immediately (never blocks on processing)."""
    from .tasks import process_document

    process_document.apply_async(args=[document_id, file_path, source_url], task_id=task_id)


def enqueue_purge(document_id: str) -> None:
    from .tasks import purge_deleted_vectors

    purge_deleted_vectors.apply_async(args=[document_id])


def _create_or_requeue(ident: dict, document_id: str, filename: str, source_type: str,
                       source_url: str | None, file_path: str | None, payload: bytes | None,
                       department: str | None = None, access_level: str = "internal") -> UploadResponse:
    """Insert Document + IngestionJob (PENDING), then enqueue. Handles duplicates."""
    task_id = str(uuid.uuid4())
    try:
        with database.session_scope() as s:
            doc = s.scalar(select(Document).where(Document.document_id == document_id))
            if doc is not None and doc.status != Status.FAILED.value and not doc.is_deleted:
                return UploadResponse(document_id=document_id, filename=doc.filename, status=doc.status,
                                      duplicate=True, message="Document already exists")
            if doc is None:
                doc = Document(document_id=document_id, filename=filename, source_type=source_type,
                               source_url=source_url, tenant_id=ident["tenant_id"],
                               uploaded_by=ident["user_id"], status=Status.PENDING.value,
                               department=department, access_level=access_level)
                s.add(doc)
            else:  # previously FAILED or soft-deleted -> (re)ingest
                doc.status, doc.error_message = Status.PENDING.value, None
                doc.is_deleted, doc.deleted_at = False, None
                doc.department, doc.access_level = department, access_level
            if payload is not None:  # write the file only once we know we will process it
                with open(file_path, "wb") as fh:  # type: ignore[arg-type]
                    fh.write(payload)
                doc.file_path = file_path
            s.flush()
            s.add(IngestionJob(task_id=task_id, document_id=document_id, status=Status.PENDING.value))
    except IntegrityError:  # two identical uploads raced; the other one won
        return UploadResponse(document_id=document_id, filename=filename, status=Status.PENDING.value,
                              duplicate=True, message="Document already exists")

    try:
        enqueue_ingestion(document_id, file_path, source_url, task_id)
    except Exception:
        logger.exception("could not enqueue %s", document_id)
        with database.session_scope() as s:
            doc = s.scalar(select(Document).where(Document.document_id == document_id))
            doc.status, doc.error_message = Status.FAILED.value, "Task queue unavailable"
        raise HTTPException(503, "Task queue unavailable; please retry later")
    return UploadResponse(document_id=document_id, filename=filename, status=Status.PENDING.value,
                          task_id=task_id, message="Queued for background ingestion")


# ---------------------------------------------------------------- endpoints
@app.post("/documents/upload", response_model=UploadResponse, status_code=202)
async def upload_document(request: Request, file: UploadFile = File(...),
                          department: str | None = Form(default=None, max_length=128),
                          access_level: str = Form(default="internal"), ident: dict = Depends(identity)):
    _check_access_level(access_level)
    department = (department or "").strip() or None
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    source_type = EXT_TO_TYPE.get(ext)
    if not source_type:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(EXT_TO_TYPE)}")

    settings = request.app.state.settings
    max_bytes = settings.max_upload_mb * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
    if not payload:
        raise HTTPException(400, "Empty file")
    if source_type in MAGIC and not payload.startswith(MAGIC[source_type]):
        raise HTTPException(400, f"File content does not look like a valid {source_type.upper()}")

    document_id = _doc_id(ident["tenant_id"], payload)
    os.makedirs(settings.upload_dir, exist_ok=True)
    path = os.path.join(settings.upload_dir, f"{document_id}{ext}")
    return await run_in_threadpool(
        _create_or_requeue, ident, document_id, name, source_type, None, path, payload, department, access_level
    )


@app.post("/documents/url", response_model=UploadResponse, status_code=202)
def ingest_url(body: UrlIngestRequest, ident: dict = Depends(identity)):
    url = str(body.url)
    try:
        _assert_public_url(url)  # fail fast; the worker checks again
    except LoaderError as exc:
        raise HTTPException(400, str(exc))
    document_id = _doc_id(ident["tenant_id"], url.encode())
    return _create_or_requeue(ident, document_id, url, "url", url, None, None,
                              (body.department or "").strip() or None, body.access_level)



@app.get("/documents", response_model=DocumentListResponse)
def list_documents(
    ident: dict = Depends(identity),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(
        default=None,
        pattern="^(PENDING|PROCESSING|COMPLETED|FAILED)$",
    ),
    source_type: str | None = Query(default=None, min_length=1, max_length=32),
    department: str | None = Query(default=None, min_length=1, max_length=128),
):
    """
    List documents uploaded by the authenticated user.

    Security:
    - Documents are restricted to the current tenant.
    - Documents are restricted to the current user.
    - Internal file paths are never exposed.

    Supports pagination and optional status/source-type filtering.
    """

    filters = [
        Document.tenant_id == ident["tenant_id"],
        Document.uploaded_by == ident["user_id"],
        Document.is_deleted.is_(False),
    ]
    if department:
        filters.append(Document.department == department)

    if status:
        filters.append(Document.status == status)

    if source_type:
        filters.append(Document.source_type == source_type)

    with database.session_scope() as s:

        # 1. Get the total number of matching documents.
        total = s.scalar(
            select(func.count())
            .select_from(Document)
            .where(*filters)
        ) or 0

        # 2. Fetch the requested page.
        documents = s.scalars(
            select(Document)
            .where(*filters)
            .order_by(
                Document.created_at.desc(),
                Document.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        ).all()

        # 3. Fetch the latest ingestion job for each document.
        document_ids = [doc.document_id for doc in documents]

        latest_jobs = {}

        if document_ids:
            jobs = s.scalars(
                select(IngestionJob)
                .where(IngestionJob.document_id.in_(document_ids))
                .order_by(
                    IngestionJob.document_id,
                    IngestionJob.id.desc(),
                )
            ).all()

            # The first job encountered for each document is its latest job.
            for job in jobs:
                latest_jobs.setdefault(job.document_id, job)

        # 4. Build the response.
        results = []

        for doc in documents:
            job = latest_jobs.get(doc.document_id)

            results.append(
                DocumentListItem(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    source_type=doc.source_type,
                    source_url=doc.source_url,
                    uploaded_by=doc.uploaded_by,
                    department=doc.department,
                    access_level=doc.access_level or "internal",
                    status=doc.status,
                    total_chunks=doc.total_chunks,
                    task_id=job.task_id if job else None,
                    progress=job.progress if job else 0,
                    retries=job.retries if job else 0,
                    error_message=(
                        doc.error_message
                        or (job.error_message if job else None)
                    ),
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
            )

        return DocumentListResponse(
            total=total,
            limit=limit,
            offset=offset,
            documents=results,
        )

@app.get("/documents/{document_id}/status", response_model=DocumentStatus)
def document_status(document_id: str, ident: dict = Depends(identity)):
    with database.session_scope() as s:
        doc = s.scalar(select(Document).where(Document.document_id == document_id))
        # 404 (not 403) for other tenants' documents: do not reveal that the ID exists
        if doc is None or doc.tenant_id != ident["tenant_id"] or doc.is_deleted:
            raise HTTPException(404, "Document not found")
        job = s.scalar(select(IngestionJob).where(IngestionJob.document_id == document_id)
                       .order_by(IngestionJob.id.desc()))
        return DocumentStatus(
            document_id=doc.document_id, filename=doc.filename, source_type=doc.source_type,
            status=doc.status, progress=job.progress if job else 0, total_chunks=doc.total_chunks,
            task_id=job.task_id if job else None, retries=job.retries if job else 0,
            error_message=doc.error_message, created_at=doc.created_at, updated_at=doc.updated_at,
        )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, request: Request, ident: dict = Depends(identity)):
    """Soft-delete in PostgreSQL FIRST (hides it from status/list/keyword search), then remove its vectors.
    If the Pinecone call fails, a Celery task retries the removal (with backoff)."""
    with database.session_scope() as s:
        doc = s.scalar(select(Document).where(Document.document_id == document_id))
        if doc is None or doc.tenant_id != ident["tenant_id"]:
            raise HTTPException(404, "Document not found")  # same answer for other tenants' documents
        doc.is_deleted, doc.deleted_at = True, utcnow()
        ids = list(s.scalars(select(DocumentChunk.chunk_id).where(
            DocumentChunk.document_id == document_id, DocumentChunk.chunk_type != "parent")))
    try:
        request.app.state.pinecone.delete_ids(ids, namespace=request.app.state.settings.pinecone_namespace)
        return DeleteResponse(document_id=document_id, vectors_removed=True, chunks=len(ids))
    except Exception:
        logger.exception("vector deletion failed for %s; queueing background purge", document_id)
    queued = True
    try:
        enqueue_purge(document_id)
    except Exception:
        logger.exception("could not queue purge for %s", document_id)
        queued = False
    return DeleteResponse(document_id=document_id, vectors_removed=False, purge_queued=queued, chunks=len(ids))


def _guard_question(question: str, st, *, pii_only: bool = False) -> tuple[GuardrailConfig, InputCheck]:
    """Run the question guardrails once per request. A prompt-injection attempt becomes HTTP 400 (not 502, and the
    detail lists rule names only). With `pii_only` (used by /search, which has no LLM) injection is not checked."""
    gcfg = GuardrailConfig.from_settings(st.settings)
    if pii_only:
        gcfg = dataclasses.replace(gcfg, injection_check_enabled=False)
    try:
        return gcfg, check_question(question, gcfg)
    except GuardrailBlocked as exc:
        logger.warning("question blocked by guardrails: %s %s", exc.reason, exc.matches)
        raise HTTPException(400, {"error": "guardrail_blocked", "reason": exc.reason, "message": BLOCKED_MESSAGE,
                                  "matches": exc.matches})


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, ident: dict = Depends(identity)):
    """Cited answer. Guardrails: prompt-injection question -> 400; PII is redacted from the question before
    retrieval and from the answer; poisoned chunks are dropped; the `guardrails` field reports what happened."""
    st = request.app.state
    flt = _search_filter(body, ident)
    gcfg, gate = _guard_question(body.question, st)
    try:
        return answer_question(
            body.question, st.pinecone, st.gemini, top_k=body.top_k,
            model=st.settings.gemini_model, metadata_filter=flt, rewrite_query=body.rewrite_query,
            guardrails=gcfg, input_check=gate,
            **_retrieval_deps(st), **_llm_deps(st, ident, "ask"),
        )
    except Exception:
        logger.exception("ask failed")
        raise HTTPException(502, "The answer service is temporarily unavailable")


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request, ident: dict = Depends(identity)):
    """Retrieval only (no LLM): dense + BM25 + RRF + rerank. `mode` exposes a single stage for comparison:
    dense | keyword | hybrid | hybrid_rerank (default). `stats` reports candidate counts and per-stage latency."""
    st = request.app.state
    flt = _search_filter(body, ident)
    llm = _llm_deps(st, ident, "search")
    _, gate = _guard_question(body.question, st, pii_only=True)  # redact PII before it reaches Pinecone / rewrite
    try:
        ctx = retrieve_context(gate.question, st.pinecone, flt, body.top_k, mode=body.mode,
                               gemini_client=st.gemini, rewrite_model=st.settings.gemini_model,
                               rewrite_query=body.rewrite_query, llm_config=llm["llm_config"],
                               **_retrieval_deps(st))
    except Exception:
        logger.exception("search failed")
        raise HTTPException(502, "Search is temporarily unavailable")
    record_calls(ctx["llm_calls"], llm["usage_context"])  # the query-rewrite call (if any); never raises
    return SearchResponse(query=gate.question, mode=body.mode,
                          results=[SearchHit(**c) for c in ctx["chunks"]], stats=ctx["stats"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request, ident: dict = Depends(identity)):
    """Server-Sent Events: [`query_rewrite`] -> `retrieval` -> `rerank` -> `sources` -> `token`* -> [`replace`] -> `done`.

    A prompt-injection question is rejected with HTTP 400 before the stream starts. `replace` (only when output
    validation changed the answer) carries the clean text that must replace the streamed tokens; `done` includes
    the `guardrails` report.

    `query_rewrite` (only when rewriting is on) lists the original + rewritten queries; `retrieval` and `rerank`
    carry stage counts and latencies; older clients can ignore all of them.

    The blocking generator (Pinecone + Gemini) runs in a worker thread so the event loop stays free.
    """
    st = request.app.state
    flt = _search_filter(body, ident)
    gcfg, gate = _guard_question(body.question, st)
    events = stream_rag_answer(
        body.question, st.pinecone, st.gemini, top_k=body.top_k,
        model=st.settings.gemini_model, metadata_filter=flt, rewrite_query=body.rewrite_query,
        guardrails=gcfg, input_check=gate,
        **_retrieval_deps(st), **_llm_deps(st, ident, "ask_stream"),
    )

    async def event_source():
        try:
            async for ev in iterate_in_threadpool(events):
                if await request.is_disconnected():  # client closed the tab -> stop paying for tokens
                    logger.info("client disconnected, stopping stream")
                    break
                yield _sse(ev["event"], ev["data"])
        except Exception:
            logger.exception("stream crashed")
            yield _sse("error", {"message": "Internal streaming error", "stage": "stream"})
        finally:
            events.close()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    st = request.app.state

    def safe(fn) -> bool:
        try:
            return bool(fn())
        except Exception:
            return False

    def redis_ping():
        import redis
        return redis.Redis.from_url(st.settings.redis_url, socket_connect_timeout=2).ping()

    db_ok, redis_ok, pc_ok = safe(database.ping_db), safe(redis_ping), safe(st.pinecone.ping)
    return HealthResponse(status="ok" if (db_ok and redis_ok and pc_ok) else "degraded",
                          database=db_ok, redis=redis_ok, pinecone=pc_ok, model=st.settings.gemini_model)
