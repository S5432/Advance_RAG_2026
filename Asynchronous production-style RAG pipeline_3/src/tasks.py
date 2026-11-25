"""Celery tasks: asynchronous document ingestion.

`run_ingestion` holds the pipeline logic (plain function -> easy to unit test);
`process_document` is the thin Celery wrapper that adds retries and cleanup.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from sqlalchemy import select

from .celery_app import celery_app
from .chunking import chunk_documents
from .database import init_db, session_scope
from .loaders import load_document
from .models import Document, DocumentChunk, IngestionJob, Status, utcnow
from .pinecone_service import upsert_chunks_to_pinecone

logger = logging.getLogger(__name__)

# Embeddings are generated INSIDE Pinecone (integrated embedding on `chunk_text`, llama-text-embed-v2),
# so there is no client-side embedding step. We upsert in slices to report progress on big documents;
# PineconeService.upsert_documents() further batches each slice (50 documents per request).
UPSERT_SLICE = 200


class PermanentIngestionError(Exception):
    """Failure that retrying cannot fix (empty document, unsupported type...)."""


@lru_cache
def _deps():
    """Created once per worker process: settings + schema-validated Pinecone service."""
    from .config import get_settings
    from .pinecone_service import PineconeService

    settings = get_settings()
    init_db()
    return settings, PineconeService.from_settings(settings)


# ------------------------------------------------------------------ DB helpers
def _update_job(task_id: str | None, **fields) -> None:
    if not task_id:
        return
    with session_scope() as s:
        job = s.scalar(select(IngestionJob).where(IngestionJob.task_id == task_id))
        if job:
            for k, v in fields.items():
                setattr(job, k, v)


def _progress(task_id: str | None, value: int) -> None:
    _update_job(task_id, progress=value)
    logger.info("task %s progress %s%%", task_id, value)


def mark_failed(document_id: str, task_id: str | None, error: str, retries: int = 0) -> None:
    with session_scope() as s:
        doc = s.scalar(select(Document).where(Document.document_id == document_id))
        if doc:
            doc.status = Status.FAILED.value
            doc.error_message = error[:2000]
    _update_job(task_id, status=Status.FAILED.value, error_message=error[:2000], retries=retries,
                completed_at=utcnow())


# --------------------------------------------------------------- the pipeline
def run_ingestion(document_id: str, file_path: str | None, source_url: str | None, *,
                  settings, pinecone, task_id: str | None = None) -> dict:
    """PROCESSING -> load(+OCR) -> chunk -> Pinecone upsert (embeds server-side) -> save chunks -> COMPLETED."""
    # 1) PROCESSING
    with session_scope() as s:
        doc = s.scalar(select(Document).where(Document.document_id == document_id))
        if doc is None:
            raise PermanentIngestionError(f"Document {document_id} not found")
        doc.status, doc.error_message = Status.PROCESSING.value, None
        filename, source_type = doc.filename, doc.source_type
        tenant_id, uploaded_by, created_at = doc.tenant_id, doc.uploaded_by, doc.created_at
    _update_job(task_id, status=Status.PROCESSING.value, progress=5, started_at=utcnow(),
                completed_at=None, error_message=None)

    # 2-3) load (PDF pages without text are OCR'd inside load_pdf)
    target = source_url or file_path
    if not target:
        raise PermanentIngestionError("Neither file_path nor source_url given")
    if file_path and not source_url and not os.path.exists(file_path):
        raise PermanentIngestionError("Uploaded file is no longer on disk; please upload again")
    docs = load_document(target, source_type, document_id)
    if not any(d["text"].strip() for d in docs):
        raise PermanentIngestionError("No extractable text found (empty or unreadable document)")
    for d in docs:
        d["metadata"]["source"] = filename
    _progress(task_id, 15)

    # 4) clean + chunk
    chunks = chunk_documents(
        docs, settings.chunk_size, settings.chunk_overlap,
        extra_metadata={
            "document_name": filename, "uploaded_by": uploaded_by, "tenant_id": tenant_id,
            "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    if not chunks:
        raise PermanentIngestionError("Document produced no chunks")
    _progress(task_id, 30)

    # 5-6) Pinecone upsert. Pinecone embeds `chunk_text` itself (no client embeddings). Deterministic
    #      `_id` = chunk_id => idempotent: a replay after a crash overwrites instead of duplicating.
    new_ids: list[str] = []
    for i in range(0, len(chunks), UPSERT_SLICE):
        new_ids.extend(
            upsert_chunks_to_pinecone(chunks[i : i + UPSERT_SLICE], pinecone, settings.pinecone_namespace)
        )
        _progress(task_id, 30 + int(60 * min(i + UPSERT_SLICE, len(chunks)) / len(chunks)))

    # 7-8) save chunk rows + COMPLETED (single transaction)
    with session_scope() as s:
        old_ids = set(s.scalars(select(DocumentChunk.chunk_id).where(DocumentChunk.document_id == document_id)))
        s.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        s.add_all(
            DocumentChunk(
                document_id=document_id,
                chunk_id=c["metadata"]["chunk_id"],
                chunk_index=c["metadata"]["chunk_index"],
                content=c["text"],
                page_number=c["metadata"].get("page_number"),
                section=c["metadata"].get("section") or None,
                token_count=len(c["text"].split()),
                embedding_status="EMBEDDED",
            )
            for c in chunks
        )
        doc = s.scalar(select(Document).where(Document.document_id == document_id))
        doc.status, doc.total_chunks, doc.error_message = Status.COMPLETED.value, len(chunks), None
    stale = sorted(old_ids - set(new_ids))  # re-ingested document produced fewer chunks than before
    if stale:
        pinecone.delete_ids(stale, namespace=settings.pinecone_namespace)
    _update_job(task_id, status=Status.COMPLETED.value, progress=100, completed_at=utcnow())
    logger.info("document %s completed (%d chunks)", document_id, len(chunks))
    return {"document_id": document_id, "status": Status.COMPLETED.value, "chunks": len(chunks)}


def _cleanup(file_path: str | None) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            logger.warning("could not delete %s", file_path)


# ---------------------------------------------------------------- Celery tasks
@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_document(self, document_id: str, file_path: str | None = None, source_url: str | None = None):
    task_id = self.request.id
    retries = self.request.retries
    try:
        settings, pinecone = _deps()
        result = run_ingestion(document_id, file_path, source_url, settings=settings,
                               pinecone=pinecone, task_id=task_id)
        _cleanup(file_path)
        return result
    except PermanentIngestionError as exc:
        logger.error("document %s failed permanently: %s", document_id, exc)
        mark_failed(document_id, task_id, str(exc), retries)
        _cleanup(file_path)
        return {"document_id": document_id, "status": Status.FAILED.value, "error": str(exc)}
    except Exception as exc:
        logger.exception("document %s failed (attempt %d)", document_id, retries + 1)
        mark_failed(document_id, task_id, f"{type(exc).__name__}: {exc}", retries)
        if retries >= self.max_retries:
            _cleanup(file_path)
            raise  # lets Celery/Flower record the task as FAILURE
        raise self.retry(exc=exc, countdown=self.default_retry_delay * 2**retries)  # 10s, 20s, 40s


@celery_app.task
def requeue_stuck_documents() -> int:
    """Celery beat job (every 10 min): re-queue documents stuck in PENDING/PROCESSING,
    e.g. after a worker crash or a lost message."""
    from .config import get_settings

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=get_settings().stuck_after_minutes)
    requeued = 0
    with session_scope() as s:
        stuck = s.scalars(
            select(Document).where(Document.status.in_([Status.PENDING.value, Status.PROCESSING.value]),
                                   Document.updated_at < cutoff)
        ).all()
        for doc in stuck:
            source_ok = bool(doc.source_url) or (doc.file_path and os.path.exists(doc.file_path))
            if not source_ok:
                doc.status, doc.error_message = Status.FAILED.value, "Stuck and source file missing; re-upload"
                continue
            task_id = str(uuid.uuid4())
            doc.status = Status.PENDING.value
            s.add(IngestionJob(task_id=task_id, document_id=doc.document_id, status=Status.PENDING.value))
            s.flush()
            process_document.apply_async(args=[doc.document_id, doc.file_path, doc.source_url], task_id=task_id)
            requeued += 1
    return requeued
