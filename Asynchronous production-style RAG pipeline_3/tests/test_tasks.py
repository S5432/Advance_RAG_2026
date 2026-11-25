"""Ingestion pipeline tests on in-memory SQLite with fake Pinecone / embedder."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src import database, tasks
from src.models import Document, DocumentChunk, IngestionJob, Status
from tests.fakes import FakePinecone

DATA = Path(__file__).resolve().parent.parent / "data"
SETTINGS = SimpleNamespace(chunk_size=800, chunk_overlap=150, pinecone_namespace="documents")


@pytest.fixture()
def db():
    database.configure("sqlite:///:memory:")
    database.init_db()
    return database


def _make_doc(tmp_path, name="sample_notes.md", doc_id="doc_1", tenant="t1", content=None, source_type="markdown"):
    path = tmp_path / name
    if content is None:
        shutil.copy(DATA / name, path)
    else:
        path.write_bytes(content)
    with database.session_scope() as s:
        s.add(Document(document_id=doc_id, filename=name, source_type=source_type, tenant_id=tenant,
                       uploaded_by="u1", status=Status.PENDING.value, file_path=str(path)))
        s.flush()
        s.add(IngestionJob(task_id="task-1", document_id=doc_id, status=Status.PENDING.value))
    return str(path)


def _run(path, pc, doc_id="doc_1"):
    return tasks.run_ingestion(doc_id, path, None, settings=SETTINGS,
                               pinecone=pc, task_id="task-1")


def test_run_ingestion_completes_and_saves_everything(db, tmp_path):
    path, pc = _make_doc(tmp_path), FakePinecone()
    result = _run(path, pc)
    assert result["status"] == "COMPLETED" and result["chunks"] >= 3

    with database.session_scope() as s:
        doc = s.scalar(select(Document))
        job = s.scalar(select(IngestionJob))
        rows = s.scalars(select(DocumentChunk).order_by(DocumentChunk.chunk_index)).all()
    assert doc.status == "COMPLETED" and doc.total_chunks == len(rows) == len(pc.store)
    assert job.status == "COMPLETED" and job.progress == 100 and job.completed_at is not None
    assert rows[0].chunk_id == "doc_1_chunk_0" and rows[0].embedding_status == "EMBEDDED"
    # stored_doc = pc.store["doc_1_chunk_0"]
    # assert stored_doc["tenant_id"] == "t1"
    # assert stored_doc["uploaded_by"] == "u1"
    # assert stored_doc["document_name"] == "sample_notes.md"

    # Verify the Pinecone document structure.
    stored_doc = pc.store["doc_1_chunk_0"]

    # Verify document identity.
    assert stored_doc["_id"] == "doc_1_chunk_0"

    # Verify integrated embedding input.
    assert stored_doc["chunk_text"]
    assert stored_doc["title"] == "sample_notes.md"

    # Verify tenant isolation metadata.
    assert stored_doc["tenant_id"] == "t1"
    assert stored_doc["uploaded_by"] == "u1"

    # Verify document metadata.
    assert stored_doc["document_name"] == "sample_notes.md"
    assert stored_doc["document_id"] == "doc_1"
    assert stored_doc["source_type"] == "markdown"
    assert stored_doc["chunk_index"] == 0
    assert stored_doc["chunk_id"] == "doc_1_chunk_0"

    # Verify required metadata is stored at the top level.
    required_fields = {
        "document_id",
        "document_name",
        "source_type",
        "page_number",
        "chunk_index",
        "chunk_id",
        "section",
        "uploaded_by",
        "tenant_id",
        "created_at",
    }

    assert required_fields.issubset(stored_doc.keys())

    # Ensure legacy nested metadata is not being used.
    assert "metadata" not in stored_doc


def test_rerun_is_idempotent_no_duplicate_vectors_or_rows(db, tmp_path):
    path, pc = _make_doc(tmp_path), FakePinecone()
    _run(path, pc)
    n_vectors = len(pc.store)
    _run(path, pc)  # e.g. redelivered after a worker crash
    with database.session_scope() as s:
        assert len(s.scalars(select(DocumentChunk)).all()) == n_vectors
    assert len(pc.store) == n_vectors


def test_empty_document_is_permanent_failure(db, tmp_path):
    path = _make_doc(tmp_path, "empty.md", content=b"   \n  ")
    with pytest.raises(tasks.PermanentIngestionError):
        _run(path, FakePinecone())


def test_mark_failed_records_error_on_document_and_job(db, tmp_path):
    _make_doc(tmp_path)
    tasks.mark_failed("doc_1", "task-1", "boom", retries=2)
    with database.session_scope() as s:
        doc, job = s.scalar(select(Document)), s.scalar(select(IngestionJob))
    assert doc.status == "FAILED" and doc.error_message == "boom"
    assert job.status == "FAILED" and job.retries == 2


def test_celery_task_marks_failed_on_permanent_error(db, tmp_path, monkeypatch):
    path = _make_doc(tmp_path, "empty.md", content=b"  ")
    monkeypatch.setattr(tasks, "_deps", lambda: (SETTINGS, FakePinecone()))
    res = tasks.process_document.apply(args=["doc_1", path, None], task_id="task-1")
    assert res.get()["status"] == "FAILED"
    with database.session_scope() as s:
        assert s.scalar(select(Document)).status == "FAILED"
    assert not Path(path).exists()  # temp upload cleaned up


def test_celery_task_transient_error_ends_failed_with_message(db, tmp_path, monkeypatch):
    path = _make_doc(tmp_path)
    monkeypatch.setattr(tasks, "_deps", lambda: (SETTINGS, FakePinecone()))
    monkeypatch.setattr(tasks, "run_ingestion", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(tasks.process_document, "default_retry_delay", 0)
    res = tasks.process_document.apply(args=["doc_1", path, None], task_id="task-1")
    assert res.failed()
    with database.session_scope() as s:
        doc = s.scalar(select(Document))
    assert doc.status == "FAILED" and "boom" in doc.error_message
