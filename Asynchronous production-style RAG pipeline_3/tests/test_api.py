"""API tests: SQLite DB, fake Pinecone/Gemini/embedder, Celery enqueue replaced by a recorder."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src import database, main, tasks
from src.models import Document
from tests.fakes import FakeGemini, FakePinecone

DATA = Path(__file__).resolve().parent.parent / "data"
H = {"X-Tenant-ID": "company_001", "X-User-ID": "user_001"}
OTHER = {"X-Tenant-ID": "company_002", "X-User-ID": "user_002"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    database.configure("sqlite:///:memory:")
    database.init_db()
    settings = SimpleNamespace(
        gemini_model="gemini-2.5-flash", chunk_size=800, chunk_overlap=150, pinecone_namespace="documents",
        max_upload_mb=1, upload_dir=str(tmp_path), redis_url="redis://localhost:6379/0",
    )
    main.app.state.settings = settings
    main.app.state.pinecone = FakePinecone()
    main.app.state.gemini = FakeGemini()
    queued = []
    monkeypatch.setattr(main, "enqueue_ingestion", lambda *a: queued.append(a))
    return SimpleNamespace(client=TestClient(main.app), queued=queued, settings=settings)


def _upload(env, name="sample_notes.md", headers=H):
    with open(DATA / name, "rb") as fh:
        return env.client.post("/documents/upload", files={"file": (name, fh)}, headers=headers)


def _ingest_now(env, doc_id, name):
    """Simulate the worker for a queued document."""
    with database.session_scope() as s:
        path = s.scalar(select(Document).where(Document.document_id == doc_id)).file_path
    tasks.run_ingestion(doc_id, path, None, settings=env.settings,
                        pinecone=main.app.state.pinecone)


def test_upload_returns_immediately_and_queues_task(env):
    r = _upload(env)
    body = r.json()
    assert r.status_code == 202 and body["status"] == "PENDING" and body["task_id"] and not body["duplicate"]
    assert len(env.queued) == 1 and env.queued[0][0] == body["document_id"]
    st = env.client.get(f"/documents/{body['document_id']}/status", headers=H).json()
    assert st["status"] == "PENDING" and st["progress"] == 0 and st["task_id"] == body["task_id"]


def test_duplicate_upload_not_queued_twice(env):
    _upload(env)
    again = _upload(env).json()
    assert again["duplicate"] is True and len(env.queued) == 1


def test_failed_document_can_be_reuploaded(env):
    doc_id = _upload(env).json()["document_id"]
    tasks.mark_failed(doc_id, None, "boom")
    assert _upload(env).json()["duplicate"] is False and len(env.queued) == 2


def test_enqueue_failure_marks_failed_and_returns_503(env, monkeypatch):
    def boom(*a):
        raise ConnectionError("redis down")
    monkeypatch.setattr(main, "enqueue_ingestion", boom)
    assert _upload(env).status_code == 503
    with database.session_scope() as s:
        assert s.scalar(select(Document)).status == "FAILED"


def test_upload_validation(env):
    c = env.client
    assert c.post("/documents/upload", files={"file": ("x.exe", b"abc")}, headers=H).status_code == 415
    assert c.post("/documents/upload", files={"file": ("x.pdf", b"not pdf")}, headers=H).status_code == 400
    assert c.post("/documents/upload", files={"file": ("x.md", b"")}, headers=H).status_code == 400
    big = c.post("/documents/upload", files={"file": ("x.md", b"a" * (1024 * 1024 + 1))}, headers=H)
    assert big.status_code == 413
    assert c.post("/ask", json={"question": "hello there"}).status_code == 401


def test_url_endpoint(env):
    assert env.client.post("/documents/url", json={"url": "http://127.0.0.1/x"}, headers=H).status_code == 400
    assert env.client.post("/documents/url", json={"url": "not a url"}, headers=H).status_code == 422


def test_status_is_tenant_scoped(env):
    doc_id = _upload(env).json()["document_id"]
    assert env.client.get(f"/documents/{doc_id}/status", headers=OTHER).status_code == 404
    assert env.client.get("/documents/doc_nope/status", headers=H).status_code == 404


def test_ask_search_and_tenant_isolation_after_ingestion(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    st = env.client.get(f"/documents/{doc_id}/status", headers=H).json()
    assert st["status"] == "COMPLETED" and st["total_chunks"] >= 3

    ask = env.client.post("/ask", json={"question": "What is the probation period?"}, headers=H).json()
    assert "[Source 1]" in ask["answer"] and ask["citations"][0]["source_id"] == 1

    hits = env.client.post("/search", json={"question": "probation", "top_k": 3}, headers=H).json()["results"]
    assert 0 < len(hits) <= 3 and hits[0]["source_id"] == 1
    assert env.client.post("/search", json={"question": "probation"}, headers=OTHER).json()["results"] == []
    widened = env.client.post("/search", json={"question": "probation", "document_ids": [doc_id]}, headers=OTHER)
    assert widened.json()["results"] == []


def _parse_sse(text):
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_ask_stream_sse(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    r = env.client.post("/ask/stream", json={"question": "probation period?"}, headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _parse_sse(r.text)
    names = [n for n, _ in ev]
    assert names[0] == "sources" and names[-1] == "done" and names.count("token") == 4
    assert ev[-1][1]["citations"][0]["source_id"] == 1


def test_ask_stream_reports_gemini_failure(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    main.app.state.gemini = FakeGemini(fail_after=2)
    ev = _parse_sse(env.client.post("/ask/stream", json={"question": "probation?"}, headers=H).text)
    assert [n for n, _ in ev][-2:] == ["error", "done"] and ev[-1][1]["status"] == "failed"


def test_ask_stream_requires_identity_and_valid_body(env):
    assert env.client.post("/ask/stream", json={"question": "hello there"}).status_code == 401
    assert env.client.post("/ask/stream", json={"question": "hi"}, headers=H).status_code == 422
