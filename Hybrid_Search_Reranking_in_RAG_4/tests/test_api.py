"""API tests: SQLite DB, fake Pinecone/Gemini/embedder, Celery enqueue replaced by a recorder."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src import database, main, tasks
from src.models import Document
from src.rag_chain import NOT_FOUND
from tests.fakes import FakeGemini, FakePinecone, FakeReranker

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
        rerank_min_score=0.0,  # the fake reranker scores by exact word overlap; thresholds are tested separately
    )
    main.app.state.settings = settings
    main.app.state.pinecone = FakePinecone()
    main.app.state.gemini = FakeGemini()
    main.app.state.reranker = FakeReranker()
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
    assert names[:3] == ["retrieval", "rerank", "sources"] and names[-1] == "done" and names.count("token") == 4
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


# ------------------------------------------------------------------ Day 11 / Phase 3
def _upload_with(env, name="sample_notes.md", headers=H, **form):
    with open(DATA / name, "rb") as fh:
        return env.client.post("/documents/upload", files={"file": (name, fh)}, data=form, headers=headers)


def test_department_and_access_level_stored_and_filterable(env):
    doc_id = _upload_with(env, department=" HR ", access_level="confidential").json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    stored = next(iter(main.app.state.pinecone.store.values()))
    assert stored["department"] == "HR" and stored["access_level"] == "confidential" and stored["created_at_ts"] > 0
    listed = env.client.get("/documents?department=HR", headers=H).json()
    assert listed["total"] == 1 and listed["documents"][0]["access_level"] == "confidential"
    assert env.client.get("/documents?department=IT", headers=H).json()["total"] == 0
    hr = env.client.post("/search", json={"question": "probation", "filters": {"department": "HR"}}, headers=H)
    it = env.client.post("/search", json={"question": "probation", "filters": {"department": "IT"}}, headers=H)
    assert hr.json()["results"] and it.json()["results"] == []


def test_invalid_access_level_and_bad_filters_return_400(env):
    assert _upload_with(env, access_level="secret").status_code == 400
    for flt in ({"tenant_id": "company_002"}, {"source_type": "exe"}, {"date_from": "nope"}):
        r = env.client.post("/search", json={"question": "probation", "filters": flt}, headers=H)
        assert r.status_code == 400


def test_client_filter_cannot_cross_tenants(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    r = env.client.post("/search", json={"question": "probation", "filters": {"document_id": doc_id}}, headers=OTHER)
    assert r.status_code == 200 and r.json()["results"] == []


def test_delete_hides_document_everywhere_and_removes_vectors(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    assert env.client.delete(f"/documents/{doc_id}", headers=OTHER).status_code == 404   # other tenant
    r = env.client.delete(f"/documents/{doc_id}", headers=H)
    assert r.status_code == 200 and r.json()["vectors_removed"] is True and r.json()["chunks"] >= 3
    assert main.app.state.pinecone.store == {}
    assert env.client.post("/search", json={"question": "probation"}, headers=H).json()["results"] == []
    assert env.client.get(f"/documents/{doc_id}/status", headers=H).status_code == 404
    assert env.client.get("/documents", headers=H).json()["total"] == 0


def test_failed_vector_deletion_queues_background_purge(env, monkeypatch):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    monkeypatch.setattr(main.app.state.pinecone, "delete_ids",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pinecone down")))
    purged = []
    monkeypatch.setattr(main, "enqueue_purge", purged.append)
    r = env.client.delete(f"/documents/{doc_id}", headers=H).json()
    assert r["vectors_removed"] is False and r["purge_queued"] is True and purged == [doc_id]
    assert env.client.get(f"/documents/{doc_id}/status", headers=H).status_code == 404   # hidden in PostgreSQL anyway


def test_deleted_document_can_be_reuploaded(env):
    doc_id = _upload(env).json()["document_id"]
    env.client.delete(f"/documents/{doc_id}", headers=H)
    again = _upload(env).json()
    assert again["document_id"] == doc_id and again["duplicate"] is False and len(env.queued) == 2
    assert env.client.get(f"/documents/{doc_id}/status", headers=H).status_code == 200


# ------------------------------------------------------------------ Day 11 / Phase 6
def _ready(env):
    doc_id = _upload(env).json()["document_id"]
    _ingest_now(env, doc_id, "sample_notes.md")
    return doc_id


def test_search_returns_stage_scores_and_stats(env):
    _ready(env)
    body = env.client.post("/search", json={"question": "probation period", "top_k": 2}, headers=H).json()
    assert body["mode"] == "hybrid_rerank" and 0 < len(body["results"]) <= 2
    top = body["results"][0]
    assert top["source_id"] == 1 and top["rerank_score"] is not None and top["rrf_score"] > 0
    assert top["score"] == top["rerank_score"] and "probation" in top["text"].lower()
    st = body["stats"]
    assert st["dense_candidates"] >= 1 and st["rerank"]["candidates_in"] >= 1 and st["rerank"]["fallback"] is False


def test_search_modes(env):
    _ready(env)
    for mode in ("dense", "keyword", "hybrid", "hybrid_rerank"):
        r = env.client.post("/search", json={"question": "probation period", "mode": mode}, headers=H)
        assert r.status_code == 200 and r.json()["mode"] == mode and r.json()["results"], mode
    keyword = env.client.post("/search", json={"question": "probation", "mode": "keyword"}, headers=H).json()
    assert keyword["results"][0]["keyword_score"] > 0 and keyword["results"][0]["rerank_score"] is None
    assert env.client.post("/search", json={"question": "probation", "mode": "magic"}, headers=H).status_code == 422


def test_search_survives_reranker_outage(env):
    _ready(env)
    main.app.state.reranker = FakeReranker(always_fail=True, status=400)
    body = env.client.post("/search", json={"question": "probation period"}, headers=H).json()
    assert body["stats"]["rerank"]["fallback"] is True
    assert body["results"] and body["results"][0]["rerank_score"] is None and body["results"][0]["score"] > 0


def test_search_nothing_relevant_returns_empty(env):
    _ready(env)
    env.settings.rerank_min_score = 0.05
    body = env.client.post("/search", json={"question": "quantum chromodynamics lattice"}, headers=H).json()
    assert body["results"] == [] and body["stats"]["rerank"]["below_threshold"] >= 1


def test_ask_includes_retrieval_stats_and_rerank_scores(env):
    _ready(env)
    body = env.client.post("/ask", json={"question": "What is the probation period?"}, headers=H).json()
    assert body["retrieval"]["rerank"]["returned"] == body["retrieved_chunks"] >= 1
    assert body["citations"][0]["rerank_score"] is not None


def test_ask_not_found_without_llm_when_nothing_relevant(env):
    _ready(env)
    env.settings.rerank_min_score = 0.05
    r = env.client.post("/ask", json={"question": "quantum chromodynamics lattice"}, headers=H)
    # FakeGemini would answer "... [Source 1]" if it were called, so NOT_FOUND proves the LLM was skipped
    assert r.status_code == 200 and r.json()["answer"] == NOT_FOUND
    assert r.json()["citations"] == [] and r.json()["retrieved_chunks"] == 0


def test_ask_stream_emits_retrieval_and_rerank_events(env):
    _ready(env)
    ev = _parse_sse(env.client.post("/ask/stream", json={"question": "probation period"}, headers=H).text)
    names = [n for n, _ in ev]
    assert names == ["retrieval", "rerank", "sources"] + ["token"] * 4 + ["done"]
    data = dict(ev)
    assert data["retrieval"]["dense_candidates"] >= 1 and data["rerank"]["fallback"] is False
    assert data["sources"]["sources"][0]["rerank_score"] is not None
    assert data["done"]["status"] == "completed" and data["done"]["latency_ms"] >= 0


def test_ask_stream_is_tenant_scoped(env):
    _ready(env)
    ev = _parse_sse(env.client.post("/ask/stream", json={"question": "probation period"}, headers=OTHER).text)
    assert dict(ev)["sources"]["sources"] == []
