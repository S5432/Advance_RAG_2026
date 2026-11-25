import io, json
import pytest
from fastapi.testclient import TestClient
import src.main as main
from src.gemini_client import GeminiQuotaError, GeminiUpstreamError
from tests.conftest import FakeGemini, GOOD_ANSWER, NO_INFO, SAMPLE_PDF

@pytest.fixture()
def env(store, monkeypatch):
    fake = FakeGemini(GOOD_ANSWER)
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main, "get_gemini_client", lambda: fake)
    return TestClient(main.app), fake

def upload(client, data=None, name="hr_policy.pdf", ctype="application/pdf"):
    return client.post("/documents/upload", files={"file": (name, data if data is not None else SAMPLE_PDF.read_bytes(), ctype)})

def events(text):
    return [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in text.strip().split("\n\n")]

def test_health(env):
    b = env[0].get("/health").json(); assert b["status"] == "ok" and b["vector_store"] is True

def test_upload_get_delete_flow(env):
    c, _ = env
    r = upload(c); assert r.status_code == 201
    body = r.json(); doc = body["document_id"]
    assert body["page_count"] == 3 and body["chunk_count"] >= 3 and body["status"] == "ingested"
    info = c.get(f"/documents/{doc}").json()
    assert info["document_name"] == "hr_policy.pdf" and info["chunk_count"] == body["chunk_count"]
    assert [d["document_id"] for d in c.get("/documents").json()] == [doc]
    d = c.delete(f"/documents/{doc}"); assert d.status_code == 200 and d.json()["deleted_chunks"] == body["chunk_count"]
    assert c.get(f"/documents/{doc}").status_code == 404 and c.delete(f"/documents/{doc}").status_code == 404

@pytest.mark.parametrize("kwargs,status", [({"name": "notes.txt", "ctype": "text/plain"}, 415), ({"data": b"not a pdf at all"}, 415),
                                           ({"data": b"%PDF-1.4 broken"}, 400), ({"name": "x.pdf", "ctype": "image/png"}, 415)])
def test_upload_rejections(env, kwargs, status):
    assert upload(env[0], **kwargs).status_code == status

def test_upload_too_large_is_413(env, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0"); main.get_settings.cache_clear()
    try: assert upload(env[0], data=b"%PDF-" + b"0" * 10).status_code == 413
    finally: main.get_settings.cache_clear()

def test_upload_missing_file_is_400(env):
    assert env[0].post("/documents/upload").status_code == 400

def test_search_returns_chunks_with_pages(env):
    c, fake = env; upload(c)
    r = c.post("/search", json={"query": "How many paid leaves do employees get?", "top_k": 3}).json()
    assert len(r["results"]) == 3 and r["results"][0]["page"] == 1 and set(r["results"][0]) >= {"chunk_id", "document_id", "text", "page", "score"}
    assert fake.calls == 0                                          # /search never calls the LLM

def test_ask_returns_grounded_answer_with_citations(env):
    c, fake = env; doc = upload(c).json()["document_id"]
    r = c.post("/ask", json={"question": "How many paid leaves do employees get?", "top_k": 4, "document_id": doc})
    b = r.json()
    assert r.status_code == 200 and b["insufficient_context"] is False and b["retrieved_chunks"] == 4
    assert b["citations"][0]["document_name"] == "hr_policy.pdf" and b["citations"][0]["page"] == 1 and b["citations"][0]["chunk_id"].startswith(doc)
    assert "hr_policy.pdf, p.1" in b["answer"]

def test_ask_insufficient_context_flag(env):
    c, fake = env; upload(c); fake.structured = [NO_INFO]
    b = c.post("/ask", json={"question": "What is the company's policy on remote work?"}).json()
    assert b["insufficient_context"] is True and b["citations"] == [] and "don't have enough information" in b["answer"]

def test_ask_on_empty_store_refuses_without_llm(env):
    c, fake = env
    b = c.post("/ask", json={"question": "anything?"}).json()
    assert b["insufficient_context"] is True and b["retrieved_chunks"] == 0 and fake.calls == 0

def test_ask_unknown_document_is_404(env):
    assert env[0].post("/ask", json={"question": "q", "document_id": "doc_nope"}).status_code == 404

@pytest.mark.parametrize("path,payload", [("/ask", {"question": ""}), ("/ask", {"question": "  "}), ("/ask", {"question": "x" * 1001}),
                                          ("/ask", {"question": "q", "top_k": 0}), ("/ask", {"question": "q", "top_k": 21}),
                                          ("/ask", {"question": "q", "document_id": "../etc"}), ("/search", {"query": ""}), ("/ask/stream", {})])
def test_validation_errors_are_400(env, path, payload):
    assert env[0].post(path, json=payload).status_code == 400

def test_llm_failure_is_502_and_safe(env):
    c, fake = env; upload(c); fake.error = GeminiQuotaError("raw provider text sk-SECRET")
    r = c.post("/ask", json={"question": "How many paid leaves do employees get?"})
    assert r.status_code == 502 and r.json()["detail"]["error_type"] == "quota_exceeded" and "SECRET" not in r.text

def test_ask_stream_events(env):
    c, _ = env; upload(c)
    r = c.post("/ask/stream", json={"question": "How many paid leaves do employees get?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r.text); names = [e for e, _ in ev]
    assert names[0] == "sources" and names[-1] == "done" and "chunk" in names
    assert ev[-1][1]["citations"][0]["page"] == 1 and "hr_policy.pdf, p.1" in ev[-1][1]["answer"]

def test_ask_stream_midway_failure_emits_error_event(env):
    c, fake = env; upload(c)
    def broken(*a, **k):
        yield "partial "; raise GeminiUpstreamError("boom SECRET")
    fake.stream = broken
    ev = events(c.post("/ask/stream", json={"question": "How many paid leaves do employees get?"}).text)
    assert [e for e, _ in ev][-1] == "error" and "SECRET" not in json.dumps(ev)

def test_llm_not_configured_is_503_but_search_works(store, monkeypatch):
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setenv("GEMINI_API_KEY", ""); main.get_settings.cache_clear()
    try:
        c = TestClient(main.app); upload(c)
        assert c.post("/search", json={"query": "paid leaves"}).status_code == 200
        assert c.post("/ask", json={"question": "How many paid leaves do employees get?"}).status_code == 503
    finally: main.get_settings.cache_clear()

def test_request_id_header(env):
    assert env[0].get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"
