"""API tests with fake Pinecone / embedding model / Gemini (no network, no API keys)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app

DATA = Path(__file__).resolve().parent.parent / "data"
H = {"X-Tenant-ID": "company_001", "X-User-ID": "user_001"}
OTHER = {"X-Tenant-ID": "company_002", "X-User-ID": "user_002"}


class FakeEmbedder:
    def encode(self, texts, **_):
        return np.array([[float(len(t) % 7), 1.0, 0.5] for t in texts])


class FakePinecone:
    def __init__(self):
        self.store = {}

    def upsert_vectors(self, vectors, namespace=None, batch_size=100):
        for v in vectors:
            self.store[v["id"]] = v
        return len(vectors)

    def query(self, vector, top_k=5, metadata_filter=None, namespace=None):
        f = metadata_filter or {}
        out = []
        for v in self.store.values():
            md = v["metadata"]
            if "tenant_id" in f and md["tenant_id"] != f["tenant_id"]:
                continue
            if "document_id" in f and md["document_id"] not in f["document_id"]["$in"]:
                continue
            out.append({"id": v["id"], "score": 0.9, "metadata": md})
        return out[:top_k]

    def delete_ids(self, ids, namespace=None):
        for i in ids:
            self.store.pop(i, None)

    def ping(self):
        return True


class FakeGemini:
    def __init__(self):
        self.models = SimpleNamespace(generate_content=self._gen)
        self.calls = 0

    def _gen(self, model, contents, config=None):
        self.calls += 1
        assert "Answer only using the provided context" in contents
        return SimpleNamespace(text="Employees get 18 days of annual leave. [Source 1]")


@pytest.fixture()
def client():
    app.state.settings = SimpleNamespace(
        gemini_model="gemini-2.5-flash", chunk_size=800, chunk_overlap=150,
        pinecone_namespace="documents", max_upload_mb=1,
    )
    app.state.embedding_model = FakeEmbedder()
    app.state.pinecone = FakePinecone()
    app.state.gemini = FakeGemini()
    app.state.registry = {}
    return TestClient(app)  # no `with` -> lifespan (real clients) is not started


def _upload(client, name="sample_notes.md", headers=H):
    with open(DATA / name, "rb") as fh:
        return client.post("/documents/upload", files={"file": (name, fh)}, headers=headers)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_requires_identity_headers(client):
    assert client.post("/ask", json={"question": "hello there"}).status_code == 401


def test_upload_ingests_and_status_completed(client):
    r = _upload(client)
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    s = client.get(f"/documents/{doc_id}/status", headers=H).json()
    assert s["status"] == "completed" and s["chunks"] >= 3
    meta = next(iter(app.state.pinecone.store.values()))["metadata"]
    for key in ("document_id", "document_name", "source_type", "page_number", "chunk_index",
                "chunk_id", "section", "uploaded_by", "tenant_id", "created_at"):
        assert key in meta


def test_pdf_and_csv_upload(client):
    for name in ("sample_policy.pdf", "sample_faq.csv"):
        assert _upload(client, name).status_code == 202
    statuses = [e["status"] for e in app.state.registry.values()]
    assert statuses == ["completed", "completed"]


def test_duplicate_upload(client):
    _upload(client)
    assert _upload(client).json()["status"] == "duplicate"


def test_invalid_uploads(client):
    bad_ext = client.post("/documents/upload", files={"file": ("x.exe", b"abc")}, headers=H)
    assert bad_ext.status_code == 415
    fake_pdf = client.post("/documents/upload", files={"file": ("x.pdf", b"not a pdf")}, headers=H)
    assert fake_pdf.status_code == 400
    empty = client.post("/documents/upload", files={"file": ("x.md", b"")}, headers=H)
    assert empty.status_code == 400
    too_big = client.post("/documents/upload", files={"file": ("x.md", b"a" * (1024 * 1024 + 1))}, headers=H)
    assert too_big.status_code == 413


def test_ingestion_failure_is_recorded(client):
    r = client.post("/documents/upload", files={"file": ("bad.csv", b"\x00\x01")}, headers=H)
    doc_id = r.json()["document_id"]
    s = client.get(f"/documents/{doc_id}/status", headers=H).json()
    assert s["status"] == "failed" and s["error"]


def test_ask_returns_cited_answer(client):
    _upload(client, "sample_policy.pdf")
    r = client.post("/ask", json={"question": "How many annual leave days?"}, headers=H)
    body = r.json()
    assert r.status_code == 200
    assert "[Source 1]" in body["answer"]
    assert body["citations"][0]["source_id"] == 1
    assert body["citations"][0]["document_name"] == "sample_policy.pdf"
    assert body["model"] == "gemini-2.5-flash"


def test_ask_with_nothing_retrieved_skips_llm(client):
    r = client.post("/ask", json={"question": "anything at all"}, headers=H)
    assert r.json()["answer"].startswith("I could not find")
    assert app.state.gemini.calls == 0


def test_tenant_isolation(client):
    doc_id = _upload(client).json()["document_id"]
    assert client.post("/search", json={"question": "probation period"}, headers=OTHER).json()["results"] == []
    assert client.get(f"/documents/{doc_id}/status", headers=OTHER).status_code == 404
    # a caller cannot widen the filter via document_ids
    r = client.post("/search", json={"question": "probation", "document_ids": [doc_id]}, headers=OTHER)
    assert r.json()["results"] == []


def test_search_and_validation(client):
    _upload(client)
    r = client.post("/search", json={"question": "learning budget", "top_k": 3}, headers=H)
    assert r.status_code == 200 and 0 < len(r.json()["results"]) <= 3
    assert client.post("/search", json={"question": "hi"}, headers=H).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}, headers=H).status_code == 422


def test_url_endpoint_blocks_private_hosts(client):
    r = client.post("/documents/url", json={"url": "http://127.0.0.1/secret"}, headers=H)
    doc_id = r.json()["document_id"]
    s = client.get(f"/documents/{doc_id}/status", headers=H).json()
    assert s["status"] == "failed"
