import pytest
from src.vector_store import ChromaVectorStore, EmbeddingMismatchError

CHUNKS = ["Employees are eligible for 12 paid leaves per year.", "Refunds are issued within 7 business days.", "Passwords must contain 12 characters."]
METAS = [{"page": 1, "document_name": "a.pdf", "page_count": 3}, {"page": 2, "document_name": "a.pdf", "page_count": 3},
         {"page": 3, "document_name": "a.pdf", "page_count": 3}]

def test_add_and_search_returns_required_fields(store):
    assert store.add_document("doc_1", CHUNKS, METAS) == 3
    hits = store.search("how many paid leaves per year", top_k=2)
    assert len(hits) == 2
    top = hits[0]
    assert top["chunk_id"] == "doc_1_chunk_0" and top["document_id"] == "doc_1" and top["page"] == 1 and top["document_name"] == "a.pdf"
    assert "12 paid leaves" in top["text"] and 0 < top["score"] <= 1 and top["chunk_index"] == 0
    assert hits[0]["score"] >= hits[1]["score"]

def test_metadata_filter_by_document(store):
    store.add_document("doc_1", CHUNKS, METAS)
    store.add_document("doc_2", ["Refund policy of the other company is 14 days."], [{"page": 1, "document_name": "b.pdf"}])
    hits = store.search("refund policy", top_k=5, metadata_filter={"document_id": "doc_2"})
    assert {h["document_id"] for h in hits} == {"doc_2"}
    both = store.search("refund", top_k=5, metadata_filter={"document_id": "doc_1", "page": 2})     # two conditions -> $and
    assert [h["chunk_id"] for h in both] == ["doc_1_chunk_1"]

def test_get_list_delete_document(store):
    store.add_document("doc_1", CHUNKS, METAS)
    info = store.get_document("doc_1")
    assert (info["document_name"], info["page_count"], info["chunk_count"]) == ("a.pdf", 3, 3) and info["uploaded_at"]
    assert [d["document_id"] for d in store.list_documents()] == ["doc_1"]
    assert store.delete_document("doc_1") == 3 and store.get_document("doc_1") is None and store.count() == 0
    assert store.delete_document("doc_1") == 0

def test_search_empty_store_and_unknown_filter(store):
    assert store.search("anything") == []
    store.add_document("doc_1", CHUNKS, METAS)
    assert store.search("paid leaves", metadata_filter={"document_id": "nope"}) == []

def test_top_k_larger_than_collection(store):
    store.add_document("doc_1", CHUNKS, METAS)
    assert len(store.search("policy", top_k=20)) == 3

def test_length_mismatch_and_none_metadata(store):
    with pytest.raises(ValueError):
        store.add_document("d", ["a", "b"], [{"page": 1}])
    assert store.add_document("d", ["alpha beta"], [{"page": None, "document_name": "x.pdf"}]) == 1     # None values are dropped, not rejected

def test_persistence_across_instances(tmp_path, embedder):
    path = str(tmp_path / "persist")
    ChromaVectorStore(path, "docs", embedder).add_document("doc_1", CHUNKS, METAS)
    again = ChromaVectorStore(path, "docs", embedder)                     # new client on the same folder = "restart"
    assert again.count() == 3 and again.search("paid leaves", 1)[0]["document_id"] == "doc_1"

def test_embedding_model_mismatch_is_detected(tmp_path, embedder):
    path = str(tmp_path / "m")
    ChromaVectorStore(path, "docs", embedder)
    other = type("Other", (), {"model_name": "another-model", "encode": embedder.encode})()
    with pytest.raises(EmbeddingMismatchError):
        ChromaVectorStore(path, "docs", other)
