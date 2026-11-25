"""Day 9 `upsert_chunks_to_pinecone` (schema-based, integrated embedding) feeding the Day 10 chain."""
from src.chunking import chunk_documents
from src.loaders import load_markdown
from src.pinecone_service import REQUIRED_METADATA, upsert_chunks_to_pinecone
from src.rag_chain import retrieve_chunks
from tests.fakes import FakePinecone

from pathlib import Path
DATA = Path(__file__).resolve().parent.parent / "data"


def _chunks(tenant="t1"):
    docs = load_markdown(str(DATA / "sample_notes.md"), "doc_1")
    return chunk_documents(docs, 800, 150, extra_metadata={
        "document_name": "sample_notes.md", "uploaded_by": "u1", "tenant_id": tenant,
        "created_at": "2026-11-18T10:30:00Z"})


def test_documents_have_schema_fields_and_all_required_metadata():
    pc = FakePinecone()
    ids = upsert_chunks_to_pinecone(_chunks(), pc, "documents")
    assert ids[0] == "doc_1_chunk_0" and len(pc.store) == len(ids)
    doc = pc.store[ids[0]]
    assert doc["_id"] == ids[0] and doc["chunk_text"] and doc["title"] == "sample_notes.md"
    assert "text" not in doc                      # text lives in chunk_text, not metadata
    assert all(doc.get(f) is not None for f in REQUIRED_METADATA)


def test_reupsert_is_idempotent_and_retrieval_is_tenant_scoped():
    pc = FakePinecone()
    upsert_chunks_to_pinecone(_chunks(), pc); n = len(pc.store)
    upsert_chunks_to_pinecone(_chunks(), pc)
    assert len(pc.store) == n
    assert retrieve_chunks("probation", pc, 5, {"tenant_id": {"$eq": "t1"}})
    assert retrieve_chunks("probation", pc, 5, {"tenant_id": {"$eq": "t2"}}) == []
