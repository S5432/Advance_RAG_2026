import io
import pytest
from pypdf import PdfWriter
from src.ingestion import IngestionError, ingest_pdf, make_document_id

def test_ingest_sample_pdf(ingested):
    store, result = ingested
    assert result["page_count"] == 3 and result["status"] == "ingested" and result["document_name"] == "hr_policy.pdf"
    assert result["chunk_count"] == store.count() >= 3 and result["document_id"].startswith("doc_")
    info = store.get_document(result["document_id"])
    assert info["chunk_count"] == result["chunk_count"] and info["page_count"] == 3

def test_chunks_keep_exact_page_numbers(ingested):
    store, result = ingested
    top = store.search("how many paid leaves do employees get", 1)[0]
    assert top["page"] == 1
    assert store.search("what is the refund policy and return window", 1)[0]["page"] == 2
    assert store.search("password breach report security team", 1)[0]["page"] == 3

def test_reingesting_same_file_does_not_duplicate(store, sample_bytes):
    a = ingest_pdf(sample_bytes, "x.pdf", store)
    b = ingest_pdf(sample_bytes, "x.pdf", store)
    assert a["document_id"] == b["document_id"] == make_document_id(sample_bytes) and store.count() == b["chunk_count"]

def test_pdf_without_text_is_rejected(store):
    w = PdfWriter(); w.add_blank_page(100, 100)
    buf = io.BytesIO(); w.write(buf)
    with pytest.raises(IngestionError):
        ingest_pdf(buf.getvalue(), "scan.pdf", store)

def test_smaller_chunks_make_more_chunks(store, sample_bytes):
    n400 = ingest_pdf(sample_bytes, "s.pdf", store, 400, 50, document_id="doc_400")["chunk_count"]
    n1200 = ingest_pdf(sample_bytes, "s.pdf", store, 1200, 0, document_id="doc_1200")["chunk_count"]
    assert n400 > n1200 == 3
