from pathlib import Path

import pytest

from src import loaders
from src.loaders import LoaderError, load_csv, load_document, load_html, load_markdown, load_pdf

DATA = Path(__file__).resolve().parent.parent / "data"


def _check(docs, source_type):
    assert docs
    for d in docs:
        assert d["text"].strip()
        m = d["metadata"]
        assert m["source_type"] == source_type
        assert {"source", "page_number", "document_id"} <= set(m)


def test_load_pdf_text_pages():
    docs = load_pdf(str(DATA / "sample_policy.pdf"), "doc_x")
    _check(docs, "pdf")
    assert len(docs) == 4 and docs[1]["metadata"]["page_number"] == 2
    assert "18 days" in docs[1]["text"]


def test_load_pdf_falls_back_to_ocr_for_blank_pages(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    p = tmp_path / "scan.pdf"
    w = PdfWriter(); w.add_blank_page(200, 200)
    with open(p, "wb") as fh:
        w.write(fh)
    called = {}

    def fake_ocr(path, document_id=None, pages=None):
        called["pages"] = pages
        return [{"text": "OCR text", "metadata": {"source": "scan.pdf", "source_type": "pdf",
                                                  "page_number": 1, "document_id": document_id, "ocr": True}}]

    monkeypatch.setattr(loaders, "load_scanned_pdf", fake_ocr)
    docs = load_pdf(str(p), "doc_y")
    assert called["pages"] == [1] and docs[0]["metadata"]["ocr"] is True


def test_load_markdown_sections():
    docs = load_markdown(str(DATA / "sample_notes.md"), "doc_m")
    _check(docs, "markdown")
    assert {d["metadata"]["section"] for d in docs} >= {"First Week", "Probation"}


def test_load_csv_row_per_document():
    docs = load_csv(str(DATA / "sample_faq.csv"), "doc_c")
    _check(docs, "csv")
    assert len(docs) == 5 and "question:" in docs[0]["text"]


def test_load_html_strips_scripts(tmp_path):
    f = tmp_path / "p.html"
    f.write_text("<html><body><script>evil()</script><h1>Title</h1><p>Hello world</p></body></html>")
    docs = load_html(str(f), "doc_h")
    _check(docs, "html")
    assert "evil" not in docs[0]["text"] and "Hello world" in docs[0]["text"]


def test_website_blocks_private_addresses():
    with pytest.raises(LoaderError):
        loaders.load_website("http://127.0.0.1:8000/admin")
    with pytest.raises(LoaderError):
        loaders.load_website("file:///etc/passwd")


def test_dispatcher():
    assert load_document(str(DATA / "sample_faq.csv"), "csv")[0]["metadata"]["source_type"] == "csv"
    with pytest.raises(LoaderError):
        load_document("x", "exe")
