import io
import pytest
from pypdf import PdfWriter
from src.pdf_loader import PDFExtractionError, extract_pdf_text
from tests.conftest import SAMPLE_PDF

def test_sample_pdf_has_three_pages_with_numbers_and_text():
    pages = extract_pdf_text(str(SAMPLE_PDF))
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "12 paid leaves" in pages[0]["text"] and "Refund" in pages[1]["text"] and "password" in pages[2]["text"].lower()
    assert all(set(p) == {"page", "text"} for p in pages)

def test_accepts_file_like_object():
    assert len(extract_pdf_text(io.BytesIO(SAMPLE_PDF.read_bytes()))) == 3

def test_blank_page_returns_empty_text_and_keeps_numbering():
    w = PdfWriter(); w.add_blank_page(200, 200); w.add_blank_page(200, 200)
    buf = io.BytesIO(); w.write(buf); buf.seek(0)
    assert extract_pdf_text(buf) == [{"page": 1, "text": ""}, {"page": 2, "text": ""}]

@pytest.mark.parametrize("payload", [b"this is not a pdf", b"", b"%PDF-1.4\ngarbage"])
def test_corrupt_files_raise_pdf_extraction_error(payload):
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(io.BytesIO(payload))

def test_missing_file_raises_pdf_extraction_error():
    with pytest.raises(PDFExtractionError):
        extract_pdf_text("/no/such/file.pdf")

def test_encrypted_pdf_is_rejected():
    w = PdfWriter(); w.add_blank_page(100, 100); w.encrypt("secret")
    buf = io.BytesIO(); w.write(buf); buf.seek(0)
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(buf)
