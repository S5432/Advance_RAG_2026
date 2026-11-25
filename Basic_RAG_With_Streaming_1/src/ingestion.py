import hashlib
import io
from src.chunking import chunk_text
from src.pdf_loader import extract_pdf_text

class IngestionError(ValueError):
    """The PDF is readable but cannot be ingested (e.g. no extractable text)."""

def make_document_id(data: bytes) -> str:
    """Content-based id: uploading the same file again replaces it instead of duplicating chunks."""
    return "doc_" + hashlib.sha256(data).hexdigest()[:12]

def ingest_pdf(data: bytes, filename: str, store, chunk_size: int = 800, chunk_overlap: int = 150,
               document_id: str | None = None) -> dict:
    """PDF bytes -> pages -> chunks (never spanning pages, so every chunk has an exact page) -> embeddings -> Chroma."""
    pages = extract_pdf_text(io.BytesIO(data))
    chunks, metas = [], []
    for page in pages:
        for piece in chunk_text(page["text"], chunk_size, chunk_overlap):
            chunks.append(piece)
            metas.append({"page": page["page"], "document_name": filename, "page_count": len(pages)})
    if not chunks:
        raise IngestionError("The PDF has no extractable text (it may be a scanned image; OCR is required).")
    doc_id = document_id or make_document_id(data)
    store.delete_document(doc_id)                                   # idempotent re-upload
    count = store.add_document(doc_id, chunks, metas)
    return {"document_id": doc_id, "document_name": filename, "page_count": len(pages), "chunk_count": count,
            "pages_without_text": [p["page"] for p in pages if not p["text"]], "status": "ingested"}
