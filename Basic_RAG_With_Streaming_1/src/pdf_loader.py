import logging
import re
from pathlib import Path
from typing import BinaryIO
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger("pdf_loader")

class PDFExtractionError(Exception):
    """The file could not be opened as a readable PDF."""

def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)              # collapse spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pdf_text(file_path: str | Path | BinaryIO) -> list[dict]:
    """One record per page: {"page": 1-based number, "text": cleaned text}.
    Pages without extractable text (blank or scanned images) are returned with text "" so page numbers stay aligned;
    a failure on a single page is logged and treated as empty. Raises PDFExtractionError for unreadable / encrypted files."""
    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFExtractionError("PDF is password protected")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = _clean(page.extract_text() or "")
            except Exception as exc:                      # one bad page must not lose the whole document
                logger.warning("pdf_page_extraction_failed page=%d error=%s", number, type(exc).__name__)
                text = ""
            pages.append({"page": number, "text": text})
        return pages
    except PDFExtractionError:
        raise
    except FileNotFoundError as exc:
        raise PDFExtractionError(f"File not found: {file_path}") from exc
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as exc:
        raise PDFExtractionError(f"Could not read PDF: {type(exc).__name__}") from exc
