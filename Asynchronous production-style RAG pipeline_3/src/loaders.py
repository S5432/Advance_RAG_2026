"""Document loaders. Every loader returns:
[{"text": str, "metadata": {"source", "source_type", "page_number", "document_id", ...}}]
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
import uuid
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MIN_OCR_CONFIDENCE = 60.0  # mean Tesseract confidence below this is flagged as low quality
REQUEST_TIMEOUT = 15
MAX_HTML_BYTES = 5 * 1024 * 1024


class LoaderError(Exception):
    """Raised when a document cannot be loaded."""


def _new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


def _record(text: str, source: str, source_type: str, page: int, document_id: str, **extra) -> dict:
    meta = {
        "source": source,
        "source_type": source_type,
        "page_number": page,
        "document_id": document_id,
    }
    meta.update({k: v for k, v in extra.items() if v is not None})
    return {"text": text, "metadata": meta}


# --------------------------------------------------------------------- PDF
def load_pdf(path: str, document_id: str | None = None) -> list[dict]:
    """Extract text page by page. Pages with no selectable text are OCR'd."""
    from pypdf import PdfReader

    document_id = document_id or _new_document_id()
    name = os.path.basename(path)
    try:
        reader = PdfReader(path)
        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise LoaderError(f"Cannot read PDF '{name}': {exc}") from exc

    docs: list[dict] = []
    empty_pages: list[int] = []
    for i, text in enumerate(page_texts, start=1):
        if text:
            docs.append(_record(text, name, "pdf", i, document_id))
        else:
            empty_pages.append(i)

    if empty_pages:
        logger.info("%s: %d page(s) without text, falling back to OCR", name, len(empty_pages))
        docs.extend(load_scanned_pdf(path, document_id, pages=empty_pages))
        docs.sort(key=lambda d: d["metadata"]["page_number"])
    return docs


def load_scanned_pdf(
    path: str, document_id: str | None = None, pages: list[int] | None = None
) -> list[dict]:
    """OCR a scanned/image PDF with pdf2image + pytesseract (page-wise).

    `pages` restricts OCR to specific 1-based page numbers (used for mixed PDFs);
    None means every page.
    """
    import pytesseract
    from pdf2image import convert_from_path
    from pytesseract import Output

    document_id = document_id or _new_document_id()
    name = os.path.basename(path)

    images: list[tuple[int, object]] = []
    try:
        if pages is None:
            images = list(enumerate(convert_from_path(path, dpi=300), start=1))
        else:
            for p in pages:
                images.extend((p, img) for img in convert_from_path(path, dpi=300, first_page=p, last_page=p))
    except Exception as exc:
        raise LoaderError(f"OCR rasterisation failed for '{name}': {exc}") from exc

    docs: list[dict] = []
    for page_no, image in images:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        pairs = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        text = " ".join(w for w, _ in pairs).strip()
        if not text:
            continue
        mean_conf = round(sum(c for _, c in pairs) / len(pairs), 1)
        docs.append(
            _record(
                text, name, "pdf", page_no, document_id,
                ocr=True,
                ocr_confidence=mean_conf,
                low_ocr_quality=mean_conf < MIN_OCR_CONFIDENCE,
            )
        )
    return docs


# -------------------------------------------------------------------- DOCX
def load_docx(path: str, document_id: str | None = None) -> list[dict]:
    import docx2txt

    document_id = document_id or _new_document_id()
    name = os.path.basename(path)
    try:
        text = (docx2txt.process(path) or "").strip()
    except Exception as exc:
        raise LoaderError(f"Cannot read DOCX '{name}': {exc}") from exc
    text = "\n".join(line.rstrip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return [_record(text, name, "docx", 1, document_id)] if text else []


# -------------------------------------------------------------------- HTML
def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def load_html(path_or_url: str, document_id: str | None = None) -> list[dict]:
    if path_or_url.lower().startswith(("http://", "https://")):
        return load_website(path_or_url, document_id)
    document_id = document_id or _new_document_id()
    name = os.path.basename(path_or_url)
    try:
        with open(path_or_url, "r", encoding="utf-8", errors="ignore") as fh:
            text = _html_to_text(fh.read())
    except OSError as exc:
        raise LoaderError(f"Cannot read HTML '{name}': {exc}") from exc
    return [_record(text, name, "html", 1, document_id)] if text else []


# ----------------------------------------------------------------- Website
# def _assert_public_url(url: str) -> None:
#     """SSRF guard: only http(s) URLs that resolve to public IP addresses."""
#     parsed = urlparse(url)
#     if parsed.scheme not in ("http", "https") or not parsed.hostname:
#         raise LoaderError("Only http(s) URLs are allowed")
#     try:
#         infos = socket.getaddrinfo(parsed.hostname, None)
#     except socket.gaierror as exc:
#         raise LoaderError(f"Cannot resolve host '{parsed.hostname}'") from exc
#     for info in infos:
#         ip = ipaddress.ip_address(info[4][0])
#         if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
#             raise LoaderError("URL resolves to a non-public address")



def _assert_public_url(url: str) -> None:
    """
    Reject URLs that do not resolve exclusively to public IP addresses.
    """

    parsed = urlparse(url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise LoaderError("Only valid public HTTP(S) URLs are allowed.")

    try:
        infos = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise LoaderError(
            f"Cannot resolve host '{parsed.hostname}'"
        ) from exc

    if not infos:
        raise LoaderError("Hostname did not resolve to an IP address.")

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])

        if not ip.is_global:
            raise LoaderError(
                "URL resolves to a non-public IP address."
            )


def load_website(url: str, document_id: str | None = None) -> list[dict]:
    _assert_public_url(url)
    document_id = document_id or _new_document_id()
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "DocQA-Bot/1.0"}, stream=True)
        resp.raise_for_status()
        content = resp.raw.read(MAX_HTML_BYTES + 1, decode_content=True)
    except requests.RequestException as exc:
        raise LoaderError(f"Cannot fetch '{url}': {exc}") from exc
    if len(content) > MAX_HTML_BYTES:
        raise LoaderError("Web page is too large")
    html = content.decode(resp.encoding or "utf-8", errors="ignore")
    # soup = BeautifulSoup(html, "html.parser")
    # title = soup.title.string.strip() if soup.title and soup.title.string else None
    # text = _html_to_text(html)
    # return [_record(text, url, "url", 1, document_id, title=title)] if text else []
    
    soup = BeautifulSoup(html, "html.parser")

    title = (soup.title.get_text(strip=True) if soup.title else None)

    text = _html_to_text(html)
    return [_record(text, url, "url", 1, document_id, page_title=title,)] if text else []

    
# ---------------------------------------------------------------- Markdown
def load_markdown(path: str, document_id: str | None = None) -> list[dict]:
    """Split by headings so each record carries its `section`."""
    document_id = document_id or _new_document_id()
    name = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise LoaderError(f"Cannot read Markdown '{name}': {exc}") from exc

    docs: list[dict] = []
    section: str | None = None
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            docs.append(_record(text, name, "markdown", 1, document_id, section=section))

    for line in lines:
        if line.lstrip().startswith("#"):
            flush()
            buf = [line]
            section = line.lstrip("# ").strip() or section
        else:
            buf.append(line)
    flush()
    return docs


# --------------------------------------------------------------------- CSV
def load_csv(path: str, document_id: str | None = None) -> list[dict]:
    """One record per row: 'col: value | col: value'. `page_number` holds the row number."""
    document_id = document_id or _new_document_id()
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        raise LoaderError(f"Cannot read CSV '{name}': {exc}") from exc

    docs = []
    for i, row in enumerate(df.to_dict(orient="records"), start=1):
        text = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        if text:
            docs.append(_record(text, name, "csv", i, document_id, row_number=i))
    return docs


# --------------------------------------------------------------- Dispatcher
def load_document(path_or_url: str, source_type: str, document_id: str | None = None) -> list[dict]:
    """Automatically select the correct loader from `source_type`."""
    st = source_type.lower().strip()
    loaders = {
        "pdf": load_pdf,
        "scanned_pdf": load_scanned_pdf,
        "docx": load_docx,
        "html": load_html,
        "url": load_website,
        "website": load_website,
        "markdown": load_markdown,
        "md": load_markdown,
        "csv": load_csv,
    }
    if st not in loaders:
        raise LoaderError(f"Unsupported source_type '{source_type}'")
    return loaders[st](path_or_url, document_id)
