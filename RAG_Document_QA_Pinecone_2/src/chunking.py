"""Sentence-aware chunking that preserves metadata and adds chunk_index / chunk_id."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 &/,'-]{2,60}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 60 and not line.endswith((".", ",", ";")) and bool(_HEADING.match(line))


def _split_units(text: str, limit: int) -> list[str]:
    """Break text into paragraph -> sentence -> word units, each <= limit characters."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            units.append(para)
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= limit:
                units.append(sent)
            else:  # very long sentence: fall back to word windows
                words, cur = sent.split(), ""
                for w in words:
                    if len(cur) + len(w) + 1 > limit and cur:
                        units.append(cur)
                        cur = w
                    else:
                        cur = f"{cur} {w}".strip()
                if cur:
                    units.append(cur)
    return units


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Pack whole sentences/paragraphs into chunks of <= chunk_size characters.

    The last sentences of a chunk (up to `chunk_overlap` characters) are repeated at the
    start of the next one so context is not lost at boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    units = _split_units(text, chunk_size)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        add = len(unit) + (1 if current else 0)
        if current and length + add > chunk_size:
            chunks.append(" ".join(current))
            # build overlap from trailing units
            overlap: list[str] = []
            total = 0
            for u in reversed(current):
                if total + len(u) + 1 > chunk_overlap:
                    break
                overlap.insert(0, u)
                total += len(u) + 1
            current = overlap
            length = sum(len(u) + 1 for u in current) - (1 if current else 0)
            add = len(unit) + (1 if current else 0)
            if length + add > chunk_size:  # overlap + unit too big -> drop overlap
                current, length, add = [], 0, len(unit)
        current.append(unit)
        length += add
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """Return chunks ready for embedding.

    Each chunk: {"text": ..., "metadata": {<source metadata>, "chunk_index", "chunk_id",
    "section" (when known), **extra_metadata}}. `chunk_index` is continuous per document_id.
    """
    chunks: list[dict] = []
    counters: dict[str, int] = {}
    last_section: dict[str, str] = {}

    for doc in documents:
        meta = doc.get("metadata", {})
        doc_id = meta.get("document_id")
        if not doc_id:
            raise ValueError("every document needs metadata['document_id']")
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        # a loader-provided section (Markdown) wins; otherwise detect headings in the text
        if meta.get("section"):
            last_section[doc_id] = meta["section"]

        for piece in split_text(text, chunk_size, chunk_overlap):
            if not meta.get("section"):
                first_lines = [ln for ln in piece.splitlines()[:2] if ln.strip()]
                for ln in first_lines:
                    if _looks_like_heading(ln):
                        last_section[doc_id] = ln.strip()
                        break
            idx = counters.get(doc_id, 0)
            counters[doc_id] = idx + 1

            new_meta = dict(meta)
            new_meta.update(extra_metadata or {})
            new_meta["chunk_index"] = idx
            new_meta["chunk_id"] = f"{doc_id}_chunk_{idx}"
            if new_meta.get("page_number") is None:
               new_meta["page_number"] = 1
            if doc_id in last_section:
                new_meta["section"] = last_section[doc_id]
            chunks.append({"text": piece, "metadata": new_meta})
    return chunks
