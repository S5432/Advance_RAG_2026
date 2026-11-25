"""Day 11 chunking: fixed, recursive (default), heading-aware, parent-child + metadata enrichment."""
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


# ============================================================ Day 11 strategies
_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_STRICT_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+\S.*|[A-Z][A-Za-z0-9&/,'-]*(?: [A-Za-z0-9&/,'-]+){0,6})$")


def count_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per word); good enough for metadata/observability."""
    return max(1, round(len(text.split()) * 1.3)) if text.strip() else 0


def _check(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")


def fixed_size_chunking(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Plain sliding character window (baseline; may cut sentences)."""
    _check(chunk_size, overlap)
    text = text.strip()
    step = chunk_size - overlap
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


def recursive_chunking(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """DEFAULT. paragraph -> sentence -> word packing; fenced code blocks stay whole (<= 2x chunk_size)."""
    _check(chunk_size, overlap)
    out: list[str] = []
    for part in _CODE_FENCE.split(text):
        part = part.strip()
        if not part:
            continue
        if part.startswith("```") and len(part) <= chunk_size * 2:
            out.append(part)  # never split a code block
        else:
            out.extend(split_text(part, chunk_size, overlap))
    return out


def _is_heading(line: str, prev_blank: bool, next_line: str | None) -> str | None:
    """Return the heading title when `line` is a heading, else None."""
    m = _MD_HEADING.match(line.strip())
    if m:
        return m.group(1).strip()
    st = line.strip()
    if (prev_blank and next_line and next_line.strip() and len(st) <= 60
            and not st.endswith((".", ",", ";", ":")) and _STRICT_HEADING.match(st)
            and len(next_line.strip()) > len(st)):
        return st
    return None


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """[(heading or None, body)] - Markdown '#', numbered or Title-Case headings; code fences are skipped."""
    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    in_code, prev_blank = False, True
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
        title = None if in_code else _is_heading(line, prev_blank, lines[i + 1] if i + 1 < len(lines) else None)
        if title:
            sections.append((title, []))
        else:
            sections[-1][1].append(line)
        prev_blank = not line.strip()
    return [(t, "\n".join(b).strip()) for t, b in sections if t or "".join(b).strip()]


def _heading_pieces(text: str, max_chunk_size: int) -> list[tuple[str | None, str]]:
    """[(section title, chunk text)]; a section is kept whole when it fits, else split + title repeated."""
    pieces: list[tuple[str | None, str]] = []
    for title, body in split_sections(text):
        whole = f"{title}\n{body}".strip() if title else body
        if len(whole) <= max_chunk_size:
            pieces.append((title, whole))
            continue
        budget = max(max_chunk_size - (len(title) + 1 if title else 0), 100)
        for part in recursive_chunking(body, budget, min(150, budget // 4)):
            pieces.append((title, f"{title}\n{part}" if title else part))
    return pieces


def heading_aware_chunking(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Markdown / HTML / structured-PDF friendly: one chunk per section, big sections split with title kept."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")
    return [t for _, t in _heading_pieces(text, max_chunk_size)]


def _meta_for(meta: dict, extra: dict | None, **fields) -> dict:
    m = dict(meta)
    m.update(extra or {})
    m.update({k: v for k, v in fields.items()})
    if m.get("page_number") is None:
        m["page_number"] = 1
    return m


def parent_child_chunking(documents: list[dict], parent_size: int = 2000, child_size: int = 400,
                          child_overlap: int = 60, extra_metadata: dict | None = None) -> list[dict]:
    """Large PARENT chunks (context for the LLM) + small CHILD chunks (precise retrieval).

    Returns parents and children (chunk_type='parent'|'child'). Children carry `parent_chunk_id`;
    ids are continuous per document: `<doc>_parent_<n>` and `<doc>_chunk_<i>`.
    Only children should be embedded; parents are stored in PostgreSQL and fetched at answer time.
    """
    out: list[dict] = []
    parents_n: dict[str, int] = {}
    kids_n: dict[str, int] = {}
    for doc in documents:
        meta, text = doc.get("metadata", {}), (doc.get("text") or "").strip()
        doc_id = meta.get("document_id")
        if not doc_id:
            raise ValueError("every document needs metadata['document_id']")
        if not text:
            continue
        for title, parent_text in _heading_pieces(text, parent_size):
            pn = parents_n.get(doc_id, 0)
            parents_n[doc_id] = pn + 1
            pid = f"{doc_id}_parent_{pn}"
            base = _meta_for(meta, extra_metadata, chunking_strategy="parent_child")
            if title or meta.get("section"):
                base["section"] = title or meta["section"]
            out.append({"text": parent_text, "metadata": {**base, "chunk_type": "parent", "chunk_id": pid,
                                                          "chunk_index": pn, "parent_chunk_id": None,
                                                          "token_count": count_tokens(parent_text)}})
            for child in recursive_chunking(parent_text, child_size, child_overlap):
                i = kids_n.get(doc_id, 0)
                kids_n[doc_id] = i + 1
                out.append({"text": child, "metadata": {**base, "chunk_type": "child", "chunk_id": f"{doc_id}_chunk_{i}",
                                                        "chunk_index": i, "parent_chunk_id": pid,
                                                        "token_count": count_tokens(child)}})
    return out


def choose_best_chunks(chunks: list[dict]) -> list[dict]:
    """Quality gate: drop empty / punctuation-only chunks and exact duplicates (per document)."""
    seen: set[tuple] = set()
    best: list[dict] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not re.search(r"\w", text):
            continue
        meta = c.get("metadata", {})
        key = (meta.get("document_id"), meta.get("chunk_type"), " ".join(text.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        best.append(c)
    return best


def choose_strategy(meta: dict, text: str, requested: str = "recursive") -> str:
    """recursive is the default; Markdown/HTML/URL and structured PDFs use heading-aware chunking."""
    if requested in ("fixed", "parent_child", "heading"):
        return requested
    if meta.get("source_type") in ("markdown", "html", "url"):
        return "heading"
    if meta.get("source_type") == "pdf":
        lines = text.splitlines()
        heads = sum(1 for i, ln in enumerate(lines)
                    if _is_heading(ln, i == 0 or not lines[i - 1].strip(), lines[i + 1] if i + 1 < len(lines) else None))
        if heads >= 2:
            return "heading"
    return "recursive"


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    extra_metadata: dict | None = None,
    strategy: str = "recursive",
) -> list[dict]:
    """Chunks ready for Pinecone: {"text", "metadata": {..., chunk_index, chunk_id, parent_chunk_id,
    chunk_type, chunking_strategy, token_count, language, access_level, section, page_number}}."""
    _check(chunk_size, chunk_overlap)
    if strategy == "parent_child":
        docs = [d for d in documents]
        chunks = parent_child_chunking(docs, extra_metadata=extra_metadata)
        for c in chunks:
            c["metadata"].setdefault("language", "en")
            c["metadata"].setdefault("access_level", "internal")
        return chunks

    raw: list[dict] = []
    last_section: dict[str, str] = {}
    for doc in documents:
        meta = doc.get("metadata", {})
        doc_id = meta.get("document_id")
        if not doc_id:
            raise ValueError("every document needs metadata['document_id']")
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        used = choose_strategy(meta, text, strategy)
        if meta.get("section"):  # loader-provided section (Markdown) wins
            last_section[doc_id] = meta["section"]
        if used == "heading":
            pieces = _heading_pieces(text, max(chunk_size, 1000))
        else:
            fn = fixed_size_chunking if used == "fixed" else recursive_chunking
            pieces = [(None, p) for p in fn(text, chunk_size, chunk_overlap)]
        for title, piece in pieces:
            if title:
                last_section[doc_id] = title
            elif not meta.get("section"):
                for ln in [l for l in piece.splitlines()[:2] if l.strip()]:
                    if _looks_like_heading(ln):
                        last_section[doc_id] = ln.strip()
                        break
            m = _meta_for(meta, extra_metadata, chunking_strategy=used, chunk_type="standalone", parent_chunk_id=None)
            if doc_id in last_section:
                m["section"] = last_section[doc_id]
            raw.append({"text": piece, "metadata": m})

    counters: dict[str, int] = {}
    chunks = choose_best_chunks(raw)
    for c in chunks:
        m = c["metadata"]
        idx = counters.get(m["document_id"], 0)
        counters[m["document_id"]] = idx + 1
        m["chunk_index"], m["chunk_id"] = idx, f"{m['document_id']}_chunk_{idx}"
        m["token_count"] = count_tokens(c["text"])
        m.setdefault("language", "en")
        m.setdefault("access_level", "internal")
    return chunks
