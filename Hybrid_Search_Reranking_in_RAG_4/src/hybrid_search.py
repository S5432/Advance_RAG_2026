"""Hybrid retrieval: dense (Pinecone, llama-text-embed-v2) + keyword (BM25 over PostgreSQL chunks) + RRF.

    filter -> dense top-20  \
                             >-- Reciprocal Rank Fusion -> top-20 candidates (-> reranker)
    filter -> keyword top-20/

Every candidate dict has the same shape:
  chunk_id, text, document_id, document_name, source_type, page_number, section, parent_chunk_id,
  chunk_type, dense_score, keyword_score, rrf_score, dense_rank, keyword_rank, metadata
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter

from .filters import epoch_of, matches_filter

logger = logging.getLogger(__name__)

BM25_K1, BM25_B = 1.5, 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_STOP = frozenset("a an and are as at be by do does for how in is it many of on or that the this to was what when "
                  "which with".split())


# ------------------------------------------------------------------ helpers
def _tokenize(text: str) -> list[str]:
    """Lower-case terms; codes such as 'ERR-5021' keep the whole token AND its parts ('err', '5021')."""
    out: list[str] = []
    for tok in _TOKEN.findall((text or "").lower()):
        out.append(tok)
        if re.search(r"[-_.]", tok):
            out.extend(p for p in re.split(r"[-_.]+", tok) if p)
    return [t for t in out if t not in _STOP]


def _candidate(chunk_id: str, text: str, meta: dict, **scores) -> dict:
    return {
        "chunk_id": chunk_id, "text": text,
        "document_id": meta.get("document_id"), "document_name": meta.get("document_name", "unknown"),
        "source_type": meta.get("source_type"), "page_number": meta.get("page_number"),
        "section": meta.get("section") or None, "parent_chunk_id": meta.get("parent_chunk_id"),
        "chunk_type": meta.get("chunk_type", "standalone"),
        "dense_score": 0.0, "keyword_score": 0.0, "rrf_score": 0.0, "dense_rank": None, "keyword_rank": None,
        "metadata": {k: v for k, v in meta.items() if k != "text"},
        **scores,
    }


def _meta(doc: dict) -> dict:
    """Metadata of a keyword document (nested `metadata` wins over top-level fields)."""
    flat = {k: v for k, v in doc.items() if k not in ("text", "metadata")}
    return {**flat, **(doc.get("metadata") or {})}


# -------------------------------------------------------------------- dense
def dense_search(query: str, pinecone_service, top_k: int = 20, metadata_filter: dict | None = None) -> list[dict]:
    """Semantic search: Pinecone embeds the query with llama-text-embed-v2 (cosine). Filter applied inside Pinecone."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    matches = pinecone_service.query(query_text=query.strip(), top_k=top_k, metadata_filter=metadata_filter)
    results = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata") or {}
        text = md.get("text", "")
        if not text.strip():
            logger.warning("dense hit without text skipped: %s", m.get("id"))
            continue
        cid = md.get("chunk_id") or m.get("id")
        results.append(_candidate(cid, text, md, dense_score=float(m.get("score", 0.0)), dense_rank=i))
    return results


# ------------------------------------------------------------------ keyword
def keyword_search(query: str, documents: list[dict], top_k: int = 20,
                   metadata_filter: dict | None = None) -> list[dict]:
    """BM25 over `documents` ({chunk_id, text, metadata}); the metadata filter is applied BEFORE scoring."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    pool = [d for d in documents if matches_filter(_meta(d), metadata_filter)]
    terms = sorted(set(_tokenize(query)))
    if not pool or not terms:
        return []

    tokenised = [_tokenize(f"{_meta(d).get('section') or ''} {d.get('text', '')}") for d in pool]
    n = len(pool)
    avg_len = sum(len(t) for t in tokenised) / n or 1.0
    df = Counter(term for toks in tokenised for term in set(toks))

    scored = []
    for d, toks in zip(pool, tokenised):
        tf = Counter(toks)
        score = 0.0
        for term in terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            score += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * len(toks) / avg_len))
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("chunk_id"))))

    results = []
    for rank, (score, d) in enumerate(scored[:top_k], start=1):
        meta = _meta(d)
        results.append(_candidate(d.get("chunk_id") or meta.get("chunk_id"), d.get("text", ""), meta,
                                  keyword_score=score, keyword_rank=rank))
    return results


# ---------------------------------------------------------------------- RRF
def reciprocal_rank_fusion(dense_results: list[dict], keyword_results: list[dict],
                           k: int = 60, top_k: int = 20) -> list[dict]:
    """RRF(d) = sum over lists of 1 / (k + rank). Dedupes by chunk_id, keeps both original scores."""
    if k <= 0:
        raise ValueError("k must be > 0")
    fused: dict[str, dict] = {}
    for results, rank_key, score_key in ((dense_results, "dense_rank", "dense_score"),
                                         (keyword_results, "keyword_rank", "keyword_score")):
        for rank, r in enumerate(results, start=1):  # rank = position in THIS list (1-based)
            cid = r["chunk_id"]
            item = fused.get(cid)
            if item is None:
                item = fused[cid] = {**r, "dense_score": 0.0, "keyword_score": 0.0, "rrf_score": 0.0,
                                     "dense_rank": None, "keyword_rank": None}
            elif not item.get("text"):
                item["text"] = r.get("text", "")
            item[score_key] = r.get(score_key, 0.0)
            item[rank_key] = rank
            item["rrf_score"] += 1.0 / (k + rank)

    def order(it):  # score desc; tie -> better single rank, then chunk_id (deterministic)
        ranks = [x for x in (it["dense_rank"], it["keyword_rank"]) if x is not None]
        return (-it["rrf_score"], min(ranks), it["chunk_id"])

    return sorted(fused.values(), key=order)[:top_k]


# ------------------------------------------------------------------- hybrid
def hybrid_search_detailed(query: str, pinecone_service, documents: list[dict], top_k: int = 20,
                           metadata_filter: dict | None = None, dense_top_k: int = 20,
                           keyword_top_k: int = 20, rrf_k: int = 60) -> dict:
    """Dense + keyword + RRF with all intermediate lists and timings (used by POST /search)."""
    t0 = time.perf_counter()
    live_ids = {(_meta(d).get("chunk_id") or d.get("chunk_id")) for d in documents
                if not _meta(d).get("is_deleted")}
    live_docs = [d for d in documents if (_meta(d).get("chunk_id") or d.get("chunk_id")) in live_ids]

    dense_raw = dense_search(query, pinecone_service, dense_top_k, metadata_filter)
    t1 = time.perf_counter()
    # deleted-document guard: a vector with no live PostgreSQL chunk (soft-deleted, purge pending, stale) is dropped
    dense = [r for r in dense_raw if r["chunk_id"] in live_ids]
    stale = len(dense_raw) - len(dense)
    if stale:
        logger.warning("dropped %d dense hit(s) without a live PostgreSQL chunk (deleted/stale)", stale)

    keyword = keyword_search(query, live_docs, keyword_top_k, metadata_filter)
    t2 = time.perf_counter()
    hybrid = reciprocal_rank_fusion(dense, keyword, k=rrf_k, top_k=top_k)
    t3 = time.perf_counter()

    stats = {
        "dense_candidates": len(dense), "keyword_candidates": len(keyword), "fused_candidates": len(hybrid),
        "stale_dropped": stale,
        "latency_ms": {"dense": round((t1 - t0) * 1000, 1), "keyword": round((t2 - t1) * 1000, 1),
                       "fusion": round((t3 - t2) * 1000, 2), "total": round((t3 - t0) * 1000, 1)},
    }
    logger.info("hybrid q=%r dense=%d keyword=%d fused=%d stale=%d total=%sms", query[:60], len(dense),
                len(keyword), len(hybrid), stale, stats["latency_ms"]["total"])
    return {"dense_results": dense, "keyword_results": keyword, "hybrid_results": hybrid, "stats": stats}


def hybrid_search(query: str, pinecone_service, documents: list[dict], top_k: int = 20,
                  metadata_filter: dict | None = None) -> list[dict]:
    """Top fused candidates (default 20) before reranking."""
    return hybrid_search_detailed(query, pinecone_service, documents, top_k, metadata_filter)["hybrid_results"]


# ------------------------------------------------- PostgreSQL chunk loader
def load_documents_from_db(metadata_filter: dict) -> list[dict]:
    """Live, searchable chunks for the keyword leg: COMPLETED + not deleted + child/standalone only.

    The tenant (mandatory) and simple equality filters are pushed into SQL; the full filter is re-applied
    by keyword_search through `matches_filter`, so both retrieval legs obey identical rules.
    """
    from sqlalchemy import select

    from . import database
    from .models import Document, DocumentChunk, Status

    tenant = ((metadata_filter or {}).get("tenant_id") or {}).get("$eq")
    if not tenant:
        raise ValueError("metadata_filter must contain tenant_id")

    cond = [Document.tenant_id == tenant, Document.is_deleted.is_(False),
            Document.status == Status.COMPLETED.value, DocumentChunk.chunk_type != "parent"]
    for field, col in (("source_type", Document.source_type), ("department", Document.department),
                       ("access_level", Document.access_level)):
        v = (metadata_filter.get(field) or {}).get("$eq")
        if v:
            cond.append(col == v)
    did = metadata_filter.get("document_id") or {}
    if "$eq" in did:
        cond.append(Document.document_id == did["$eq"])
    elif "$in" in did:
        cond.append(Document.document_id.in_(did["$in"]))

    with database.session_scope() as s:
        rows = s.execute(select(DocumentChunk, Document)
                         .join(Document, Document.document_id == DocumentChunk.document_id)
                         .where(*cond)).all()
        return [{
            "chunk_id": c.chunk_id, "text": c.content,
            "metadata": {
                "chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "document_id": d.document_id,
                "document_name": d.filename, "source_type": d.source_type, "source_url": d.source_url,
                "page_number": c.page_number, "section": c.section or "", "parent_chunk_id": c.parent_chunk_id,
                "chunk_type": c.chunk_type, "tenant_id": d.tenant_id, "uploaded_by": d.uploaded_by,
                "department": d.department, "access_level": d.access_level or "internal",
                "created_at_ts": epoch_of(d.created_at), "is_deleted": bool(d.is_deleted),
            },
        } for c, d in rows]
