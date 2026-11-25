"""Stage-2 reranking with Pinecone's hosted cross-encoder (bge-reranker-v2-m3).

    hybrid top-20 candidates --(query + chunk text read together)--> top-5 chunks with `rerank_score`

Original metadata, citations and the dense / keyword / RRF scores are preserved on every result.
If the reranker finds nothing relevant (all scores below `min_score`) the result is EMPTY, so the RAG layer
can answer "not found" instead of grounding on irrelevant text. If the rerank API itself is down, we degrade
to the RRF order (`rerank_score` = None, `fallback` = True) instead of failing the request.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-reranker-v2-m3"
MAX_DOCS = 100            # hosted model limit per request
MAX_CHARS = 3000          # keep query + passage inside the model's 1024-token window
ATTEMPTS = 3


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return status in (408, 425, 429) or status >= 500


@lru_cache(maxsize=1)
def _default_client():
    """Pinecone Inference client (created once per process)."""
    from pinecone import Pinecone

    from .config import get_settings

    return Pinecone(api_key=get_settings().pinecone_api_key).inference


def _prepare(candidates: list[dict]) -> list[dict]:
    """Drop blank texts and duplicate chunk_ids (first occurrence wins), cap at MAX_DOCS."""
    seen, out = set(), []
    for c in candidates:
        cid, text = c.get("chunk_id"), (c.get("text") or "").strip()
        if not cid or not text or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out[:MAX_DOCS]


def _call_with_retry(client, model: str, query: str, docs: list[dict]):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return client.rerank(
                model=model, query=query,
                documents=[{"id": str(c["chunk_id"]), "text": c["text"][:MAX_CHARS]} for c in docs],
                rank_fields=["text"], top_n=len(docs), return_documents=False,
            )
        except Exception as exc:
            if attempt == ATTEMPTS or not _retryable(exc):
                raise
            logger.warning("rerank transient failure (%s); retry %d/%d", exc, attempt, ATTEMPTS - 1)
            time.sleep(0.5 * attempt)


def rerank_chunks_detailed(query: str, candidates: list[dict], top_k: int = 5, *, client=None,
                           model: str = DEFAULT_MODEL, min_score: float = 0.0) -> dict:
    """Rerank and report: {"results", "latency_ms", "fallback", "candidates_in", "below_threshold"}."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    docs = _prepare(candidates)
    info = {"results": [], "latency_ms": 0.0, "fallback": False, "candidates_in": len(docs), "below_threshold": 0}
    if not docs:
        return info

    t0 = time.perf_counter()
    try:
        response = _call_with_retry(client or _default_client(), model, query.strip(), docs)
        scored = []
        for item in _get(response, "data", []) or []:
            idx, score = int(_get(item, "index")), float(_get(item, "score", 0.0))
            if 0 <= idx < len(docs):
                scored.append((score, docs[idx]))
        scored.sort(key=lambda x: -x[0])
        kept = [(s, c) for s, c in scored if s >= min_score]
        info["below_threshold"] = len(scored) - len(kept)
        results = [{**c, "rerank_score": s} for s, c in kept[:top_k]]
    except Exception:
        logger.exception("reranker unavailable; falling back to RRF order")
        info["fallback"] = True
        ordered = sorted(docs, key=lambda c: -float(c.get("rrf_score") or 0.0))
        results = [{**c, "rerank_score": None} for c in ordered[:top_k]]

    info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    info["results"] = results
    logger.info("rerank candidates=%d returned=%d below_threshold=%d fallback=%s latency_ms=%s",
                len(docs), len(results), info["below_threshold"], info["fallback"], info["latency_ms"])
    return info


def rerank_chunks(query: str, candidates: list[dict], top_k: int = 5, *, client=None,
                  model: str = DEFAULT_MODEL, min_score: float = 0.0) -> list[dict]:
    """Top `top_k` chunks by relevance to `query` (empty list = nothing relevant)."""
    return rerank_chunks_detailed(query, candidates, top_k, client=client, model=model,
                                  min_score=min_score)["results"]
