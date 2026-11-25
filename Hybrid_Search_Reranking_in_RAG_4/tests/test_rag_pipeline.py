"""Phase 6: hybrid + rerank pipeline inside the RAG chain (fakes only; no network, no database)."""
import pytest

from src.filters import build_metadata_filter
from src.rag_chain import (MODES, NOT_FOUND, RetrievalConfig, answer_question, retrieve_context,
                           stream_rag_answer)
from tests.fakes import FakeGemini, FakePinecone, FakeReranker, keyword_docs

T1 = build_metadata_filter("t1")
NO_MIN = RetrievalConfig(rerank_min_score=0.0)

TEXTS = {
    "c0": "Remote work is allowed two days per week with manager approval",
    "c1": "Employees are entitled to 18 days of paid annual leave per year",
    "c2": "Error ERR-5021 means the payroll export failed",
    "c3": "Expense reports are due monthly",
}


def _pc(tenant="t1", extra=None):
    pc = FakePinecone()
    for cid, text in {**TEXTS, **(extra or {})}.items():
        pc.store[cid] = {"_id": cid, "chunk_text": text, "chunk_id": cid, "document_id": "doc_1",
                         "document_name": "policy.pdf", "page_number": 3, "section": "Leave",
                         "source_type": "pdf", "tenant_id": tenant}
    return pc


def _ctx(q, pc=None, mode="hybrid_rerank", top_k=5, config=NO_MIN, reranker=None, docs=None):
    pc = pc or _pc()
    return retrieve_context(q, pc, T1, top_k, config=config, mode=mode,
                            documents=keyword_docs(pc) if docs is None else docs,
                            rerank_client=reranker or FakeReranker())


# --------------------------------------------------------------- retrieval
def test_reranker_promotes_the_relevant_chunk_and_scores_are_kept():
    ctx = _ctx("how many days of annual leave")
    top = ctx["chunks"][0]
    assert top["chunk_id"] == "c1" and top["source_id"] == 1 and top["rerank_score"] > 0
    assert top["score"] == top["rerank_score"]                       # ordering score = rerank score
    assert top["dense_score"] == 0.9 and top["keyword_score"] > 0 and top["rrf_score"] > 0
    assert (top["document_name"], top["page_number"], top["section"]) == ("policy.pdf", 3, "Leave")
    assert [c["source_id"] for c in ctx["chunks"]] == list(range(1, len(ctx["chunks"]) + 1))


def test_final_top_k_is_respected_and_stats_are_reported():
    ctx = _ctx("annual leave days", top_k=2)
    st = ctx["stats"]
    assert len(ctx["chunks"]) == 2 and st["mode"] == "hybrid_rerank"
    assert st["dense_candidates"] == 4 and st["fused_candidates"] == 4 and st["keyword_candidates"] >= 1
    assert st["rerank"]["candidates_in"] == 4 and st["rerank"]["returned"] == 2 and st["rerank"]["fallback"] is False
    assert {"dense", "keyword", "fusion", "total"} <= set(st["latency_ms"])


def test_single_stage_modes():
    q = "ERR-5021 payroll export"
    dense = _ctx(q, mode="dense")["chunks"]
    keyword = _ctx(q, mode="keyword")["chunks"]
    hybrid = _ctx(q, mode="hybrid")
    assert len(dense) == 4 and all(c["rerank_score"] is None for c in dense)     # fake dense: all tenant chunks
    assert keyword[0]["chunk_id"] == "c2" and keyword[0]["score"] == keyword[0]["keyword_score"] > 0
    assert hybrid["chunks"][0]["chunk_id"] == "c2" and hybrid["stats"]["rerank"] is None
    assert hybrid["chunks"][0]["score"] == hybrid["chunks"][0]["rrf_score"]
    with pytest.raises(ValueError):
        _ctx(q, mode="nope")
    assert set(MODES) == {"dense", "keyword", "hybrid", "hybrid_rerank"}


def test_min_score_returns_nothing_relevant():
    ctx = _ctx("quantum chromodynamics lattice", config=RetrievalConfig(rerank_min_score=0.05))
    assert ctx["chunks"] == [] and ctx["stats"]["rerank"]["below_threshold"] == 4


def test_reranker_outage_falls_back_to_rrf_order_with_rrf_score():
    ctx = _ctx("annual leave", reranker=FakeReranker(always_fail=True, status=400))
    assert ctx["stats"]["rerank"]["fallback"] is True
    top = ctx["chunks"][0]
    assert top["rerank_score"] is None and top["score"] == top["rrf_score"] > 0


def test_tenant_isolation_and_deleted_guard_hold_through_the_pipeline():
    pc = _pc(extra={"x9": "annual leave days for another tenant"})
    pc.store["x9"]["tenant_id"] = "t2"
    ids = [c["chunk_id"] for c in _ctx("annual leave days", pc=pc)["chunks"]]
    assert "x9" not in ids
    # a vector whose PostgreSQL chunk is gone (soft-deleted / purge pending) must not be used
    docs = [d for d in keyword_docs(pc) if d["chunk_id"] != "c1"]
    ids = [c["chunk_id"] for c in _ctx("annual leave days", pc=pc, docs=docs)["chunks"]]
    assert "c1" not in ids
    assert _ctx("annual leave", docs=[])["chunks"] == []             # nothing live -> nothing retrieved


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        _ctx("   ")


# ------------------------------------------------------------------ answers
def _stream(q, pc=None, gem=None, config=NO_MIN, reranker=None):
    pc = pc or _pc()
    return list(stream_rag_answer(q, pc, gem or FakeGemini(), 5, metadata_filter=T1, documents=keyword_docs(pc),
                                  config=config, rerank_client=reranker or FakeReranker()))


def test_answer_question_returns_citations_and_retrieval_stats():
    pc = _pc()
    out = answer_question("annual leave days", pc, FakeGemini(), 5, metadata_filter=T1,
                          config=NO_MIN, documents=keyword_docs(pc), rerank_client=FakeReranker())
    assert out["retrieved_chunks"] == 4 and out["citations"][0]["source_id"] == 1
    assert out["citations"][0]["rerank_score"] is not None and out["retrieval"]["rerank"]["returned"] == 4


def test_answer_question_skips_llm_when_nothing_is_relevant():
    pc = _pc()
    out = answer_question("quantum chromodynamics", pc, FakeGemini(fail_after=0), 5, metadata_filter=T1,
                          config=RetrievalConfig(rerank_min_score=0.05), documents=keyword_docs(pc),
                          rerank_client=FakeReranker())
    assert out["answer"] == NOT_FOUND and out["citations"] == [] and out["retrieved_chunks"] == 0


def test_stream_event_contract():
    ev = _stream("annual leave days")
    names = [e["event"] for e in ev]
    assert names == ["retrieval", "rerank", "sources"] + ["token"] * 4 + ["done"]
    retrieval, rerank, sources, done = ev[0]["data"], ev[1]["data"], ev[2]["data"]["sources"], ev[-1]["data"]
    assert retrieval["dense_candidates"] == 4 and "latency_ms" in retrieval
    assert rerank["model"] == "bge-reranker-v2-m3" and rerank["candidates_in"] == 4 and rerank["fallback"] is False
    assert sources[0]["rerank_score"] is not None and "text" not in sources[0]       # sources never carry full text
    assert done["status"] == "completed" and done["citations"][0]["source_id"] == 1 and done["latency_ms"] >= 0


def test_stream_nothing_relevant_skips_llm():
    ev = _stream("quantum chromodynamics", gem=FakeGemini(fail_after=0), config=RetrievalConfig(rerank_min_score=0.05))
    assert [e["event"] for e in ev] == ["retrieval", "rerank", "sources", "token", "done"]
    assert ev[2]["data"]["sources"] == [] and ev[3]["data"]["content"] == NOT_FOUND


def test_stream_rerank_fallback_is_reported_not_fatal():
    ev = _stream("annual leave", reranker=FakeReranker(always_fail=True, status=400))
    assert ev[1]["event"] == "rerank" and ev[1]["data"]["fallback"] is True and ev[-1]["data"]["status"] == "completed"


def test_stream_retrieval_failure_emits_single_error():
    class Broken(FakePinecone):
        def query(self, *a, **k):
            raise RuntimeError("pinecone down")
    ev = _stream("annual leave", pc=Broken())
    assert [e["event"] for e in ev] == ["error"] and ev[0]["data"]["stage"] == "retrieval"
