"""Phase 5: Pinecone hosted reranker (fake client; no network)."""
import pytest

from src import reranker
from src.reranker import rerank_chunks, rerank_chunks_detailed
from tests.fakes import FakeReranker


def _cand(cid, text, rrf=0.02, **extra):
    return {"chunk_id": cid, "text": text, "document_id": "doc_1", "document_name": "employee_policy.pdf",
            "page_number": 4, "section": "Leave Policy", "dense_score": 0.8, "keyword_score": 3.2,
            "rrf_score": rrf, "metadata": {"tenant_id": "t1", "chunk_id": cid}, **extra}


CANDS = [
    _cand("c0", "Remote work is allowed two days per week", rrf=0.033),
    _cand("c1", "Employees are entitled to 18 days of paid annual leave", rrf=0.020),
    _cand("c2", "Expense reports are due monthly", rrf=0.015),
]
Q = "how many days of annual leave"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(reranker.time, "sleep", lambda s: None)


def test_reorders_by_relevance_and_preserves_everything():
    fake = FakeReranker()
    out = rerank_chunks(Q, CANDS, top_k=2, client=fake)
    assert [r["chunk_id"] for r in out] == ["c1", "c0"]            # c1 was only #2 after RRF
    top = out[0]
    assert top["rank"] == 1 and top["rerank_score"] > out[1]["rerank_score"]
    assert (top["dense_score"], top["keyword_score"], top["rrf_score"]) == (0.8, 3.2, 0.020)
    assert (top["document_name"], top["page_number"], top["section"]) == ("employee_policy.pdf", 4, "Leave Policy")
    assert top["metadata"]["tenant_id"] == "t1"
    assert fake.calls[0]["model"] == "bge-reranker-v2-m3" and fake.calls[0]["documents"][0] == {"id": "c0", "text": CANDS[0]["text"]}
    assert "rerank_score" not in CANDS[0]                           # input candidates are not mutated


def test_top_k_default_five_and_empty_inputs():
    many = [_cand(f"c{i}", f"annual leave days policy {i}") for i in range(20)]
    assert len(rerank_chunks(Q, many, client=FakeReranker())) == 5
    assert rerank_chunks(Q, [], client=FakeReranker()) == []


def test_nothing_relevant_returns_empty_list():
    out = rerank_chunks_detailed("quantum chromodynamics lattice", CANDS, client=FakeReranker(), min_score=0.05)
    assert out["results"] == [] and out["below_threshold"] == 3 and out["fallback"] is False


def test_min_score_filters_only_weak_chunks():
    out = rerank_chunks(Q, CANDS, top_k=5, client=FakeReranker(), min_score=0.3)
    assert [r["chunk_id"] for r in out] == ["c1"]


def test_transient_error_is_retried_then_succeeds():
    fake = FakeReranker(fail_times=2, status=503)
    out = rerank_chunks_detailed(Q, CANDS, client=fake)
    assert len(fake.calls) == 3 and out["fallback"] is False and out["results"][0]["chunk_id"] == "c1"


def test_permanent_error_falls_back_to_rrf_order():
    fake = FakeReranker(always_fail=True, status=400)
    out = rerank_chunks_detailed(Q, CANDS, top_k=2, client=fake)
    assert len(fake.calls) == 1 and out["fallback"] is True            # 400 is not retried
    assert [r["chunk_id"] for r in out["results"]] == ["c0", "c1"]      # RRF order
    assert all(r["rerank_score"] is None for r in out["results"]) and out["results"][0]["rank"] == 1


def test_dedupes_skips_blank_and_truncates_long_text():
    cands = [_cand("a", "x " * 5000), _cand("a", "duplicate"), _cand("b", "   "), _cand("c", "annual leave")]
    fake = FakeReranker()
    out = rerank_chunks_detailed(Q, cands, client=fake)
    sent = fake.calls[0]["documents"]
    assert [d["id"] for d in sent] == ["a", "c"] and len(sent[0]["text"]) == reranker.MAX_CHARS
    assert out["candidates_in"] == 2


def test_dict_style_response_is_supported():
    class DictClient:
        def rerank(self, **kw):
            return {"data": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]}
    out = rerank_chunks(Q, CANDS[:2], client=DictClient())
    assert [r["chunk_id"] for r in out] == ["c1", "c0"] and out[0]["rerank_score"] == 0.9


def test_latency_reported_and_validation():
    assert rerank_chunks_detailed(Q, CANDS, client=FakeReranker())["latency_ms"] >= 0
    with pytest.raises(ValueError):
        rerank_chunks(" ", CANDS, client=FakeReranker())
    with pytest.raises(ValueError):
        rerank_chunks(Q, CANDS, top_k=0, client=FakeReranker())
