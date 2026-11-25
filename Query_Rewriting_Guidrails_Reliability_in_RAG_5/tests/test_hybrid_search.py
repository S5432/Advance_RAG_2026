"""Phase 4: dense search, BM25 keyword search, RRF, deleted-document guard, PostgreSQL chunk loader."""
import pytest

from src.filters import build_metadata_filter
from src.hybrid_search import (dense_search, hybrid_search, hybrid_search_detailed, keyword_search,
                               load_documents_from_db, reciprocal_rank_fusion)
from tests.fakes import FakePinecone

T1 = build_metadata_filter("t1")


def _doc(cid, text, tenant="t1", **meta):
    return {"chunk_id": cid, "text": text,
            "metadata": {"chunk_id": cid, "document_id": "doc_1", "document_name": "p.pdf", "page_number": 1,
                         "section": "", "tenant_id": tenant, "source_type": "pdf", **meta}}


CORPUS = [
    _doc("c0", "Employees are entitled to 18 days of paid annual leave per year."),
    _doc("c1", "Error ERR-5021 means the payroll export failed; restart the export job."),
    _doc("c2", "Remote work is allowed two days per week with manager approval."),
    _doc("c3", "Staff may take vacation days after probation.", tenant="t2"),
]


def _item(cid, **kw):
    return {"chunk_id": cid, "text": f"text {cid}", "dense_score": 0.0, "keyword_score": 0.0, **kw}


# ------------------------------------------------------------------ keyword
def test_keyword_exact_code_match_ranks_first_with_scores():
    res = keyword_search("what does ERR-5021 mean?", CORPUS, 20, T1)
    assert res[0]["chunk_id"] == "c1" and res[0]["keyword_score"] > 0 and res[0]["keyword_rank"] == 1
    assert res[0]["document_name"] == "p.pdf" and res[0]["metadata"]["tenant_id"] == "t1"


def test_keyword_filter_applied_before_scoring_and_top_k():
    assert all(r["metadata"]["tenant_id"] == "t1" for r in keyword_search("vacation days leave", CORPUS, 20, T1))
    assert [r["chunk_id"] for r in keyword_search("days", CORPUS, 1, T1)] and len(keyword_search("days", CORPUS, 1, T1)) == 1
    assert keyword_search("vacation", CORPUS, 20, build_metadata_filter("t2"))[0]["chunk_id"] == "c3"
    assert keyword_search("vacation", CORPUS, 20, build_metadata_filter("t1")) == []


def test_keyword_no_match_and_validation():
    assert keyword_search("zzzunknownzzz", CORPUS, 20, T1) == []
    assert keyword_search("the of and", CORPUS, 20, T1) == []          # only stop-words
    with pytest.raises(ValueError):
        keyword_search("  ", CORPUS)


# -------------------------------------------------------------------- dense
def test_dense_search_returns_scores_ranks_and_respects_filter():
    pc = FakePinecone()
    for d in CORPUS:
        pc.store[d["chunk_id"]] = {"_id": d["chunk_id"], "chunk_text": d["text"], **d["metadata"]}
    res = dense_search("annual leave", pc, 20, T1)
    assert [r["chunk_id"] for r in res] == ["c0", "c1", "c2"]            # tenant t2 chunk filtered inside Pinecone
    assert res[0]["dense_score"] == 0.9 and res[0]["dense_rank"] == 1 and res[0]["text"]
    assert len(dense_search("annual leave", pc, 2, T1)) == 2
    with pytest.raises(ValueError):
        dense_search("", pc)


# ---------------------------------------------------------------------- RRF
def test_rrf_assignment_example_k60():
    dense = [_item("A", dense_score=.9), _item("C", dense_score=.8), _item("B", dense_score=.7)]
    kw = [_item("B", keyword_score=5.), _item("C", keyword_score=4.), _item("A", keyword_score=3.)]
    out = {r["chunk_id"]: r for r in reciprocal_rank_fusion(dense, kw, k=60)}
    assert out["A"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 63)
    assert out["B"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 63)
    assert out["C"]["rrf_score"] == pytest.approx(2 / 62)
    ordered = [r["chunk_id"] for r in reciprocal_rank_fusion(dense, kw)]
    assert ordered == ["A", "B", "C"]                                    # A and B tie; C is lower


def test_rrf_dedupes_preserves_scores_and_handles_single_list():
    dense = [_item("X", dense_score=.9), _item("Y", dense_score=.5)]
    kw = [_item("X", keyword_score=7.)]
    out = reciprocal_rank_fusion(dense, kw)
    assert [r["chunk_id"] for r in out] == ["X", "Y"]
    x = out[0]
    assert (x["dense_score"], x["keyword_score"], x["dense_rank"], x["keyword_rank"]) == (.9, 7., 1, 1)
    assert out[1]["keyword_score"] == 0.0 and out[1]["keyword_rank"] is None
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_top_k_and_invalid_k():
    dense = [_item(f"d{i}") for i in range(30)]
    assert len(reciprocal_rank_fusion(dense, [], top_k=20)) == 20
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(dense, [], k=0)


# ------------------------------------------------------------------- hybrid
def _pc_with(docs):
    pc = FakePinecone()
    for d in docs:
        pc.store[d["chunk_id"]] = {"_id": d["chunk_id"], "chunk_text": d["text"], **d["metadata"]}
    return pc


def test_hybrid_combines_both_legs_and_reports_stats():
    out = hybrid_search_detailed("ERR-5021 payroll", _pc_with(CORPUS), CORPUS, 20, T1)
    h = out["hybrid_results"]
    assert h[0]["chunk_id"] == "c1" and h[0]["dense_rank"] and h[0]["keyword_rank"] == 1   # found by both legs
    assert {r["chunk_id"] for r in h} == {"c0", "c1", "c2"}
    s = out["stats"]
    assert s["dense_candidates"] == 3 and s["keyword_candidates"] >= 1 and s["fused_candidates"] == 3
    assert s["latency_ms"]["total"] >= 0
    assert [r["chunk_id"] for r in hybrid_search("ERR-5021", _pc_with(CORPUS), CORPUS, 1, T1)] == ["c1"]


def test_deleted_or_unknown_vectors_never_returned():
    docs = [dict(d) for d in CORPUS]
    pc = _pc_with(docs)
    pc.store["ghost"] = {"_id": "ghost", "chunk_text": "annual leave ghost", **docs[0]["metadata"], "chunk_id": "ghost"}
    live = [d for d in docs if d["chunk_id"] != "c0"]                    # c0's document was soft-deleted in PostgreSQL
    ids = {r["chunk_id"] for r in hybrid_search("annual leave", pc, live, 20, T1)}
    assert "ghost" not in ids and "c0" not in ids and "c2" in ids
    flagged = [_doc("z", "annual leave", is_deleted=True)]
    assert hybrid_search("annual leave", _pc_with(flagged), flagged, 20, T1) == []
    assert hybrid_search_detailed("annual leave", pc, live, 20, T1)["stats"]["stale_dropped"] == 2


# ------------------------------------------------- PostgreSQL chunk loader
@pytest.fixture()
def db():
    from src import database
    database.configure("sqlite:///:memory:")
    database.init_db()
    return database


def _seed(database, doc_id, tenant, *, deleted=False, status="COMPLETED", dept=None, chunks=(("child", "hello"),)):
    from src.models import Document, DocumentChunk
    with database.session_scope() as s:
        s.add(Document(document_id=doc_id, filename=f"{doc_id}.pdf", source_type="pdf", tenant_id=tenant,
                       uploaded_by="u", status=status, is_deleted=deleted, department=dept, access_level="internal"))
        s.flush()
        for i, (ctype, text) in enumerate(chunks):
            s.add(DocumentChunk(document_id=doc_id, chunk_id=f"{doc_id}_c{i}", chunk_index=i, content=text,
                                chunk_type=ctype, page_number=2, section="Leave"))


def test_loader_tenant_isolation_and_exclusions(db):
    _seed(db, "d1", "t1", chunks=(("child", "alpha"), ("parent", "alpha parent"), ("standalone", "beta")))
    _seed(db, "d2", "t2")
    _seed(db, "d3", "t1", deleted=True)
    _seed(db, "d4", "t1", status="PROCESSING")
    docs = load_documents_from_db(build_metadata_filter("t1"))
    assert sorted(d["chunk_id"] for d in docs) == ["d1_c0", "d1_c2"]     # no parent, no deleted, no other tenant
    m = docs[0]["metadata"]
    assert m["tenant_id"] == "t1" and m["document_name"] == "d1.pdf" and m["page_number"] == 2 and m["created_at_ts"] > 0


def test_loader_filters_and_requires_tenant(db):
    _seed(db, "d1", "t1", dept="HR")
    _seed(db, "d5", "t1", dept="IT")
    assert [d["metadata"]["department"] for d in load_documents_from_db(build_metadata_filter("t1", department="HR"))] == ["HR"]
    assert len(load_documents_from_db(build_metadata_filter("t1", document_ids=["d1", "d5"]))) == 2
    with pytest.raises(ValueError):
        load_documents_from_db({"source_type": {"$eq": "pdf"}})


def test_end_to_end_keyword_leg_from_postgres(db):
    _seed(db, "d1", "t1", chunks=(("child", "Error ERR-5021 payroll export failed"), ("child", "Annual leave is 18 days")))
    flt = build_metadata_filter("t1")
    docs = load_documents_from_db(flt)
    assert keyword_search("ERR-5021", docs, 20, flt)[0]["chunk_id"] == "d1_c0"
