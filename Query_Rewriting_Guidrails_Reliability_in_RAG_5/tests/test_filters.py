"""Phase 3: metadata filter builder, request whitelist, local evaluator, tenant isolation."""
import pytest

from src.filters import build_metadata_filter, filters_from_request, matches_filter, to_epoch
from tests.fakes import FakePinecone

DAY = 86400


def test_tenant_always_present_and_deleted_excluded():
    f = build_metadata_filter("company_001")
    assert f["tenant_id"] == {"$eq": "company_001"} and f["is_deleted"] == {"$ne": True}
    assert set(f) == {"tenant_id", "is_deleted"}
    assert "is_deleted" not in build_metadata_filter("c", include_deleted=True)


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_blank_tenant_rejected(bad):
    with pytest.raises(ValueError):
        build_metadata_filter(bad)


def test_all_optional_filters():
    f = build_metadata_filter("c1", source_type="pdf", document_id="doc_1", access_level="internal",
                              department="HR", date_from="2026-11-01", date_to="2026-11-30")
    assert f["source_type"] == {"$eq": "pdf"} and f["document_id"] == {"$eq": "doc_1"}
    assert f["access_level"] == {"$eq": "internal"} and f["department"] == {"$eq": "HR"}
    assert f["created_at_ts"]["$lte"] - f["created_at_ts"]["$gte"] == 30 * DAY - 1   # whole last day included


def test_document_ids_use_in_operator():
    assert build_metadata_filter("c", document_ids=["a", "b"])["document_id"] == {"$in": ["a", "b"]}


def test_invalid_values_rejected():
    for kw in (dict(source_type="exe"), dict(access_level="secret"), dict(date_from="yesterday"),
               dict(date_from="2026-12-01", date_to="2026-11-01")):
        with pytest.raises(ValueError):
            build_metadata_filter("c", **kw)


def test_to_epoch_formats():
    assert to_epoch("2026-11-23") == to_epoch("2026-11-23T00:00:00Z") == to_epoch("2026-11-23T00:00:00")
    assert to_epoch("2026-11-23", end_of_day=True) == to_epoch("2026-11-23") + DAY - 1


def test_client_cannot_override_tenant_or_deleted_flag():
    for key in ("tenant_id", "is_deleted", "uploaded_by"):
        with pytest.raises(ValueError):
            filters_from_request("c1", {key: "x"})
    f = filters_from_request("c1", {"source_type": "pdf", "department": "HR"}, ["doc_1"])
    assert f["tenant_id"] == {"$eq": "c1"} and f["department"] == {"$eq": "HR"} and f["document_id"] == {"$eq": "doc_1"}
    with pytest.raises(ValueError):
        filters_from_request("c1", {"source_type": ["pdf"]})


def test_matches_filter_semantics():
    md = {"tenant_id": "t1", "created_at_ts": 100, "department": "HR"}
    assert matches_filter(md, {"tenant_id": {"$eq": "t1"}, "is_deleted": {"$ne": True}})   # missing field passes $ne
    assert not matches_filter({**md, "is_deleted": True}, {"is_deleted": {"$ne": True}})
    assert matches_filter(md, {"created_at_ts": {"$gte": 50, "$lte": 100}})
    assert not matches_filter(md, {"created_at_ts": {"$gt": 100}})
    assert not matches_filter(md, {"missing": {"$eq": "x"}}) and matches_filter(md, {"missing": {"$nin": ["x"]}})
    assert matches_filter(md, {"$or": [{"department": "IT"}, {"department": "HR"}]})
    assert matches_filter(md, None)


def _store():
    pc = FakePinecone()
    base = dict(chunk_text="leave policy text", source_type="pdf", access_level="internal", created_at_ts=to_epoch("2026-11-10"))
    for i, (tenant, dept, st, deleted) in enumerate([
            ("A", "HR", "pdf", False), ("B", "HR", "pdf", False), ("A", "IT", "pdf", False),
            ("A", "HR", "markdown", False), ("A", "HR", "pdf", True)]):
        pc.store[f"c{i}"] = {"_id": f"c{i}", **base, "tenant_id": tenant, "department": dept,
                             "source_type": st, "is_deleted": deleted}
    return pc


def _ids(pc, **kw):
    return sorted(m["id"] for m in pc.query("leave", 10, build_metadata_filter(**kw)))


def test_tenant_isolation_and_query_level_filters():
    pc = _store()
    assert _ids(pc, tenant_id="A") == ["c0", "c2", "c3"]            # B's chunk and the deleted one are invisible
    assert _ids(pc, tenant_id="B") == ["c1"]
    assert _ids(pc, tenant_id="A", source_type="pdf") == ["c0", "c2"]
    assert _ids(pc, tenant_id="A", department="HR") == ["c0", "c3"]
    assert _ids(pc, tenant_id="A", source_type="pdf", department="HR") == ["c0"]
    assert _ids(pc, tenant_id="A", date_from="2026-11-11") == []
    assert _ids(pc, tenant_id="A", date_from="2026-11-01", date_to="2026-11-10") == ["c0", "c2", "c3"]
    assert _ids(pc, tenant_id="A", include_deleted=True) == ["c0", "c2", "c3", "c4"]
