"""Metadata filters: Pinecone-compatible builder, request whitelist, and a local evaluator.

`matches_filter` applies the SAME semantics client-side (keyword search, test fakes), so a chunk is
visible to dense and keyword retrieval under exactly the same rules.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

ALLOWED_SOURCE_TYPES = {"pdf", "docx", "html", "url", "markdown", "csv"}
ACCESS_LEVELS = ("public", "internal", "confidential")
# keys a CALLER may send in `filters`; tenant_id / is_deleted are never client-controlled
CLIENT_FILTER_KEYS = {"source_type", "department", "document_id", "document_ids", "access_level",
                      "date_from", "date_to"}


def to_epoch(value: str, end_of_day: bool = False) -> int:
    """'2026-11-01' or ISO datetime ('...Z' ok) -> UTC epoch seconds. Date-only upper bounds include the whole day."""
    try:
        v = str(value).strip()
        if len(v) == 10:
            dt = datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end_of_day:
                dt += timedelta(days=1) - timedelta(seconds=1)
        else:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid date '{value}': use YYYY-MM-DD or ISO-8601") from exc


def epoch_of(dt: datetime) -> int:
    """datetime (naive = UTC) -> epoch seconds; same value ingestion stores as `created_at_ts`."""
    return int((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp())


def build_metadata_filter(
    tenant_id: str,
    source_type: str | None = None,
    document_id: str | None = None,
    access_level: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    department: str | None = None,
    document_ids: list[str] | None = None,
    include_deleted: bool = False,
) -> dict:
    """Pinecone metadata filter (implicit AND). `tenant_id` is ALWAYS present; soft-deleted docs are excluded."""
    if not tenant_id or not str(tenant_id).strip():
        raise ValueError("tenant_id is required")
    f: dict = {"tenant_id": {"$eq": str(tenant_id).strip()}}

    if source_type:
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}")
        f["source_type"] = {"$eq": source_type}
    if access_level:
        if access_level not in ACCESS_LEVELS:
            raise ValueError(f"access_level must be one of {list(ACCESS_LEVELS)}")
        f["access_level"] = {"$eq": access_level}
    if department and department.strip():
        f["department"] = {"$eq": department.strip()}

    ids = ([document_id] if document_id else []) + list(document_ids or [])
    if len(ids) == 1:
        f["document_id"] = {"$eq": ids[0]}
    elif ids:
        f["document_id"] = {"$in": ids}

    if date_from or date_to:  # numeric epoch field: Pinecone range operators only work on numbers
        lo = to_epoch(date_from) if date_from else None
        hi = to_epoch(date_to, end_of_day=True) if date_to else None
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("date_from must not be after date_to")
        rng = {}
        if lo is not None:
            rng["$gte"] = lo
        if hi is not None:
            rng["$lte"] = hi
        f["created_at_ts"] = rng

    if not include_deleted:
        f["is_deleted"] = {"$ne": True}  # also matches vectors that never had the field
    return f


def filters_from_request(tenant_id: str, filters: dict | None = None,
                         document_ids: list[str] | None = None) -> dict:
    """Whitelist client `filters`; the tenant ALWAYS comes from the authenticated identity."""
    f = dict(filters or {})
    bad = sorted(set(f) - CLIENT_FILTER_KEYS)
    if bad:
        raise ValueError(f"Unsupported filter(s): {bad}")

    def scalar(key):
        v = f.get(key)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"filter '{key}' must be a string")
        return v

    ids = list(document_ids or [])
    extra = f.get("document_ids")
    if extra:
        if not isinstance(extra, list):
            raise ValueError("filter 'document_ids' must be a list")
        ids += [str(x) for x in extra]
    return build_metadata_filter(
        tenant_id, source_type=scalar("source_type"), document_id=scalar("document_id"),
        access_level=scalar("access_level"), date_from=scalar("date_from"), date_to=scalar("date_to"),
        department=scalar("department"), document_ids=ids or None,
    )


# --------------------------------------------------------------- local evaluator
def _op(value, op: str, arg) -> bool:
    if op == "$eq":
        return value == arg
    if op == "$ne":
        return value != arg                    # missing field counts as "not equal"
    if op == "$in":
        return value in arg
    if op == "$nin":
        return value not in arg
    if value is None or isinstance(value, bool) or isinstance(arg, bool):
        return False
    try:
        return {"$gt": value > arg, "$gte": value >= arg, "$lt": value < arg, "$lte": value <= arg}[op]
    except (TypeError, KeyError):
        return False


def matches_filter(metadata: dict, flt: dict | None) -> bool:
    """Evaluate a Pinecone-style filter against one chunk's metadata."""
    for field, cond in (flt or {}).items():
        if field == "$and":
            ok = all(matches_filter(metadata, c) for c in cond)
        elif field == "$or":
            ok = any(matches_filter(metadata, c) for c in cond)
        elif isinstance(cond, dict):
            ok = all(_op(metadata.get(field), op, arg) for op, arg in cond.items())
        else:
            ok = metadata.get(field) == cond
        if not ok:
            return False
    return True
