"""Test doubles that mimic the Day 9 schema-based Pinecone service (Documents API, integrated embedding)."""
from types import SimpleNamespace


def _match(md: dict, flt: dict | None) -> bool:
    for field, cond in (flt or {}).items():
        if isinstance(cond, dict):
            if "$eq" in cond and md.get(field) != cond["$eq"]:
                return False
            if "$in" in cond and md.get(field) not in cond["$in"]:
                return False
        elif md.get(field) != cond:
            return False
    return True


class FakePinecone:
    """Same public surface as src.pinecone_service.PineconeService:
    upsert_documents(documents) and query(query_text, ...). No vectors, no client-side embedding."""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def upsert_documents(self, documents, namespace=None, batch_size=50):
        for d in documents:
            self.store[d["_id"]] = d  # same _id => replace (like Pinecone)
        return len(documents)

    def query(self, query_text, top_k=5, metadata_filter=None, namespace=None):
        assert isinstance(query_text, str) and query_text.strip()
        out = []
        for _id, d in self.store.items():
            md = {k: v for k, v in d.items() if k not in ("_id", "chunk_text")}
            if not _match(md, metadata_filter):
                continue
            md["text"] = d["chunk_text"]  # same shape PineconeService.query() returns
            out.append({"id": _id, "score": 0.9, "metadata": md})
        return out[:top_k]

    def delete_ids(self, ids, namespace=None):
        for i in ids:
            self.store.pop(i, None)

    def ping(self):
        return True


class FakeGemini:
    """Streams a fixed answer in pieces; can fail after N pieces."""

    PIECES = ["Employees get ", "18 days ", "of annual leave. ", "[Source 1]"]

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.models = SimpleNamespace(generate_content=self._gen, generate_content_stream=self._stream)

    def _gen(self, model, contents, config=None):
        return SimpleNamespace(text="".join(self.PIECES))

    def _stream(self, model, contents, config=None):
        assert "Answer only using the provided context" in contents
        for i, p in enumerate(self.PIECES):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("gemini boom")
            yield SimpleNamespace(text=p)
