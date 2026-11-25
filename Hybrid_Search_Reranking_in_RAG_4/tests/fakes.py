"""Test doubles that mimic the Day 9 schema-based Pinecone service (Documents API, integrated embedding)."""
from types import SimpleNamespace


from src.filters import matches_filter as _match


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


class FakeReranker:
    """Mimics pc.inference.rerank: scores = share of query words found in the passage (0..1)."""

    def __init__(self, fail_times: int = 0, status: int | None = None, always_fail: bool = False):
        self.calls, self.fail_times, self.status, self.always_fail = [], fail_times, status, always_fail

    def rerank(self, model, query, documents, rank_fields, top_n, return_documents):
        self.calls.append({"model": model, "query": query, "documents": documents, "top_n": top_n})
        if self.always_fail or len(self.calls) <= self.fail_times:
            err = RuntimeError("rerank boom")
            if self.status:
                err.status = self.status
            raise err
        words = set(query.lower().split())
        scored = [(i, len(words & set(d["text"].lower().split())) / max(len(words), 1)) for i, d in enumerate(documents)]
        scored.sort(key=lambda x: -x[1])
        return SimpleNamespace(data=[SimpleNamespace(index=i, score=s) for i, s in scored[:top_n]])


def keyword_docs(pc: "FakePinecone") -> list[dict]:
    """BM25 corpus (the shape load_documents_from_db returns) built from a FakePinecone store."""
    return [{"chunk_id": d["chunk_id"], "text": d["chunk_text"],
             "metadata": {k: v for k, v in d.items() if k not in ("_id", "chunk_text")}}
            for d in pc.store.values()]
