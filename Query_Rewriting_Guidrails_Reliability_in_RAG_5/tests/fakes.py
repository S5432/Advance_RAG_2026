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


class QueryAwarePinecone(FakePinecone):
    """Dense fake whose ranking depends on the query (word overlap), unlike FakePinecone which returns every
    chunk with the same score. Lets tests show that a rewritten query surfaces chunks a vague one misses.
    `fail_on` = query texts that raise, to test a failing variant."""

    def __init__(self, fail_on=()):
        super().__init__()
        self.fail_on, self.queries = set(fail_on), []

    def query(self, query_text, top_k=5, metadata_filter=None, namespace=None):
        self.queries.append(query_text)
        if query_text in self.fail_on:
            raise RuntimeError("pinecone boom")
        words = set(query_text.lower().split())
        out = super().query(query_text, top_k=10_000, metadata_filter=metadata_filter)
        for m in out:
            m["score"] = len(words & set(m["metadata"]["text"].lower().split())) / max(len(words), 1)
        out.sort(key=lambda m: (-m["score"], m["id"]))
        return out[:top_k]


class FakeRewriteGemini(FakeGemini):
    """FakeGemini that also answers the query-rewrite prompt (recognised by <user_question>) with JSON."""

    def __init__(self, rewrites=None, raw=None, fail=False, delay=0.0, **kw):
        super().__init__(**kw)
        self.rewrites, self.raw, self.fail, self.delay, self.rewrite_prompts = rewrites or [], raw, fail, delay, []

    def _gen(self, model, contents, config=None):
        if "<user_question>" not in contents:
            return super()._gen(model, contents, config)
        import json
        import time
        self.rewrite_prompts.append(contents)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("gemini rewrite boom")
        return SimpleNamespace(text=self.raw if self.raw is not None else json.dumps({"queries": self.rewrites}))


class FakeAPIError(Exception):
    """Like google.genai.errors.APIError: carries an HTTP status in `.code`."""

    def __init__(self, code: int, message: str = "boom"):
        super().__init__(f"{code} {message}")
        self.code = code


class FlakyGemini:
    """Scripted Gemini for retry / fallback / usage tests.

    `script[model]` is a list of outcomes consumed one per call to that model (an empty list = succeed):
      * an Exception instance -> raised at once (before any token)
      * ("after", n, exc)     -> streaming only: yield n pieces, then raise exc
    Successful calls report usage_metadata (blocking: on the response; streaming: on the last chunk).
    `calls` records (model, kind, timeout_ms) for every call."""

    PIECES = ["Employees get ", "18 days ", "of annual leave. ", "[Source 1]"]

    def __init__(self, script=None, usage=(120, 12, 5), report_usage=True):
        self.script = {m: list(v) for m, v in (script or {}).items()}
        self.usage, self.report_usage, self.calls = usage, report_usage, []
        self.models = SimpleNamespace(generate_content=self._gen, generate_content_stream=self._stream)

    def _meta(self):
        if not self.report_usage:
            return None
        p, c, t = self.usage
        return SimpleNamespace(prompt_token_count=p, candidates_token_count=c, thoughts_token_count=t,
                               total_token_count=p + c + t, cached_content_token_count=0)

    def _record(self, model, kind, config):
        timeout = getattr(getattr(config, "http_options", None), "timeout", None)
        self.calls.append((model, kind, timeout))
        queue = self.script.get(model) or []
        return queue.pop(0) if queue else None

    def _gen(self, model, contents, config=None):
        outcome = self._record(model, "generate", config)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(text="".join(self.PIECES), usage_metadata=self._meta())

    def _stream(self, model, contents, config=None):
        outcome = self._record(model, "stream", config)
        if isinstance(outcome, BaseException):
            raise outcome
        after, exc = (outcome[1], outcome[2]) if isinstance(outcome, tuple) else (None, None)
        for i, piece in enumerate(self.PIECES):
            if after is not None and i == after:
                raise exc
            last = i == len(self.PIECES) - 1
            yield SimpleNamespace(text=piece, usage_metadata=self._meta() if last else None)
