"""Retries, timeouts, model fallback, token usage and usage logging (fakes only; no network)."""
from types import SimpleNamespace

import pytest

from src import gemini_client as gc
from src import usage as usage_mod
from src.gemini_client import LLMCallFailed, LLMConfig, generate, generate_text, stream
from src.rag_chain import NOT_FOUND, RetrievalConfig, answer_question, stream_rag_answer
from tests.fakes import FakeAPIError, FakeGemini, FakePinecone, FakeReranker, FakeRewriteGemini, FlakyGemini, keyword_docs

P, F = "primary-flash", "fallback-lite"
T1 = {"tenant_id": {"$eq": "t1"}}
CFG = LLMConfig(max_retries=2, backoff_base_s=1.0, fallback_models=(F,))


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Virtual time: sleeping advances the clock instantly, jitter is fixed at 1.0."""
    state = SimpleNamespace(now=0.0, sleeps=[])
    monkeypatch.setattr(gc, "_clock", lambda: state.now)

    def fake_sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(gc, "_sleep", fake_sleep)
    monkeypatch.setattr(gc.random, "uniform", lambda a, b: 1.0)
    return state


# ---------------------------------------------------------------- usage
def test_usage_comes_from_the_api_when_reported():
    res = generate(FlakyGemini(), P, "hello", config=CFG)
    assert res.usage == {"prompt_tokens": 120, "completion_tokens": 12, "thinking_tokens": 5,
                         "cached_tokens": 0, "total_tokens": 137, "source": "api"}
    assert res.status == "ok" and res.model_used == P and res.fallback_used is False and res.attempts == 1


def test_usage_is_estimated_when_the_api_reports_nothing():
    res = generate(FlakyGemini(report_usage=False), P, "x" * 400, config=CFG)
    u = res.usage
    assert u["source"] == "estimated" and u["prompt_tokens"] == 100 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_result_dict_never_contains_the_answer_text():
    d = generate(FlakyGemini(), P, "secret question", config=CFG).to_dict()
    assert "text" not in d and "secret" not in str(d)


# ---------------------------------------------------------------- retries
def test_transient_error_is_retried_with_backoff(clock):
    gem = FlakyGemini({P: [FakeAPIError(503), FakeAPIError(429)]})
    res = generate(gem, P, "q", config=CFG)
    assert res.status == "ok" and res.attempts == 3 and res.model_used == P and not res.fallback_used
    assert clock.sleeps == [1.0, 2.0]                      # exponential: base * 2**n
    assert [a["outcome"] for a in res.attempt_log] == ["error", "error", "ok"]


def test_backoff_is_capped(clock):
    cfg = LLMConfig(max_retries=4, backoff_base_s=1.0, backoff_max_s=3.0, total_timeout_s=999)
    gem = FlakyGemini({P: [FakeAPIError(500)] * 4})
    assert generate(gem, P, "q", config=cfg).attempts == 5
    assert clock.sleeps == [1.0, 2.0, 3.0, 3.0]               # 4.0 and 8.0 are capped at backoff_max_s


def test_unknown_errors_are_retried_like_network_failures():
    gem = FlakyGemini({P: [ConnectionError("reset")]})
    assert generate(gem, P, "q", config=CFG).attempts == 2


@pytest.mark.parametrize("code", [400, 401, 403])
def test_request_errors_fail_at_once_without_retry_or_fallback(code):
    gem = FlakyGemini({P: [FakeAPIError(code)]})
    with pytest.raises(LLMCallFailed) as err:
        generate(gem, P, "q", config=CFG)
    assert [c[0] for c in gem.calls] == [P] and err.value.result.attempts == 1


# --------------------------------------------------------------- fallback
def test_fallback_model_answers_after_primary_retries_are_exhausted():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3})
    res = generate(gem, P, "q", config=CFG)
    assert [c[0] for c in gem.calls] == [P, P, P, F]
    assert res.model_used == F and res.fallback_used is True and res.attempts == 4 and res.status == "ok"


def test_model_not_found_skips_retries_but_uses_fallback():
    gem = FlakyGemini({P: [FakeAPIError(404)]})
    res = generate(gem, P, "q", config=CFG)
    assert [c[0] for c in gem.calls] == [P, F] and res.fallback_used is True


def test_fallback_chain_is_tried_in_order_and_duplicates_are_skipped():
    cfg = LLMConfig(max_retries=0, fallback_models=(P, "m2", "m3"))
    gem = FlakyGemini({P: [FakeAPIError(500)], "m2": [FakeAPIError(500)]})
    res = generate(gem, P, "q", config=cfg)
    assert [c[0] for c in gem.calls] == [P, "m2", "m3"] and res.model_used == "m3"


def test_all_models_failing_raises_with_a_full_attempt_log():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3, F: [FakeAPIError(500)] * 3})
    with pytest.raises(LLMCallFailed) as err:
        generate(gem, P, "q", config=CFG)
    r = err.value.result
    assert r.status == "failed" and r.model_used is None and r.attempts == 6 and "500" in r.error
    assert [a["model"] for a in r.attempt_log] == [P] * 3 + [F] * 3
    assert isinstance(err.value, RuntimeError) and err.value.__cause__ is not None


def test_no_fallback_configured_means_no_fallback_call():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3})
    with pytest.raises(LLMCallFailed):
        generate(gem, P, "q", config=LLMConfig())
    assert {c[0] for c in gem.calls} == {P}


# --------------------------------------------------------------- timeouts
def test_each_attempt_gets_an_sdk_timeout_in_milliseconds():
    gem = FlakyGemini()
    generate(gem, P, "q", config=LLMConfig(timeout_s=12.5))
    assert gem.calls[0][2] == 12500


def test_total_budget_stops_retrying_and_shrinks_the_attempt_timeout(clock):
    cfg = LLMConfig(max_retries=5, backoff_base_s=4.0, backoff_max_s=100.0, timeout_s=30.0, total_timeout_s=10.0)
    gem = FlakyGemini({P: [FakeAPIError(503)] * 10})
    with pytest.raises(LLMCallFailed) as err:
        generate(gem, P, "q", config=cfg)
    assert err.value.result.attempts == 2 and "total timeout" in err.value.result.error
    assert gem.calls[0][2] == 10000                            # min(timeout_s, remaining budget)
    assert gem.calls[1][2] == 6000                             # 10s budget - 4s backoff
    assert clock.now <= 10.0


# -------------------------------------------------------------- streaming
def _drain(s):
    return list(s)


def test_stream_reports_usage_from_the_last_chunk():
    s = stream(FlakyGemini(), P, "q", config=CFG)
    assert "".join(_drain(s)) == "Employees get 18 days of annual leave. [Source 1]"
    assert s.result.usage["source"] == "api" and s.result.usage["total_tokens"] == 137
    assert s.result.status == "ok" and s.result.stream is True


def test_stream_failing_before_the_first_token_retries_then_falls_back():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3})
    s = stream(gem, P, "q", config=CFG)
    text = "".join(_drain(s))
    assert text == "".join(FlakyGemini.PIECES)                  # one clean answer, nothing duplicated
    assert [c[0] for c in gem.calls] == [P, P, P, F]
    assert s.result.model_used == F and s.result.fallback_used and s.result.attempts == 4


def test_stream_failing_midway_is_not_retried_and_keeps_partial_usage():
    gem = FlakyGemini({P: [("after", 2, FakeAPIError(503))]})
    s = stream(gem, P, "q", config=CFG)
    got = []
    with pytest.raises(LLMCallFailed):
        for piece in s:
            got.append(piece)
    assert got == FlakyGemini.PIECES[:2]
    assert [c[0] for c in gem.calls] == [P]                     # no retry, no fallback: would duplicate text
    r = s.result
    assert r.status == "failed" and r.text == "".join(got) and r.usage["source"] == "estimated"


def test_stream_all_models_failing_raises():
    gem = FlakyGemini({P: [FakeAPIError(500)] * 3, F: [FakeAPIError(500)] * 3})
    with pytest.raises(LLMCallFailed):
        _drain(stream(gem, P, "q", config=CFG))


def test_closing_a_stream_early_marks_the_call_cancelled():
    s = stream(FlakyGemini(), P, "q", config=CFG)
    it = iter(s)
    next(it)
    s.close()
    assert s.result.status == "cancelled" and s.result.text == FlakyGemini.PIECES[0]
    assert s.result.model_used == P and s.result.usage["source"] == "estimated"


def test_stream_attempt_deadline_is_enforced_between_chunks(clock):
    class Slow(FlakyGemini):
        def _stream(self, model, contents, config=None):
            self._record(model, "stream", config)
            for piece in self.PIECES:
                clock.now += 25.0                                # every chunk takes 25 virtual seconds
                yield SimpleNamespace(text=piece, usage_metadata=None)

    s = stream(Slow(), P, "q", config=LLMConfig(stream_timeout_s=60.0, max_retries=0))
    got = []
    with pytest.raises(LLMCallFailed):
        for piece in s:
            got.append(piece)
    assert len(got) == 2 and "stream exceeded" in s.result.error


# ------------------------------------------------------- legacy wrappers
def test_legacy_generate_text_still_returns_a_string_and_retries():
    gem = FlakyGemini({P: [FakeAPIError(503)]})
    assert generate_text(gem, P, "q") == "".join(FlakyGemini.PIECES) and len(gem.calls) == 2


def test_legacy_wrapper_never_uses_a_fallback_model():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3})
    with pytest.raises(RuntimeError):
        generate_text(gem, P, "q")


# ------------------------------------------------------------ settings
def test_config_from_settings_defaults_and_fallback_chain():
    d = LLMConfig.from_settings(SimpleNamespace())
    assert d == LLMConfig()
    s = SimpleNamespace(gemini_fallback_model=" a , b ,,", llm_max_retries=4, llm_timeout_s=5)
    c = LLMConfig.from_settings(s)
    assert c.fallback_models == ("a", "b") and c.max_retries == 4 and c.timeout_s == 5
    assert LLMConfig.from_settings(SimpleNamespace(gemini_fallback_model="")).fallback_models == ()


# ------------------------------------------------- usage helpers / logging
def _call(**kw):
    base = {"purpose": "answer", "model_requested": P, "model_used": P, "status": "ok", "stream": False,
            "fallback_used": False, "attempts": 1, "latency_ms": 10.0, "error": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "thinking_tokens": 1, "cached_tokens": 0,
                      "total_tokens": 16, "source": "api"}}
    return {**base, **kw}


def test_summarize_adds_up_tokens_attempts_and_flags():
    t = usage_mod.summarize([_call(), _call(purpose="query_rewrite", fallback_used=True, attempts=3, status="failed")])
    assert t["total_tokens"] == 32 and t["prompt_tokens"] == 20 and t["llm_calls"] == 2
    assert t["failed_calls"] == 1 and t["fallback_used"] is True and t["attempts"] == 4


def test_each_call_is_logged_as_one_json_line_and_fallback_is_a_warning(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="docqa.usage"):
        usage_mod.record_calls([_call(), _call(fallback_used=True, model_used=F)],
                               {"request_id": "r1", "tenant_id": "t1", "user_id": "u1", "endpoint": "ask"})
    recs = [r for r in caplog.records if r.name == "docqa.usage"]
    assert len(recs) == 2 and recs[0].levelno == logging.INFO and recs[1].levelno == logging.WARNING
    assert '"request_id": "r1"' in recs[0].getMessage() and '"total_tokens": 16' in recs[0].getMessage()


def test_recording_never_raises_even_if_persistence_breaks(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(usage_mod, "persist_calls", boom)
    usage_mod.record_calls([_call()], {"tenant_id": "t1"})   # must not raise


def test_calls_are_persisted_to_the_llm_usage_table():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import select
    from src import database
    from src.models import LLMUsage

    database.configure("sqlite:///:memory:")
    database.init_db()
    ctx = {"request_id": "r9", "tenant_id": "t1", "user_id": "u1", "endpoint": "ask", "persist": True}
    usage_mod.record_calls([_call(), _call(purpose="query_rewrite", status="failed", model_used=None,
                                           error="timeout")], ctx)
    with database.session_scope() as s:
        rows = s.scalars(select(LLMUsage).order_by(LLMUsage.id)).all()
        assert [(r.request_id, r.purpose, r.status, r.total_tokens) for r in rows] == \
               [("r9", "answer", "ok", 16), ("r9", "query_rewrite", "failed", 16)]
        assert rows[1].model_used is None and rows[1].error == "timeout"
    usage_mod.record_calls([_call()], {**ctx, "persist": False})      # opt-out writes nothing more
    with database.session_scope() as s:
        assert len(s.scalars(select(LLMUsage)).all()) == 2


# -------------------------------------------------------- RAG end to end
def _seed(pc):
    pc.upsert_documents([{"_id": "t1_d_chunk_0", "chunk_text": "Employees get 18 days of annual leave",
                          "title": "policy.pdf", "chunk_id": "t1_d_chunk_0", "document_id": "t1_d",
                          "document_name": "policy.pdf", "page_number": 1, "section": "Leave", "tenant_id": "t1"}])


CTX = {"request_id": "req-1", "tenant_id": "t1", "user_id": "u1", "endpoint": "ask", "persist": False}


def _ask(gem, **kw):
    pc = FakePinecone(); _seed(pc)
    return answer_question("annual leave days", pc, gem, 5, P, T1, config=RetrievalConfig(rerank_min_score=0.0),
                           documents=keyword_docs(pc), rerank_client=FakeReranker(), usage_context=CTX, **kw)


def test_answer_question_falls_back_and_reports_the_model_that_answered():
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3})
    out = _ask(gem, llm_config=CFG)
    assert out["model"] == F and "[Source 1]" in out["answer"]
    u = out["usage"]
    assert u["request_id"] == "req-1" and u["totals"]["fallback_used"] is True and u["totals"]["total_tokens"] == 137
    assert [c["purpose"] for c in u["calls"]] == ["answer"] and u["calls"][0]["attempts"] == 4


def test_answer_question_records_usage_even_when_every_model_fails(monkeypatch):
    seen = []
    monkeypatch.setattr("src.rag_chain.record_calls", lambda calls, ctx: seen.append((calls, ctx)))
    gem = FlakyGemini({P: [FakeAPIError(503)] * 3, F: [FakeAPIError(503)] * 3})
    with pytest.raises(LLMCallFailed):
        _ask(gem, llm_config=CFG)
    calls, ctx = seen[0]
    assert calls[0]["status"] == "failed" and calls[0]["attempts"] == 6 and ctx["request_id"] == "req-1"


def test_rewrite_and_answer_calls_are_both_tracked():
    gem = FakeRewriteGemini(rewrites=["annual leave policy"])
    out = _ask(gem, llm_config=LLMConfig(), rewrite_query=True)
    assert [c["purpose"] for c in out["usage"]["calls"]] == ["query_rewrite", "answer"]
    assert out["usage"]["totals"]["llm_calls"] == 2 and out["usage"]["totals"]["total_tokens"] > 0


def test_failed_rewrite_is_still_logged_and_does_not_block_the_answer():
    out = _ask(FakeRewriteGemini(fail=True), llm_config=LLMConfig(), rewrite_query=True)
    rewrite = out["usage"]["calls"][0]
    assert rewrite["purpose"] == "query_rewrite" and rewrite["status"] == "failed"
    assert out["answer"] and out["retrieval"]["query_rewrite"]["fallback"] is True


def test_no_chunks_means_no_answer_call_but_usage_object_exists():
    pc = FakePinecone()
    out = answer_question("q", pc, FlakyGemini(), 5, P, T1, config=RetrievalConfig(), documents=[],
                          rerank_client=FakeReranker(), usage_context=CTX)
    assert out["answer"] == NOT_FOUND and out["usage"]["totals"]["llm_calls"] == 0


def _stream(gem, **kw):
    pc = FakePinecone(); _seed(pc)
    return list(stream_rag_answer("annual leave days", pc, gem, 5, P, T1,
                                  config=RetrievalConfig(rerank_min_score=0.0), documents=keyword_docs(pc),
                                  rerank_client=FakeReranker(), usage_context=CTX, **kw))


def test_stream_done_event_carries_usage_and_the_fallback_model():
    ev = _stream(FlakyGemini({P: [FakeAPIError(503)] * 3}), llm_config=CFG)
    assert [e["event"] for e in ev] == ["retrieval", "rerank", "sources"] + ["token"] * 4 + ["done"]
    done = ev[-1]["data"]
    assert done["status"] == "completed" and done["model"] == F
    assert done["usage"]["totals"]["fallback_used"] is True and done["usage"]["totals"]["total_tokens"] == 137


def test_stream_failure_midway_still_reports_usage_and_the_event_order_is_unchanged():
    ev = _stream(FlakyGemini({P: [("after", 2, FakeAPIError(503))]}), llm_config=CFG)
    assert [e["event"] for e in ev][-2:] == ["error", "done"]
    assert ev[-2]["data"]["partial"] is True and ev[-1]["data"]["status"] == "failed"
    call = ev[-1]["data"]["usage"]["calls"][0]
    assert call["status"] == "failed" and call["usage"]["source"] == "estimated"


def test_stream_usage_is_recorded_once_even_if_the_client_disconnects(monkeypatch):
    seen = []
    monkeypatch.setattr("src.rag_chain.record_calls", lambda calls, ctx: seen.append(list(calls)))
    pc = FakePinecone(); _seed(pc)
    gen = stream_rag_answer("annual leave days", pc, FlakyGemini(), 5, P, T1,
                            config=RetrievalConfig(rerank_min_score=0.0), documents=keyword_docs(pc),
                            rerank_client=FakeReranker(), usage_context=CTX)
    for ev in gen:
        if ev["event"] == "token":
            break
    gen.close()                                                  # what the API does when the tab is closed
    assert len(seen) == 1 and seen[0][0]["status"] == "cancelled"


def test_fake_gemini_without_usage_metadata_still_works_end_to_end():
    ev = _stream(FakeGemini())
    assert ev[-1]["data"]["usage"]["calls"][0]["usage"]["source"] == "estimated"
