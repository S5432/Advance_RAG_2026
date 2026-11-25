"""Gemini wrapper: retries with backoff, timeouts, model fallback and token-usage tracking.

    primary model --(transient error)--> retry with exponential backoff + jitter (up to `max_retries`)
                  --(still failing)----> next model in `fallback_models`, same retry policy
                  --(all failed)-------> LLMCallFailed (carries the attempt log for usage logging)

Rules of this module:
  * Only errors that can succeed on a second try are retried (timeouts, network errors, 408/425/429/5xx).
    400 / 401 / 403 fail at once on EVERY model (bad request, bad key), so they skip retries and fallback.
    404 (model not found / retired) skips the retries but DOES go to the fallback model.
  * Timeouts: every attempt gets `timeout_s` (blocking) or `stream_timeout_s` (streaming), sent to the SDK as
    `http_options.timeout`. `total_timeout_s` is one budget for ALL attempts + backoff sleeps + fallback, so a
    request can never run for (attempts x models x timeout).
  * Streaming: retry / fallback only happen BEFORE the first token reaches the caller. After that a retry
    would duplicate text, so the error is raised instead (`LLMCallResult.status == "failed"`, partial text kept).
  * Token usage comes from Gemini's `usage_metadata`. When the API does not report it (or a stream was cut)
    it is estimated (~4 characters per token) and flagged with `usage["source"] == "estimated"`.

`generate_text` / `stream_text` keep their old signatures for existing callers; new code should use
`generate` / `stream`, which return the answer AND the usage / attempt details.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

# indirection so tests can run without real sleeping / waiting
_sleep = time.sleep
_clock = time.monotonic

MAX_ATTEMPT_LOG = 20       # attempt entries kept per call (bounded: logs and API responses stay small)


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class LLMConfig:
    """Resilience knobs. Built from Settings; the defaults mean 'retry, but no fallback model'."""

    max_retries: int = 2                 # retries per model AFTER the first try (2 => up to 3 tries)
    backoff_base_s: float = 1.0          # delay before retry n = base * 2**n (capped, with jitter)
    backoff_max_s: float = 10.0
    timeout_s: float = 30.0              # one blocking attempt
    stream_timeout_s: float = 60.0       # one streaming attempt (whole stream)
    total_timeout_s: float = 90.0        # budget for all attempts + sleeps + fallbacks of one call
    fallback_models: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings) -> "LLMConfig":
        d = cls()
        raw = getattr(settings, "gemini_fallback_model", None) or ""
        fallbacks = tuple(m.strip() for m in str(raw).split(",") if m.strip())
        return cls(
            max_retries=getattr(settings, "llm_max_retries", d.max_retries),
            backoff_base_s=getattr(settings, "llm_retry_backoff_s", d.backoff_base_s),
            backoff_max_s=getattr(settings, "llm_retry_backoff_max_s", d.backoff_max_s),
            timeout_s=getattr(settings, "llm_timeout_s", d.timeout_s),
            stream_timeout_s=getattr(settings, "llm_stream_timeout_s", d.stream_timeout_s),
            total_timeout_s=getattr(settings, "llm_total_timeout_s", d.total_timeout_s),
            fallback_models=fallbacks,
        )


# ------------------------------------------------------------------ errors
class LLMCallFailed(RuntimeError):
    """Every attempt on every model failed. `.result` has the attempt log + usage for logging."""

    def __init__(self, message: str, result: "LLMCallResult"):
        super().__init__(message)
        self.result = result


class LLMTimeout(TimeoutError):
    """A streaming attempt ran past its time limit."""


def _status_code(exc: BaseException) -> int | None:
    """HTTP status of an SDK / HTTP error, if it has one (google-genai: `.code`; requests/httpx: `.status_code`)."""
    for attr in ("code", "status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def is_retryable(exc: BaseException) -> bool:
    """Unknown errors (network drop, timeout, ...) are retried; HTTP errors only when they are transient."""
    status = _status_code(exc)
    if status is None:
        return True
    return status in (408, 425, 429) or status >= 500


def allows_fallback(exc: BaseException) -> bool:
    """400 / 401 / 403 would fail identically on another model (same request, same key): stop."""
    return _status_code(exc) not in (400, 401, 403)


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


# ------------------------------------------------------------------- usage
def _num(value) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for when the API reports nothing."""
    return math.ceil(len(text) / 4) if text else 0


def empty_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0,
            "total_tokens": 0, "source": "none"}


def extract_usage(obj) -> dict | None:
    """Gemini `usage_metadata` (on a response or on a stream chunk) -> usage dict, or None if absent."""
    meta = getattr(obj, "usage_metadata", None)
    if meta is None:
        return None
    prompt = _num(getattr(meta, "prompt_token_count", 0))
    completion = _num(getattr(meta, "candidates_token_count", 0))
    thinking = _num(getattr(meta, "thoughts_token_count", 0))
    cached = _num(getattr(meta, "cached_content_token_count", 0))
    total = _num(getattr(meta, "total_token_count", 0)) or (prompt + completion + thinking)
    if not (prompt or completion or thinking or total):
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "thinking_tokens": thinking,
            "cached_tokens": cached, "total_tokens": total, "source": "api"}


def _usage_or_estimate(api_usage: dict | None, prompt: str, text: str) -> dict:
    if api_usage:
        return api_usage
    p, c = estimate_tokens(prompt), estimate_tokens(text)
    return {"prompt_tokens": p, "completion_tokens": c, "thinking_tokens": 0, "cached_tokens": 0,
            "total_tokens": p + c, "source": "estimated"}


# ------------------------------------------------------------------ result
@dataclass
class LLMCallResult:
    """What happened in one logical LLM call (all retries and fallbacks included)."""

    purpose: str                          # "answer" | "query_rewrite" | ...
    model_requested: str
    model_used: str | None = None         # model that produced the text (None if every model failed)
    status: str = "pending"               # ok | failed | cancelled
    stream: bool = False
    text: str = ""
    fallback_used: bool = False
    attempts: int = 0                     # total tries over all models
    latency_ms: float = 0.0
    error: str | None = None
    usage: dict = field(default_factory=empty_usage)
    attempt_log: list[dict] = field(default_factory=list)

    def log_attempt(self, model: str, attempt: int, outcome: str, started: float, error: BaseException | None = None):
        if len(self.attempt_log) < MAX_ATTEMPT_LOG:
            self.attempt_log.append({"model": model, "attempt": attempt + 1, "outcome": outcome,
                                     "latency_ms": round((_clock() - started) * 1000, 1),
                                     "error": _short(error) if error else None})

    def to_dict(self) -> dict:
        """JSON-safe summary. The answer text is NOT included (usage logs must not store user content)."""
        return {"purpose": self.purpose, "model_requested": self.model_requested, "model_used": self.model_used,
                "status": self.status, "stream": self.stream, "fallback_used": self.fallback_used,
                "attempts": self.attempts, "latency_ms": self.latency_ms, "error": self.error,
                "usage": dict(self.usage), "attempt_log": list(self.attempt_log)}


def _model_chain(model: str, fallbacks: tuple[str, ...]) -> list[str]:
    chain = [model]
    for m in fallbacks:
        if m and m not in chain:
            chain.append(m)
    return chain


class _Plan:
    """Walks (model, attempt) pairs under the retry policy and the total time budget.

    The caller runs one attempt per yielded tuple and, if it failed, calls `fail(exc, attempt)` before
    asking for the next tuple. `fail` decides: retry (sleeps the backoff), skip to the next model, or stop.
    """

    def __init__(self, models: list[str], cfg: LLMConfig, attempt_timeout_s: float):
        self.models, self.cfg, self.attempt_timeout_s = models, cfg, attempt_timeout_s
        self.deadline = _clock() + cfg.total_timeout_s
        self.budget_exhausted = False
        self._skip_model = False
        self._stop = False

    def attempts(self):
        for model in self.models:
            self._skip_model = False
            for n in range(self.cfg.max_retries + 1):
                if self._stop or self._skip_model:
                    break
                remaining = self.deadline - _clock()
                if remaining <= 0:
                    self.budget_exhausted = True
                    return
                yield model, n, min(self.attempt_timeout_s, remaining)
            if self._stop:
                return

    def fail(self, exc: BaseException, attempt: int) -> None:
        if not allows_fallback(exc):
            self._stop = True
        elif not is_retryable(exc):
            self._skip_model = True
        elif attempt < self.cfg.max_retries:
            delay = min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * (2 ** attempt)) * random.uniform(0.5, 1.0)
            _sleep(max(0.0, min(delay, self.deadline - _clock())))


def _config(max_tokens: int, thinking_budget: int | None = None, timeout_s: float | None = None):
    from google.genai import types

    kwargs = {}
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    if timeout_s is not None:
        kwargs["http_options"] = types.HttpOptions(timeout=max(1, int(timeout_s * 1000)))  # milliseconds
    return types.GenerateContentConfig(temperature=0.1, max_output_tokens=max_tokens, **kwargs)


def build_gemini_client(settings):
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def _fail(res: LLMCallResult, plan: _Plan, last_exc: BaseException | None, t0: float) -> LLMCallFailed:
    res.status, res.model_used = "failed", None
    res.latency_ms = round((_clock() - t0) * 1000, 1)
    if plan.budget_exhausted:
        res.error = f"total timeout of {plan.cfg.total_timeout_s}s exhausted" + (
            f" (last error: {_short(last_exc)})" if last_exc else "")
    else:
        res.error = _short(last_exc) if last_exc else "no attempt was made"
    err = LLMCallFailed(f"LLM call failed after {res.attempts} attempt(s) on {plan.models}: {res.error}", res)
    if last_exc is not None:
        err.__cause__ = last_exc
    return err


# ---------------------------------------------------------------- blocking
def generate(client, model: str, prompt: str, *, config: LLMConfig | None = None, max_tokens: int = 1024,
             thinking_budget: int | None = None, purpose: str = "answer") -> LLMCallResult:
    """Blocking generation with retries, timeout and fallback. Raises LLMCallFailed if every model fails.

    `thinking_budget=0` turns off Gemini 3.1 Flash's hidden reasoning (it counts against max_output_tokens);
    it is only applied to Flash models because Pro models cannot disable thinking."""
    cfg = config or LLMConfig()
    res = LLMCallResult(purpose=purpose, model_requested=model)
    plan = _Plan(_model_chain(model, cfg.fallback_models), cfg, cfg.timeout_s)
    t0, last_exc = _clock(), None

    for name, n, timeout in plan.attempts():
        started = _clock()
        res.attempts += 1
        try:
            resp = client.models.generate_content(
                model=name, contents=prompt,
                config=_config(max_tokens, thinking_budget if "flash" in name else None, timeout))
            text = (resp.text or "").strip()
        except Exception as exc:
            last_exc = exc
            res.log_attempt(name, n, "error", started, exc)
            logger.warning("Gemini call failed (model=%s attempt=%d): %s", name, n + 1, _short(exc))
            plan.fail(exc, n)
            continue

        res.log_attempt(name, n, "ok", started)
        res.text, res.model_used, res.status = text, name, "ok"
        res.fallback_used = name != model
        res.usage = _usage_or_estimate(extract_usage(resp), prompt, text)
        res.latency_ms = round((_clock() - t0) * 1000, 1)
        if res.fallback_used:
            logger.warning("Gemini fallback: %s failed, answered by %s", model, name)
        return res

    raise _fail(res, plan, last_exc, t0)


# --------------------------------------------------------------- streaming
class LLMStream:
    """Iterator of text pieces with the same retry / fallback policy as `generate`.

    After (or during) iteration `.result` holds model_used, attempts, usage and status. Call `.close()` when
    you stop early (client disconnect) so the result is finalised as 'cancelled'.
    """

    def __init__(self, client, model: str, prompt: str, *, config: LLMConfig | None = None,
                 max_tokens: int = 1024, purpose: str = "answer"):
        self._client, self._model, self._prompt = client, model, prompt
        self._cfg, self._max_tokens = config or LLMConfig(max_retries=1), max_tokens
        self.result = LLMCallResult(purpose=purpose, model_requested=model, stream=True)
        self._gen = self._run()

    def __iter__(self) -> Iterator[str]:
        return self._gen

    def close(self) -> None:
        self._gen.close()

    def _finish(self, status: str, model: str | None, parts: list[str], api_usage: dict | None, t0: float,
                error: BaseException | None = None) -> None:
        res = self.result
        res.status, res.model_used = status, model
        res.fallback_used = bool(model) and model != self._model
        res.text = "".join(parts)
        res.usage = _usage_or_estimate(api_usage, self._prompt, res.text) if model else empty_usage()
        res.latency_ms = round((_clock() - t0) * 1000, 1)
        if error is not None:
            res.error = _short(error)

    def _run(self) -> Iterator[str]:
        cfg, res = self._cfg, self.result
        plan = _Plan(_model_chain(self._model, cfg.fallback_models), cfg, cfg.stream_timeout_s)
        t0, last_exc, finished = _clock(), None, False
        parts: list[str] = []
        api_usage: dict | None = None
        current: str | None = None
        try:
            for name, n, timeout in plan.attempts():
                started, current = _clock(), name
                res.attempts += 1
                parts, api_usage, began = [], None, False
                try:
                    stream = self._client.models.generate_content_stream(
                        model=name, contents=self._prompt, config=_config(self._max_tokens, None, timeout))
                    for chunk in stream:
                        if _clock() - started > timeout:
                            raise LLMTimeout(f"stream exceeded {timeout:.0f}s")
                        usage = extract_usage(chunk)
                        if usage:
                            api_usage = usage  # usage_metadata is cumulative: the last one wins
                        piece = getattr(chunk, "text", None)
                        if piece:
                            began = True
                            parts.append(piece)
                            yield piece
                except Exception as exc:
                    last_exc = exc
                    res.log_attempt(name, n, "error", started, exc)
                    if began:  # text already reached the caller: a retry would duplicate it
                        logger.warning("Gemini stream failed mid-answer (model=%s): %s", name, _short(exc))
                        self._finish("failed", name, parts, api_usage, t0, exc)
                        finished = True
                        raise LLMCallFailed(f"stream failed after partial output: {res.error}", res) from exc
                    logger.warning("Gemini stream failed before first token (model=%s attempt=%d): %s",
                                   name, n + 1, _short(exc))
                    plan.fail(exc, n)
                    continue

                res.log_attempt(name, n, "ok", started)
                self._finish("ok", name, parts, api_usage, t0)
                finished = True
                if res.fallback_used:
                    logger.warning("Gemini fallback: %s failed, streamed from %s", self._model, name)
                return

            failed = _fail(res, plan, last_exc, t0)
            finished = True
            raise failed
        finally:
            if not finished:  # consumer stopped early (client disconnect) or generator was closed
                self._finish("cancelled", current if parts else None, parts, api_usage, t0)


def stream(client, model: str, prompt: str, *, config: LLMConfig | None = None, max_tokens: int = 1024,
           purpose: str = "answer") -> LLMStream:
    """Start a streaming call; iterate the returned object for text pieces and read `.result` afterwards."""
    return LLMStream(client, model, prompt, config=config, max_tokens=max_tokens, purpose=purpose)


# ------------------------------------------------- legacy (text-only) API
def generate_text(client, model: str, prompt: str, retries: int = 2, max_tokens: int = 1024,
                  thinking_budget: int | None = None) -> str:
    """Blocking generation returning only the text (no fallback model). Kept for existing callers."""
    return generate(client, model, prompt, config=LLMConfig(max_retries=retries), max_tokens=max_tokens,
                    thinking_budget=thinking_budget).text


def stream_text(client, model: str, prompt: str, retries: int = 1, max_tokens: int = 1024) -> Iterator[str]:
    """Yield text pieces (no fallback model). Retries only before the first token. Kept for existing callers."""
    yield from LLMStream(client, model, prompt, config=LLMConfig(max_retries=retries), max_tokens=max_tokens)
