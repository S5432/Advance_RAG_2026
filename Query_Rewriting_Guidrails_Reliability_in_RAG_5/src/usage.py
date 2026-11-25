"""LLM usage tracking: totals per request, structured log lines, optional PostgreSQL persistence.

Every Gemini call (query rewrite, answer) produces one `LLMCallResult.to_dict()`. `record_calls` writes
one `llm_usage {json}` log line per call (WARNING when the call failed or used the fallback model) and,
when a `context` with a tenant is given, one `llm_usage` row per call. Recording NEVER raises: a broken
usage table must not break an answer.

`context` = {"request_id", "tenant_id", "user_id", "endpoint", "persist": bool}; rows are written only when
`persist` is true, so unit tests and scripts that pass no context never touch a database. The answer text and the
question are never logged or stored here, only counts, models and timings.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("docqa.usage")

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "thinking_tokens", "cached_tokens", "total_tokens")


def summarize(calls: list[dict]) -> dict:
    """Totals over all LLM calls of one request."""
    totals = {k: sum(int((c.get("usage") or {}).get(k, 0) or 0) for c in calls) for k in TOKEN_KEYS}
    totals.update(
        llm_calls=len(calls),
        failed_calls=sum(1 for c in calls if c.get("status") != "ok"),
        fallback_used=any(c.get("fallback_used") for c in calls),
        attempts=sum(int(c.get("attempts", 0) or 0) for c in calls),
        latency_ms=round(sum(float(c.get("latency_ms", 0.0) or 0.0) for c in calls), 1),
    )
    return totals


def build_usage(calls: list[dict], request_id: str | None = None) -> dict:
    """The `usage` object returned by /ask and sent in the SSE `done` event."""
    return {"request_id": request_id, "totals": summarize(calls), "calls": calls}


def _flat(call: dict, context: dict | None) -> dict:
    u = call.get("usage") or {}
    ctx = context or {}
    return {
        "request_id": ctx.get("request_id"), "tenant_id": ctx.get("tenant_id"), "user_id": ctx.get("user_id"),
        "endpoint": ctx.get("endpoint"), "purpose": call.get("purpose"),
        "model_requested": call.get("model_requested"), "model_used": call.get("model_used"),
        "fallback_used": bool(call.get("fallback_used")), "attempts": call.get("attempts", 0),
        "status": call.get("status"), "stream": bool(call.get("stream")),
        "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
        "thinking_tokens": u.get("thinking_tokens", 0), "cached_tokens": u.get("cached_tokens", 0),
        "total_tokens": u.get("total_tokens", 0), "usage_source": u.get("source", "none"),
        "latency_ms": call.get("latency_ms", 0.0), "error": (call.get("error") or None),
    }


def log_calls(calls: list[dict], context: dict | None = None) -> None:
    for call in calls:
        row = _flat(call, context)
        level = logging.INFO if row["status"] == "ok" and not row["fallback_used"] else logging.WARNING
        logger.log(level, "llm_usage %s", json.dumps(row, ensure_ascii=False, default=str))


def persist_calls(calls: list[dict], context: dict | None) -> None:
    """One row per call in `llm_usage`. Opt-in: only when `context["persist"]` is truthy (main.py sets it)."""
    if not calls or not context or not context.get("persist"):
        return
    try:
        from . import database
        from .models import LLMUsage

        with database.session_scope() as s:
            for call in calls:
                row = _flat(call, context)
                row["error"] = (row["error"] or "")[:500] or None
                s.add(LLMUsage(**row))
    except Exception:
        logger.warning("could not persist LLM usage", exc_info=True)


def record_calls(calls: list[dict], context: dict | None = None) -> None:
    """Log (always) and persist (when a context is given). Never raises."""
    try:
        log_calls(calls, context)
        persist_calls(calls, context)
    except Exception:
        logger.warning("usage recording failed", exc_info=True)
