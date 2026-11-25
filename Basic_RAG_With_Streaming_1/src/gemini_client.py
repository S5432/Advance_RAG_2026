"""Gemini Flash client on the `google-genai` SDK: generate / generate_structured / stream with timeout,
exponential-backoff retries and a typed error hierarchy. The SDK object can be injected (used by the tests)."""
import json
import logging
import random
import re
import time
from urllib.parse import urlparse

logger = logging.getLogger("gemini_client")
DEFAULT_HOST = "generativelanguage.googleapis.com"

# ---------------- errors ----------------
class GeminiError(Exception):
    error_type = "upstream_error"
    retryable = False
class GeminiAuthError(GeminiError):          error_type = "auth_error"
class GeminiQuotaError(GeminiError):         error_type = "quota_exceeded"; retryable = True
class GeminiTimeoutError(GeminiError):       error_type = "timeout"; retryable = True
class GeminiBadRequestError(GeminiError):    error_type = "bad_request"
class GeminiUpstreamError(GeminiError):      error_type = "upstream_error"
class GeminiTransientError(GeminiUpstreamError):  retryable = True
class GeminiEmptyResponseError(GeminiError): error_type = "empty_response"
class GeminiInvalidOutputError(GeminiError): error_type = "invalid_output"

def split_base_url(base_url: str | None) -> tuple[str | None, str | None]:
    """'https://host/v1beta' -> (base_url only if not the Google default, api_version)."""
    if not base_url:
        return None, None
    u = urlparse(base_url)
    m = re.search(r"/(v\d+(?:beta\d*|alpha\d*)?)/?$", u.path)
    version = m.group(1) if m else None
    base = None if u.netloc == DEFAULT_HOST else f"{u.scheme}://{u.netloc}/"
    return base, version

def map_error(exc: Exception) -> GeminiError:
    """Translate SDK / network exceptions into our hierarchy (duck-typed so no SDK import is needed)."""
    if isinstance(exc, GeminiError):
        return exc
    code = getattr(exc, "code", None)
    msg = str(getattr(exc, "message", None) or exc)
    name = type(exc).__name__.lower()
    if isinstance(code, int):
        low = msg.lower()
        if code in (401, 403) or (code == 400 and ("api key" in low or "api_key" in low)):
            return GeminiAuthError("Gemini rejected the API key")
        if code == 429:
            return GeminiQuotaError("Gemini quota / rate limit exceeded")
        if code in (504, 408):
            return GeminiTimeoutError("Gemini request timed out")
        if code >= 500:
            return GeminiTransientError(f"Gemini server error {code}")
        if 400 <= code < 500:
            return GeminiBadRequestError(f"Gemini rejected the request ({code}): {msg[:200]}")
    if isinstance(exc, TimeoutError) or "timeout" in name or "timedout" in name:
        return GeminiTimeoutError("Gemini request timed out")
    if isinstance(exc, (ConnectionError, OSError)) or "connect" in name or "network" in name:
        return GeminiTransientError(f"Could not reach Gemini ({type(exc).__name__})")
    return GeminiUpstreamError(f"Unexpected Gemini client error ({type(exc).__name__})")

def parse_json_object(text: str) -> dict:
    """Parse model output as a JSON object; tolerate ```json fences."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiInvalidOutputError(f"Model returned malformed JSON ({exc.msg} at char {exc.pos})") from exc
    if not isinstance(data, dict):
        raise GeminiInvalidOutputError("Model returned JSON that is not an object")
    return data

# ---------------- client ----------------
class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", base_url: str | None = None, timeout: float = 30.0,
                 max_retries: int = 3, backoff_base: float = 0.5, backoff_max: float = 8.0,
                 thinking_budget: int | None = 0, client=None, sleep=time.sleep):
        if not api_key and client is None:
            raise ValueError("api_key is required (set GEMINI_API_KEY)")
        self.model, self.timeout, self.max_retries = model, timeout, max_retries
        self.backoff_base, self.backoff_max, self.thinking_budget = backoff_base, backoff_max, thinking_budget
        self._sleep = sleep
        if client is None:
            from google import genai                      # imported lazily so tests do not need the SDK
            base, version = split_base_url(base_url)
            http_options: dict = {"timeout": int(timeout * 1000)}            # SDK timeout is in milliseconds
            if base: http_options["base_url"] = base
            if version: http_options["api_version"] = version
            client = genai.Client(api_key=api_key, http_options=http_options)
        self._client = client

    # ----- helpers -----
    def _config(self, model: str, system_prompt, temperature, top_p, max_output_tokens, schema=None) -> dict:
        cfg: dict = {"temperature": temperature, "top_p": top_p, "max_output_tokens": max_output_tokens}
        if system_prompt:
            cfg["system_instruction"] = system_prompt
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        if self.thinking_budget is not None and model.startswith("gemini-2.5-flash"):
            cfg["thinking_config"] = {"thinking_budget": self.thinking_budget}
        return cfg

    def _delay(self, attempt: int) -> float:
        d = min(self.backoff_max, self.backoff_base * 2 ** attempt)           # 0.5, 1, 2, 4 ...
        return d + random.uniform(0, d * 0.1)

    def _call_with_retry(self, fn, op: str):
        last: GeminiError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                err = map_error(exc)
                if not err.retryable or attempt == self.max_retries:
                    logger.error("gemini_%s_failed type=%s attempts=%d detail=%s", op, err.error_type, attempt + 1, exc)
                    raise err from exc
                last = err
                delay = self._delay(attempt)
                logger.warning("gemini_%s_retry attempt=%d/%d type=%s sleeping=%.2fs", op, attempt + 1, self.max_retries + 1,
                               err.error_type, delay)
                self._sleep(delay)
        raise last or GeminiUpstreamError("Gemini request failed")

    @staticmethod
    def _text(resp) -> str:
        text = getattr(resp, "text", None)
        if not text or not text.strip():
            raise GeminiEmptyResponseError("Gemini returned no text (blocked by safety filters or token limit reached)")
        return text

    # ----- public API -----
    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.7, top_p: float = 0.95,
                 max_output_tokens: int = 1024, model: str | None = None) -> str:
        model = model or self.model
        cfg = self._config(model, system_prompt, temperature, top_p, max_output_tokens)
        start = time.perf_counter()
        resp = self._call_with_retry(lambda: self._client.models.generate_content(model=model, contents=prompt, config=cfg), "generate")
        text = self._text(resp)
        logger.info("gemini_generate_ok model=%s prompt_chars=%d reply_chars=%d latency_ms=%.0f", model, len(prompt), len(text),
                    (time.perf_counter() - start) * 1000)
        return text

    def generate_structured(self, prompt: str, response_schema: dict, system_prompt: str | None = None, temperature: float = 0.2,
                            top_p: float = 0.8, max_output_tokens: int = 1024, model: str | None = None) -> dict:
        """JSON mode: response_mime_type=application/json + response_schema. Returns a parsed dict (validate it with Pydantic)."""
        model = model or self.model
        cfg = self._config(model, system_prompt, temperature, top_p, max_output_tokens, schema=response_schema)
        start = time.perf_counter()
        resp = self._call_with_retry(lambda: self._client.models.generate_content(model=model, contents=prompt, config=cfg), "structured")
        data = parse_json_object(self._text(resp))
        logger.info("gemini_structured_ok model=%s latency_ms=%.0f", model, (time.perf_counter() - start) * 1000)
        return data

    def stream(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.7, top_p: float = 0.95,
               max_output_tokens: int = 1024, model: str | None = None):
        """Yield text chunks as they arrive. Transient failures are retried only BEFORE the first chunk (afterwards the
        caller has already received text, so the error is raised instead of silently restarting the answer)."""
        model = model or self.model
        cfg = self._config(model, system_prompt, temperature, top_p, max_output_tokens)
        attempt = 0
        while True:
            yielded = 0
            try:
                for chunk in self._client.models.generate_content_stream(model=model, contents=prompt, config=cfg):
                    text = getattr(chunk, "text", None)
                    if text:
                        yielded += 1
                        yield text
                if yielded == 0:
                    raise GeminiEmptyResponseError("Gemini returned no text (blocked by safety filters or token limit reached)")
                return
            except GeneratorExit:
                raise
            except Exception as exc:
                err = map_error(exc)
                if yielded or not err.retryable or attempt >= self.max_retries:
                    logger.error("gemini_stream_failed type=%s chunks_sent=%d attempts=%d detail=%s", err.error_type, yielded, attempt + 1, exc)
                    raise err from exc
                delay = self._delay(attempt); attempt += 1
                logger.warning("gemini_stream_retry attempt=%d type=%s sleeping=%.2fs", attempt, err.error_type, delay)
                self._sleep(delay)
