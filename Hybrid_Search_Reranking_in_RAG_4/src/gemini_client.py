"""Gemini Flash wrapper: blocking and streaming generation with retries."""
from __future__ import annotations

import logging
import time
from typing import Iterator

logger = logging.getLogger(__name__)


def build_gemini_client(settings):
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def _config(max_tokens: int):
    from google.genai import types

    return types.GenerateContentConfig(temperature=0.1, max_output_tokens=max_tokens)


def generate_text(client, model: str, prompt: str, retries: int = 2, max_tokens: int = 1024) -> str:
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=_config(max_tokens))
            return (resp.text or "").strip()
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning("Gemini call failed (%s); retrying", exc)
            time.sleep(2**attempt)
    return ""


def stream_text(client, model: str, prompt: str, retries: int = 1, max_tokens: int = 1024) -> Iterator[str]:
    """Yield text pieces as Gemini produces them.

    Retries only when the failure happens BEFORE the first token; once tokens have been
    sent to the client a retry would duplicate text, so the error is raised instead.
    """
    for attempt in range(retries + 1):
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=prompt, config=_config(max_tokens)
            ):
                piece = getattr(chunk, "text", None)
                if piece:
                    started = True
                    yield piece
            return
        except Exception as exc:
            if started or attempt == retries:
                raise
            logger.warning("Gemini stream failed before first token (%s); retrying", exc)
            time.sleep(2**attempt)
