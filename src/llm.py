"""Thin OpenRouter client wrapper.

OpenRouter is OpenAI-compatible, so we use the official ``openai`` SDK pointed
at OpenRouter's base URL. One ``chat`` function serves both email generation
and the LLM-as-judge metrics — the only difference is the model and prompt.
"""
from __future__ import annotations

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config


class LLMError(RuntimeError):
    """Raised when the model returns no usable content."""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily build the shared client so importing this module needs no key.

    ``default_headers`` are optional OpenRouter niceties (they show up on the
    dashboard) and are safe to omit.
    """
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise LLMError("OPENROUTER_API_KEY is not set. Copy .env.example to .env.")
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            default_headers={"X-Title": config.APP_TITLE},
        )
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.4,
    max_tokens: int = 1200,
) -> str:
    """Send a chat-completions request and return the assistant text.

    Retries transient failures with exponential backoff. Some reasoning models
    reject a custom ``temperature``; if that happens we retry once without it
    rather than failing the whole run.
    """
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - inspect message for temp rejection
        if "temperature" in str(exc).lower():
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
        else:
            raise

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise LLMError(f"Model {model} returned empty content.")
    return content
