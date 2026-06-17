"""Generate an email for one scenario with one model."""
from __future__ import annotations

from . import config, llm, prompt


def generate_email(model: str, scenario: dict) -> dict:
    """Return ``{"raw": ..., "email": ...}`` for a scenario.

    ``raw`` is the full model response (incl. its <thinking>); ``email`` is the
    parsed, ready-to-send email used by the metrics.
    """
    messages = prompt.build_messages(
        scenario["intent"], scenario["key_facts"], scenario["tone"]
    )
    raw = llm.chat(
        model=model,
        messages=messages,
        temperature=config.GEN_TEMPERATURE,
        max_tokens=config.GEN_MAX_TOKENS,
    )
    return {"raw": raw, "email": prompt.extract_email(raw)}
