"""Minimal LLM client on the raw Anthropic SDK. No agent frameworks.

If ANTHROPIC_API_KEY is not set, `available` is False and every caller falls back to
deterministic behaviour, so the API still works for reviewers who have no key.
"""
from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger("llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 20.0):
        self.model = model
        self._client = None
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            import anthropic

            self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=1)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        if not self._client:
            raise LLMError("LLM not configured")
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=0,
                system=system, messages=[{"role": "user", "content": user}])
        except Exception as e:  # network, auth, rate limit, bad model name...
            log.warning("LLM call failed: %s", e)
            raise LLMError(str(e)) from e
        return "".join(getattr(b, "text", "") for b in resp.content).strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 400) -> dict:
        return extract_json(self.complete(system, user, max_tokens))


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a reply, tolerating ```json fences or stray prose."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"no JSON object in reply: {text[:120]!r}")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("JSON reply is not an object")
    return obj
