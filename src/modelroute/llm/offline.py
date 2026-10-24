"""Deterministic stand-ins for real LLMs.

These exist so that tests, ``make demo`` and ``make eval`` run with no API key and
no network. They do **not** understand language; quality numbers produced with
them say nothing about a real model.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from .base import LLMResponse, Message, estimate_tokens

Handler = Callable[[list[Message], "str | None", "dict[str, Any] | None"], str]


def _echo_handler(messages: list[Message], system: str | None,
                  json_schema: dict[str, Any] | None) -> str:
    last = messages[-1].content if messages else ""
    return "{}" if json_schema else f"[offline] {last[:200]}"


class OfflineProvider:
    """Deterministic provider whose answer is computed by a pure ``handler`` function."""

    def __init__(self, handler: Handler | None = None, model: str = "offline-heuristic",
                 simulated_latency_ms: float = 0.0) -> None:
        self.model = model
        self._handler = handler or _echo_handler
        self.simulated_latency_ms = simulated_latency_ms

    def complete(self, messages: list[Message], *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        text = self._handler(messages, system, json_schema)
        prompt = (system or "") + "".join(m.content for m in messages)
        measured = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            latency_ms=measured + self.simulated_latency_ms,
            model=self.model,
            provider="offline",
        )


class ScriptedLLM:
    """Returns pre-scripted responses in order; records every call. For tests."""

    def __init__(self, responses: Iterable[str], model: str = "scripted") -> None:
        self.model = model
        self._responses: deque[str] = deque(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        text = self._responses.popleft()
        prompt = (system or "") + "".join(m.content for m in messages)
        return LLMResponse(text=text, input_tokens=estimate_tokens(prompt),
                           output_tokens=estimate_tokens(text), latency_ms=0.0,
                           model=self.model, provider="scripted")
