"""Real LLM providers. SDK imports are lazy so the package imports without extras."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .base import LLMError, LLMResponse, Message, estimate_tokens


def _schema_instruction(json_schema: dict[str, Any] | None) -> str:
    if not json_schema:
        return ""
    return (
        "\n\nRespond with a single JSON object only (no prose, no code fences) that "
        f"matches this JSON schema:\n{json.dumps(json_schema)}"
    )


class AnthropicProvider:
    """Anthropic Messages API via the official SDK (``pip install modelroute[anthropic]``)."""

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None,
                 max_tokens: int = 512, timeout_s: float = 30.0) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError(
                "anthropic SDK not installed: pip install 'modelroute[anthropic]'") from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)

    def complete(self, messages: list[Message], *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        sys_prompt = (system or "") + _schema_instruction(json_schema)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=sys_prompt or None,  # type: ignore[arg-type]
                messages=[m.model_dump() for m in messages if m.role != "system"],
            )
        except Exception as exc:  # pragma: no cover - network
            raise LLMError(f"anthropic call failed: {exc}") from exc
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
            provider="anthropic",
        )


class OpenAIProvider:
    """OpenAI Chat Completions via the official SDK (``pip install modelroute[openai]``)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 timeout_s: float = 30.0) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError("openai SDK not installed: pip install 'modelroute[openai]'") from exc
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)

    def complete(self, messages: list[Message], *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        payload: list[dict[str, str]] = []
        sys_prompt = (system or "") + _schema_instruction(json_schema)
        if sys_prompt:
            payload.append({"role": "system", "content": sys_prompt})
        payload.extend(m.model_dump() for m in messages)
        kwargs: dict[str, Any] = {}
        if json_schema:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=payload, temperature=temperature, **kwargs  # type: ignore[arg-type]
            )
        except Exception as exc:  # pragma: no cover - network
            raise LLMError(f"openai call failed: {exc}") from exc
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
            provider="openai",
        )


class OllamaProvider:
    """Local models served by Ollama over HTTP (``/api/chat``)."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434",
                 timeout_s: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def complete(self, messages: list[Message], *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        payload_msgs: list[dict[str, str]] = []
        sys_prompt = (system or "") + _schema_instruction(json_schema)
        if sys_prompt:
            payload_msgs.append({"role": "system", "content": sys_prompt})
        payload_msgs.extend(m.model_dump() for m in messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema:
            body["format"] = "json"
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # pragma: no cover - network
            raise LLMError(f"ollama call failed: {exc}") from exc
        text = data.get("message", {}).get("content", "")
        prompt_text = " ".join(m["content"] for m in payload_msgs)
        return LLMResponse(
            text=text,
            input_tokens=int(data.get("prompt_eval_count") or estimate_tokens(prompt_text)),
            output_tokens=int(data.get("eval_count") or estimate_tokens(text)),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
            provider="ollama",
        )
