"""Provider-agnostic LLM interface.

Every provider implements :class:`LLMClient`. Callers only ever depend on this
protocol, which keeps the rest of the code base testable with deterministic
stand-ins (see :mod:`.offline`).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class LLMResponse(BaseModel):
    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    model: str
    provider: str = "unknown"


class LLMError(RuntimeError):
    """Raised when a provider call fails after the provider's own retries."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion contract shared by all providers."""

    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 characters per token for English)."""
    return max(1, round(len(text) / 4))
