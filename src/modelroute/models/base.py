"""The single interface every ticket classifier implements (small, fine-tuned or LLM)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    intent: str
    category: str
    priority: str
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated P(intent is correct)")
    model: str
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = 0
    output_tokens: int = 0
    top_intents: list[tuple[str, float]] = Field(default_factory=list)


@runtime_checkable
class Classifier(Protocol):
    name: str

    def predict(self, text: str) -> Prediction: ...

    def predict_batch(self, texts: list[str]) -> list[Prediction]: ...
