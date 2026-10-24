"""Configurable price table -> cost per request / per 1,000 requests."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelPrice(BaseModel):
    input_per_mtok: float = Field(default=0.0, ge=0, description="USD per 1M input tokens")
    output_per_mtok: float = Field(default=0.0, ge=0, description="USD per 1M output tokens")
    per_request: float = Field(default=0.0, ge=0, description="USD fixed cost per request")
    note: str = ""


class PriceTable(BaseModel):
    models: dict[str, ModelPrice]
    default_small: str = "small-tfidf-svm"
    default_large: str = "simulated-large-llm"

    @classmethod
    def from_yaml(cls, path: Path) -> PriceTable:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def price(self, model: str) -> ModelPrice:
        if model in self.models:
            return self.models[model]
        # prefix match lets "llm-claude-sonnet-5-5" use the "claude-sonnet-5-5" entry, etc.
        for name, p in self.models.items():
            if name in model:
                return p
        raise KeyError(f"no price configured for model {model!r}")

    def cost(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
        p = self.price(model)
        return (p.per_request + input_tokens * p.input_per_mtok / 1e6
                + output_tokens * p.output_per_mtok / 1e6)


DEFAULT_PRICES_PATH = Path(__file__).resolve().parents[3] / "configs" / "prices.yaml"


def load_prices(path: Path | None = None) -> PriceTable:
    path = path or DEFAULT_PRICES_PATH
    if path.exists():
        return PriceTable.from_yaml(path)
    # Packaged fallback so an installed wheel still works without the repo's configs/.
    return PriceTable(models={
        "small-tfidf": ModelPrice(per_request=0.000002, note="CPU serving, assumed"),
        "small-lora": ModelPrice(per_request=0.00004, note="GPU serving, assumed"),
        "simulated-large-llm": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
        "offline": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
        "claude-sonnet": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
        "claude-haiku": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
        "gpt-4o-mini": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
        "llama3.1": ModelPrice(per_request=0.0002),
    })
