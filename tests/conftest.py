from __future__ import annotations

from pathlib import Path

import pytest

from modelroute.data import Ticket, clean_tickets, generate_tickets, stratified_split
from modelroute.data.taxonomy import category_of
from modelroute.evaluation.pricing import load_prices
from modelroute.models import Prediction, SmallClassifier

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def splits() -> dict[str, list[Ticket]]:
    tickets, _ = clean_tickets(generate_tickets(n=2000, seed=3))
    return stratified_split(tickets, seed=3)


@pytest.fixture(scope="session")
def small_model(splits: dict[str, list[Ticket]]) -> SmallClassifier:
    return SmallClassifier(seed=3).fit(splits["train"])


@pytest.fixture()
def prices():
    return load_prices(REPO / "configs" / "prices.yaml")


class StubClassifier:
    """Returns a fixed prediction; optionally raises to simulate an outage."""

    def __init__(self, name: str, intent: str = "refund_request", confidence: float = 0.9,
                 fail: bool = False, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.name = name
        self.intent = intent
        self.confidence = confidence
        self.fail = fail
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return Prediction(intent=self.intent, category=category_of(self.intent),
                          priority="medium", confidence=self.confidence, model=self.name,
                          input_tokens=self.input_tokens, output_tokens=self.output_tokens)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]
