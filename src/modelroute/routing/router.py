"""Confidence- and complexity-gated router: small model first, large model on demand.

Policy (in order):
1. ``force`` overrides everything (useful for shadow tests and debugging).
2. Run the small model (always - it is ~1 ms and provides the confidence signal).
3. If the request is flagged complex -> fall back to the large model.
4. If calibrated confidence < threshold -> fall back.
5. Otherwise answer with the small model.
If the large model errors, degrade gracefully to the small model's answer.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, Field

from ..evaluation.pricing import PriceTable
from ..models.base import Classifier, Prediction
from .complexity import ComplexityDetector, ComplexityReport

log = logging.getLogger("modelroute.router")

Route = Literal["small", "large"]


class RouterConfig(BaseModel):
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    use_complexity: bool = True


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str]
    prediction: Prediction
    small_prediction: Prediction
    complexity: ComplexityReport
    latency_ms: float
    cost_usd: float
    degraded: bool = False


class Router:
    def __init__(self, small: Classifier, large: Classifier, prices: PriceTable,
                 config: RouterConfig | None = None,
                 detector: ComplexityDetector | None = None) -> None:
        self.small = small
        self.large = large
        self.prices = prices
        self.config = config or RouterConfig()
        self.detector = detector or ComplexityDetector()

    def _cost(self, p: Prediction) -> float:
        return self.prices.cost(p.model, p.input_tokens, p.output_tokens)

    def fallback_reasons(self, small_pred: Prediction, complexity: ComplexityReport) -> list[str]:
        reasons: list[str] = []
        if self.config.use_complexity and complexity.is_complex:
            reasons.extend(f"complex:{r}" for r in complexity.reasons)
        if small_pred.confidence < self.config.threshold:
            reasons.append("low_confidence")
        return reasons

    def route(self, text: str, force: Route | None = None) -> RouteDecision:
        start = time.perf_counter()
        small_pred = self.small.predict(text)
        complexity = self.detector.assess(text)
        reasons = [f"forced:{force}"] if force else self.fallback_reasons(small_pred, complexity)
        use_large = force == "large" or (force is None and bool(reasons))

        cost = self._cost(small_pred)
        final, degraded = small_pred, False
        if use_large:
            try:
                final = self.large.predict(text)
                cost += self._cost(final)
            except Exception as exc:
                degraded = True
                reasons.append("large_failed")
                log.error("large model failed, serving small-model answer",
                          extra={"error": str(exc)})
        route: Route = "large" if use_large and not degraded else "small"
        return RouteDecision(route=route, reasons=reasons, prediction=final,
                             small_prediction=small_pred, complexity=complexity,
                             latency_ms=(time.perf_counter() - start) * 1000,
                             cost_usd=cost, degraded=degraded)
