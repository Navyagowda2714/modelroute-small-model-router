"""Offline policy simulation and threshold tuning.

Both models' predictions for a labelled split are computed once (:func:`precompute`); every
routing policy is then a cheap boolean mask over those predictions. This mirrors exactly what
:class:`~modelroute.routing.router.Router` does online (same detector, same cost model), so a
threshold tuned here behaves the same in serving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.generator import Ticket
from ..evaluation.metrics import StrategyResult, summarize
from ..evaluation.pricing import PriceTable
from ..models.base import Prediction
from .complexity import ComplexityDetector


@dataclass
class Precomputed:
    tickets: list[Ticket]
    small: list[Prediction]
    large: list[Prediction]
    complex_flags: np.ndarray
    small_cost: np.ndarray
    large_cost: np.ndarray

    @property
    def confidence(self) -> np.ndarray:
        return np.array([p.confidence for p in self.small])


def precompute(tickets: list[Ticket], small: list[Prediction], large: list[Prediction],
               prices: PriceTable, detector: ComplexityDetector | None = None) -> Precomputed:
    detector = detector or ComplexityDetector()
    flags = np.array([detector.assess(t.text).is_complex for t in tickets])
    sc = np.array([prices.cost(p.model, p.input_tokens, p.output_tokens) for p in small])
    lc = np.array([prices.cost(p.model, p.input_tokens, p.output_tokens) for p in large])
    return Precomputed(tickets, small, large, flags, sc, lc)


def evaluate_policy(pc: Precomputed, name: str, *, threshold: float | None,
                    use_complexity: bool = True, mode: str = "routed") -> StrategyResult:
    """mode: ``small`` (never fall back), ``large`` (always), ``routed`` (policy)."""
    n = len(pc.tickets)
    if mode == "small":
        fb = np.zeros(n, dtype=bool)
    elif mode == "large":
        fb = np.ones(n, dtype=bool)
    else:
        assert threshold is not None
        fb = pc.confidence < threshold
        if use_complexity:
            fb = fb | pc.complex_flags
    chosen = [lg if f else sm for f, sm, lg in zip(fb, pc.small, pc.large, strict=True)]
    small_lat = np.array([p.latency_ms for p in pc.small])
    large_lat = np.array([p.latency_ms for p in pc.large])
    if mode == "large":
        # large-only deployments do not run the small model first
        latency, cost = large_lat, pc.large_cost
    else:
        latency = small_lat + np.where(fb, large_lat, 0.0)
        cost = pc.small_cost + np.where(fb, pc.large_cost, 0.0)
    return summarize(
        name,
        [t.intent for t in pc.tickets], [p.intent for p in chosen],
        [t.category for t in pc.tickets], [p.category for p in chosen],
        [t.priority for t in pc.tickets], [p.priority for p in chosen],
        latency.tolist(), cost.tolist(), fb.tolist(), threshold,
    )


def sweep(pc: Precomputed, thresholds: list[float] | None = None,
          use_complexity: bool = True) -> list[StrategyResult]:
    thresholds = thresholds if thresholds is not None else [round(x, 2) for x in
                                                             np.linspace(0, 1, 41)]
    return [evaluate_policy(pc, f"routed@{t:.2f}", threshold=t, use_complexity=use_complexity)
            for t in thresholds]


def choose_threshold(points: list[StrategyResult], reference_f1: float,
                     tolerance: float = 0.005) -> StrategyResult:
    """Cheapest point whose macro-F1 is within ``tolerance`` of ``reference_f1``.

    Falls back to the best-F1 point if none qualifies.
    """
    ok = [p for p in points if p.intent_macro_f1 >= reference_f1 - tolerance]
    if not ok:
        return max(points, key=lambda p: (p.intent_macro_f1, -p.cost_per_1k_usd))
    return min(ok, key=lambda p: (p.cost_per_1k_usd, -p.intent_macro_f1))
