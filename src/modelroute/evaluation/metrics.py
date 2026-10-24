"""Quality, latency and cost metrics for a set of routed predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0


class StrategyResult(BaseModel):
    strategy: str
    n: int
    intent_accuracy: float
    intent_macro_f1: float
    category_accuracy: float
    priority_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_usd: float
    fallback_rate: float
    threshold: float | None = None


def summarize(strategy: str, gold_intent: Sequence[str], pred_intent: Sequence[str],
              gold_category: Sequence[str], pred_category: Sequence[str],
              gold_priority: Sequence[str], pred_priority: Sequence[str],
              latencies_ms: Sequence[float], costs_usd: Sequence[float],
              fallbacks: Sequence[bool], threshold: float | None = None) -> StrategyResult:
    n = len(gold_intent)
    return StrategyResult(
        strategy=strategy,
        n=n,
        intent_accuracy=float(accuracy_score(gold_intent, pred_intent)),
        intent_macro_f1=float(f1_score(gold_intent, pred_intent, average="macro", zero_division=0)),
        category_accuracy=float(accuracy_score(gold_category, pred_category)),
        priority_accuracy=float(accuracy_score(gold_priority, pred_priority)),
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        cost_per_1k_usd=float(np.sum(costs_usd) / max(1, n) * 1000),
        fallback_rate=float(np.mean(fallbacks)) if n else 0.0,
        threshold=threshold,
    )


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool],
                               bins: int = 10) -> float:
    """ECE: bin-size-weighted mean |accuracy - confidence| over equal-width bins."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)
