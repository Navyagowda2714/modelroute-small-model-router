from __future__ import annotations

import numpy as np
import pytest

from modelroute.evaluation.metrics import StrategyResult, expected_calibration_error
from modelroute.models import SimulatedLLMClassifier
from modelroute.routing import (
    ComplexityDetector,
    Router,
    RouterConfig,
    choose_threshold,
    evaluate_policy,
    precompute,
    sweep,
)
from tests.conftest import StubClassifier


@pytest.mark.parametrize(("text", "reason"), [
    ("I'm not asking for a refund yet, I was charged twice.", "negation"),
    ("My package is broken. Also, I need an invoice.", "multi_intent"),
    ("Is it shipped? Where is it? When will it come?", "multi_question"),
    ("word " * 80, "long"),
])
def test_complexity_detector_flags_hard_requests(text: str, reason: str) -> None:
    report = ComplexityDetector().assess(text)
    assert reason in report.reasons


@pytest.mark.parametrize("text", ["I can't log in to the iOS app.",
                                  "Please send me the invoice for order #A12345.",
                                  "The app crashes on startup."])
def test_complexity_detector_passes_simple_requests(text: str) -> None:
    assert not ComplexityDetector().assess(text).is_complex


def _router(prices, small_conf: float, large_fail: bool = False) -> tuple[Router, StubClassifier]:
    small = StubClassifier("small-tfidf-svm", "track_order", small_conf)
    large = StubClassifier("simulated-large-llm", "shipping_delay", 0.95, fail=large_fail,
                           input_tokens=600, output_tokens=24)
    return Router(small, large, prices, RouterConfig(threshold=0.8)), large


def test_router_uses_small_model_when_confident(prices) -> None:
    router, large = _router(prices, 0.93)
    d = router.route("Where is my order?")
    assert d.route == "small" and d.reasons == [] and large.calls == 0
    assert d.cost_usd == pytest.approx(0.000002)


def test_router_falls_back_on_low_confidence_and_accounts_cost(prices) -> None:
    router, _large = _router(prices, 0.42)
    d = router.route("Where is my order?")
    assert d.route == "large" and "low_confidence" in d.reasons
    assert d.prediction.intent == "shipping_delay"
    expected = 0.000002 + 600 * 3.0 / 1e6 + 24 * 15.0 / 1e6
    assert d.cost_usd == pytest.approx(expected)


def test_router_falls_back_on_complex_even_if_confident(prices) -> None:
    router, _ = _router(prices, 0.99)
    d = router.route("I don't want to cancel anything, my card was declined twice.")
    assert d.route == "large" and "complex:negation" in d.reasons


def test_router_force_and_graceful_degradation(prices) -> None:
    router, _ = _router(prices, 0.2)
    assert router.route("x", force="small").route == "small"
    router, _ = _router(prices, 0.2, large_fail=True)
    d = router.route("Where is my order?")
    assert d.degraded and d.route == "small" and "large_failed" in d.reasons
    assert d.prediction.intent == "track_order"


def test_policy_simulation_and_threshold_choice(small_model, splits, prices) -> None:
    val = splits["val"]
    pc = precompute(val, small_model.predict_batch([t.text for t in val]),
                    [SimulatedLLMClassifier().predict_ticket(t) for t in val], prices)
    small = evaluate_policy(pc, "s", threshold=None, mode="small")
    large = evaluate_policy(pc, "l", threshold=None, mode="large")
    assert small.fallback_rate == 0.0 and large.fallback_rate == 1.0
    assert large.cost_per_1k_usd > 100 * small.cost_per_1k_usd
    points = sweep(pc, [0.0, 0.5, 0.8, 1.01])
    rates = [p.fallback_rate for p in points]
    assert rates == sorted(rates), "fallback rate must grow with the threshold"
    chosen = choose_threshold(points, reference_f1=large.intent_macro_f1, tolerance=0.02)
    assert chosen.cost_per_1k_usd <= large.cost_per_1k_usd


def test_choose_threshold_picks_cheapest_within_tolerance() -> None:
    def pt(t: float, f1: float, cost: float) -> StrategyResult:
        return StrategyResult(strategy=str(t), n=1, intent_accuracy=f1, intent_macro_f1=f1,
                              category_accuracy=1, priority_accuracy=1, latency_p50_ms=1,
                              latency_p95_ms=1, cost_per_1k_usd=cost, fallback_rate=0,
                              threshold=t)
    pts = [pt(0.2, 0.90, 0.1), pt(0.5, 0.945, 0.5), pt(0.8, 0.96, 0.9)]
    assert choose_threshold(pts, reference_f1=0.95, tolerance=0.01).threshold == 0.5
    assert choose_threshold(pts, reference_f1=0.99, tolerance=0.0).threshold == 0.8


def test_expected_calibration_error() -> None:
    assert expected_calibration_error([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0] * 4, [False] * 4) == pytest.approx(1.0)
    assert np.isclose(expected_calibration_error([], []), 0.0)
