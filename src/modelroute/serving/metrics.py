"""Prometheus instruments. One registry per app instance keeps tests isolated."""

from __future__ import annotations

from collections import deque

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class RouterMetrics:
    def __init__(self, registry: CollectorRegistry | None = None, window: int = 500) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("modelroute_requests_total", "Classified requests by route",
                                ["route"], registry=r)
        self.fallbacks = Counter("modelroute_fallbacks_total",
                                 "Fallbacks to the large model by reason", ["reason"], registry=r)
        self.errors = Counter("modelroute_errors_total", "Request errors by type", ["type"],
                              registry=r)
        self.latency = Histogram(
            "modelroute_request_latency_seconds", "End-to-end classification latency",
            ["route"], registry=r,
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5))
        self.confidence = Histogram(
            "modelroute_small_confidence", "Calibrated confidence of the small model",
            registry=r, buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
        self.cost = Counter("modelroute_cost_usd_total", "Estimated spend in USD by model",
                            ["model"], registry=r)
        self.fallback_ratio = Gauge("modelroute_fallback_ratio",
                                    f"Share of the last {window} requests sent to the large model",
                                    registry=r)
        self._window: deque[bool] = deque(maxlen=window)

    def observe(self, route: str, reasons: list[str], latency_s: float, confidence: float,
                costs: dict[str, float]) -> None:
        self.requests.labels(route=route).inc()
        for reason in reasons:
            self.fallbacks.labels(reason=reason.split(":")[0] if reason.startswith("forced")
                                  else reason).inc()
        self.latency.labels(route=route).observe(latency_s)
        self.confidence.observe(confidence)
        for model, usd in costs.items():
            self.cost.labels(model=model).inc(usd)
        self._window.append(route == "large")
        self.fallback_ratio.set(sum(self._window) / len(self._window))
