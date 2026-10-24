"""FastAPI service: ``POST /classify`` with real-time routing, ``/metrics`` for Prometheus."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..evaluation.pricing import load_prices
from ..llm.factory import get_llm_client
from ..models.large import LLMClassifier, keyword_handler
from ..models.small import SmallClassifier
from ..routing.router import Router, RouterConfig
from .metrics import RouterMetrics

log = logging.getLogger("modelroute.api")
fallback_log = logging.getLogger("modelroute.fallback")


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    force_route: Literal["small", "large"] | None = None


class ClassifyResponse(BaseModel):
    request_id: str
    intent: str
    category: str
    priority: str
    confidence: float
    route: Literal["small", "large"]
    reasons: list[str]
    model: str
    latency_ms: float
    cost_usd: float
    degraded: bool


def resolve_threshold(settings: Settings) -> float:
    """Tuned threshold from ``make eval`` if present, else the configured default."""
    if settings.tuned_threshold_path.exists():
        return float(json.loads(settings.tuned_threshold_path.read_text())["threshold"])
    return settings.threshold


def build_router(settings: Settings) -> Router:
    if not settings.small_model_path.exists():
        raise RuntimeError(f"small model not found at {settings.small_model_path}; "
                           "run `modelroute train` (or `make train`) first")
    small = SmallClassifier.load(settings.small_model_path)
    large = LLMClassifier(get_llm_client(offline_handler=keyword_handler))
    cfg = RouterConfig(threshold=resolve_threshold(settings),
                       use_complexity=settings.use_complexity)
    return Router(small, large, load_prices(settings.prices_path), cfg)


def create_app(router: Router | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="ModelRoute", version="0.1.0",
                  description="Small model first, large LLM only when needed.")
    metrics = RouterMetrics()
    state: dict[str, Router] = {}
    if router is not None:
        state["router"] = router

    def get_router() -> Router:
        if "router" not in state:
            state["router"] = build_router(settings)
        return state["router"]

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            r = get_router()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ok", "small_model": r.small.name, "large_model": r.large.name,
                "threshold": r.config.threshold, "use_complexity": r.config.use_complexity}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        request_id = uuid.uuid4().hex[:16]
        try:
            r = get_router()
        except RuntimeError as exc:
            metrics.errors.labels(type="not_ready").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        start = time.perf_counter()
        decision = r.route(req.text, force=req.force_route)
        latency_s = time.perf_counter() - start
        small_cost = r.prices.cost(decision.small_prediction.model)
        costs = {decision.small_prediction.model: small_cost}
        if decision.route == "large":
            costs[decision.prediction.model] = decision.cost_usd - small_cost
        metrics.observe(decision.route, decision.reasons, latency_s,
                        decision.small_prediction.confidence, costs)
        if decision.reasons:
            # Structured, PII-safe: no raw ticket text, only a hash + length.
            fallback_log.info("route.fallback", extra={
                "request_id": request_id, "reasons": decision.reasons,
                "route": decision.route, "small_intent": decision.small_prediction.intent,
                "small_confidence": round(decision.small_prediction.confidence, 4),
                "final_intent": decision.prediction.intent,
                "complexity_score": decision.complexity.score,
                "text_sha256": hashlib.sha256(req.text.encode()).hexdigest()[:16],
                "text_chars": len(req.text), "degraded": decision.degraded})
        p = decision.prediction
        return ClassifyResponse(
            request_id=request_id, intent=p.intent, category=p.category, priority=p.priority,
            confidence=round(p.confidence, 4), route=decision.route, reasons=decision.reasons,
            model=p.model, latency_ms=round(latency_s * 1000, 3),
            cost_usd=round(decision.cost_usd, 8), degraded=decision.degraded)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.state.metrics = metrics
    return app
