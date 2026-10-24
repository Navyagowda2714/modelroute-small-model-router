"""End-to-end offline benchmark: small-only vs large-only vs routed.

What is measured vs assumed
---------------------------
* Small model: trained here; accuracy, calibration and per-request latency are **measured**.
* Large model: :class:`SimulatedLLMClassifier` - accuracy and latency are **assumptions**
  (see ``SimulationAssumptions``); its token counts come from the real prompt.
* Costs: token counts x the price table in ``configs/prices.yaml`` (illustrative prices).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..data.generator import Ticket
from ..data.preprocess import build_dataset, read_jsonl
from ..evaluation.pricing import PriceTable, load_prices
from ..models.base import Prediction
from ..models.large import SimulatedLLMClassifier, SimulationAssumptions
from ..models.small import SmallClassifier
from ..routing.tuning import Precomputed, choose_threshold, evaluate_policy, precompute, sweep
from .metrics import StrategyResult, expected_calibration_error
from .plots import pareto_chart, reliability_chart, threshold_chart

log = logging.getLogger(__name__)


def ensure_dataset(settings: Settings) -> dict[str, list[Ticket]]:
    d = settings.data_dir
    if not (d / "train.jsonl").exists():
        log.info("generating dataset", extra={"dir": str(d), "n": settings.n_tickets})
        build_dataset(d, n=settings.n_tickets, seed=settings.seed)
    return {s: read_jsonl(d / f"{s}.jsonl") for s in ("train", "val", "test")}


def train_small(settings: Settings, train: list[Ticket]) -> SmallClassifier:
    start = time.perf_counter()
    model = SmallClassifier(seed=settings.seed).fit(train)
    model.save(settings.small_model_path)
    log.info("trained small model", extra={"seconds": round(time.perf_counter() - start, 2),
                                           "path": str(settings.small_model_path)})
    return model


def small_predictions(model: SmallClassifier, tickets: list[Ticket]) -> list[Prediction]:
    """Labels from one batched call; latency measured per request (single-item calls)."""
    preds = model.predict_batch([t.text for t in tickets])
    for t in tickets[:20]:  # warm-up
        model.predict(t.text)
    for p, t in zip(preds, tickets, strict=True):
        start = time.perf_counter()
        model.predict(t.text)
        p.latency_ms = (time.perf_counter() - start) * 1000
    return preds


def _breakdown(pc: Precomputed, res_threshold: float) -> dict[str, dict[str, Any]]:
    fb = (pc.confidence < res_threshold) | pc.complex_flags
    out: dict[str, dict[str, Any]] = {}
    for key in ("difficulty", "phrasing"):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(pc.tickets):
            groups[getattr(t, key)].append(i)
        for g, idx in sorted(groups.items()):
            small_acc = np.mean([pc.small[i].intent == pc.tickets[i].intent for i in idx])
            routed = [pc.large[i] if fb[i] else pc.small[i] for i in idx]
            routed_acc = np.mean([p.intent == pc.tickets[i].intent
                                  for p, i in zip(routed, idx, strict=True)])
            out[f"{key}={g}"] = {"n": len(idx), "small_acc": round(float(small_acc), 4),
                                 "routed_acc": round(float(routed_acc), 4),
                                 "fallback_rate": round(float(fb[idx].mean()), 4)}
    return out


def run_benchmark(settings: Settings, docs_dir: Path = Path("docs"),
                  tolerance: float = 0.01) -> dict[str, Any]:
    prices: PriceTable = load_prices(settings.prices_path)
    splits = ensure_dataset(settings)
    model = train_small(settings, splits["train"])
    large = SimulatedLLMClassifier(SimulationAssumptions())

    pcs: dict[str, Precomputed] = {}
    for name in ("val", "test"):
        tickets = splits[name]
        pcs[name] = precompute(tickets, small_predictions(model, tickets),
                               [large.predict_ticket(t) for t in tickets], prices)

    # 1) tune the threshold on validation only
    val_large = evaluate_policy(pcs["val"], "large-only", threshold=None, mode="large")
    val_sweep = sweep(pcs["val"])
    tuned = choose_threshold(val_sweep, reference_f1=val_large.intent_macro_f1,
                             tolerance=tolerance)
    t_star = float(tuned.threshold or settings.threshold)

    # 2) report on test
    test = pcs["test"]
    results: list[StrategyResult] = [
        evaluate_policy(test, "small-only", threshold=None, mode="small"),
        evaluate_policy(test, "large-only", threshold=None, mode="large"),
        evaluate_policy(test, f"routed (default t={settings.threshold:.2f})",
                        threshold=settings.threshold),
        evaluate_policy(test, f"routed (tuned t={t_star:.2f})", threshold=t_star),
        evaluate_policy(test, f"routed, confidence only (t={t_star:.2f})", threshold=t_star,
                        use_complexity=False),
    ]
    test_sweep = sweep(test)

    # 3) sensitivity: how do conclusions change if the large model is better / worse?
    sensitivity = []
    for acc in (0.90, 0.95, 0.98):
        sim = SimulatedLLMClassifier(SimulationAssumptions(intent_accuracy={
            "simple": acc, "long": acc, "negation": acc, "multi_intent": acc}))
        pc = precompute(test.tickets, test.small, [sim.predict_ticket(t) for t in test.tickets],
                        prices)
        lg = evaluate_policy(pc, "large-only", threshold=None, mode="large")
        rt = evaluate_policy(pc, "routed", threshold=t_star)
        sensitivity.append({"assumed_large_accuracy": acc,
                            "large_only_macro_f1": round(lg.intent_macro_f1, 4),
                            "routed_macro_f1": round(rt.intent_macro_f1, 4),
                            "routed_cost_per_1k": round(rt.cost_per_1k_usd, 4),
                            "large_cost_per_1k": round(lg.cost_per_1k_usd, 4)})

    conf = test.confidence
    correct = np.array([p.intent == t.intent for p, t in zip(test.small, test.tickets,
                                                              strict=True)])
    images = docs_dir / "images"
    pareto_chart(test_sweep, results[0], results[1], results[3], results[2],
                 images / "pareto.png")
    threshold_chart(val_sweep, t_star, images / "threshold_sweep.png")
    reliability_chart(conf, correct, images / "reliability.png")

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.tuned_threshold_path.write_text(json.dumps({"threshold": t_star}))

    large_only, tuned_test = results[1], results[3]
    report: dict[str, Any] = {
        "dataset": {k: len(v) for k, v in splits.items()},
        "tuned_threshold": t_star,
        "tuning_rule": f"cheapest threshold with val macro-F1 >= large-only - {tolerance}",
        "small_model_ece": round(expected_calibration_error(conf, correct), 4),
        "results": [r.model_dump() for r in results],
        "savings_vs_large": {
            "cost_reduction": round(1 - tuned_test.cost_per_1k_usd / large_only.cost_per_1k_usd, 4),
            "p50_latency_reduction": round(
                1 - tuned_test.latency_p50_ms / large_only.latency_p50_ms, 4),
            "macro_f1_delta": round(tuned_test.intent_macro_f1 - large_only.intent_macro_f1, 4),
        },
        "breakdown_test": _breakdown(test, t_star),
        "sensitivity": sensitivity,
        "assumptions": SimulationAssumptions().model_dump(),
        "val_sweep": [p.model_dump() for p in val_sweep],
    }
    (docs_dir / "results").mkdir(parents=True, exist_ok=True)
    (docs_dir / "results" / "eval_results.json").write_text(json.dumps(report, indent=2))
    return report


def markdown_table(results: list[dict[str, Any]]) -> str:
    head = ("| Strategy | Intent acc | Intent macro-F1 | Priority acc | p50 ms | p95 ms "
            "| $/1k req | Fallback |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = [
        f"| {r['strategy']} | {r['intent_accuracy']:.3f} | {r['intent_macro_f1']:.3f} | "
        f"{r['priority_accuracy']:.3f} | {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} | "
        f"{r['cost_per_1k_usd']:.4f} | {r['fallback_rate']:.1%} |"
        for r in results
    ]
    return "\n".join([head, *rows])
