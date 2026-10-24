from __future__ import annotations

import numpy as np
import pytest

from modelroute.data import Ticket
from modelroute.llm import OfflineProvider, ScriptedLLM
from modelroute.models import (
    Classifier,
    LLMClassifier,
    SimulatedLLMClassifier,
    SmallClassifier,
    keyword_classify,
    keyword_handler,
)


def test_small_model_learns_and_is_confident_when_right(small_model, splits) -> None:
    test = splits["test"]
    preds = small_model.predict_batch([t.text for t in test])
    correct = np.array([p.intent == t.intent for p, t in zip(preds, test, strict=True)])
    seen = np.array([t.phrasing == "seen" for t in test])
    assert correct[seen].mean() > 0.85
    conf = np.array([p.confidence for p in preds])
    # calibrated confidence must separate right from wrong answers
    assert conf[correct].mean() > conf[~correct].mean() + 0.15


def test_small_model_probabilities_are_valid(small_model) -> None:
    probs, classes = small_model.predict_proba(["I was charged twice", "app crashes"])
    assert probs.shape == (2, len(classes))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    p = small_model.predict("I was charged twice for the same order")
    assert isinstance(small_model, Classifier)
    assert p.intent == "billing_double_charge"
    assert p.category == "billing"
    assert 0 <= p.confidence <= 1 and len(p.top_intents) == 3


def test_small_model_save_load_roundtrip(small_model, tmp_path) -> None:
    path = tmp_path / "m.joblib"
    small_model.save(path)
    loaded = SmallClassifier.load(path)
    texts = ["where is my package", "please delete my account"]
    a = [(p.intent, round(p.confidence, 6)) for p in small_model.predict_batch(texts)]
    b = [(p.intent, round(p.confidence, 6)) for p in loaded.predict_batch(texts)]
    assert a == b


def test_unfitted_small_model_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        SmallClassifier().predict("hello")


def test_llm_classifier_parses_fenced_json_and_retries_bad_output() -> None:
    llm = ScriptedLLM(["sorry, I cannot", '```json\n{"intent": "track_order", '
                                          '"priority": "low", "confidence": 0.8}\n```'])
    pred = LLMClassifier(llm).predict("where is my parcel")
    assert pred.intent == "track_order" and pred.category == "shipping"
    assert pred.confidence == 0.8
    assert len(llm.calls) == 2
    assert pred.input_tokens > 0


def test_llm_classifier_rejects_unknown_intent() -> None:
    llm = ScriptedLLM(['{"intent": "make_coffee", "priority": "low"}'] * 2)
    with pytest.raises(ValueError):
        LLMClassifier(llm).predict("hi")


def test_offline_keyword_stand_in_respects_negation() -> None:
    intent, _ = keyword_classify("I'm not asking for a refund yet, I was charged twice for it")
    assert intent == "billing_double_charge"
    pred = LLMClassifier(OfflineProvider(handler=keyword_handler)).predict(
        "My order is stuck in transit and very late. This is urgent.")
    assert pred.intent == "shipping_delay" and pred.priority == "high"


def test_simulated_llm_is_deterministic_and_close_to_assumed_accuracy(splits) -> None:
    sim = SimulatedLLMClassifier()
    tickets: list[Ticket] = splits["train"]
    a = [sim.predict_ticket(t).intent for t in tickets]
    b = [sim.predict_ticket(t).intent for t in tickets]
    assert a == b
    acc = np.mean([p == t.intent for p, t in zip(a, tickets, strict=True)])
    assert 0.92 < acc < 0.99
    p = sim.predict_ticket(tickets[0])
    assert p.input_tokens > 300  # real prompt incl. taxonomy is token-heavy
    assert p.latency_ms > 0
