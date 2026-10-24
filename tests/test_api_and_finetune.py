from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from modelroute.data import Ticket
from modelroute.evaluation.pricing import PriceTable
from modelroute.finetune import FinetuneConfig, export_sft, target_json, to_chat
from modelroute.finetune.train_lora import tokenize_example
from modelroute.llm import OfflineProvider
from modelroute.models import LLMClassifier, keyword_handler
from modelroute.routing import Router, RouterConfig
from tests.conftest import REPO


@pytest.fixture()
def client(small_model, prices) -> TestClient:
    from modelroute.serving import create_app

    large = LLMClassifier(OfflineProvider(handler=keyword_handler))
    router = Router(small_model, large, prices, RouterConfig(threshold=0.8))
    return TestClient(create_app(router=router))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["threshold"] == 0.8


def test_classify_and_prometheus_metrics(client: TestClient, caplog) -> None:
    r = client.post("/classify", json={"text": "I was charged twice for my order, $49 two times"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "billing_double_charge" and body["category"] == "billing"
    with caplog.at_level(logging.INFO, logger="modelroute.fallback"):
        r2 = client.post("/classify", json={"text": "hello", "force_route": "large"})
    assert r2.json()["route"] == "large"
    record = next(rec for rec in caplog.records if rec.getMessage() == "route.fallback")
    assert record.reasons == ["forced:large"]  # type: ignore[attr-defined]
    assert not hasattr(record, "text"), "raw ticket text must not be logged"

    metrics = client.get("/metrics").text
    assert 'modelroute_requests_total{route="large"} 1.0' in metrics
    assert 'modelroute_fallbacks_total{reason="forced"} 1.0' in metrics
    assert "modelroute_request_latency_seconds_bucket" in metrics
    assert 'modelroute_cost_usd_total{model="llm-offline-heuristic"}' in metrics
    assert "modelroute_fallback_ratio 0.5" in metrics


def test_classify_validates_input(client: TestClient) -> None:
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "x", "force_route": "huge"}).status_code == 422


def test_price_table_lookup() -> None:
    table = PriceTable.model_validate({"models": {"claude-sonnet": {"input_per_mtok": 3,
                                                                    "output_per_mtok": 15}}})
    assert table.cost("llm-claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(3.0)
    with pytest.raises(KeyError):
        table.cost("unknown-model")


def test_sft_formatting_and_export(tmp_path) -> None:
    t = Ticket(id="T1", text="I was charged twice", intent="billing_double_charge",
               category="billing", priority="high")
    msgs = to_chat(t)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert json.loads(msgs[-1]["content"]) == {"intent": "billing_double_charge",
                                               "category": "billing", "priority": "high"}
    assert target_json(t) == (
        '{"intent":"billing_double_charge","category":"billing","priority":"high"}')
    assert export_sft([t, t], tmp_path / "sft.jsonl") == 2


@pytest.mark.parametrize("name", ["lora_qwen2.5-0.5b.yaml", "lora_llama3.1-8b.yaml"])
def test_finetune_configs_are_valid(name: str) -> None:
    cfg = FinetuneConfig.from_yaml(REPO / "configs" / name)
    assert cfg.method == "qlora" and cfg.lora.r > 0 and "q_proj" in cfg.lora.target_modules


def test_sft_tokenisation_masks_prompt_tokens() -> None:
    roles = {"system": 1, "user": 2, "assistant": 9}

    class FakeTokenizer:
        """Each message -> [role_token, *chars]; generation prompt = assistant role token."""

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            ids = [tok for m in messages for tok in [roles[m["role"]], *map(ord, m["content"])]]
            return ids + ([roles["assistant"]] if add_generation_prompt else [])

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "{}"}]
    out = tokenize_example(FakeTokenizer(), msgs, max_len=64)
    assert out["input_ids"] == [1, ord("s"), 2, ord("h"), ord("i"), 9, ord("{"), ord("}")]
    # loss only on the assistant answer tokens
    assert out["labels"] == [-100] * 6 + [ord("{"), ord("}")]
    assert len(out["attention_mask"]) == len(out["input_ids"])
    truncated = tokenize_example(FakeTokenizer(), msgs, max_len=7)
    assert truncated["labels"] == [-100] * 6 + [ord("{")]
