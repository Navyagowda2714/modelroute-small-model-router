"""Instruction formatting for supervised fine-tuning (SFT) and the YAML training config.

Pure Python - no torch/transformers needed - so it is unit-tested and can export the SFT
dataset on any machine (``modelroute export-sft``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..data.generator import Ticket
from ..data.taxonomy import INTENT_NAMES, PRIORITIES

INSTRUCTION = (
    "Classify the support ticket. Return JSON with keys intent, category, priority.\n"
    f"intent is one of: {', '.join(INTENT_NAMES)}.\n"
    f"priority is one of: {', '.join(PRIORITIES)}."
)


def target_json(ticket: Ticket) -> str:
    # Fixed key order and compact separators -> short, deterministic targets.
    return json.dumps({"intent": ticket.intent, "category": ticket.category,
                       "priority": ticket.priority}, separators=(",", ":"))


def to_chat(ticket: Ticket, include_answer: bool = True) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": ticket.text}]
    if include_answer:
        msgs.append({"role": "assistant", "content": target_json(ticket)})
    return msgs


def export_sft(tickets: Iterable[Ticket], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for t in tickets:
            fh.write(json.dumps({"id": t.id, "messages": to_chat(t)}) + "\n")
            n += 1
    return n


class LoraSettings(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])


class TrainSettings(BaseModel):
    epochs: float = 2.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 8
    grad_accum_steps: int = 4
    max_seq_len: int = 384
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    eval_steps: int = 100
    eval_generate_samples: int = 200
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 13


class FinetuneConfig(BaseModel):
    base_model: str
    method: Literal["lora", "qlora"] = "qlora"
    output_dir: str = "artifacts/lora"
    train_file: str = "data/generated/sft_train.jsonl"
    val_file: str = "data/generated/sft_val.jsonl"
    lora: LoraSettings = Field(default_factory=LoraSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> FinetuneConfig:
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        return cls.model_validate(data)
