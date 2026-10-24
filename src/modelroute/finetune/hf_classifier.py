"""Serve a LoRA fine-tuned causal LM behind the same :class:`Classifier` interface.

Confidence is not the model's self-report: we score every candidate intent by the
log-likelihood of the answer prefix ``{"intent":"<name>"`` and take a temperature-scaled
softmax over the 20 candidates. ``temperature`` should be fitted on the validation split
(minimise NLL) so the router's threshold has the same meaning as for the TF-IDF model.

Requires ``pip install 'modelroute[finetune]'`` and the trained adapter; not run in CI.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from ..data.taxonomy import INTENT_NAMES, INTENTS, category_of
from ..models.base import Prediction
from .formatting import INSTRUCTION


class HFLoraClassifier:  # pragma: no cover - requires GPU + weights
    def __init__(self, base_model: str, adapter_dir: str, temperature: float = 1.0,
                 device: str | None = None) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = "small-lora"
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                    device_map=device or "auto")
        self.model = PeftModel.from_pretrained(base, adapter_dir).eval()

    def _prompt(self, text: str) -> str:
        msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _intent_logprobs(self, prompt: str) -> list[float]:
        torch = self._torch
        cands = [f'{{"intent":"{name}"' for name in INTENT_NAMES]
        enc = self.tokenizer([prompt + c for c in cands], return_tensors="pt", padding=True)
        enc = enc.to(self.model.device)
        p_len = len(self.tokenizer(prompt)["input_ids"])
        with torch.no_grad():
            logits = self.model(**enc).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = enc["input_ids"][:, 1:]
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, : p_len - 1] = 0  # only score the candidate continuation
        return (tok_lp * mask).sum(-1).tolist()

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        prompt = self._prompt(text)
        scores = [s / self.temperature for s in self._intent_logprobs(prompt)]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        probs = [e / z for e in exps]
        best = max(range(len(probs)), key=probs.__getitem__)
        intent = INTENT_NAMES[best]
        priority = self._generate_priority(prompt, intent)
        ranked = sorted(zip(INTENT_NAMES, probs, strict=True), key=lambda x: -x[1])[:3]
        return Prediction(intent=intent, category=category_of(intent), priority=priority,
                          confidence=probs[best], model=self.name,
                          latency_ms=(time.perf_counter() - start) * 1000,
                          top_intents=[(n, round(p, 4)) for n, p in ranked])

    def _generate_priority(self, prompt: str, intent: str) -> str:
        prefix = f'{{"intent":"{intent}","category":"{category_of(intent)}","priority":"'
        enc = self.tokenizer(prompt + prefix, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=6, do_sample=False)
        tail = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            value: Any = json.loads(prefix + tail.split('"')[0] + '"}')["priority"]
        except (json.JSONDecodeError, KeyError):
            value = None
        return value if value in ("low", "medium", "high", "urgent") else \
            INTENTS[intent].base_priority

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]
