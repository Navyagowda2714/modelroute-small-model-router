"""Large-model side of the router.

* :class:`LLMClassifier` - the production path: a prompted LLM (Anthropic / OpenAI / Ollama)
  returning JSON. In offline mode it is backed by :func:`keyword_handler`, a transparent
  lexicon heuristic so the API works with no key.
* :class:`SimulatedLLMClassifier` - used **only for offline evaluation**. It does not read the
  text: given the gold label it answers correctly with an *assumed* accuracy, and samples a
  latency from an *assumed* distribution. It exists so the router's cost/latency mechanics
  can be measured end-to-end here; it is not evidence of how good any real LLM is.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any

from pydantic import BaseModel

from ..data.generator import Ticket
from ..data.taxonomy import INTENT_NAMES, INTENTS, PRIORITIES, category_of, shift_priority
from ..llm.base import LLMClient, Message, estimate_tokens
from .base import Prediction

SYSTEM_PROMPT = (
    "You are a support-ticket triage model. Classify the customer's ticket.\n"
    "Pick exactly one intent - the customer's PRIMARY request. Ignore things the customer "
    "explicitly says they are NOT asking for.\nIntents:\n"
    + "\n".join(f"- {n}: {INTENTS[n].description}" for n in INTENT_NAMES)
    + "\nPriority is one of low|medium|high|urgent. Raise it for urgency, repeated contacts, "
    "business impact, chargeback or legal threats; lower it if the customer says it is not urgent."
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_NAMES},
        "priority": {"type": "string", "enum": PRIORITIES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["intent", "priority"],
}


def build_user_message(text: str) -> str:
    return f"Ticket:\n<<<\n{text}\n>>>"


def prompt_tokens(text: str) -> int:
    return estimate_tokens(SYSTEM_PROMPT + json.dumps(OUTPUT_SCHEMA) + build_user_message(text))


def parse_llm_output(raw: str) -> dict[str, Any]:
    """Extract and validate the JSON object; tolerate code fences and surrounding prose."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM output: {raw[:120]!r}")
    data = json.loads(match.group(0))
    if data.get("intent") not in INTENTS:
        raise ValueError(f"unknown intent {data.get('intent')!r}")
    if data.get("priority") not in PRIORITIES:
        data["priority"] = INTENTS[data["intent"]].base_priority
    return data


# --------------------------------------------------------------------------- offline handler
_NEGATORS = {"not", "don't", "dont", "isn't", "no", "never", "aren't", "i'm not", "do"}
LEXICON: dict[str, list[str]] = {
    "billing_double_charge": ["twice", "two times", "double", "duplicate", "same charge"],
    "refund_request": ["refund", "money back"],
    "invoice_request": ["invoice", "receipt", "vat", "billing documents", "tax id"],
    "payment_failed": ["declined", "payment failed", "failed", "could not be processed",
                       "rejected"],
    "update_payment_method": ["change the card", "new card", "payment method", "card on file",
                              "update my billing", "add a new credit"],
    "login_issue": ["log in", "login", "sign in", "two-factor", "sso", "session expired"],
    "password_reset": ["password", "reset"],
    "account_locked": ["locked", "suspended", "disabled", "deactivated", "frozen", "on hold"],
    "update_account_details": ["change the email", "phone number", "change the name", "username",
                               "profile details", "contact email", "company address"],
    "delete_account": ["delete", "erase", "gdpr", "close my account", "wiped", "remove my account"],
    "cancel_subscription": ["cancel", "unsubscribe", "auto-renew", "auto renewal", "end my"],
    "upgrade_plan": ["upgrade", "more seats", "higher plan", "business tier"],
    "downgrade_plan": ["downgrade", "cheaper", "move down", "free tier", "reduce our seats",
                       "too expensive", "something smaller", "fewer"],
    "shipping_delay": ["late", "still not here", "stuck in transit", "pushed back",
                       "still waiting"],
    "track_order": ["tracking", "track", "shipped yet", "status of order", "where my package"],
    "wrong_item": ["wrong item", "wrong size", "different product", "isn't what i ordered",
                   "different model", "doesn't match"],
    "damaged_item": ["broken", "damaged", "cracked", "shattered", "dent"],
    "change_shipping_address": ["address", "zip code", "redirect"],
    "app_crash": ["crash", "freezes", "blank white screen", "error 500", "hang"],
    "feature_request": ["would be great", "feature request", "please add", "suggestion",
                        "would love", "would be nice", "any plans"],
}


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 30) : start]
    return any(re.search(rf"\b{re.escape(n)}\b", window) for n in _NEGATORS - {"do"})


def keyword_classify(text: str) -> tuple[str, str]:
    """Transparent lexicon heuristic. First (primary) un-negated match wins ties."""
    low = text.lower()
    best: tuple[float, str] = (0.0, "feature_request")
    for intent, words in LEXICON.items():
        score = 0.0
        for w in words:
            for m in re.finditer(re.escape(w), low):
                if not _negated(low, m.start()):
                    score += 1.0 + 0.5 * (1 - m.start() / max(1, len(low)))  # earlier = primary
        if score > best[0]:
            best = (score, intent)
    intent = best[1]
    prio = INTENTS[intent].base_priority
    if re.search(r"chargeback|lawyer|completely down", low):
        prio = "urgent"
    elif re.search(r"urgent|asap|today|third time|blocked", low) and "not urgent" not in low:
        prio = shift_priority(prio, +1)
    elif re.search(r"no rush|whenever you get a chance|not urgent", low):
        prio = shift_priority(prio, -1)
    return intent, prio


def keyword_handler(messages: list[Message], system: str | None,
                    json_schema: dict[str, Any] | None) -> str:
    content = messages[-1].content
    m = re.search(r"<<<\n(.*)\n>>>", content, re.DOTALL)
    intent, prio = keyword_classify(m.group(1) if m else content)
    return json.dumps({"intent": intent, "priority": prio, "confidence": 0.9})


# --------------------------------------------------------------------------- real LLM path
class LLMClassifier:
    """Prompted LLM classifier. Any :class:`LLMClient` works (real or offline)."""

    def __init__(self, client: LLMClient, max_attempts: int = 2) -> None:
        self.client = client
        self.name = f"llm-{client.model}"
        self.max_attempts = max_attempts

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        in_tok = out_tok = 0
        last_err: Exception | None = None
        for _ in range(self.max_attempts):
            resp = self.client.complete([Message(role="user", content=build_user_message(text))],
                                        system=SYSTEM_PROMPT, json_schema=OUTPUT_SCHEMA)
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            try:
                data = parse_llm_output(resp.text)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = exc
        else:
            raise ValueError(f"LLM returned unparseable output: {last_err}")
        intent = data["intent"]
        conf = float(data.get("confidence", 0.9))
        return Prediction(intent=intent, category=category_of(intent), priority=data["priority"],
                          confidence=min(1.0, max(0.0, conf)), model=self.name,
                          latency_ms=(time.perf_counter() - start) * 1000,
                          input_tokens=in_tok, output_tokens=out_tok)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]


# --------------------------------------------------------------------------- offline simulator
class SimulationAssumptions(BaseModel):
    """Every number here is an assumption, not a measurement. Edit to match your provider."""

    model: str = "simulated-large-llm"
    intent_accuracy: dict[str, float] = {
        "simple": 0.97, "long": 0.96, "negation": 0.94, "multi_intent": 0.90}
    priority_accuracy: float = 0.90
    latency_median_ms: float = 850.0
    latency_sigma: float = 0.35  # lognormal shape -> p95 ~ 1.8x median
    output_tokens: int = 24


class SimulatedLLMClassifier:
    """Oracle-with-noise stand-in for a large LLM, deterministic per ticket text."""

    def __init__(self, assumptions: SimulationAssumptions | None = None, seed: int = 7) -> None:
        self.a = assumptions or SimulationAssumptions()
        self.seed = seed
        self.name = self.a.model

    def _rng(self, text: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def predict_ticket(self, ticket: Ticket) -> Prediction:
        r = self._rng(ticket.text)
        intent = ticket.intent
        # Gold labels that were deliberately corrupted cannot be "known" by any model.
        acc = self.a.intent_accuracy.get(ticket.difficulty, 0.95)
        intent_correct = "intent" not in ticket.label_noise and r.random() < acc
        if not intent_correct:
            same_cat = [i for i in INTENT_NAMES
                        if category_of(i) == category_of(intent) and i != intent]
            intent = r.choice(same_cat)
        priority = ticket.priority
        if "priority" in ticket.label_noise or r.random() >= self.a.priority_accuracy:
            priority = shift_priority(priority, r.choice([-1, 1]))
            if priority == ticket.priority:  # clipped at an end of the scale
                priority = shift_priority(priority, -1 if priority == "urgent" else 1)
        latency = self.a.latency_median_ms * math.exp(self.a.latency_sigma * r.gauss(0, 1))
        return Prediction(intent=intent, category=category_of(intent), priority=priority,
                          confidence=0.95, model=self.name, latency_ms=latency,
                          input_tokens=prompt_tokens(ticket.text),
                          output_tokens=self.a.output_tokens)
