"""Synthetic-but-realistic labelled support-ticket generator.

Design goals
------------
* **Learnable but not trivial** - paraphrases, typos, filler context, multi-intent and
  negation tickets create a long tail that a bag-of-words model gets wrong.
* **Labels follow explicit rules** - priority = intent base priority adjusted by urgency /
  calm cues, so a model can learn it from text; a small amount of annotator noise is
  injected deliberately (documented in ``docs/DATA_CARD.md``).
* **Deterministic** - everything flows from one ``random.Random(seed)``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

from .taxonomy import INTENT_NAMES, INTENTS, PRIORITIES, shift_priority
from .templates import (
    CALM_CUES,
    CLOSERS,
    FILLERS,
    NEGATIONS,
    OPENERS,
    PARAPHRASES,
    SEVERE_CUES,
    SHORT,
    SLOTS,
    TEMPLATES,
    URGENT_CUES,
)

Difficulty = Literal["simple", "multi_intent", "negation", "long"]


class Ticket(BaseModel):
    id: str
    text: str
    intent: str
    category: str
    priority: str
    difficulty: Difficulty = "simple"
    template: str = Field(default="", description="<intent>/<index> of the body template")
    phrasing: Literal["seen", "holdout"] = Field(
        default="seen", description="holdout templates never appear in the training split")
    label_noise: list[Literal["intent", "priority"]] = Field(
        default_factory=list, description="Labels deliberately perturbed (annotator noise)")


class GeneratorConfig(BaseModel):
    n: int = 6000
    seed: int = 13
    p_multi_intent: float = 0.10
    p_negation: float = 0.07
    p_long: float = 0.06
    p_typo: float = 0.15
    p_lowercase: float = 0.12
    p_strip_punct: float = 0.08
    p_paraphrase_word: float = 0.5
    p_urgent: float = 0.20
    p_severe: float = 0.04
    p_calm: float = 0.14
    p_priority_noise: float = 0.03
    p_intent_noise: float = 0.01
    p_duplicate: float = 0.02  # exact re-submissions, removed again by cleaning
    holdout_templates_per_intent: int = 2
    p_holdout: float = 0.08  # share of tickets written with a held-out phrasing
    p_full_secondary: float = 0.5  # multi-intent: secondary request is a full sentence


class TicketGenerator:
    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.cfg = config or GeneratorConfig()
        self.rng = random.Random(self.cfg.seed)
        picker = random.Random(self.cfg.seed + 1)
        self.holdout: dict[str, set[int]] = {
            intent: set(picker.sample(range(len(TEMPLATES[intent])),
                                      k=self.cfg.holdout_templates_per_intent))
            for intent in INTENT_NAMES
        }

    def _template(self, intent: str, holdout: bool) -> tuple[str, int]:
        pool = [i for i in range(len(TEMPLATES[intent])) if (i in self.holdout[intent]) == holdout]
        idx = self.rng.choice(pool)
        return TEMPLATES[intent][idx], idx

    # ------------------------------------------------------------------ slots / noise
    def _fill(self, template: str) -> str:
        r = self.rng

        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key == "amount":
                return f"${r.choice([9.99, 14.99, 19.0, 29.99, 49.0, 99.0, 129.5, 240.0]):.2f}"
            if key == "order":
                return f"#{r.choice('ABCDEFGH')}{r.randint(10000, 99999)}"
            if key == "days":
                return str(r.randint(2, 15))
            return r.choice(SLOTS[key])

        return re.sub(r"\{(\w+)\}", repl, template)

    def _paraphrase(self, text: str) -> str:
        for word, options in PARAPHRASES.items():
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            if pattern.search(text) and self.rng.random() < self.cfg.p_paraphrase_word:
                text = pattern.sub(self.rng.choice(options), text, count=1)
        return text

    def _typo(self, text: str) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        for i in self.rng.sample(candidates, k=min(len(candidates), self.rng.randint(1, 2))):
            w = words[i]
            j = self.rng.randint(1, len(w) - 2)
            op = self.rng.random()
            if op < 0.4:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]  # swap
            elif op < 0.7:
                w = w[:j] + w[j + 1 :]  # drop
            else:
                w = w[:j] + w[j] + w[j:]  # double
            words[i] = w
        return " ".join(words)

    def _surface_noise(self, text: str) -> str:
        c, r = self.cfg, self.rng
        if r.random() < c.p_typo:
            text = self._typo(text)
        if r.random() < c.p_strip_punct:
            text = re.sub(r"[.,!?;:]", "", text)
        if r.random() < c.p_lowercase:
            text = text.lower()
        return text

    # ------------------------------------------------------------------ body builders
    def _body(self, intent: str, holdout: bool) -> tuple[str, Difficulty, str, str]:
        """Return (body, difficulty, intent, template id); negation may override the intent."""
        r, c = self.rng, self.cfg
        roll = r.random()
        if roll < c.p_negation:
            decoy, intent = r.choice(NEGATIONS)
            tpl, idx = self._template(intent, holdout)
            body = self._fill(tpl)
            joiner = r.choice([", ", ". ", " - "])
            text = f"{decoy}{joiner}{body[0].lower() + body[1:]}"
            return text, "negation", intent, f"{intent}/{idx}"
        tpl, idx = self._template(intent, holdout)
        body, tid = self._fill(tpl), f"{intent}/{idx}"
        if roll < c.p_negation + c.p_multi_intent:
            other = r.choice([i for i in INTENT_NAMES if i != intent])
            connector = r.choice(["Also,", "Another thing:", "Side note -", "Separately,",
                                  "On top of that,"])
            if r.random() < c.p_full_secondary:
                sec, _ = self._template(other, holdout=False)
                sec = self._fill(sec)
                secondary = sec[0].lower() + sec[1:]
            else:
                secondary = r.choice(SHORT[other]) + "."
            return f"{body} {connector} {secondary}", "multi_intent", intent, tid
        if roll < c.p_negation + c.p_multi_intent + c.p_long:
            fillers = [self._fill(f) for f in r.sample(FILLERS, k=r.randint(2, 4))]
            pos = r.randint(0, len(fillers))
            parts = [*fillers[:pos], body, *fillers[pos:]]
            return " ".join(parts), "long", intent, tid
        return body, "simple", intent, tid

    def _priority(self, intent: str) -> tuple[str, str]:
        """Return (cue sentence, priority) following the documented rules."""
        r, c = self.rng, self.cfg
        base = INTENTS[intent].base_priority
        roll = r.random()
        if roll < c.p_severe:
            return r.choice(SEVERE_CUES), "urgent"
        if roll < c.p_severe + c.p_urgent:
            return r.choice(URGENT_CUES), shift_priority(base, +1)
        if roll < c.p_severe + c.p_urgent + c.p_calm:
            return r.choice(CALM_CUES), shift_priority(base, -1)
        return "", base

    # ------------------------------------------------------------------ public API
    def generate_one(self, idx: int) -> Ticket:
        r, c = self.rng, self.cfg
        intent = r.choice(INTENT_NAMES)
        holdout = r.random() < c.p_holdout
        body, difficulty, intent, template = self._body(intent, holdout)
        cue, priority = self._priority(intent)
        body = self._paraphrase(body)
        opener = self._fill(r.choice(OPENERS))
        closer = self._fill(r.choice(CLOSERS))
        text = " ".join(p for p in [opener, body, cue, closer] if p)
        text = self._surface_noise(text)

        noisy: list[Literal["intent", "priority"]] = []
        if r.random() < c.p_priority_noise:
            priority = shift_priority(priority, r.choice([-1, 1]))
            noisy.append("priority")
        if r.random() < c.p_intent_noise:
            same_cat = [i for i in INTENT_NAMES
                        if INTENTS[i].category == INTENTS[intent].category and i != intent]
            intent = r.choice(same_cat)
            noisy.append("intent")
        return Ticket(id=f"T{idx:06d}", text=text, intent=intent,
                      category=INTENTS[intent].category, priority=priority,
                      difficulty=difficulty, label_noise=noisy, template=template,
                      phrasing="holdout" if holdout else "seen")

    def generate(self) -> Iterator[Ticket]:
        previous: Ticket | None = None
        for i in range(self.cfg.n):
            if previous is not None and self.rng.random() < self.cfg.p_duplicate:
                yield previous.model_copy(update={"id": f"T{i:06d}"})
                continue
            previous = self.generate_one(i)
            yield previous


def generate_tickets(n: int = 6000, seed: int = 13) -> list[Ticket]:
    return list(TicketGenerator(GeneratorConfig(n=n, seed=seed)).generate())


assert set(TEMPLATES) == set(INTENT_NAMES) == set(SHORT), "templates out of sync with taxonomy"
assert PRIORITIES == ["low", "medium", "high", "urgent"]
