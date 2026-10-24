"""Cheap, explainable detector for requests a small model should not answer alone.

Signals (each contributes a weight; ``is_complex`` when the total >= ``threshold``):

* ``long``          - many words: more context, more room for distractors.
* ``multi_intent``  - discourse markers that introduce a second request ("also", "side note").
* ``negation``      - negated request verbs ("I'm not asking for a refund"), a classic
                      bag-of-words failure mode.
* ``multi_question``- two or more questions in one ticket.

The detector is deliberately rule-based: it is fast (<0.1 ms), auditable, and its reasons are
logged with every fallback so operators can see *why* traffic went to the expensive model.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_MULTI = re.compile(
    r"\b(also|another thing|side note|separately|on top of that|additionally|in addition|"
    r"as well as|plus i)\b", re.IGNORECASE)
_NEG = re.compile(
    r"(?:\b(?:not|never|no longer)\b|\b(?:don't|dont|isn't|is not|do not|i'm not|im not)\b)"
    r"(?:\s+\S+){0,3}?\s+(?:asking|trying|looking|want|wanting|need|cancel\w*|refund\w*|delete|"
    r"upgrade|downgrade|change|damaged|wrong|crashing|locked|about)\b",
    re.IGNORECASE)


class ComplexityConfig(BaseModel):
    max_words: int = 55
    threshold: float = 1.0
    weights: dict[str, float] = Field(default_factory=lambda: {
        "long": 1.0, "multi_intent": 1.0, "negation": 1.0, "multi_question": 0.5})


class ComplexityReport(BaseModel):
    score: float
    reasons: list[str]
    is_complex: bool


class ComplexityDetector:
    def __init__(self, config: ComplexityConfig | None = None) -> None:
        self.cfg = config or ComplexityConfig()

    def assess(self, text: str) -> ComplexityReport:
        reasons: list[str] = []
        if len(text.split()) > self.cfg.max_words:
            reasons.append("long")
        if _MULTI.search(text):
            reasons.append("multi_intent")
        if _NEG.search(text):
            reasons.append("negation")
        if text.count("?") >= 2:
            reasons.append("multi_question")
        score = sum(self.cfg.weights.get(r, 0.0) for r in reasons)
        return ComplexityReport(score=score, reasons=reasons,
                                is_complex=score >= self.cfg.threshold)
