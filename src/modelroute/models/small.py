"""Locally-trainable "small model": TF-IDF features + calibrated linear classifiers.

This is the runnable baseline behind the same :class:`Classifier` interface as the LoRA
fine-tuned model in :mod:`modelroute.finetune`. Two heads share the feature extractor:

* intent head - calibrated linear SVM (or logistic regression); its max probability is the
  routing confidence.
* priority head - logistic regression (urgency is mostly lexical cues).

Category is derived deterministically from the intent via the taxonomy, so the three labels
can never be mutually inconsistent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from ..data.generator import Ticket
from ..data.taxonomy import category_of
from .base import Prediction

Head = Literal["svm", "logreg"]


def build_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True, max_features=60000)),
    ])


class SmallClassifier:
    """TF-IDF + calibrated linear model. ``fit`` -> ``predict`` -> ``save``/``load``."""

    def __init__(self, head: Head = "svm", calibration: Literal["sigmoid", "isotonic"] = "sigmoid",
                 C: float = 1.0, seed: int = 13) -> None:
        self.head = head
        self.calibration = calibration
        self.C = C
        self.seed = seed
        self.name = f"small-tfidf-{head}"
        self.features = build_features()
        self.intent_clf: CalibratedClassifierCV | LogisticRegression | None = None
        self.priority_clf: LogisticRegression | None = None

    # ----------------------------------------------------------------- training
    def fit(self, tickets: list[Ticket]) -> SmallClassifier:
        texts = [t.text for t in tickets]
        X = self.features.fit_transform(texts)
        y_intent = [t.intent for t in tickets]
        if self.head == "svm":
            base = LinearSVC(C=self.C, random_state=self.seed)
            self.intent_clf = CalibratedClassifierCV(base, method=self.calibration, cv=5)
        else:
            self.intent_clf = LogisticRegression(C=self.C * 10, max_iter=2000)
        self.intent_clf.fit(X, y_intent)
        self.priority_clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
        self.priority_clf.fit(X, [t.priority for t in tickets])
        return self

    # ----------------------------------------------------------------- inference
    def _check(self) -> None:
        if self.intent_clf is None or self.priority_clf is None:
            raise RuntimeError("SmallClassifier is not fitted; call fit() or load()")

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        self._check()
        X = self.features.transform(texts)
        assert self.intent_clf is not None
        return self.intent_clf.predict_proba(X), list(self.intent_clf.classes_)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        self._check()
        assert self.intent_clf is not None and self.priority_clf is not None
        start = time.perf_counter()
        X = self.features.transform(texts)
        p_int = self.intent_clf.predict_proba(X)
        prio = self.priority_clf.predict(X)
        per_item_ms = (time.perf_counter() - start) * 1000 / max(1, len(texts))
        classes = list(self.intent_clf.classes_)
        out: list[Prediction] = []
        for row, pr in zip(p_int, prio, strict=True):
            order = np.argsort(row)[::-1][:3]
            intent = classes[order[0]]
            out.append(Prediction(
                intent=intent, category=category_of(intent), priority=str(pr),
                confidence=float(row[order[0]]), model=self.name, latency_ms=per_item_ms,
                top_intents=[(classes[i], round(float(row[i]), 4)) for i in order],
            ))
        return out

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        pred = self.predict_batch([text])[0]
        pred.latency_ms = (time.perf_counter() - start) * 1000
        return pred

    # ----------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        self._check()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"head": self.head, "calibration": self.calibration, "C": self.C,
                     "seed": self.seed, "features": self.features,
                     "intent_clf": self.intent_clf, "priority_clf": self.priority_clf}, path)

    @classmethod
    def load(cls, path: Path) -> SmallClassifier:
        blob = joblib.load(path)
        obj = cls(head=blob["head"], calibration=blob["calibration"], C=blob["C"],
                  seed=blob["seed"])
        obj.features = blob["features"]
        obj.intent_clf = blob["intent_clf"]
        obj.priority_clf = blob["priority_clf"]
        return obj
