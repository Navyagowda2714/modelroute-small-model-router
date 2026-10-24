"""Cleaning, de-duplication, stratified splitting and JSONL I/O."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from .generator import Ticket

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Unicode-normalise, collapse whitespace, strip. Case is preserved (models lower-case)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", "")
    return _WS.sub(" ", text).strip()


class CleaningReport(BaseModel):
    input_rows: int
    dropped_empty: int
    dropped_duplicates: int
    output_rows: int


def clean_tickets(tickets: Iterable[Ticket],
                  min_chars: int = 3) -> tuple[list[Ticket], CleaningReport]:
    seen: set[str] = set()
    out: list[Ticket] = []
    n_in = n_empty = n_dup = 0
    for t in tickets:
        n_in += 1
        text = clean_text(t.text)
        if len(text) < min_chars:
            n_empty += 1
            continue
        key = text.lower()
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out.append(t.model_copy(update={"text": text}))
    return out, CleaningReport(input_rows=n_in, dropped_empty=n_empty,
                               dropped_duplicates=n_dup, output_rows=len(out))


def stratified_split(tickets: list[Ticket], val_frac: float = 0.15, test_frac: float = 0.15,
                     seed: int = 13) -> dict[str, list[Ticket]]:
    """Intent-stratified split with a template-group holdout.

    Tickets written with *held-out* phrasings (``phrasing == "holdout"``) are only placed in
    val/test, so evaluation measures generalisation to wordings the model never saw - the
    situation a production classifier is actually in. Seen-phrasing tickets are split so that
    the overall split sizes still match ``val_frac`` / ``test_frac``.
    """
    seen = [t for t in tickets if t.phrasing == "seen"]
    held = [t for t in tickets if t.phrasing == "holdout"]
    n_eval = round(len(tickets) * (val_frac + test_frac))
    n_eval_seen = max(0, n_eval - len(held))
    train, eval_seen = train_test_split(seen, test_size=n_eval_seen,
                                        stratify=[t.intent for t in seen], random_state=seed)
    pool = eval_seen + held
    val, test = train_test_split(pool, test_size=test_frac / (val_frac + test_frac),
                                 stratify=[t.intent for t in pool], random_state=seed)
    return {"train": train, "val": val, "test": test}


def write_jsonl(path: Path, tickets: Iterable[Ticket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in tickets:
            fh.write(t.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Ticket]:
    with path.open(encoding="utf-8") as fh:
        return [Ticket.model_validate(json.loads(line)) for line in fh if line.strip()]


def dataset_stats(splits: dict[str, list[Ticket]]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for name, rows in splits.items():
        lengths = sorted(len(t.text.split()) for t in rows)
        stats[name] = {
            "rows": len(rows),
            "intents": dict(sorted(Counter(t.intent for t in rows).items())),
            "categories": dict(sorted(Counter(t.category for t in rows).items())),
            "priorities": dict(sorted(Counter(t.priority for t in rows).items())),
            "difficulty": dict(sorted(Counter(t.difficulty for t in rows).items())),
            "phrasing": dict(sorted(Counter(t.phrasing for t in rows).items())),
            "intent_noise_rows": sum("intent" in t.label_noise for t in rows),
            "priority_noise_rows": sum("priority" in t.label_noise for t in rows),
            "median_words": lengths[len(lengths) // 2] if lengths else 0,
            "max_words": lengths[-1] if lengths else 0,
        }
    return stats


def build_dataset(out_dir: Path, n: int = 6000, seed: int = 13) -> dict[str, object]:
    """Generate -> clean -> split -> write ``train/val/test.jsonl`` + ``stats.json``."""
    from .generator import generate_tickets

    raw = generate_tickets(n=n, seed=seed)
    cleaned, report = clean_tickets(raw)
    splits = stratified_split(cleaned, seed=seed)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    stats = {"cleaning": report.model_dump(), "splits": dataset_stats(splits),
             "seed": seed, "requested_rows": n}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
