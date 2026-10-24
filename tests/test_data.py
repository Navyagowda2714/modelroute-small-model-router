from __future__ import annotations

from collections import Counter

from modelroute.data import (
    INTENT_NAMES,
    INTENTS,
    GeneratorConfig,
    Ticket,
    TicketGenerator,
    clean_text,
    clean_tickets,
    generate_tickets,
)
from modelroute.data.templates import SEVERE_CUES


def test_generator_is_deterministic() -> None:
    a = [t.model_dump() for t in generate_tickets(n=200, seed=5)]
    b = [t.model_dump() for t in generate_tickets(n=200, seed=5)]
    c = [t.model_dump() for t in generate_tickets(n=200, seed=6)]
    assert a == b
    assert a != c


def test_labels_are_consistent_with_taxonomy() -> None:
    tickets = generate_tickets(n=1500, seed=1)
    assert {t.intent for t in tickets} == set(INTENT_NAMES)
    for t in tickets:
        assert t.category == INTENTS[t.intent].category
        assert t.priority in {"low", "medium", "high", "urgent"}
    difficulties = Counter(t.difficulty for t in tickets)
    assert set(difficulties) == {"simple", "multi_intent", "negation", "long"}
    assert difficulties["simple"] > len(tickets) * 0.6


def test_severe_cue_forces_urgent_priority() -> None:
    gen = TicketGenerator(GeneratorConfig(n=3000, seed=2, p_priority_noise=0.0))
    severe = [t for t in gen.generate() if any(c.lower() in t.text.lower() for c in SEVERE_CUES)]
    assert severe, "expected some tickets with a severe cue"
    assert all(t.priority == "urgent" for t in severe)


def test_clean_text_normalises_whitespace_and_unicode() -> None:
    assert clean_text("  Hi​  there\n\tteam！ ") == "Hi there team!"


def test_cleaning_drops_duplicates_and_empty() -> None:
    rows = [Ticket(id="1", text="Refund please", intent="refund_request", category="billing",
                   priority="medium"),
            Ticket(id="2", text="  refund   PLEASE ", intent="refund_request",
                   category="billing", priority="medium"),
            Ticket(id="3", text=" ", intent="refund_request", category="billing",
                   priority="medium")]
    out, report = clean_tickets(rows)
    assert [t.id for t in out] == ["1"]
    assert report.dropped_duplicates == 1
    assert report.dropped_empty == 1


def test_split_is_stratified_and_holdout_phrasings_never_train(splits) -> None:
    train, val, test = splits["train"], splits["val"], splits["test"]
    ids = [t.id for t in train + val + test]
    assert len(ids) == len(set(ids)), "a ticket leaked into two splits"
    assert all(t.phrasing == "seen" for t in train)
    assert any(t.phrasing == "holdout" for t in test)
    for split in (train, val, test):
        assert {t.intent for t in split} == set(INTENT_NAMES)
    train_templates = {t.template for t in train}
    holdout_templates = {t.template for t in test if t.phrasing == "holdout"}
    assert not train_templates & holdout_templates
