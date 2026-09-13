"""PPE compliance (SH17 config): body-part <-> PPE matching, question forms, guardrail rule R6."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.detector import Detection
from app.reasoning import policy
from app.reasoning.facts import build_facts
from app.reasoning.router import Route, RuleRouter
from app.reasoning.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[1]
IDS = {"person": 0, "head": 1, "helmet": 2, "safety-vest": 3, "hands": 4, "gloves": 5}


@pytest.fixture()
def vocab() -> Vocabulary:
    v = Vocabulary.load(ROOT / "configs" / "classes.yaml")
    v.reliability = {"head": 0.9, "helmet": 0.9, "hands": 0.7, "gloves": 0.45, "person": 0.95, "safety-vest": 0.9}
    return v


def img():
    return Image.fromarray(np.random.default_rng(0).integers(60, 200, (720, 1280, 3), dtype=np.uint8))


def d(name, conf, box):
    return Detection(IDS[name], name, conf, tuple(map(float, box)))


def worker(x, helmet=True, helmet_conf=0.9):
    """A head at x with (optionally) a helmet sitting on top of it, like SH17 geometry."""
    dets = [d("head", 0.9, (x, 200, x + 60, 270))]
    if helmet:
        dets.append(d("helmet", helmet_conf, (x - 5, 170, x + 65, 215)))  # mostly above the head box
    return dets


def facts(vocab, dets):
    return build_facts(dets, img(), vocab)


def test_helmet_above_head_counts_as_worn(vocab):
    c = facts(vocab, worker(100) + worker(400) + worker(700, helmet=False))["compliance"]["helmet"]
    assert (c["checked"], c["with"], c["without"], c["unclear"]) == (3, 2, 1, 0)


def test_one_helmet_cannot_cover_two_heads(vocab):
    dets = [d("head", 0.9, (100, 200, 160, 270)), d("head", 0.9, (150, 200, 210, 270)),
            d("helmet", 0.9, (120, 170, 190, 215))]
    c = facts(vocab, dets)["compliance"]["helmet"]
    assert (c["with"], c["without"]) == (1, 1)


def test_low_conf_helmet_makes_head_unclear(vocab):
    c = facts(vocab, worker(100, helmet_conf=0.35))["compliance"]["helmet"]
    assert (c["with"], c["without"], c["unclear"]) == (0, 0, 1)


@pytest.mark.parametrize("q,form", [
    ("Is anyone not wearing a helmet?", "any_without"),
    ("Is anyone without gloves?", "any_without"),
    ("Is everyone wearing a hard hat?", "all_with"),
    ("Are all the workers wearing safety vests?", "all_with"),
    ("How many people are not wearing helmets?", "count_without"),
    ("How many workers are wearing gloves?", "count_with"),
    ("Is anyone wearing a vest?", "any_with"),
])
def test_router_compliance_forms(vocab, q, form):
    r = RuleRouter(vocab).route(q)
    assert (r.intent, r.form) == ("compliance", form), r.reason


@pytest.mark.parametrize("q,intent", [
    ("How many helmets are there?", "count"),
    ("Is there a helmet on the floor?", "existence"),
    ("What colour are the helmets?", "unsupported"),
])
def test_plain_questions_are_not_compliance(vocab, q, intent):
    assert RuleRouter(vocab).route(q).intent == intent


def ask(vocab, dets, form, ppe="helmet"):
    return policy.decide(Route("compliance", [ppe], form=form), facts(vocab, dets), vocab)


def test_violation_found(vocab):
    r = ask(vocab, worker(100) + worker(400, helmet=False), "any_without")
    assert r.status == policy.ANSWERED
    assert r.answer.startswith("Yes. Of 2 heads detected, 1 has a helmet and 1 has no helmet.")


def test_everyone_compliant_with_reliable_head_detector(vocab):
    r = ask(vocab, worker(100) + worker(400), "all_with")
    assert r.status == policy.ANSWERED and r.answer.startswith("Yes.")


def test_all_compliant_but_dark_image_is_refused(vocab):
    dark = Image.new("RGB", (1280, 720), (5, 5, 5))
    r = policy.decide(Route("compliance", ["helmet"], form="any_without"),
                      build_facts(worker(100), dark, vocab), vocab)
    assert r.status == policy.INSUFFICIENT and "R3" in r.rules_fired


def test_violation_qualified_when_ppe_recall_is_low(vocab):
    # gloves recall 0.45 < 0.6: a hand without a detected glove may just be a missed glove (R6)
    dets = [d("hands", 0.9, (100, 400, 150, 450))]
    r = ask(vocab, dets, "any_without", ppe="gloves")
    assert r.status == policy.UNCERTAIN and "R6" in r.rules_fired and "45%" in " ".join(r.caveats)


def test_unclear_only_is_insufficient(vocab):
    r = ask(vocab, worker(100, helmet_conf=0.35), "any_without")
    assert r.status == policy.INSUFFICIENT


def test_no_people_detected(vocab):
    r = ask(vocab, [], "any_without")
    assert r.status == policy.ANSWERED and "nobody to check" in r.answer


def test_vest_caveat_mentions_cropped_torsos(vocab):
    dets = [d("person", 0.9, (100, 100, 300, 600))]
    r = ask(vocab, dets, "any_without", ppe="safety-vest")
    assert any("torso" in c for c in r.caveats)


def test_implausible_detection_count_is_refused(vocab):
    dets = [d("head", 0.9, (i % 40 * 30, i // 40 * 30, i % 40 * 30 + 25, i // 40 * 30 + 25)) for i in range(200)]
    r = ask(vocab, dets, "any_without")
    assert r.status == policy.INSUFFICIENT and r.rules_fired == ["R7"]


def test_location_list_is_capped(vocab):
    heads = [worker(x, helmet=False)[0] for x in (50, 300, 600, 900, 1150)]
    heads += [d("head", 0.9, (x, 600, x + 60, 670)) for x in (50, 600, 1150)]
    r = ask(vocab, heads, "any_without")
    assert "elsewhere" in r.answer and r.answer.count(" at the ") <= 4
