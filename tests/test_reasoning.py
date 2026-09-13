"""Tests for Part B. No model weights or API key needed: the detector and LLM are faked.

Run: pytest -q
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.detector import Detection
from app.reasoning import narrate, policy
from app.reasoning.facts import build_facts
from app.reasoning.router import Route, Router, RuleRouter
from app.reasoning.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def vocab() -> Vocabulary:
    v = Vocabulary.load(ROOT / "tests" / "fixtures" / "classes_example.yaml")
    v.reliability = {"helmet": 0.9, "head": 0.85, "vest": 0.5}  # vest recall below min_class_recall (0.6)
    return v


def textured(w=640, h=480) -> Image.Image:
    """A sharp, mid-brightness image, so no quality flag fires."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(60, 200, (h, w, 3), dtype=np.uint8))


def det(name: str, conf: float, x=100.0, y=100.0) -> Detection:
    ids = {"helmet": 0, "head": 1, "vest": 2}
    return Detection(ids[name], name, conf, (x, y, x + 40, y + 40))


def facts_for(vocab, dets, image=None):
    return build_facts(dets, image or textured(), vocab)


# ---- vocabulary / router -------------------------------------------------------------------------
def test_longest_phrase_wins(vocab):
    assert vocab.match_classes("Is anyone not wearing a helmet?") == ["head"]
    assert vocab.match_classes("Are there more helmets than vests?") == ["helmet", "vest"]
    assert vocab.match_classes("more vests than hard hats") == ["vest", "helmet"]


@pytest.mark.parametrize("q,intent,targets", [
    ("How many helmets are in this image?", "count", ["helmet"]),
    ("Is anyone not wearing a helmet?", "existence", ["head"]),
    ("What's the most common object here?", "most_common", []),
    ("Are there more helmets than vests?", "compare", ["helmet", "vest"]),
    ("Where are the vests?", "location", ["vest"]),
    ("What's in this image?", "list", []),
    ("What can you detect?", "meta", []),
    ("What's the capital of France?", "general", []),
    ("What is the weather in Paris?", "general", []),
    ("What colour is the helmet?", "unsupported", ["helmet"]),
    ("How many people are in this image?", "unsupported", []),     # no 'person' class in this domain
    ("Is there a dog in this photo?", "unsupported", []),
    ("Is everyone wearing a helmet?", "unsupported", ["helmet"]),  # rules refuse universal questions
])
def test_rule_router(vocab, q, intent, targets):
    r = RuleRouter(vocab).route(q)
    assert (r.intent, r.target_classes) == (intent, targets), r.reason


@pytest.mark.parametrize("q,intent", [
    ("Is anyone wearing gloves?", "unsupported"),       # about the scene, object not in vocabulary
    ("What colour is the bus?", "unsupported"),
    ("Is Paris in France?", "general"),
    ("Do you like pizza?", "general"),
])
def test_rule_router_unknown_objects_vs_small_talk(vocab, q, intent):
    assert RuleRouter(vocab).route(q).intent == intent


def test_llm_route_is_validated(vocab):
    router = Router(vocab, llm=None)
    r = router.validate({"intent": "count", "target_classes": ["helmet", "gloves"], "reason": "x"})
    assert r.target_classes == ["helmet"] and "gloves" in r.unsupported_targets
    assert router.validate({"intent": "count", "target_classes": ["gloves"]}).intent == "unsupported"
    assert router.validate({"intent": "compare", "target_classes": ["helmet"]}).intent == "unsupported"
    assert router.validate({"intent": "count", "target_classes": ["helmet"], "negate": True}).negate is False
    with pytest.raises(ValueError):
        router.validate({"intent": "launch_missiles"})


# ---- guardrail -----------------------------------------------------------------------------------
def test_count_exact(vocab):
    d = policy.decide(Route("count", ["helmet"]), facts_for(vocab, [det("helmet", 0.9)] * 3), vocab)
    assert d.status == policy.ANSWERED and "3 helmets" in d.answer


def test_count_range_when_uncertain_boxes_exist(vocab):
    f = facts_for(vocab, [det("helmet", 0.9)] * 3 + [det("helmet", 0.3)] * 2)
    d = policy.decide(Route("count", ["helmet"]), f, vocab)
    assert d.status == policy.UNCERTAIN and "at least 3" in d.answer and "up to 5" in d.answer


def test_count_all_uncertain_is_insufficient(vocab):
    d = policy.decide(Route("count", ["helmet"]), facts_for(vocab, [det("helmet", 0.3)] * 2), vocab)
    assert d.status == policy.INSUFFICIENT


def test_negative_refused_on_dark_image(vocab):
    dark = Image.new("RGB", (640, 480), (5, 5, 5))
    d = policy.decide(Route("existence", ["head"]), facts_for(vocab, [], dark), vocab)
    assert d.status == policy.INSUFFICIENT and "R3" in d.rules_fired


def test_negative_refused_for_low_recall_class(vocab):
    d = policy.decide(Route("existence", ["vest"]), facts_for(vocab, []), vocab)
    assert d.status == policy.INSUFFICIENT and "50%" in d.answer


def test_negative_allowed_for_reliable_class(vocab):
    d = policy.decide(Route("existence", ["head"]), facts_for(vocab, [det("helmet", 0.9)]), vocab)
    assert d.status == policy.ANSWERED and d.answer.startswith("No,") and "85%" in " ".join(d.caveats)


def test_universal_negated_existence(vocab):
    # "Is everyone wearing a helmet?" routed (by the LLM) as existence(head, negate=True)
    d = policy.decide(Route("existence", ["head"], negate=True), facts_for(vocab, [det("head", 0.8)]), vocab)
    assert d.answer.startswith("No,")


def test_most_common_refuses_when_ranking_could_flip(vocab):
    f = facts_for(vocab, [det("helmet", 0.9)] * 3 + [det("vest", 0.9)] * 2 + [det("vest", 0.3)] * 2)
    d = policy.decide(Route("most_common"), f, vocab)
    assert d.status == policy.INSUFFICIENT and "R4" in d.rules_fired


def test_compare_overlapping_ranges_is_insufficient(vocab):
    f = facts_for(vocab, [det("helmet", 0.9)] * 2 + [det("vest", 0.9)] * 2 + [det("vest", 0.3)])
    d = policy.decide(Route("compare", ["helmet", "vest"]), f, vocab)
    assert d.status == policy.INSUFFICIENT


def test_unsupported_target_adds_caveat(vocab):
    d = policy.decide(Route("count", ["helmet"], ["gloves"]), facts_for(vocab, [det("helmet", 0.9)]), vocab)
    assert d.status == policy.UNCERTAIN and any("gloves" in c for c in d.caveats)


# ---- LLM safety net ------------------------------------------------------------------------------
class FakeLLM:
    available = True

    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, *a, **k):
        return self.reply

    def complete_json(self, *a, **k):
        import json
        return json.loads(self.reply)


def test_polish_rejects_new_numbers():
    d = policy.Decision(policy.ANSWERED, "I count 3 helmets.")
    out, src = narrate.polish(FakeLLM("There are seven helmets."), "how many?", d)
    assert out.answer == "I count 3 helmets." and "rejected" in src
    out, src = narrate.polish(FakeLLM("There are 3 helmets in the picture."), "how many?", d)
    assert src == "llm"


def test_refusals_are_never_polished():
    d = policy.Decision(policy.INSUFFICIENT, "I can't tell.")
    out, src = narrate.polish(FakeLLM("Yes, definitely 4."), "q", d)
    assert out.answer == "I can't tell."


def test_reason_other_allows_derived_sums(vocab):
    f = facts_for(vocab, [det("helmet", 0.9)] * 3 + [det("head", 0.9)] * 2)
    out, _ = narrate.reason_other(FakeLLM('{"sufficient": true, "answer": "About 5 people: 3 with helmets."}'),
                                  "how many workers?", f, vocab)
    assert out.status == policy.ANSWERED
    out, _ = narrate.reason_other(FakeLLM('{"sufficient": true, "answer": "There are 9 people."}'), "q", f, vocab)
    assert out.status == policy.INSUFFICIENT


# ---- API -----------------------------------------------------------------------------------------
class FakeDetector:
    names = {0: "helmet", 1: "head", 2: "vest"}
    weights = Path("fake.pt")

    def predict(self, image, conf):
        dets = [det("helmet", 0.92, 50, 50), det("helmet", 0.81, 400, 60), det("head", 0.35, 250, 70)]
        return [d for d in dets if d.confidence >= conf]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from fastapi.testclient import TestClient
    from app import main

    main.state.detector = None
    main.build_state(detector=FakeDetector(), classes_config=ROOT / "tests" / "fixtures" / "classes_example.yaml")
    with TestClient(main.app) as c:
        yield c


def png_bytes() -> bytes:
    buf = io.BytesIO()
    textured().save(buf, format="PNG")
    return buf.getvalue()


def test_detect_endpoint(client):
    r = client.post("/detect", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["conf_threshold"] == 0.5
    assert {"x1", "y1", "x2", "y2"} <= set(body["detections"][0]["box"])


def test_ask_endpoint_with_uncertain_head(client):
    r = client.post("/ask", data={"question": "Is anyone not wearing a helmet?"},
                    files={"file": ("x.png", png_bytes(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["used_detector"] is True
    assert body["status"] == "insufficient_information"   # one bare head at 0.35 conf: could be, can't say


def test_ask_general_skips_detector(client):
    r = client.post("/ask", data={"question": "What's the capital of France?"})
    assert r.json()["used_detector"] is False


def test_bad_upload(client):
    r = client.post("/detect", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    r = client.post("/detect", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
