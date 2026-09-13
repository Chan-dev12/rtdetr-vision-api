"""The whole Part B decision layer in one readable function: route -> detect -> facts -> guardrail -> phrase."""
from __future__ import annotations

import logging
import time

from PIL import Image

from . import narrate, policy
from .facts import build_facts
from .llm import LLMClient
from .policy import INSUFFICIENT, Decision
from .router import Router
from .vocab import Vocabulary

log = logging.getLogger("reasoner")


class Reasoner:
    def __init__(self, vocab: Vocabulary, detector, llm: LLMClient | None,
                 router_mode: str = "auto", answer_mode: str = "auto"):
        self.vocab, self.detector, self.llm = vocab, detector, llm
        self.router = Router(vocab, llm, router_mode)
        self.answer_mode = answer_mode

    def ask(self, question: str, image: Image.Image | None) -> dict:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        route = self.router.route(question)
        timings["route_ms"] = _ms(t)

        facts, used_detector, answer_source = None, False, "template"
        if route.intent == "meta":
            decision = policy.meta(self.vocab)
        elif route.intent == "general":
            decision, answer_source = narrate.general(self.llm, question, self.vocab)
        elif route.intent == "unsupported":
            decision = policy.unsupported(route, self.vocab)
        elif image is None:
            decision = Decision(INSUFFICIENT, "This question is about image content, but no image was provided.")
        else:
            t = time.perf_counter()
            # Run at the LOW threshold: uncertain boxes are needed for the guardrail even though they're never counted.
            dets = self.detector.predict(image, conf=self.vocab.thresholds.uncertain_conf)
            timings["detect_ms"] = _ms(t)
            used_detector = True
            facts = build_facts(dets, image, self.vocab)
            if route.intent == "other":
                decision, answer_source = narrate.reason_other(self.llm, question, facts, self.vocab)
            else:
                decision = policy.decide(route, facts, self.vocab)
                if self._polish_enabled():
                    t = time.perf_counter()
                    decision, answer_source = narrate.polish(self.llm, question, decision)
                    timings["polish_ms"] = _ms(t)

        log.info("ask intent=%s source=%s detector=%s status=%s rules=%s",
                 route.intent, route.source, used_detector, decision.status, decision.rules_fired)
        return {
            "question": question,
            "answer": " ".join([decision.answer, *decision.caveats]).strip(),
            "status": decision.status,
            "route": route.to_dict(),
            "used_detector": used_detector,
            "answer_source": answer_source,
            "guardrail_rules_fired": decision.rules_fired,
            "facts": None if facts is None else {k: facts[k] for k in
                                                 ("counts_confident", "counts_uncertain", "regions_confident",
                                                  "compliance", "image", "thresholds")},
            "detections": None if facts is None else facts["detections"],
            "timings_ms": timings,
        }

    def _polish_enabled(self) -> bool:
        if self.answer_mode == "template" or self.llm is None or not self.llm.available:
            return False
        return True


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)
