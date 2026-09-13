"""LLM-backed steps, each with a deterministic safety net.

  polish()       rewrites a VERIFIED answer into natural language. Rejected if it introduces
                 any number that isn't in the verified answer. Refusals are never rewritten.
  reason_other() open-ended questions ("are the workers standing close together?"): the LLM
                 reasons over the structured facts and must say whether they are sufficient.
                 Any number it uses must be derivable from the facts.
  general()      questions unrelated to the image.
"""
from __future__ import annotations

import itertools
import json
import logging
import re

from .llm import LLMClient, LLMError
from .policy import ANSWERED, INSUFFICIENT, NO_DETECTION, UNCERTAIN, Decision
from .vocab import Vocabulary

log = logging.getLogger("narrate")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORD_NUM = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    """Digits AND spelled-out numbers, so the LLM can't slip in "seven" instead of "7"."""
    return set(NUM.findall(text)) | {WORDS[w.lower()] for w in WORD_NUM.findall(text)}


def allowed_numbers(facts: dict) -> set[str]:
    """Every number an answer may legitimately contain: counts, sums of counts over any subset
    of classes (e.g. helmets + bare heads = people), region counts, confidences as percentages."""
    conf, unc = facts["counts_confident"], facts["counts_uncertain"]
    nums = set(conf.values()) | set(unc.values()) | {conf[k] + unc[k] for k in conf}
    classes = [k for k in conf if conf[k] or unc[k]][:10]
    for r in range(2, len(classes) + 1):
        for combo in itertools.combinations(classes, r):
            nums.add(sum(conf[k] for k in combo))
            nums.add(sum(conf[k] + unc[k] for k in combo))
    for regions in facts["regions_confident"].values():
        nums |= set(regions.values())
    for c in (facts.get("compliance") or {}).values():
        nums |= {c["checked"], c["with"], c["without"], c["unclear"], c["uncertain_bases"],
                 c["without"] + c["unclear"], c["with"] + c["unclear"]}
    pct = {round(d["confidence"] * 100) for d in facts["detections"]}
    return {str(n) for n in nums | pct} | {"0", "1"}


POLISH_SYSTEM = """You rewrite answers from an object-detection system into one to three clear, natural sentences.
Rules: keep the meaning and the yes/no verdict exactly; use ONLY the numbers in the verified answer and caveats;
do not add objects, attributes, or speculation; keep every caveat's substance. Reply with the rewritten text only."""


def polish(llm: LLMClient, question: str, d: Decision) -> tuple[Decision, str]:
    if d.status not in (ANSWERED, UNCERTAIN):
        return d, "template (refusals are never rewritten)"
    verified = " ".join([d.answer, *d.caveats])
    try:
        text = llm.complete(POLISH_SYSTEM, json.dumps({"question": question, "verified_answer": d.answer,
                                                       "caveats": d.caveats}), max_tokens=250)
    except LLMError:
        return d, "template (LLM unavailable)"
    extra = numbers_in(text) - numbers_in(verified)
    if not text or extra:
        log.warning("polish rejected: new numbers %s", extra)
        return d, f"template (LLM output rejected: introduced numbers {sorted(extra)})"
    return Decision(d.status, text, [], d.rules_fired), "llm"


OTHER_SYSTEM = """You answer questions about an image using ONLY structured object-detection output.
You cannot see the image. The detector knows only these classes: {classes}.
"confident" detections are facts. "uncertain" ones must not be treated as facts.
Boxes are pixel x1,y1,x2,y2; image size and a quality check are included.
If the facts do not let you answer with confidence, say so. Do not guess.
Reply with ONLY JSON: {{"sufficient": true|false, "answer": "<one to three sentences>"}}"""


def reason_other(llm: LLMClient | None, question: str, facts: dict, vocab: Vocabulary) -> tuple[Decision, str]:
    if llm is None or not llm.available:
        return Decision(INSUFFICIENT, "I can't interpret that question reliably. I can count, check for, locate, "
                                      f"or compare these objects: {vocab.describe()}.", rules_fired=["R5"]), "template"
    compact = {k: v for k, v in facts.items() if k != "thresholds"}
    try:
        out = llm.complete_json(OTHER_SYSTEM.format(classes=vocab.describe()),
                                json.dumps({"question": question, "facts": compact}), max_tokens=350)
        answer, sufficient = str(out["answer"]).strip(), bool(out["sufficient"])
    except (LLMError, KeyError):
        return Decision(INSUFFICIENT, "I couldn't reason about that question reliably right now.",
                        rules_fired=["R5"]), "template (LLM failed)"
    extra = numbers_in(answer) - allowed_numbers(facts)
    if extra:
        return Decision(INSUFFICIENT, "I can't answer that reliably from the detections.",
                        [f"(LLM answer rejected: numbers {sorted(extra)} not supported by detections)"],
                        ["R1"]), "llm (rejected)"
    return Decision(ANSWERED if sufficient else INSUFFICIENT, answer), "llm"


GENERAL_SYSTEM = ("You are the assistant behind an image-analysis API for {domain}. The user asked something "
                  "unrelated to the image. Answer briefly (max 3 sentences). If it needs live or personal data "
                  "you don't have, say so.")


def general(llm: LLMClient | None, question: str, vocab: Vocabulary) -> tuple[Decision, str]:
    if llm is not None and llm.available:
        try:
            return Decision(NO_DETECTION, llm.complete(GENERAL_SYSTEM.format(domain=vocab.domain), question,
                                                       max_tokens=250)), "llm"
        except LLMError:
            pass
    return Decision(NO_DETECTION, f"That question isn't about the image, so I didn't run the detector. I answer "
                                  f"questions about {vocab.domain} images (objects I know: {vocab.describe()})."
                    ), "template"
