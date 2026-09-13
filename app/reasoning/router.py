"""Intent routing: does this question need the detector, and what exactly is it asking?

Two routers produce the same `Route` object:
  * RuleRouter: deterministic regex and synonym matching. Always available and fully testable.
  * LLM router: handles paraphrases ("any bare heads on site?", "is everyone wearing a hard hat?").
    Its output is VALIDATED: unknown classes are stripped, the intent must be from a fixed set,
    and `needs_detection` is derived from the intent instead of trusted from the LLM.
In `auto` mode the LLM is tried first and the rules take over if it fails or is not configured.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from .llm import LLMClient, LLMError
from .vocab import Vocabulary, normalize

log = logging.getLogger("router")

DETECTION_INTENTS = {"count", "existence", "most_common", "list", "location", "compare", "compliance", "other"}
ALL_INTENTS = DETECTION_INTENTS | {"meta", "general", "unsupported"}
NEEDS_TARGET = {"count", "existence", "location", "compare", "compliance"}
# compliance question forms: which set the question asks about, and how to phrase the verdict
COMPLIANCE_FORMS = {"any_without", "any_with", "all_with", "count_without", "count_with"}


@dataclass
class Route:
    intent: str
    target_classes: list[str] = field(default_factory=list)
    unsupported_targets: list[str] = field(default_factory=list)
    negate: bool = False          # "is everyone wearing X?" == NOT exists(class meaning "without X")
    form: str = ""                # compliance only: one of COMPLIANCE_FORMS
    reason: str = ""
    source: str = "rules"

    @property
    def needs_detection(self) -> bool:
        return self.intent in DETECTION_INTENTS

    def to_dict(self) -> dict:
        return asdict(self) | {"needs_detection": self.needs_detection}


# --- deterministic router -------------------------------------------------------------------------
META = re.compile(r"\b(what|which) (classes|objects|things|categories)\b.*\b(can|do) you (detect|see|recogni[sz]e)"
                  r"|\bwhat can you (detect|see|recogni[sz]e)|\bwhat do you detect\b|\byour (classes|labels)\b")
# Attributes a detector's boxes can never provide. Visual ones only make sense about a picture,
# so they're refused outright; the rest only when the question also refers to the image.
VISUAL_ATTR = re.compile(r"\b(colou?rs?|colou?red|logo|brand|written|text on|facial|expression|emotion"
                         r"|wearing what|what (is|are) (he|she|they) (doing|wearing))\b")
CONTEXT_ATTR = re.compile(r"\b(who (is|are) (this|that|they|he|she|these|those)|whose|name|read|text|how old|age"
                          r"|gender|male|female|happy|sad|angry|doing|why|weather|time of day|model of)\b")
NEG_WEAR = re.compile(r"\b(not|without|no|missing|lacking|lacks?|isn't|aren't|doesn't|don't|haven't|hasn't|unprotected)\b")
WEAR = re.compile(r"\b(wear|wears|wearing|wore|have|has|having|with|use|uses|using)\b")
COUNT_Q = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b")
UNIVERSAL = re.compile(r"\b(everyone|everybody|every \w+|all (the )?\w+|each \w+|nobody|no one|none of)\b")
# Words that signal the question is about what's IN the picture (people, "the X", "this"), so an
# unknown object should be refused as unsupported rather than treated as small talk.
IMAGE_REF = re.compile(r"\b(image|picture|photo|pic|frame|scene|here|see|visible|shown|this|these|those"
                       r"|any(one|body)|some(one|body)|every(one|body)|people|persons?|workers?|they|them)\b")
INTENT_PATTERNS = [
    ("compare", re.compile(r"\b(more|fewer|less)\b.*\bthan\b|\bas many\b|\bcompare")),
    ("count", re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\bhow much\b")),
    ("most_common", re.compile(r"\bmost (common|frequent|numerous)\b|\bmajority\b|\bmost of (them|the)\b")),
    ("location", re.compile(r"\bwhere\b|\bwhich side\b|\bon the (left|right)\b|\bat the (top|bottom)\b")),
    ("list", re.compile(r"\bwhat (objects|things|items)\b|\bwhat do you see\b|\b(list|describe)\b"
                        r"|\bwhat('s| is| are) (in|on) (this|the)\b|\bdetect(ed)? anything\b")),
    ("existence", re.compile(r"^(is|are|does|do|did|can|could|has|have)\b|\bany(one|body|thing)?\b|\bthere (is|are)\b|\bpresent\b")),
]


class RuleRouter:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def route(self, question: str) -> Route:
        q = normalize(question)
        targets = self.vocab.match_classes(q)
        if META.search(q):
            return Route("meta", reason="question is about the system's capabilities")
        intent = next((name for name, pat in INTENT_PATTERNS if pat.search(q)), None)

        ppe = [t for t in targets if t in self.vocab.compliance]
        if ppe and not VISUAL_ATTR.search(q):
            neg, wear, univ, cnt = (bool(NEG_WEAR.search(q)), bool(WEAR.search(q)),
                                    bool(UNIVERSAL.search(q)), bool(COUNT_Q.search(q)))
            if neg or wear:
                if cnt:
                    form = "count_without" if neg else "count_with"
                elif univ and not neg:
                    form = "all_with"
                elif neg:
                    form = "any_without"
                else:
                    form = "any_with"
                return Route("compliance", [ppe[0]], form=form,
                             reason=f"PPE question about '{ppe[0]}' ({form})")

        if VISUAL_ATTR.search(q) or (CONTEXT_ATTR.search(q) and (IMAGE_REF.search(q) or targets)):
            return Route("unsupported", targets, reason="asks for an attribute (colour, identity, text, action...) "
                                                        "that bounding boxes cannot provide")
        if UNIVERSAL.search(q) and targets and intent in {"existence", None}:
            # "Is everyone wearing a helmet?" needs knowing which class means "without X".
            # The LLM router can resolve that from class descriptions; the rules refuse to guess.
            return Route("unsupported", targets, reason="universal question ('everyone/all'); rephrase as "
                                                        "'is anyone ...' or enable the LLM router")
        if intent is None:
            if targets:
                return Route("other", targets, reason="mentions a detectable class but no recognised question type")
            if IMAGE_REF.search(q):
                return Route("other", reason="about the image but no recognised question type or class")
            return Route("general", reason="no reference to the image or any detectable class")

        if intent == "compare" and len(targets) < 2:
            return Route("unsupported", targets, reason="comparison needs two detectable classes")
        if intent in NEEDS_TARGET and not targets:
            if IMAGE_REF.search(q) or intent in {"count", "location"}:
                return Route("unsupported", reason=f"asks about something outside the detector's classes "
                                                   f"({self.vocab.describe()})")
            return Route("general", reason="no detectable class and no reference to the image")
        return Route(intent, targets, reason=f"matched '{intent}' pattern")


# --- LLM router -----------------------------------------------------------------------------------
ROUTER_SYSTEM = """You are the intent router for an image question-answering API.
The API's object detector can ONLY detect these classes (name: description):
{classes}

Classify the user's question. Reply with ONLY a JSON object, no prose, with keys:
  "intent": one of {intents}
  "target_classes": list of class NAMES from the list above that the question is about
  "unsupported_targets": list of objects the question asks about that are NOT detectable classes
  "negate": true only for universal questions answered by the ABSENCE of a class
            (e.g. "is everyone wearing X?" -> existence of the class meaning "without X", negate=true)
  "reason": a short justification

Intent meanings:
  count       - how many of a class
  existence   - whether any instance of a class is present
  most_common - which detectable class appears most
  list        - which detectable objects are present / describe the image in terms of the classes
  location    - where instances are (left / right / top / bottom)
  compare     - more / fewer of one class than another (exactly 2 target classes, in the order asked)
  compliance  - whether people are / aren't wearing a PPE item. target_classes = [the PPE class], one of:
                {ppe}. Also set "form": any_without ("is anyone not wearing X?"), any_with ("is anyone
                wearing X?"), all_with ("is everyone wearing X?"), count_without ("how many aren't wearing X?"),
                count_with ("how many are wearing X?")
  other       - about the image and answerable from boxes, counts and positions, but none of the above
  meta        - about the system itself (e.g. "what can you detect?")
  general     - not about the image at all
  unsupported - about the image but needs something boxes can't give (colour, identity, text,
                actions, attributes) or ONLY undetectable objects, or too ambiguous to map to a class
Map paraphrases to classes using the descriptions. Never invent class names."""


class Router:
    def __init__(self, vocab: Vocabulary, llm: LLMClient | None, mode: str = "auto"):
        self.vocab, self.llm, self.mode = vocab, llm, mode
        self.rules = RuleRouter(vocab)

    def route(self, question: str) -> Route:
        use_llm = self.mode == "llm" or (self.mode == "auto" and self.llm is not None and self.llm.available)
        if use_llm and self.llm is not None and self.llm.available:
            try:
                return self._llm_route(question)
            except (LLMError, KeyError, TypeError, ValueError) as e:
                log.warning("LLM router failed (%s); falling back to rules", e)
        return self.rules.route(question)

    def _llm_route(self, question: str) -> Route:
        classes = "\n".join(f"  - {n}: {i.get('description', '')}" for n, i in self.vocab.classes.items())
        system = ROUTER_SYSTEM.format(classes=classes, intents=sorted(ALL_INTENTS),
                                      ppe=", ".join(self.vocab.compliance) or "(none configured)")
        raw = self.llm.complete_json(system, json.dumps({"question": question}), max_tokens=300)
        return self.validate(raw)

    def validate(self, raw: dict) -> Route:
        intent = str(raw.get("intent", "")).strip().lower()
        if intent not in ALL_INTENTS:
            raise ValueError(f"unknown intent {intent!r}")
        targets = [t for t in raw.get("target_classes") or [] if t in self.vocab.classes]
        dropped = [t for t in raw.get("target_classes") or [] if t not in self.vocab.classes]
        unsupported = [str(t) for t in (raw.get("unsupported_targets") or [])] + dropped
        route = Route(intent, targets, unsupported, bool(raw.get("negate", False)),
                      form=str(raw.get("form", "")), reason=str(raw.get("reason", ""))[:200], source="llm")
        if route.intent == "compliance":
            route.target_classes = [t for t in route.target_classes if t in self.vocab.compliance][:1]
            if route.form not in COMPLIANCE_FORMS:
                route.form = "any_without"
            route.negate = False
        else:
            route.form = ""
        # structural checks the LLM can't talk its way past
        if route.intent in NEEDS_TARGET and not route.target_classes:
            route.intent, route.reason = "unsupported", route.reason or "no detectable class in question"
        if route.intent == "compare" and len(route.target_classes) != 2:
            route.intent, route.reason = "unsupported", "comparison needs exactly two detectable classes"
        if route.negate and route.intent != "existence":
            route.negate = False
        return route
