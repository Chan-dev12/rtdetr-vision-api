"""The confidence guardrail. Given a route and the facts, decide what can honestly be said.

Every rule is deterministic, so it can be unit-tested and defended line by line:

  R1  Confident detections (>= operating_conf) are the only thing counted as fact.
  R2  Uncertain detections ([uncertain_conf, operating_conf)) are never counted, but they
      turn exact answers into ranges ("at least 3, possibly 5") and block confident "none" answers.
  R3  A NEGATIVE answer ("no X") is refused when the image is dark/blurry/low-res, or when the
      class's measured test-set recall is below min_class_recall. Absence of a detection is
      only evidence of absence if the detector is known to find that class reliably.
  R4  Comparisons and "most common" are refused when uncertain detections could flip the result.
  R5  Anything outside the detector's vocabulary (other objects, colours, identities, actions)
      gets an explicit "insufficient information" answer rather than a guess.
  R6  PPE compliance cuts both ways. A reported VIOLATION ("no helmet") is only as good as the
      detector's recall for the PPE item (a missed helmet looks like a violation). "Everyone is
      compliant" is only as good as its recall for the body part (a missed head is never checked).
      Each direction is qualified or refused using the matching class's measured test recall.
  R7  If an image yields an implausible number of confident boxes, the detector output itself is
      not trusted (wrong weights, broken config, degenerate image), so nothing is answered from it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .router import Route
from .vocab import Vocabulary

ANSWERED = "answered"
UNCERTAIN = "answered_with_uncertainty"
INSUFFICIENT = "insufficient_information"
NO_DETECTION = "no_detection_needed"
_SEVERITY = {ANSWERED: 0, UNCERTAIN: 1, INSUFFICIENT: 2}


@dataclass
class Decision:
    status: str
    answer: str
    caveats: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _join(parts: list[Decision]) -> Decision:
    worst = max((p.status for p in parts), key=_SEVERITY.get)
    # one insufficient part among answered ones -> the whole answer is partial, not a refusal
    if worst == INSUFFICIENT and any(p.status != INSUFFICIENT for p in parts):
        worst = UNCERTAIN
    return Decision(worst, " ".join(p.answer for p in parts),
                    [c for p in parts for c in p.caveats], sorted({r for p in parts for r in p.rules_fired}))


def negative_blockers(cls: str, facts: dict, vocab: Vocabulary) -> list[str]:
    """R3: reasons a 'none detected' result can't be trusted for this class."""
    reasons = []
    flags = facts["image"]["quality_flags"]
    if flags:
        reasons.append(f"the image looks {' and '.join(flags)}, which is a known cause of missed detections")
    rec = vocab.reliability.get(cls)
    if rec is not None and rec < vocab.thresholds.min_class_recall:
        reasons.append(f"the detector only finds {rec:.0%} of {vocab.plural(cls, 2)} on its test set")
    return reasons


def _negative(cls: str, facts: dict, vocab: Vocabulary, yes_no: str | None = None) -> Decision:
    """Nothing detected for `cls`, not even uncertain boxes. Answer 'none' only if R3 allows it."""
    blockers = negative_blockers(cls, facts, vocab)
    if blockers:
        return Decision(INSUFFICIENT, f"I didn't detect any {vocab.plural(cls, 2)}, but I can't be confident "
                                      f"there are none: {'; '.join(blockers)}.", rules_fired=["R3"])
    lead = f"{yes_no} " if yes_no else ""
    d = Decision(ANSWERED, f"{lead}I didn't detect any {vocab.plural(cls, 2)}.")
    rec = vocab.reliability.get(cls)
    if rec is not None:
        d.caveats.append(f"On its test set the detector finds {rec:.0%} of {vocab.plural(cls, 2)}, "
                         f"so a missed one is possible.")
    return d


def _count_one(cls: str, facts: dict, vocab: Vocabulary) -> Decision:
    c, u = facts["counts_confident"][cls], facts["counts_uncertain"][cls]
    if c == 0 and u == 0:
        return _negative(cls, facts, vocab)
    if c == 0:
        return Decision(INSUFFICIENT, f"I can't count {vocab.plural(cls, 2)} confidently: there are {u} possible "
                                      f"{vocab.plural(cls, u)}, all below the confidence threshold.", rules_fired=["R2"])
    d = Decision(ANSWERED, f"I count {vocab.phrase(cls, c)}.")
    if u:
        d = Decision(UNCERTAIN, f"I count at least {vocab.phrase(cls, c)}, possibly up to {c + u} "
                                f"({u} more low-confidence {'detection' if u == 1 else 'detections'}).",
                     rules_fired=["R2"])
    if c >= vocab.thresholds.dense_scene_count:
        d.caveats.append("This is a crowded scene; overlapping objects are often missed, so the true number may be higher.")
    return d


def count(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    return _join([_count_one(c, facts, vocab) for c in route.target_classes])


def existence(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    parts = []
    for cls in route.target_classes:
        c, u = facts["counts_confident"][cls], facts["counts_uncertain"][cls]
        yes, no = ("No,", "Yes,") if route.negate else ("Yes,", "No,")
        if c:
            parts.append(Decision(ANSWERED, f"{yes} I detected {vocab.phrase(cls, c)} "
                                            f"(highest confidence {facts['max_confidence'][cls]:.0%})."))
        elif u:
            parts.append(Decision(INSUFFICIENT, f"Possibly: there {'is' if u == 1 else 'are'} {u} low-confidence "
                                                f"{vocab.plural(cls, u)}, which isn't enough to say yes or no.",
                                  rules_fired=["R2"]))
        else:
            parts.append(_negative(cls, facts, vocab, yes_no=no))
    return _join(parts)


def _ranking_is_stable(leader: str, facts: dict, names: list[str]) -> list[str]:
    """R4: classes whose uncertain detections could tie or overtake the leader."""
    lead = facts["counts_confident"][leader]
    return [n for n in names if n != leader
            and facts["counts_confident"][n] + facts["counts_uncertain"][n] >= lead]


def most_common(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    counts = facts["counts_confident"]
    if not any(counts.values()):
        return _nothing_found(facts, vocab)
    ranked = sorted(counts, key=lambda n: -counts[n])
    leader = ranked[0]
    ties = [n for n in ranked if counts[n] == counts[leader]]
    if len(ties) > 1:
        return Decision(UNCERTAIN, "It's a tie: " + " and ".join(vocab.phrase(n, counts[n]) for n in ties) + ".",
                        rules_fired=["R4"])
    rivals = _ranking_is_stable(leader, facts, vocab.names)
    if rivals:
        r = rivals[0]
        return Decision(INSUFFICIENT, f"{vocab.plural(leader, 2).capitalize()} look most common "
                                      f"({counts[leader]} confident), but {vocab.plural(r, 2)} have {counts[r]} confident "
                                      f"plus {facts['counts_uncertain'][r]} low-confidence detections, so the ranking "
                                      f"could change. I can't say for sure.", rules_fired=["R4"])
    others = ", ".join(vocab.phrase(n, counts[n]) for n in ranked[1:] if counts[n])
    return Decision(ANSWERED, f"Most common: {vocab.phrase(leader, counts[leader])}"
                              + (f", followed by {others}." if others else ". Nothing else I know was detected."))


def _nothing_found(facts: dict, vocab: Vocabulary) -> Decision:
    unc = {n: v for n, v in facts["counts_uncertain"].items() if v}
    if unc:
        return Decision(INSUFFICIENT, "I have no confident detections, only low-confidence ones ("
                        + ", ".join(vocab.phrase(n, v) for n, v in unc.items()) + "), so I can't answer reliably.",
                        rules_fired=["R2"])
    flags = facts["image"]["quality_flags"]
    if flags:
        return Decision(INSUFFICIENT, f"I didn't detect any {vocab.describe()}, but the image looks "
                                      f"{' and '.join(flags)}, so I may be missing things.", rules_fired=["R3"])
    return Decision(ANSWERED, f"I didn't detect any of the objects I know about ({vocab.describe()}).")


def list_objects(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    counts = {n: v for n, v in facts["counts_confident"].items() if v}
    if not counts:
        return _nothing_found(facts, vocab)
    d = Decision(ANSWERED, "I detected " + ", ".join(vocab.phrase(n, v) for n, v in
                                                     sorted(counts.items(), key=lambda kv: -kv[1])) + ".")
    unc = {n: v for n, v in facts["counts_uncertain"].items() if v}
    if unc:
        d.status = UNCERTAIN
        d.caveats.append("There are also low-confidence detections I'm not counting: "
                         + ", ".join(vocab.phrase(n, v) for n, v in unc.items()) + ".")
        d.rules_fired.append("R2")
    d.caveats.append(f"I can only detect: {vocab.describe()}. Other objects in the image are not reported.")
    return d


def location(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    parts = []
    for cls in route.target_classes:
        regions = facts["regions_confident"][cls]
        if not regions:
            if facts["counts_uncertain"][cls]:
                parts.append(Decision(INSUFFICIENT, f"I can't locate {vocab.plural(cls, 2)} confidently; I only have "
                                                    f"low-confidence detections.", rules_fired=["R2"]))
            else:
                parts.append(_negative(cls, facts, vocab))
            continue
        where = ", ".join(f"{n} at the {r.replace('-', ' ')}" for r, n in sorted(regions.items(), key=lambda kv: -kv[1]))
        parts.append(Decision(ANSWERED, f"{vocab.plural(cls, 2).capitalize()}: {where}."))
    return _join(parts)


def compare(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    a, b = route.target_classes[:2]
    ca, ua = facts["counts_confident"][a], facts["counts_uncertain"][a]
    cb, ub = facts["counts_confident"][b], facts["counts_uncertain"][b]
    summary = f"{vocab.phrase(a, ca)} (+{ua} uncertain) vs {vocab.phrase(b, cb)} (+{ub} uncertain)"
    if ca > cb + ub:
        return Decision(ANSWERED, f"Yes, there are more {vocab.plural(a, 2)} than {vocab.plural(b, 2)}: {summary}.")
    if cb > ca + ua:
        return Decision(ANSWERED, f"No, there are more {vocab.plural(b, 2)} than {vocab.plural(a, 2)}: {summary}.")
    if ca == cb and ua == ub == 0:
        if ca == 0:
            return _nothing_found(facts, vocab)
        return Decision(ANSWERED, f"No, they're equal: {summary}.")
    return Decision(INSUFFICIENT, f"I can't tell: {summary}. The uncertain detections could change the answer.",
                    rules_fired=["R4"])


def unsupported(route: Route, vocab: Vocabulary) -> Decision:
    what = f" ({', '.join(route.unsupported_targets)})" if route.unsupported_targets else ""
    return Decision(INSUFFICIENT, f"I don't have enough information to answer that{what}. My detector only "
                                  f"finds these objects: {vocab.describe()}, as boxes with confidence scores. "
                                  f"It can't tell colours, identities, text or actions. Reason: {route.reason}.",
                    rules_fired=["R5"])


def meta(vocab: Vocabulary) -> Decision:
    desc = "; ".join(f"{n} ({i.get('description', '').strip()})" for n, i in vocab.classes.items())
    extra = (f" I can also check PPE compliance ({', '.join(f'{p} on {r['base']}' for p, r in vocab.compliance.items())})."
             if vocab.compliance else "")
    return Decision(NO_DETECTION, f"I analyse {vocab.domain} images and can detect: {desc}. I can count them, "
                                  f"say whether they're present, where they are, and which is most common.{extra}")


def compliance(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    ppe = route.target_classes[0]
    c = facts["compliance"][ppe]
    base, n, yes_c, no_c, unclear = c["base"], c["checked"], c["with"], c["without"], c["unclear"]
    form = route.form or "any_without"
    ppe_word, base_n = vocab.plural(ppe, 1), lambda k: vocab.phrase(base, k)
    rule_caveat = vocab.compliance[ppe].get("caveat")

    if n == 0:
        if c["uncertain_bases"]:
            return Decision(INSUFFICIENT, f"I can't check for {vocab.plural(ppe, 2)}: I only have "
                                          f"{c['uncertain_bases']} low-confidence {vocab.plural(base, c['uncertain_bases'])}.",
                            rules_fired=["R2"])
        blockers = negative_blockers(base, facts, vocab)
        if blockers:
            return Decision(INSUFFICIENT, f"I didn't detect any {vocab.plural(base, 2)} to check, but I may be missing "
                                          f"them: {'; '.join(blockers)}.", rules_fired=["R3"])
        return Decision(ANSWERED, f"I didn't detect any {vocab.plural(base, 2)}, so there is nobody to check for "
                                  f"{vocab.plural(ppe, 2)}.")

    parts = []
    if yes_c:
        parts.append(f"{yes_c} {'has' if yes_c == 1 else 'have'} a {ppe_word}")
    if no_c:
        parts.append(f"{no_c} {'has' if no_c == 1 else 'have'} no {ppe_word}")
    if unclear:
        parts.append(f"{unclear} {'is' if unclear == 1 else 'are'} unclear (only a low-confidence {ppe_word})")
    summary = f"Of {base_n(n)} detected, " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1
                                              else parts[0]) + "."
    regions = sorted(c["without_regions"].items(), key=lambda kv: -kv[1])
    where = ", ".join(f"{k} at the {r.replace('-', ' ')}" for r, k in regions[:4])
    if len(regions) > 4:
        where += f" and {sum(k for _, k in regions[4:])} elsewhere"
    rules: list[str] = []
    caveats: list[str] = []

    verdict = {  # form -> (lead if any violations, lead if all compliant)
        "any_without": ("Yes.", "No."), "all_with": ("No.", "Yes."),
        "any_with": ("Yes." if yes_c else "No.", "Yes."),
        "count_without": (f"{no_c}.", "0."), "count_with": (f"{yes_c}.", f"{yes_c}."),
    }[form]

    if no_c > 0:
        status = UNCERTAIN if unclear else ANSWERED
        if unclear:
            rules.append("R2")
        answer = f"{verdict[0]} {summary}" + (f" Missing {ppe_word}: {where}." if where else "")
        rec = vocab.reliability.get(ppe)
        if rec is not None and rec < vocab.thresholds.min_class_recall:
            status = UNCERTAIN
            rules.append("R6")
            caveats.append(f"The detector only finds {rec:.0%} of {vocab.plural(ppe, 2)} on its test set, so some of "
                           f"these may be missed {vocab.plural(ppe, 2)} rather than real violations.")
    elif unclear > 0:
        if form == "count_with":
            status, answer = UNCERTAIN, f"At least {yes_c}, possibly {yes_c + unclear}. {summary}"
            rules.append("R2")
        else:
            return Decision(INSUFFICIENT, f"I can't say for sure. {summary}", rules_fired=["R2"])
    else:
        blockers = negative_blockers(base, facts, vocab)
        if blockers and form in ("any_without", "all_with", "count_without"):
            return Decision(INSUFFICIENT, f"Every {vocab.plural(base, 1)} I detected has a {ppe_word} ({n} of {n}), "
                                          f"but I can't confirm nobody is missing one: {'; '.join(blockers)}.",
                            rules_fired=["R3", "R6"])
        status, answer = ANSWERED, f"{verdict[1]} {summary}"
        rec = vocab.reliability.get(base)
        if rec is not None:
            caveats.append(f"On its test set the detector finds {rec:.0%} of {vocab.plural(base, 2)}; "
                           f"anyone it missed wasn't checked.")

    if n >= vocab.thresholds.dense_scene_count:
        caveats.append("This is a crowded scene; overlapping people are often missed, so some weren't checked.")
    if c["uncertain_bases"]:
        caveats.append(f"{c['uncertain_bases']} more low-confidence {vocab.plural(base, c['uncertain_bases'])} "
                       f"{'was' if c['uncertain_bases'] == 1 else 'were'} not checked.")
        status = UNCERTAIN if status == ANSWERED else status
        rules.append("R2")
    if rule_caveat and (no_c or form in ("count_without", "any_without")):
        caveats.append(rule_caveat)
    return Decision(status, answer, caveats, sorted(set(rules)))


HANDLERS = {"count": count, "existence": existence, "most_common": most_common,
            "list": list_objects, "location": location, "compare": compare, "compliance": compliance}


def decide(route: Route, facts: dict, vocab: Vocabulary) -> Decision:
    total = sum(facts["counts_confident"].values())
    if total > vocab.thresholds.max_plausible_detections:
        return Decision(INSUFFICIENT, f"I can't answer reliably: the detector returned {total} confident objects "
                                      f"for this image, which is implausible and suggests its output is not "
                                      f"trustworthy here.", rules_fired=["R7"])
    decision = HANDLERS[route.intent](route, facts, vocab)
    if route.unsupported_targets and decision.status != INSUFFICIENT:
        decision.caveats.append(f"I can't detect {', '.join(route.unsupported_targets)}, so that part of the "
                                f"question is unanswered.")
        decision.status = UNCERTAIN
        decision.rules_fired.append("R5")
    return decision
