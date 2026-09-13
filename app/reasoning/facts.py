"""Turn raw detections into structured facts. All arithmetic lives here, in plain Python.

The LLM never counts or compares. It only receives these facts. That removes an entire class of
hallucination ("I see 7 helmets" when there are 5) and makes every number in an answer traceable.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from PIL import Image

from ..detector import Detection
from .vocab import Vocabulary


def image_quality(image: Image.Image) -> dict:
    """Cheap heuristics for the failure modes that make "nothing found" untrustworthy.
    The thresholds are rough heuristics; the night-scene failure case suggests 0.18 is too low."""
    gray = image.convert("L")
    if max(gray.size) > 640:
        gray.thumbnail((640, 640))
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean() / 255.0)
    sharpness = float(cv2.Laplacian(arr, cv2.CV_32F).var())  # variance of Laplacian: low = few edges = blur
    flags = []
    if brightness < 0.18:
        flags.append("very dark")
    if brightness > 0.92:
        flags.append("overexposed")
    if sharpness < 60:
        flags.append("blurry")
    if min(image.size) < 240:
        flags.append("low resolution")
    return {"width": image.size[0], "height": image.size[1], "brightness": round(brightness, 3),
            "sharpness": round(sharpness, 1), "quality_flags": flags}


def region(det: Detection, width: int, height: int) -> str:
    x1, y1, x2, y2 = det.box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    v = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return f"{v}-{h}"


def _coverage(base: Detection, ppe: Detection, expand: float) -> float:
    """Fraction of the PPE box that lies inside the base box grown by `expand` on every side."""
    bx1, by1, bx2, by2 = base.box
    w, h = bx2 - bx1, by2 - by1
    bx1, by1, bx2, by2 = bx1 - expand * w, by1 - expand * h, bx2 + expand * w, by2 + expand * h
    px1, py1, px2, py2 = ppe.box
    iw = max(0.0, min(bx2, px2) - max(bx1, px1))
    ih = max(0.0, min(by2, py2) - max(by1, py1))
    area = max((px2 - px1) * (py2 - py1), 1e-9)
    return iw * ih / area


def _assign(bases: list[Detection], ppes: list[Detection], rule: dict) -> set[int]:
    """Greedy one-to-one matching, best coverage first. Returns indices of bases that got a PPE item."""
    pairs = sorted(((_coverage(b, p, rule["expand"]), i, j) for i, b in enumerate(bases) for j, p in enumerate(ppes)),
                   reverse=True)
    used_b, used_p = set(), set()
    for score, i, j in pairs:
        if score < rule["min_overlap"]:
            break
        if i not in used_b and j not in used_p:
            used_b.add(i)
            used_p.add(j)
    return used_b


def compliance_facts(confident: list[Detection], uncertain: list[Detection], vocab: Vocabulary,
                     width: int, height: int) -> dict:
    """Per PPE class: how many confident base boxes have it, lack it, or are unclear.
    A base is 'unclear' when it has no confident PPE match but an uncertain PPE box would match it."""
    out = {}
    for ppe, rule in vocab.compliance.items():
        bases = [d for d in confident if d.class_name == rule["base"]]
        conf_ppe = [d for d in confident if d.class_name == ppe]
        unc_ppe = [d for d in uncertain if d.class_name == ppe]
        with_ppe = _assign(bases, conf_ppe, rule)
        rest = [b for i, b in enumerate(bases) if i not in with_ppe]
        unclear = _assign(rest, unc_ppe, rule)
        without = [b for i, b in enumerate(rest) if i not in unclear]
        out[ppe] = {"base": rule["base"], "checked": len(bases), "with": len(with_ppe), "without": len(without),
                    "unclear": len(unclear),
                    "uncertain_bases": sum(d.class_name == rule["base"] for d in uncertain),
                    "without_regions": dict(Counter(region(b, width, height) for b in without))}
    return out


def build_facts(dets: list[Detection], image: Image.Image, vocab: Vocabulary) -> dict:
    t = vocab.thresholds
    w, h = image.size
    confident = [d for d in dets if d.confidence >= t.operating_conf]
    uncertain = [d for d in dets if t.uncertain_conf <= d.confidence < t.operating_conf]
    return {
        "image": image_quality(image),
        "thresholds": {"confident": t.operating_conf, "uncertain": t.uncertain_conf},
        "counts_confident": {n: sum(d.class_name == n for d in confident) for n in vocab.names},
        "counts_uncertain": {n: sum(d.class_name == n for d in uncertain) for n in vocab.names},
        "regions_confident": {n: dict(Counter(region(d, w, h) for d in confident if d.class_name == n))
                              for n in vocab.names},
        "max_confidence": {n: round(max((d.confidence for d in dets if d.class_name == n), default=0.0), 3)
                           for n in vocab.names},
        "compliance": compliance_facts(confident, uncertain, vocab, w, h),
        "detections": [d.to_dict() | {"region": region(d, w, h),
                                      "confident": d.confidence >= t.operating_conf} for d in dets[:100]],
    }
