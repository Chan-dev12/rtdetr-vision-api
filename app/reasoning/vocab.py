"""The detector's vocabulary (what it can see) plus the guardrail thresholds, loaded from configs/classes.yaml."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Thresholds:
    operating_conf: float = 0.5
    uncertain_conf: float = 0.25
    min_class_recall: float = 0.6
    dense_scene_count: int = 25
    max_plausible_detections: int = 150


@dataclass
class Vocabulary:
    domain: str
    classes: dict[str, dict]            # name -> {description, plural, synonyms}
    thresholds: Thresholds
    reliability: dict[str, float] = field(default_factory=dict)  # class -> test-set recall
    compliance: dict[str, dict] = field(default_factory=dict)   # ppe class -> {base, expand, min_overlap, caveat}

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        cfg = yaml.safe_load(Path(path).read_text())
        vocab = cls(domain=cfg.get("domain", "images"),
                    classes={k: v or {} for k, v in cfg["classes"].items()},
                    thresholds=Thresholds(**cfg.get("thresholds", {})))
        for ppe, rule in (cfg.get("compliance") or {}).items():
            if ppe not in vocab.classes or rule.get("base") not in vocab.classes:
                raise ValueError(f"compliance rule {ppe}->{rule.get('base')} uses a class not in `classes`")
            vocab.compliance[ppe] = {"expand": 0.0, "min_overlap": 0.3, **rule}
        rel = cfg.get("reliability_csv")
        if rel and Path(rel).exists():
            vocab.reliability = load_reliability(rel)
        return vocab

    @property
    def names(self) -> list[str]:
        return list(self.classes)

    def plural(self, name: str, n: int) -> str:
        """Human wording: optional `label` (singular) and `plural` from classes.yaml, else the class name."""
        info = self.classes.get(name, {})
        label = info.get("label") or name
        return label if n == 1 else (info.get("plural") or f"{label}s")

    def phrase(self, name: str, n: int) -> str:
        return f"{n} {self.plural(name, n)}"

    def describe(self) -> str:
        return ", ".join(self.names)

    def match_classes(self, text: str) -> list[str]:
        """Find class mentions using longest-phrase-first matching, so "not wearing a helmet"
        maps to the bare-head class instead of also triggering "helmet"."""
        text = f" {normalize(text)} "
        phrases = []
        for name, info in self.classes.items():
            for s in {name, *info.get("synonyms", [])}:
                phrases.append((normalize(s), name))
        phrases.sort(key=lambda p: -len(p[0]))
        first_pos: dict[str, int] = {}
        for phrase, name in phrases:
            pat = re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])")
            for m in pat.finditer(text):
                first_pos[name] = min(first_pos.get(name, m.start()), m.start())
            # consume matched spans (same length, so positions stay valid) so shorter phrases can't re-match
            text = pat.sub(lambda m: "#" * len(m.group(0)), text)
        return sorted(first_pos, key=first_pos.get)  # in order of mention (matters for "more X than Y")


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_reliability(path: str | Path) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                if int(row.get("gt_instances", 1)) > 0:
                    out[row["class"]] = float(row["recall"])
            except (KeyError, ValueError):
                continue
    return out
