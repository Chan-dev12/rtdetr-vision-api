"""Build one YOLO-format dataset with leak-free train/val/test splits.

Why this script exists
----------------------
Public detection datasets (Roboflow exports especially) often contain
  * augmented copies of the same photo:  `img_12_jpg.rf.<hash>.jpg` x 3
  * consecutive video frames that are near-identical
  * exact duplicate files across train/valid/test
A random per-image split puts near-identical images in both train and test,
which inflates test mAP and says nothing about unseen (hidden-set) images.

So this script:
  1. collects every image and label from every source in configs/dataset.yaml
  2. remaps source class names to target classes (you must map every class)
  3. drops byte-identical duplicates (sha1)
  4. groups images that share an origin: Roboflow augmentation stem, optional
     scene/video regex, and perceptual-hash near-duplicates (union-find)
  5. splits by GROUP (never by image), so nothing near-identical crosses splits
  6. writes data.yaml, manifest.csv (per-image provenance and split) and prepare_report.json

Optional (configs/dataset.yaml):
  max_side:   downscale images so the longest side is <= this (e.g. 1280). Labels are unchanged
              (YOLO coordinates are relative). Uses JPEG draft decoding + all CPU cores, so very
              large datasets like SH17 (up to 8192 px, 14 GB) shrink to ~1-2 GB in minutes.
  fixed_test: keep a source's official evaluation list as OUR test split (so results can be
              compared with published baselines). Near-duplicates of test images found in the
              rest of the data are DROPPED from train/val (leakage) and counted in the report.

Usage:
    python scripts/prepare_data.py --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


@dataclass
class Item:
    source: str
    image: Path
    labels: list[tuple[int, float, float, float, float]]  # target_cls, cx, cy, w, h (normalised)
    sha1: str = ""
    origin: str = ""
    phash: np.ndarray | None = None
    group: int = -1
    split: str = ""
    fixes: list[str] = field(default_factory=list)
    staged: Path | None = None   # downscaled copy, if max_side is set
    fixed_test: bool = False


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def label_path_for(img: Path) -> Path:
    """YOLO convention: .../images/.../x.jpg -> .../labels/.../x.txt (last 'images' component)."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")  # labels next to images


def source_names(src: dict) -> list[str]:
    if src.get("names"):
        names = src["names"]
    else:
        yml = Path(src["path"]) / "data.yaml"
        if not yml.exists():
            raise SystemExit(f"[{src['name']}] no data.yaml in {src['path']}; add `names:` to the source config")
        names = yaml.safe_load(yml.read_text())["names"]
    if isinstance(names, dict):  # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    return list(names)


def read_labels(txt: Path, id_map: dict[int, int | None], fixes: list[str]):
    out = []
    if not txt.exists():
        return out  # background image, which is fine and useful against false positives
    for ln, line in enumerate(txt.read_text().splitlines(), 1):
        vals = line.split()
        if not vals:
            continue
        if len(vals) != 5:
            fixes.append(f"line {ln}: {len(vals)} values (polygon/segment?) skipped")
            continue
        cls = int(float(vals[0]))
        if cls not in id_map:
            raise SystemExit(f"{txt}: class id {cls} is not in the source's names list")
        tgt = id_map[cls]
        if tgt is None:
            continue
        cx, cy, w, h = map(float, vals[1:])
        # convert to corners, clip to the image, convert back
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            fixes.append(f"line {ln}: degenerate box dropped")
            continue
        if (x1, y1, x2, y2) != (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2):
            fixes.append(f"line {ln}: box clipped to image")
        out.append((tgt, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


def collect(cfg: dict) -> list[Item]:
    targets = cfg["target_classes"]
    items: list[Item] = []
    for src in cfg["sources"]:
        names = source_names(src)
        cmap = src.get("class_map") or {}
        missing = [n for n in names if n not in cmap]
        if missing:
            raise SystemExit(f"[{src['name']}] class_map is missing {missing}; map each to a target class or null")
        bad = {v for v in cmap.values() if v is not None and v not in targets}
        if bad:
            raise SystemExit(f"[{src['name']}] class_map targets {bad} are not in target_classes")
        id_map = {i: (targets.index(cmap[n]) if cmap[n] is not None else None) for i, n in enumerate(names)}

        root = Path(src["path"]) / src.get("images_subdir", "")
        imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and "labels" not in p.parts)
        if not imgs:
            raise SystemExit(f"[{src['name']}] no images found under {root}")
        for img in imgs:
            fixes: list[str] = []
            labels = read_labels(label_path_for(img), id_map, fixes)
            items.append(Item(source=src["name"], image=img, labels=labels, fixes=fixes))
        print(f"[{src['name']}] {len(imgs)} images")
    return items


def assign_groups(items: list[Item], cfg: dict) -> int:
    uf = UnionFind(len(items))
    # 1) same origin stem (Roboflow augmentation copies) and optional scene regex
    key_first: dict[str, int] = {}
    regex = re.compile(cfg["group_regex"]) if cfg.get("group_regex") else None
    for i, it in enumerate(items):
        it.origin = ROBOFLOW_SUFFIX.sub("", it.image.stem)
        keys = [f"origin::{it.source}::{it.origin}"]
        if regex and (m := regex.search(it.image.name)):
            keys.append(f"scene::{it.source}::{m.group(1)}")
        for k in keys:
            if k in key_first:
                uf.union(key_first[k], i)
            else:
                key_first[k] = i

    # 2) perceptual-hash near duplicates (chunked all-pairs Hamming distance)
    maxd = int(cfg.get("phash_max_distance", 0))
    if maxd > 0:
        bits = np.stack([it.phash.flatten() for it in items]).astype(bool)  # (n, 64)
        n = len(items)
        for start in range(0, n, 256):
            chunk = bits[start:start + 256]
            dist = (chunk[:, None, :] != bits[None, :, :]).sum(-1)  # (c, n)
            ii, jj = np.nonzero(dist <= maxd)
            for a, b in zip(ii + start, jj):
                if a < b:
                    uf.union(int(a), int(b))

    roots: dict[int, int] = {}
    for i, it in enumerate(items):
        it.group = roots.setdefault(uf.find(i), len(roots))
    return len(roots)


def split_groups(items: list[Item], ratios: dict[str, float], seed: int) -> None:
    """Group-aware AND class-stratified split.

    Each group is keyed by the rarest class it contains ("background" if none). Within each key,
    groups are dealt to whichever split is furthest below its target share. So rare classes get
    spread across train/val/test too, instead of landing wherever the random shuffle puts them.
    """
    groups: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.group].append(it)
    freq = Counter(c for it in items for c, *_ in it.labels)

    def key(g: int) -> int:
        present = {c for it in groups[g] for c, *_ in it.labels}
        return min(present, key=lambda c: freq[c]) if present else -1

    strata: dict[int, list[int]] = defaultdict(list)
    for g in groups:
        strata[key(g)].append(g)
    rng = random.Random(seed)
    for _, gs in sorted(strata.items()):
        rng.shuffle(gs)
        gs.sort(key=lambda g: -len(groups[g]))  # big groups first; stable sort keeps the shuffle for ties
        total = sum(len(groups[g]) for g in gs)
        count = {s: 0 for s in ratios}
        for g in gs:
            s = max(ratios, key=lambda k: ratios[k] * total - count[k])
            for it in groups[g]:
                it.split = s
            count[s] += len(groups[g])

def _process(job: tuple) -> tuple:
    """Worker (runs in a separate process): hash, optionally downscale, and perceptual-hash one image."""
    idx, src, stage_dir, max_side, quality = job
    try:
        data = Path(src).read_bytes()
        sha1 = hashlib.sha1(data).hexdigest()
        with Image.open(src) as im:
            if max_side and im.format == "JPEG":
                im.draft("RGB", (max_side, max_side))  # decode at 1/2, 1/4 or 1/8 scale: much faster
            # Labels refer to the upright image (Ultralytics/OpenCV apply EXIF rotation when loading),
            # so bake the rotation in before saving without EXIF.
            im = ImageOps.exif_transpose(im).convert("RGB")
            staged = None
            if max_side:
                if max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.LANCZOS)
                staged = str(Path(stage_dir) / f"{idx}.jpg")
                im.save(staged, quality=quality)
            return idx, sha1, imagehash.phash(im).hash, staged, None
    except Exception as e:  # corrupt / unreadable file
        return idx, None, None, None, f"{e.__class__.__name__}: {e}"


def load_fixed_test(cfg: dict) -> tuple[str, set[str]] | None:
    ft = cfg.get("fixed_test")
    if not ft:
        return None
    lines = Path(ft["list"]).read_text().splitlines()
    names = {Path(ln.strip().replace("\\", "/")).name for ln in lines if ln.strip()}
    return ft["source"], names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    targets = cfg["target_classes"]

    items = collect(cfg)
    max_side = cfg.get("max_side")
    stage = out.parent / f"_{out.name}_staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # hash + (downscale) + perceptual hash, in parallel
    workers = int(cfg.get("workers") or os.cpu_count() or 1)
    jobs = [(i, str(it.image), str(stage), max_side, int(cfg.get("jpeg_quality", 92))) for i, it in enumerate(items)]
    unreadable = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (i, sha1, ph, staged, err) in enumerate(pool.map(_process, jobs, chunksize=16), 1):
            if err:
                unreadable.append((str(items[i].image), err))
            else:
                items[i].sha1, items[i].phash = sha1, ph
                items[i].staged = Path(staged) if staged else None
            if n % 500 == 0 or n == len(jobs):
                print(f"  processed {n}/{len(jobs)} images", flush=True)
    items = [it for it in items if it.phash is not None]

    # exact duplicates
    seen: set[str] = set()
    unique: list[Item] = []
    dup_count = 0
    for it in items:
        if it.sha1 in seen:
            dup_count += 1
            continue
        seen.add(it.sha1)
        unique.append(it)
    items = unique

    n_groups = assign_groups(items, cfg)

    # official test list (optional)
    fixed = load_fixed_test(cfg)
    leak_report = {}
    seed = cfg.get("seed", 42)
    if fixed:
        src_name, names = fixed
        for it in items:
            it.fixed_test = it.source == src_name and it.image.name in names
        found = {it.image.name for it in items if it.fixed_test}
        by_group: dict[int, list[Item]] = defaultdict(list)
        for it in items:
            by_group[it.group].append(it)
        leaky = [g for g, members in by_group.items()
                 if any(m.fixed_test for m in members) and not all(m.fixed_test for m in members)]
        dropped = [m for g in leaky for m in by_group[g] if not m.fixed_test]
        dropped_ids = {id(m) for m in dropped}
        items = [it for it in items if id(it) not in dropped_ids]
        for it in items:
            if it.fixed_test:
                it.split = "test"
        rest = [it for it in items if not it.fixed_test]
        ratios = {k: v for k, v in cfg["split"].items() if k != "test"}
        split_groups(rest, ratios, seed)
        leak_report = {"fixed_test_list": cfg["fixed_test"]["list"], "listed": len(names),
                       "listed_but_not_found": len(names - found),
                       "groups_spanning_official_split": len(leaky),
                       "train_images_dropped_as_near_duplicates_of_test": len(dropped)}
        print(f"fixed test: {len(found)} of {len(names)} listed images found; "
              f"{len(dropped)} near-duplicates of test images dropped from train/val")
    else:
        split_groups(items, cfg["split"], seed)

    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)

    rows = []
    for it in items:
        name = f"{it.source}_{it.sha1[:12]}"
        if it.staged:
            dst_img = out / it.split / "images" / f"{name}.jpg"
            shutil.move(it.staged, dst_img)
        else:
            dst_img = out / it.split / "images" / f"{name}{it.image.suffix.lower()}"
            shutil.copy2(it.image, dst_img)
        (out / it.split / "labels" / f"{name}.txt").write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in it.labels))
        rows.append({"file": dst_img.name, "split": it.split, "source": it.source, "original_path": str(it.image),
                     "group": it.group, "sha1": it.sha1, "n_boxes": len(it.labels), "fixes": "; ".join(it.fixes)})

    shutil.rmtree(stage, ignore_errors=True)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["file"])))

    data_yaml = {"path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
                 "names": {i: n for i, n in enumerate(targets)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # report
    per_split = {}
    for s in ("train", "val", "test"):
        sub = [it for it in items if it.split == s]
        inst = Counter(targets[c] for it in sub for c, *_ in it.labels)
        per_split[s] = {"images": len(sub), "background_images": sum(1 for it in sub if not it.labels),
                        "groups": len({it.group for it in sub}),
                        "instances": {n: inst.get(n, 0) for n in targets}}
    report = {"sources": cfg["sources"],
              "exact_duplicates_dropped": dup_count, "unreadable_images": unreadable,
              "near_duplicate_groups": n_groups, "images_after_dedup": len(items),
              "label_fixes": sum(len(it.fixes) for it in items), "max_side": max_side,
              "fixed_test": leak_report or None, "splits": per_split}
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{len(items)} images in {n_groups} groups (dropped {dup_count} exact duplicates, "
          f"{len(unreadable)} unreadable, {report['label_fixes']} label fixes)")
    for s, r in per_split.items():
        print(f"  {s:5s} images={r['images']:5d} groups={r['groups']:5d} instances={r['instances']}")
        for n, c in r["instances"].items():
            if c == 0:
                print(f"  WARNING: class '{n}' has no instances in '{s}'")
    print(f"\nwrote {out/'data.yaml'}, manifest.csv, prepare_report.json")


if __name__ == "__main__":
    main()
