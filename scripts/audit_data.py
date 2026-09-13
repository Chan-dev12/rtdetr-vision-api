"""Audit a YOLO dataset: per-split class balance, object sizes, background images.

Run this before training. The size breakdown tells you in advance whether small objects will be a failure mode.

Usage:
    python scripts/audit_data.py --data data/processed/data.yaml --out reports/data_audit.md --preview 24

--preview N draws the ground-truth boxes on N random training images (reports/label_preview/).
LOOK AT THEM before training: they are the only reliable check that class ids are mapped
correctly and that you understand the labelling rules (e.g. is a head under a helmet also a "head"?).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _common import load_gt, load_names, size_bucket, split_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", default="reports/data_audit.md")
    ap.add_argument("--preview", type=int, default=0, help="draw GT boxes on N random train images")
    args = ap.parse_args()
    names = load_names(args.data)

    lines = ["# Dataset audit", "", f"data: `{args.data}`", ""]
    for split in args.splits:
        try:
            imgs = split_images(args.data, split)
        except SystemExit as e:
            print(e)
            continue
        inst, sizes, bg, per_img, resolutions = Counter(), Counter(), 0, [], Counter()
        for img in imgs:
            with Image.open(img) as im:
                w, h = im.size
            resolutions[f"{w}x{h}"] += 1
            cls, boxes = load_gt(img, w, h)
            per_img.append(len(cls))
            if len(cls) == 0:
                bg += 1
            for c, b in zip(cls, boxes):
                inst[names.get(int(c), str(c))] += 1
                sizes[(names.get(int(c), str(c)), size_bucket((b[2] - b[0]) * (b[3] - b[1])))] += 1

        lines += [f"## {split}", "",
                  f"- images: {len(imgs)} (background / no boxes: {bg})",
                  f"- boxes per image: mean {sum(per_img) / max(len(per_img), 1):.2f}, max {max(per_img, default=0)}",
                  f"- most common resolutions: {', '.join(f'{r} ({n})' for r, n in resolutions.most_common(3))}", "",
                  "| class | instances | share | small | medium | large |",
                  "|---|---:|---:|---:|---:|---:|"]
        total = sum(inst.values()) or 1
        for n in names.values():
            lines.append(f"| {n} | {inst[n]} | {inst[n] / total:.1%} | {sizes[(n, 'small')]} | "
                         f"{sizes[(n, 'medium')]} | {sizes[(n, 'large')]} |")
        lines.append("")

    lines += ["Size bins are COCO's (<32² px small, <96² px medium) on original-resolution pixels."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    if args.preview:
        preview(args.data, names, args.preview, Path(args.out).parent / "label_preview")


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


def preview(data: str, names: dict[int, str], n: int, out: Path) -> None:
    imgs = split_images(data, "train")
    out.mkdir(parents=True, exist_ok=True)
    for img in random.Random(0).sample(imgs, min(n, len(imgs))):
        im = Image.open(img).convert("RGB")
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=max(12, min(im.size) // 45))
        cls, boxes = load_gt(img, *im.size)
        for c, b in zip(cls, boxes):
            col = PALETTE[int(c) % len(PALETTE)]
            d.rectangle(b.tolist(), outline=col, width=max(2, min(im.size) // 300))
            d.text((b[0] + 2, b[1] + 2), names.get(int(c), str(c)), fill=col, font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        im.save(out / img.name, quality=85)
    print(f"\nlabel previews: {out}/  <- open these and check the class names sit on the right objects")


if __name__ == "__main__":
    main()
