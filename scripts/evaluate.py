"""Evaluate a trained RT-DETR model on a held-out split (or any extra YOLO folder).

Two layers of evaluation:
  A. Ultralytics model.val(): COCO-style mAP50 / mAP50-95 per class, plus PR and confusion plots.
  B. Our own matching at the API's OPERATING confidence threshold. This is the number that
     matters for /detect and /ask, because mAP averages over all thresholds and the API uses one.
     - per-class precision / recall / F1                        -> per_class.csv (read by /ask)
     - recall by object size (COCO small/medium/large)          -> size_breakdown.csv
     - P/R/F1 at thresholds 0.05..0.95, F1-optimal threshold    -> threshold_sweep.csv
     - confusion matrix incl. background                        -> confusion_matrix.csv
     - every FP/FN with a typed cause                           -> errors.csv
     - worst images drawn with GT (green) vs predictions (red)  -> failures/*.jpg

Usage:
    python scripts/evaluate.py --weights weights/best.pt --data data/processed/data.yaml --split test
    # pseudo-hidden set: any folder with images/ and labels/ in YOLO format
    python scripts/evaluate.py --weights weights/best.pt --data data/processed/data.yaml \\
        --images-dir data/wild --tag wild
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from _common import iou_matrix, load_gt, load_names, size_bucket, split_images

SWEEP = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]


def predict_all(model, images: list[Path], imgsz: int, device, min_conf: float) -> list[dict]:
    recs = []
    for i in range(0, len(images), 16):
        batch = [str(p) for p in images[i:i + 16]]
        for path, r in zip(batch, model.predict(batch, conf=min_conf, imgsz=imgsz, device=device, verbose=False)):
            h, w = r.orig_shape
            gt_cls, gt_boxes = load_gt(Path(path), w, h)
            recs.append({"image": path, "w": w, "h": h, "gt_cls": gt_cls, "gt_boxes": gt_boxes,
                         "pr_cls": r.boxes.cls.cpu().numpy().astype(int),
                         "pr_conf": r.boxes.conf.cpu().numpy(),
                         "pr_boxes": r.boxes.xyxy.cpu().numpy()})
    return recs


def match_image(rec: dict, conf: float, iou_thr: float) -> dict:
    """Greedy, confidence-ordered, same-class matching (the standard detection protocol),
    then a second pass that explains every unmatched box."""
    keep = rec["pr_conf"] >= conf
    pc, pconf, pb = rec["pr_cls"][keep], rec["pr_conf"][keep], rec["pr_boxes"][keep]
    order = np.argsort(-pconf)
    pc, pconf, pb = pc[order], pconf[order], pb[order]
    gc, gb = rec["gt_cls"], rec["gt_boxes"]
    ious = iou_matrix(pb, gb)

    gt_matched = np.full(len(gc), -1)
    pr_matched = np.full(len(pc), -1)
    for p in range(len(pc)):
        cand = [(ious[p, g], g) for g in range(len(gc)) if gc[g] == pc[p] and gt_matched[g] < 0 and ious[p, g] >= iou_thr]
        if cand:
            g = max(cand)[1]
            gt_matched[g], pr_matched[p] = p, g

    fps, fns, confusion = [], [], []
    confused_gt = set()
    for p in np.where(pr_matched < 0)[0]:
        row = ious[p] if len(gc) else np.zeros(0)
        other = [(row[g], g) for g in range(len(gc)) if gc[g] != pc[p] and gt_matched[g] < 0 and g not in confused_gt]
        same = [(row[g], g) for g in range(len(gc)) if gc[g] == pc[p]]
        if other and max(other)[0] >= iou_thr:
            g = max(other)[1]
            confused_gt.add(g)
            kind, true_cls = "class_confusion", int(gc[g])
            confusion.append((true_cls, int(pc[p])))
        elif same and max(same)[0] >= iou_thr:
            kind, true_cls = "duplicate", None           # overlaps an already-matched GT of its class
            confusion.append((None, int(pc[p])))
        elif same and max(same)[0] >= 0.1:
            kind, true_cls = "poor_localization", None   # right class, box too far off
            confusion.append((None, int(pc[p])))
        else:
            kind, true_cls = "background", None          # nothing there
            confusion.append((None, int(pc[p])))
        area = (pb[p, 2] - pb[p, 0]) * (pb[p, 3] - pb[p, 1])
        fps.append({"type": kind, "pred_cls": int(pc[p]), "true_cls": true_cls, "conf": float(pconf[p]),
                    "size": size_bucket(area), "box": pb[p].round(1).tolist()})
    for g in np.where(gt_matched < 0)[0]:
        col = ious[:, g] if len(pc) else np.zeros(0)
        area = (gb[g, 2] - gb[g, 0]) * (gb[g, 3] - gb[g, 1])
        near = [p for p in range(len(pc)) if pc[p] == gc[g] and col[p] >= 0.1]  # same-class preds touching it
        if g in confused_gt:
            kind = "class_confusion"
        elif any(pr_matched[p] < 0 for p in near):
            kind = "poor_localization"   # a box was predicted here but it's too far off to count
        elif near:
            kind = "missed_crowded"      # its neighbour was found, this one wasn't: occlusion / overlap
            confusion.append((int(gc[g]), None))
        else:
            kind = "missed"
            confusion.append((int(gc[g]), None))
        fns.append({"type": kind, "true_cls": int(gc[g]), "size": size_bucket(area), "box": gb[g].round(1).tolist()})
    # poor-localization FNs also count as (true, background) in the confusion matrix
    for fn in fns:
        if fn["type"] == "poor_localization":
            confusion.append((fn["true_cls"], None))
    for g in np.where(gt_matched >= 0)[0]:
        confusion.append((int(gc[g]), int(gc[g])))

    tp_by_cls = Counter(int(gc[g]) for g in np.where(gt_matched >= 0)[0])
    return {"tp": tp_by_cls, "fps": fps, "fns": fns, "confusion": confusion,
            "gt_sizes": [(int(gc[g]), size_bucket((gb[g, 2] - gb[g, 0]) * (gb[g, 3] - gb[g, 1])), gt_matched[g] >= 0)
                         for g in range(len(gc))]}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(recs: list[dict], conf: float, iou_thr: float, nc: int):
    tp, fp, fn = Counter(), Counter(), Counter()
    per_image = []
    for rec in recs:
        m = match_image(rec, conf, iou_thr)
        tp.update(m["tp"])
        fp.update(x["pred_cls"] for x in m["fps"])
        fn.update(x["true_cls"] for x in m["fns"])
        per_image.append(m)
    return tp, fp, fn, per_image


def draw_failure(rec: dict, m: dict, names: dict, out: Path, conf: float) -> None:
    im = Image.open(rec["image"]).convert("RGB")
    if im.size != (rec["w"], rec["h"]):
        im = im.resize((rec["w"], rec["h"]))
    d = ImageDraw.Draw(im)
    lw = max(2, int(min(im.size) / 300))
    font = ImageFont.load_default(size=max(12, min(im.size) // 40))
    d.text((6, 6), "green=GT  red=pred  yellow=missed", fill=(255, 255, 255), font=font,
           stroke_width=2, stroke_fill=(0, 0, 0))
    for c, b in zip(rec["gt_cls"], rec["gt_boxes"]):
        d.rectangle(b.tolist(), outline=(0, 200, 0), width=lw)
        d.text((b[0] + 2, b[1] + 2), f"GT {names[int(c)]}", fill=(0, 200, 0), font=font,
               stroke_width=1, stroke_fill=(0, 0, 0))
    keep = rec["pr_conf"] >= conf
    for c, s, b in zip(rec["pr_cls"][keep], rec["pr_conf"][keep], rec["pr_boxes"][keep]):
        d.rectangle(b.tolist(), outline=(230, 30, 30), width=lw)
        d.text((b[0] + 2, b[3] + 2), f"{names[int(c)]} {s:.2f}", fill=(230, 30, 30), font=font,
               stroke_width=1, stroke_fill=(0, 0, 0))
    for fn in m["fns"]:  # dashed-ish yellow marker on misses
        b = fn["box"]
        d.rectangle([b[0] - lw * 2, b[1] - lw * 2, b[2] + lw * 2, b[3] + lw * 2], outline=(255, 210, 0), width=lw)
    im.save(out, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images-dir", help="evaluate this YOLO folder instead of a data.yaml split")
    ap.add_argument("--tag", help="report sub-folder name (default: split name)")
    ap.add_argument("--conf", type=float, default=None, help="operating threshold (default: configs/classes.yaml)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed for a match")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-failures", type=int, default=30)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    from ultralytics import RTDETR

    names = load_names(args.data)
    nc = len(names)
    conf = args.conf
    if conf is None:
        conf = yaml.safe_load(Path("configs/classes.yaml").read_text())["thresholds"]["operating_conf"]
    tag = args.tag or (Path(args.images_dir).name if args.images_dir else args.split)
    out = Path(args.out) / tag
    (out / "failures").mkdir(parents=True, exist_ok=True)

    data_yaml, split = args.data, args.split
    if args.images_dir:  # wrap the extra folder in a temporary data.yaml
        d = Path(args.images_dir).resolve()
        img_dir = d / "images" if (d / "images").exists() else d
        tmp = Path(tempfile.mkdtemp()) / "extra.yaml"
        tmp.write_text(yaml.safe_dump({"path": str(img_dir), "train": ".", "val": ".", "test": ".",
                                       "names": names}))
        data_yaml, split = str(tmp), "test"

    model = RTDETR(args.weights)
    missing = {i: n for i, n in names.items() if model.names.get(i) != n}
    if missing:
        raise SystemExit(f"class names in weights {model.names} don't match {args.data}")

    # ---- A. COCO-style metrics via Ultralytics -------------------------------------------------
    val = model.val(data=data_yaml, split=split, imgsz=args.imgsz, conf=0.001, device=args.device,
                    plots=True, project=str(out.resolve()), name="ultralytics", exist_ok=True, verbose=False)
    box = val.box
    ap50 = dict(zip(map(int, box.ap_class_index), map(float, box.ap50)))
    ap5095 = dict(zip(map(int, box.ap_class_index), map(float, box.ap)))

    # ---- B. Operating-point analysis ------------------------------------------------------------
    images = split_images(data_yaml, split)
    recs = predict_all(model, images, args.imgsz, args.device, min_conf=min(SWEEP))

    sweep_rows, best_f1 = [], (-1.0, conf)
    for t in SWEEP:
        tp, fp, fn, _ = aggregate(recs, t, args.iou, nc)
        p, r, f1 = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        sweep_rows.append({"threshold": t, "class": "all", "tp": sum(tp.values()), "fp": sum(fp.values()),
                           "fn": sum(fn.values()), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
        for c, n in names.items():
            pc, rc, fc = prf(tp[c], fp[c], fn[c])
            sweep_rows.append({"threshold": t, "class": n, "tp": tp[c], "fp": fp[c], "fn": fn[c],
                               "precision": round(pc, 4), "recall": round(rc, 4), "f1": round(fc, 4)})
        if f1 > best_f1[0]:
            best_f1 = (f1, t)

    tp, fp, fn, per_image = aggregate(recs, conf, args.iou, nc)
    per_class = []
    for c, n in names.items():
        p, r, f1 = prf(tp[c], fp[c], fn[c])
        per_class.append({"class": n, "gt_instances": tp[c] + fn[c], "tp": tp[c], "fp": fp[c], "fn": fn[c],
                          "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                          "ap50": round(ap50.get(c, float("nan")), 4), "ap50_95": round(ap5095.get(c, float("nan")), 4)})

    size_hits = defaultdict(lambda: [0, 0])  # (class, size) -> [found, total]
    for m in per_image:
        for c, s, found in m["gt_sizes"]:
            size_hits[(names[c], s)][1] += 1
            size_hits[(names[c], s)][0] += int(found)
            size_hits[("all", s)][1] += 1
            size_hits[("all", s)][0] += int(found)
    size_rows = [{"class": k[0], "size": k[1], "gt": v[1], "found": v[0],
                  "recall": round(v[0] / v[1], 4) if v[1] else None} for k, v in sorted(size_hits.items())]

    labels = [names[i] for i in range(nc)] + ["background"]
    cm = np.zeros((nc + 1, nc + 1), int)  # rows = true, cols = predicted
    for m in per_image:
        for t_cls, p_cls in m["confusion"]:
            cm[nc if t_cls is None else t_cls, nc if p_cls is None else p_cls] += 1

    err_rows = []
    for rec, m in zip(recs, per_image):
        for e in m["fps"]:
            err_rows.append({"image": Path(rec["image"]).name, "kind": "FP", "type": e["type"],
                             "class": names[e["pred_cls"]],
                             "true_class": names[e["true_cls"]] if e["true_cls"] is not None else "",
                             "conf": round(e["conf"], 3), "size": e["size"], "box": e["box"]})
        for e in m["fns"]:
            err_rows.append({"image": Path(rec["image"]).name, "kind": "FN", "type": e["type"],
                             "class": names[e["true_cls"]], "true_class": names[e["true_cls"]],
                             "conf": "", "size": e["size"], "box": e["box"]})

    ranked = sorted(range(len(recs)), key=lambda i: -(len(per_image[i]["fps"]) + len(per_image[i]["fns"])))
    fail_rows = []
    for rank, i in enumerate(ranked[:args.top_failures], 1):
        m = per_image[i]
        if not m["fps"] and not m["fns"]:
            break
        name = f"{rank:02d}_{Path(recs[i]['image']).stem}.jpg"
        draw_failure(recs[i], m, names, out / "failures" / name, conf)
        fail_rows.append({"rank": rank, "file": name, "source_image": recs[i]["image"],
                          "fp": len(m["fps"]), "fn": len(m["fns"]),
                          "error_types": dict(Counter(e["type"] for e in m["fps"] + m["fns"])),
                          "notes_for_memo": ""})  # fill in: blur / occlusion / lighting / small / label noise...

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "per_class.csv", per_class)
    write_csv(out / "size_breakdown.csv", size_rows)
    write_csv(out / "threshold_sweep.csv", sweep_rows)
    write_csv(out / "errors.csv", err_rows)
    write_csv(out / "failures" / "index.csv", fail_rows)
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + row.tolist())

    P, R, F1 = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    summary = {
        "weights": args.weights, "data": data_yaml, "split": split, "images": len(recs),
        "gt_instances": int(sum(len(r["gt_cls"]) for r in recs)),
        "coco_map50": round(float(box.map50), 4), "coco_map50_95": round(float(box.map), 4),
        "operating_conf": conf, "match_iou": args.iou,
        "operating_precision": round(P, 4), "operating_recall": round(R, 4), "operating_f1": round(F1, 4),
        "f1_optimal_conf": best_f1[1], "f1_at_optimal_conf": round(best_f1[0], 4),
        "error_type_counts": dict(Counter(f"{e['kind']}:{e['type']}" for e in err_rows)),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nper class @ conf", conf)
    for r in per_class:
        print(f"  {r['class']:15s} P={r['precision']:.3f} R={r['recall']:.3f} AP50={r['ap50']:.3f} "
              f"AP50-95={r['ap50_95']:.3f} (n={r['gt_instances']})")
    print("\nconfusion (rows=true, cols=pred):")
    print("  " + " ".join(f"{l[:10]:>10s}" for l in ["", *labels]))
    for i, row in enumerate(cm):
        print("  " + f"{labels[i][:10]:>10s} " + " ".join(f"{v:10d}" for v in row))
    print(f"\nreports written to {out}/  (failures/ holds the worst images, GT vs predictions)")


if __name__ == "__main__":
    main()
