"""Helpers shared by audit_data.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# COCO size bins, measured on box area in ORIGINAL image pixels
SMALL, MEDIUM = 32 ** 2, 96 ** 2


def size_bucket(area: float) -> str:
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def load_names(data_yaml: str | Path) -> dict[int, str]:
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def split_images(data_yaml: str | Path, split: str) -> list[Path]:
    """Resolve a split exactly the way Ultralytics does, so our numbers use the same images as model.val()."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    if not data.get(split):
        raise SystemExit(f"split '{split}' not defined in {data_yaml}")
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    imgs: list[Path] = []
    for e in entries:
        p = Path(e)
        if p.is_dir():
            imgs += sorted(x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
        elif p.suffix == ".txt":
            base = Path(data["path"])
            imgs += [(base / ln.strip()).resolve() for ln in p.read_text().splitlines() if ln.strip()]
    return imgs


def label_path(img: Path) -> Path:
    from ultralytics.data.utils import img2label_paths

    return Path(img2label_paths([str(img)])[0])


def load_gt(img: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (classes[int], boxes[N,4] xyxy in pixels) for one image."""
    txt = label_path(img)
    if not txt.exists():
        return np.zeros(0, int), np.zeros((0, 4))
    rows = [r.split() for r in txt.read_text().splitlines() if r.strip()]
    rows = [r for r in rows if len(r) == 5]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, dtype=float)
    cx, cy, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return a[:, 0].astype(int), boxes


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
