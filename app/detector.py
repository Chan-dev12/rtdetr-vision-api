"""Model loading, weights download, image decoding and inference."""
from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger("detector")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original-image pixels

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {"class_id": self.class_id, "class_name": self.class_name,
                "confidence": round(self.confidence, 4),
                "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)}}


class ImageError(ValueError):
    """Raised for uploads that are not usable images. Mapped to HTTP 4xx by the API."""


def decode_image(data: bytes, max_side: int) -> Image.Image:
    if not data:
        raise ImageError("empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                      # cheap integrity check (truncated/corrupt files)
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageError(f"not a readable image ({e.__class__.__name__})") from e
    if max(img.size) > max_side:
        raise ImageError(f"image too large: {img.size[0]}x{img.size[1]} (max side {max_side})")
    # Phone photos store rotation in EXIF. Without this the model sees a sideways image.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def ensure_weights(path: Path, url: str | None, sha256: str | None) -> Path:
    """Download the weights if they aren't on disk, and verify the checksum when one is given."""
    if not path.exists():
        if not url:
            raise FileNotFoundError(f"weights not found at {path} and MODEL_URL is not set")
        log.info("downloading weights from %s", url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    if sha256:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != sha256.lower():
            raise ValueError(f"weights checksum mismatch: expected {sha256}, got {h}")
    return path


class Detector:
    def __init__(self, weights: Path, device: str | None = None, imgsz: int = 640):
        from ultralytics import RTDETR  # imported here so the API module loads fast in tests

        self.weights = Path(weights)
        self.model = RTDETR(str(self.weights))
        self.names: dict[int, str] = dict(self.model.names)
        self.device = device
        self.imgsz = imgsz
        self._lock = threading.Lock()  # one forward pass at a time; the model is not re-entrant
        self._warmup()

    def _warmup(self) -> None:
        t = time.perf_counter()
        self.predict(Image.new("RGB", (self.imgsz, self.imgsz)), conf=0.5)
        log.info("model warm-up done in %.0f ms (classes=%s)", (time.perf_counter() - t) * 1000, self.names)

    def predict(self, image: Image.Image, conf: float) -> list[Detection]:
        with self._lock:
            result = self.model.predict(image, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        boxes = result.boxes
        out = [Detection(int(c), self.names[int(c)], float(s), tuple(float(v) for v in b))
               for c, s, b in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
        return sorted(out, key=lambda d: -d.confidence)
