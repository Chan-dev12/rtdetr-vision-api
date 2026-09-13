"""Fine-tune RT-DETR (Ultralytics implementation) and record everything needed to reproduce the run.

Writes <run_dir>/run_summary.json (and a copy next to the exported weights) containing:
hyperparameters, seed, git commit, library versions, GPU/CPU, wall-clock training time,
final metrics and the sha256 of the weights.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --set epochs=1 imgsz=320 batch=2   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_sets(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = yaml.safe_load(v)  # "1" -> 1, "true" -> True, "null" -> None
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment() -> dict:
    import ultralytics

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": gpus,
        "ultralytics": ultralytics.__version__,
        "git_commit": git_commit(),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()

    from ultralytics import RTDETR

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.update(parse_sets(args.set))
    hyp = dict(cfg)
    base_model = cfg.pop("model")
    export_to = Path(cfg.pop("export_weights", "weights/best.pt"))

    seed_everything(int(cfg.get("seed", 42)))
    env = environment()
    print(json.dumps(env, indent=2))

    model = RTDETR(base_model)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    model.train(**cfg)
    train_seconds = time.perf_counter() - t0

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    export_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_to)

    metrics = {k: float(v) for k, v in (getattr(model.trainer, "metrics", None) or {}).items()
               if isinstance(v, (int, float, np.floating))}
    summary = {
        "started_utc": started.isoformat(timespec="seconds"),
        "train_time_seconds": round(train_seconds, 1),
        "train_time_human": time.strftime("%Hh %Mm %Ss", time.gmtime(train_seconds)),
        "epochs_completed": int(getattr(model.trainer, "epoch", -1)) + 1,
        "config_file": args.config,
        "hyperparameters": hyp,
        "environment": env,
        "run_dir": str(save_dir),
        "weights": str(export_to),
        "weights_sha256": sha256(export_to),
        "final_val_metrics": metrics,
    }
    for p in (save_dir / "run_summary.json", export_to.with_name("run_summary.json")):
        p.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("train_time_human", "epochs_completed", "weights",
                                              "weights_sha256", "final_val_metrics")}, indent=2))


if __name__ == "__main__":
    main()
