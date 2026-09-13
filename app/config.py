"""Runtime settings. Everything comes from environment variables so Docker/VM deploys need no code change."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "weights/best.pt"))
    model_url: str | None = os.getenv("MODEL_URL") or None          # downloaded if model_path is missing
    model_sha256: str | None = os.getenv("MODEL_SHA256") or None    # verified after download
    classes_config: Path = Path(os.getenv("CLASSES_CONFIG", "configs/classes.yaml"))
    device: str | None = os.getenv("DEVICE") or None                # "cpu", "0", ... (None = auto)
    imgsz: int = int(os.getenv("IMGSZ", "640"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "10"))
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "8000"))  # reject absurd images (decompression bombs)
    # Part B
    llm_model: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    router_mode: str = os.getenv("ROUTER_MODE", "auto")   # auto | rules | llm
    answer_mode: str = os.getenv("ANSWER_MODE", "auto")   # auto | template | llm
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
