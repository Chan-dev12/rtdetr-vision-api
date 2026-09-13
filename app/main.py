"""FastAPI service: /detect (Part A) and /ask (Part B).

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from .config import settings
from .detector import Detector, ImageError, decode_image, ensure_weights
from .reasoning.llm import LLMClient
from .reasoning.pipeline import Reasoner
from .reasoning.vocab import Vocabulary

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}


class State:
    detector: Detector | None = None
    reasoner: Reasoner | None = None
    vocab: Vocabulary | None = None
    load_error: str | None = None


state = State()


def build_state(detector=None, classes_config=None) -> None:
    """Load config, model and reasoner. `detector`/`classes_config` can be injected (tests use fakes)."""
    state.vocab = Vocabulary.load(classes_config or settings.classes_config)
    try:
        if detector is None:
            ensure_weights(settings.model_path, settings.model_url, settings.model_sha256)
            detector = Detector(settings.model_path, settings.device, settings.imgsz)
        state.detector = detector
    except Exception as e:  # keep serving /health so the failure is visible instead of a crash loop
        state.load_error = f"{e.__class__.__name__}: {e}"
        log.exception("model failed to load")
        return
    missing = set(state.vocab.names) - set(detector.names.values())
    if missing:
        log.warning("classes.yaml lists %s which the model doesn't know (model classes: %s)",
                    sorted(missing), list(detector.names.values()))
    llm = LLMClient(settings.llm_model)
    log.info("LLM %s (router=%s, answers=%s)", "enabled: " + settings.llm_model if llm.available else "disabled",
             settings.router_mode, settings.answer_mode)
    state.reasoner = Reasoner(state.vocab, detector, llm, settings.router_mode, settings.answer_mode)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.detector is None:
        build_state()
    yield


app = FastAPI(title="RT-DETR Detection & Reasoning API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s", rid)
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
    response.headers["x-request-id"] = rid
    log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, response.status_code,
             (time.perf_counter() - t) * 1000)
    return response


# ---------------------------------------------------------------------------------------------------
class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: Box


class DetectResponse(BaseModel):
    image: dict[str, int]
    count: int
    conf_threshold: float
    detections: list[DetectionOut]
    inference_ms: float
    model: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str = Field(description="answered | answered_with_uncertainty | insufficient_information | "
                                    "no_detection_needed")
    route: dict[str, Any]
    used_detector: bool
    answer_source: str
    guardrail_rules_fired: list[str]
    facts: dict[str, Any] | None
    detections: list[dict[str, Any]] | None
    timings_ms: dict[str, float]


def require_ready() -> None:
    if state.detector is None or state.reasoner is None:
        raise HTTPException(503, f"model not loaded: {state.load_error or 'starting up'}")


async def read_image(file: UploadFile) -> Image.Image:
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported content type {file.content_type}; send a JPEG/PNG/WebP/BMP image")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
    try:
        return decode_image(data, settings.max_image_side)
    except ImageError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if state.detector else "degraded",
            "model_loaded": state.detector is not None,
            "load_error": state.load_error,
            "weights": str(settings.model_path),
            "classes": list(state.detector.names.values()) if state.detector else None,
            "operating_conf": state.vocab.thresholds.operating_conf if state.vocab else None,
            "llm_enabled": bool(state.reasoner and state.reasoner.llm and state.reasoner.llm.available)}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(..., description="image file"),
                 conf: float | None = Query(None, ge=0.0, le=1.0,
                                            description="confidence threshold (default: operating_conf)")):
    require_ready()
    image = await read_image(file)
    threshold = conf if conf is not None else state.vocab.thresholds.operating_conf
    t = time.perf_counter()
    dets = state.detector.predict(image, conf=threshold)
    return {"image": {"width": image.width, "height": image.height}, "count": len(dets),
            "conf_threshold": threshold, "detections": [d.to_dict() for d in dets],
            "inference_ms": round((time.perf_counter() - t) * 1000, 1), "model": state.detector.weights.name}


@app.post("/ask", response_model=AskResponse)
async def ask(question: str = Form(..., min_length=1, max_length=500),
              file: UploadFile | None = File(None, description="image (optional for questions not about an image)")):
    require_ready()
    image = await read_image(file) if file is not None and file.filename else None
    return state.reasoner.ask(question.strip(), image)
