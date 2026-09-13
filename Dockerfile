# Inference image for the API (CPU). Training is done outside the container, on a GPU.
#   docker build -t rtdetr-api .
#   docker run -p 8000:8000 -e MODEL_URL=<weights url> -e MODEL_SHA256=<sha256> rtdetr-api
# Or bake local weights in: put best.pt in weights/ before building.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics DEVICE=cpu

# libGL/glib: required by OpenCV (ultralytics can pull the non-headless build)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch first (~200 MB instead of several GB of CUDA libraries); pip then sees it as satisfied
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app/ app/
COPY configs/ configs/
COPY weights/ weights/
COPY reports/ reports/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,json,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
