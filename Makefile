# Every step of the pipeline, in order. `make all` reproduces the full run.
PY ?= python
WEIGHTS ?= weights/best.pt

.PHONY: data audit train eval serve test all

data:            ## merge sources, dedupe, leak-free split -> data/processed
	$(PY) scripts/prepare_data.py --config configs/dataset.yaml
audit:           ## class balance + object sizes -> reports/data_audit.md
	$(PY) scripts/audit_data.py --data data/processed/data.yaml --out reports/data_audit.md
train:           ## fine-tune RT-DETR -> weights/best.pt + run_summary.json
	$(PY) scripts/train.py --config configs/train.yaml
eval:            ## test-split metrics, threshold sweep, failure images -> reports/test/
	$(PY) scripts/evaluate.py --weights $(WEIGHTS) --data data/processed/data.yaml --split test
serve:           ## run the API on :8000
	uvicorn app.main:app --host 0.0.0.0 --port 8000
test:            ## unit + API tests (no weights or API key needed)
	$(PY) -m pytest -q
all: data audit train eval
