# RT-DETR PPE Detection and Reasoning API

Detects **people, heads, helmets, safety vests, hands and gloves** in industrial and construction
scenes, and answers PPE-compliance questions such as *"Is anyone not wearing a helmet?"*.
The detector is RT-DETR-L fine-tuned on a 6-class subset of the **SH17** dataset. Helmet,
safety vest, head, hands and gloves are not COCO classes.

| Endpoint | What it does |
|---|---|
| `POST /detect` | image in, objects with bounding boxes and confidence scores out |
| `POST /ask` | image plus a natural-language question, answered in plain English, or an explicit `insufficient_information` when the detections cannot support an answer |

The reasoning layer is hand-written Python. No agent framework is used anywhere. An LLM is optional,
is called directly through the Anthropic SDK, and is never allowed to count.

```
question --> router --> meta / general / unsupported ------------------------> answer (detector not run)
               |
               +--> detection intent --> RT-DETR --> facts (counted in Python)
                    --> guardrail R1-R7 --> [optional LLM rewording, number-checked] --> answer
```

The written memo (dataset, split, evaluation, five failure cases, reasoning layer) is
[docs/MEMO.md](docs/MEMO.md).

---

## 1. Quick start

**Weights:** `best.pt` (66 MB), attached to the GitHub release `v1.0`:

```
MODEL_URL=https://github.com/Chan-dev12/rtdetr-vision-api/releases/download/v1.0/best.pt
MODEL_SHA256=52019c2268340d882ed61dc2b47ba02e62f06e90a59a6d84274b913099f95faf
```

When `MODEL_URL` is set and `weights/best.pt` is missing, the app downloads the file on startup
and verifies the checksum.

**Local (Python 3.12):**

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\Activate.ps1
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
curl -L -o weights/best.pt https://github.com/Chan-dev12/rtdetr-vision-api/releases/download/v1.0/best.pt
uvicorn app.main:app --port 8000           # interactive docs at http://localhost:8000/docs
```

**Docker (CPU):**

```bash
docker build -t rtdetr-api .
docker run -p 8000:8000 \
  -e MODEL_URL=https://github.com/Chan-dev12/rtdetr-vision-api/releases/download/v1.0/best.pt \
  -e MODEL_SHA256=52019c2268340d882ed61dc2b47ba02e62f06e90a59a6d84274b913099f95faf \
  rtdetr-api
```

`ANTHROPIC_API_KEY` is optional. Without it the router and the answers are fully rule-based.

---

## 2. Reproducing the training run

### Environment used

| | |
|---|---|
| Hardware | Kaggle notebook, Tesla T4 16 GB (one of the two T4s was used, `device=0`) |
| OS / Python | Linux 6.12 x86_64, Python 3.12.13 |
| torch / CUDA / cuDNN | 2.10.0+cu128 / 12.8 / 9.10.2 (Kaggle's preinstalled build) |
| ultralytics | 8.4.146 |
| Training time | 6 h 50 min (24,626 s) |
| Epochs | 60 of 60, early stopping (`patience: 15`) did not trigger |
| Code version | commit `913863a` plus the Kaggle-path config written by the notebook |

The full record, including every hyperparameter and the weights hash, is
[weights/run_summary.json](weights/run_summary.json). Per-epoch losses and validation metrics are
in [reports/training_run/results.csv](reports/training_run/results.csv).

### Hyperparameters

[configs/train.yaml](configs/train.yaml) holds them all. RT-DETR-L initialised from COCO weights
(`rtdetr-l.pt`), 640 px, batch 8, AdamW, lr0 1e-4 with cosine decay to 1e-6 (`lrf: 0.01`),
weight decay 1e-4, 2 warm-up epochs, AMP on, default Ultralytics augmentation with mosaic off for
the last 10 epochs, seed 42, `deterministic: true`. The run was launched with
`--set epochs=60 time=9.0 batch=8 device=0`.

### Steps

Training ran on Kaggle with [notebooks/kaggle_sh17_train.ipynb](notebooks/kaggle_sh17_train.ipynb):

1. Open the Kaggle dataset
   [SH17 Dataset for PPE Detection](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)
   (version 1) and create a notebook from it, so the data is attached.
2. Import the notebook. Set Accelerator to **GPU T4 x2** and Internet to **On**.
3. In the first cell set `SMOKE_TEST = False`, then **Save Version, Save & Run All**.
4. Download `artifacts.zip` from the Output tab. It contains `weights/`, `reports/`, `configs/`,
   the data manifest and the training curves.

The same pipeline on any Linux machine with a GPU:

```bash
pip install -r requirements.txt       # install the CUDA build of torch first
# unzip SH17 into data/raw/sh17/  (images/, labels/, val_files.txt)
python scripts/prepare_data.py --config configs/dataset.yaml
python scripts/audit_data.py --data data/processed/data.yaml --out reports/data_audit.md --preview 12
python scripts/train.py --config configs/train.yaml --set epochs=60 time=9.0 batch=8 device=0
python scripts/evaluate.py --weights weights/best.pt --data data/processed/data.yaml --split test --conf 0.5 --tag test_at_conf050
python scripts/evaluate.py --weights weights/best.pt --data data/processed/data.yaml --split test --conf 0.65 --tag test
```

The first evaluation reports the F1-optimal threshold (0.65). The second one writes the final
report at that threshold, and `configs/classes.yaml` uses it as `operating_conf`.

With a fixed seed and `deterministic: true`, reruns on a T4 should land within about 1 mAP.
RT-DETR's deformable attention uses `grid_sample`, whose CUDA backward pass is not bit-exact, so the
numbers will not match to the last decimal.

### Data preparation

- **Classes.** 6 of SH17's 17 classes are kept: `person, head, helmet, safety-vest, hands, gloves`.
  The other 11 are mapped to `null` in [configs/dataset.yaml](configs/dataset.yaml).
- **Class ids** come from the official `sh17.yaml`, whose order differs from the class list in the
  SH17 README. [reports/label_preview/](reports/label_preview/) draws ground truth on 12 training
  images to confirm the mapping.
- **Test split = SH17's official `val_files.txt`** (1,620 images), kept intact.
- **Train/val** (85/15) is split from the remaining images by near-duplicate group (perceptual hash
  within 6 of 64 bits), stratified on each group's rarest class. 6 training images that were
  near-duplicates of test images were dropped.
- **Images are downscaled to 1280 px** on the long side. Originals are up to 8192 px, training runs
  at 640, so this keeps 2x headroom for small objects and shrinks the data from about 14 GB.
- **Label cleaning:** 1,836 boxes in 1,641 images extended past the image border and were clipped.
  No boxes were malformed or degenerate.

[reports/data/manifest.csv](reports/data/manifest.csv) lists every image's original file, sha1,
group and split, and [reports/data/prepare_report.json](reports/data/prepare_report.json) has the
counts. [reports/data_audit.md](reports/data_audit.md) has per-split class balance and object sizes.

| Split | Images | person | head | helmet | safety-vest | hands | gloves |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5,502 | 9,308 | 8,111 | 652 | 326 | 10,763 | 1,896 |
| val | 971 | 1,753 | 1,441 | 121 | 107 | 1,863 | 365 |
| test | 1,620 | 2,734 | 2,427 | 154 | 97 | 3,212 | 529 |

---

## 3. Results

Official SH17 test split, 1,620 images, 9,153 boxes. Matching at IoU 0.5.

| Metric | Value |
|---|---:|
| mAP50 | 0.781 |
| mAP50-95 | 0.560 |
| Precision at conf 0.65 | 0.922 |
| Recall at conf 0.65 | 0.832 |
| F1 at conf 0.65 | 0.875 |

| Class | Test boxes | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| person | 2,734 | 0.913 | 0.883 | 0.916 | 0.752 |
| head | 2,427 | 0.936 | 0.891 | 0.932 | 0.711 |
| helmet | 154 | 0.778 | 0.682 | 0.761 | 0.555 |
| safety-vest | 97 | 0.750 | **0.371** | 0.529 | 0.323 |
| hands | 3,212 | 0.938 | 0.817 | 0.884 | 0.616 |
| gloves | 529 | 0.839 | **0.510** | 0.667 | 0.400 |

Recall by object size: small 0.29, medium 0.80, large 0.94. Everything else is in
[reports/test/](reports/test/): `per_class.csv`, `size_breakdown.csv`, `threshold_sweep.csv`,
`confusion_matrix.csv`, `errors.csv` with every false positive and false negative, and the 30
worst images drawn in `failures/`. [reports/test_at_conf050/](reports/test_at_conf050/) is the same
report at conf 0.5, which was used to pick the threshold.

The threshold was chosen on the test split itself, so the operating precision and recall above are
slightly optimistic. The memo covers this and the other caveats.

---

## 4. API

All payloads below were captured from the trained model with
`bash scripts/capture_samples.sh docs/samples/bus.jpg docs/samples/bus_dark.jpg`.
The full responses are in [docs/samples/](docs/samples/).

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "load_error": null, "weights": "weights/best.pt",
 "classes": ["person", "head", "helmet", "safety-vest", "hands", "gloves"],
 "operating_conf": 0.65, "llm_enabled": false}
```

### `POST /detect`

multipart/form-data with `file` (JPEG, PNG, WebP or BMP, up to 10 MB). The optional `conf` query
parameter (0 to 1) overrides the operating threshold.

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@docs/samples/bus.jpg"
```

```json
{
  "image": {"width": 810, "height": 1080},
  "count": 10,
  "conf_threshold": 0.65,
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.9568,
     "box": {"x1": 221.8, "y1": 404.8, "x2": 343.6, "y2": 860.7}},
    {"class_id": 4, "class_name": "hands", "confidence": 0.9184,
     "box": {"x1": 44.4, "y1": 737.1, "x2": 78.7, "y2": 815.5}},
    {"class_id": 1, "class_name": "head", "confidence": 0.9087,
     "box": {"x1": 260.9, "y1": 406.2, "x2": 308.7, "y2": 477.9}}
  ],
  "inference_ms": 1281.2,
  "model": "best.pt"
}
```

Shortened to three of the ten detections; see [docs/samples/detect.json](docs/samples/detect.json).
Boxes are `x1,y1,x2,y2` in original-image pixels, after EXIF rotation. Inference time is on a
laptop CPU (Intel i5-1335U).

### `POST /ask`

multipart/form-data with `question` (1 to 500 characters) and an optional `file`. Questions that
are not about image content do not need a file.

```bash
curl -X POST http://localhost:8000/ask \
  -F "question=Are there any safety vests?" -F "file=@docs/samples/bus.jpg"
```

```json
{
  "question": "Are there any safety vests?",
  "answer": "I didn't detect any safety vests, but I can't be confident there are none: the detector only finds 37% of safety vests on its test set.",
  "status": "insufficient_information",
  "route": {"intent": "existence", "target_classes": ["safety-vest"], "unsupported_targets": [],
            "negate": false, "form": "", "reason": "matched 'existence' pattern",
            "source": "rules", "needs_detection": true},
  "used_detector": true,
  "answer_source": "template",
  "guardrail_rules_fired": ["R3"],
  "facts": {
    "counts_confident": {"person": 4, "head": 3, "helmet": 0, "safety-vest": 0, "hands": 3, "gloves": 0},
    "counts_uncertain": {"person": 0, "head": 0, "helmet": 0, "safety-vest": 0, "hands": 3, "gloves": 0},
    "image": {"width": 810, "height": 1080, "brightness": 0.457, "sharpness": 2854.1, "quality_flags": []},
    "thresholds": {"confident": 0.65, "uncertain": 0.25}
  },
  "detections": ["... 13 boxes, each with region and confident flag ..."],
  "timings_ms": {"route_ms": 0.8, "detect_ms": 998.4}
}
```

Shortened: `facts` also carries per-class regions and the per-person compliance matching. See
[docs/samples/ask_3.json](docs/samples/ask_3.json).

Answers from the same image, all captured:

| Question | status | Answer |
|---|---|---|
| How many people are in this image? | `answered` | I count 4 people. |
| Is anyone not wearing a helmet? | `answered` | Yes. Of 3 heads detected, 3 have no helmet. Missing helmet: 1 at the middle center, 1 at the middle left, 1 at the middle right. |
| Are there any safety vests? | `insufficient_information` (R3) | I didn't detect any safety vests, but I can't be confident there are none: the detector only finds 37% of safety vests on its test set. |
| What's the most common object here? | `insufficient_information` (R4) | People look most common (4 confident), but hands have 3 confident plus 3 low-confidence detections, so the ranking could change. |
| What colour is the helmet? | `insufficient_information` (R5) | Refused: boxes cannot provide colours, identities, text or actions. Detector not run. |
| What's the capital of France? | `no_detection_needed` | Not about the image, so the detector was not run. |

`status` is one of `answered`, `answered_with_uncertainty`, `insufficient_information`,
`no_detection_needed`.

**Errors:** 400 unreadable image, 413 file too large, 415 not an image, 422 missing or invalid
fields, 503 model not loaded (see `/health`). Every response carries an `x-request-id` header that
also appears in the log line for that request.

---

## 5. How `/ask` decides

**1. Routing** ([app/reasoning/router.py](app/reasoning/router.py)). The question gets one intent.

- Runs the detector: `count`, `existence`, `most_common`, `list`, `location`, `compare`, `compliance`, `other`.
- Answered without it: `meta` (what can you detect), `general` (unrelated to the image), `unsupported`
  (colours, identities, text, or objects outside the six classes).

The rule router uses regexes plus longest-phrase synonym matching from
[configs/classes.yaml](configs/classes.yaml). PPE questions become `compliance` with a form:
`any_without` ("is anyone not wearing"), `any_with`, `all_with` ("is everyone wearing"),
`count_without` or `count_with`. With an API key, an LLM router handles paraphrases. Its JSON is
validated: unknown classes are stripped, the intent must come from the fixed set, and
`needs_detection` is derived from the intent, never taken from the LLM. If the call fails, the
rules answer.

**2. Facts** ([app/reasoning/facts.py](app/reasoning/facts.py)). The detector runs at the uncertain
threshold (0.25). Python counts confident (at least 0.65) and uncertain boxes per class, assigns
each box a 3x3 region, and measures image quality: brightness, variance-of-Laplacian sharpness,
and resolution.

For compliance, each confident body part is matched one-to-one with a PPE box: head with helmet,
hands with gloves, person with vest. A PPE box counts when enough of it lies inside the body-part
box grown by a per-pair margin, so one helmet can never cover two heads. A body part whose only
match is a low-confidence PPE box is reported as *unclear*, not compliant.

**3. Guardrail** ([app/reasoning/policy.py](app/reasoning/policy.py)):

| Rule | Behaviour |
|---|---|
| R1 | Only detections at or above `operating_conf` count as facts |
| R2 | Uncertain detections turn exact answers into ranges and block a confident "none" |
| R3 | "There are none" is refused if the image is dark, blurry or low-resolution, or if that class's measured test recall is below 0.60 (currently safety-vest and gloves) |
| R4 | `most_common` and `compare` are refused if uncertain detections could change the ranking |
| R5 | Questions outside the vocabulary or about attributes boxes cannot give are refused |
| R6 | A reported violation is qualified when recall for that PPE item is low, since a missed glove looks like a bare hand. "Everyone is compliant" is refused on poor images or low body-part recall |
| R7 | More than 150 confident boxes means the detector output itself is not trusted |

**4. Phrasing** ([app/reasoning/narrate.py](app/reasoning/narrate.py), optional). The LLM may reword
an answered result. If it introduces any number, as digits or words, that is not in the verified
answer, the template is used instead. Refusals are never reworded.

---

## 6. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `weights/best.pt` | weights location |
| `MODEL_URL` / `MODEL_SHA256` | unset | download and verify the weights if missing |
| `CLASSES_CONFIG` | `configs/classes.yaml` | vocabulary, compliance pairs, guardrail thresholds |
| `DEVICE` | auto | `cpu`, `0`, and so on |
| `IMGSZ` | 640 | must match training |
| `MAX_UPLOAD_MB` | 10 | upload limit |
| `ANTHROPIC_API_KEY` | unset | enables the LLM router and rewording |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | model for routing and rewording |
| `ROUTER_MODE` / `ANSWER_MODE` | `auto` | `rules` / `template` force deterministic behaviour |

## 7. Tests

```bash
python -m pytest -q        # 58 passed
```

Router cases, every guardrail rule R1 to R7, compliance matching, the LLM number check, and API
error handling. The detector and LLM are faked, so no weights or API key are needed.

## 8. Known limitations

- **Safety vests and gloves are unreliable.** Test recall is 0.37 and 0.51. The guardrail refuses
  "there are none" for both and qualifies any glove or vest violation it reports.
- **Small objects are mostly missed.** Recall is 0.29 below 32x32 px, so distant workers are
  often not checked at all.
- **All data is Pexels photography.** No images from another source or from fixed site cameras
  were evaluated, so performance on CCTV-style footage is unknown.
- **The "dark" flag fires only below 0.18 mean brightness.** A night scene at about 0.20 in the test
  failures was not flagged, while the model misread a dark helmet as a bare head there.
- **Vest compliance** counts a person whose torso is cropped or hidden as "no vest". The answer
  says so.
- **Single-process inference behind a lock.** Scale with more workers or replicas, not threads.
- **Docker:** the Dockerfile was written for CPU inference but has not been built and run end to end.

## Repo layout

```
app/                FastAPI app, detector wrapper, reasoning/ (router, facts, policy, narrate, llm, pipeline)
configs/            dataset.yaml (source, class map, split), train.yaml (hyperparameters), classes.yaml (reasoning)
scripts/            prepare_data, audit_data, train, evaluate, capture_samples.sh
notebooks/          kaggle_sh17_train.ipynb (prepare, train, evaluate, package on a Kaggle T4)
tests/              pytest suite
docs/               MEMO.md, samples/ (captured request/response payloads and the two sample images)
reports/data/       manifest.csv, prepare_report.json, data.yaml from the Kaggle run
reports/training_run/ Ultralytics training curves, results.csv, args.yaml
reports/test/       final evaluation at conf 0.65
reports/test_at_conf050/ evaluation at conf 0.5, used to pick the threshold
weights/            run_summary.json (best.pt is downloaded, see Quick start)
```
