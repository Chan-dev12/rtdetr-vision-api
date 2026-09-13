#!/usr/bin/env bash
# Capture real request/response payloads from a running API into docs/samples/.
#
#   bash scripts/capture_samples.sh docs/samples/bus.jpg docs/samples/bus_dark.jpg
set -euo pipefail
API=${API:-http://localhost:8000}
IMG=${1:?usage: capture_samples.sh IMAGE [HARD_IMAGE]}
HARD=${2:-$IMG}
OUT=docs/samples
mkdir -p "$OUT"

curl -sf "$API/health" | python -m json.tool > "$OUT/health.json"
curl -sf -X POST "$API/detect" -F "file=@$IMG" | python -m json.tool > "$OUT/detect.json"

# One question per route: count, compliance, low-recall class (R3), unstable ranking (R4),
# an attribute boxes cannot provide (R5), and a question unrelated to the image.
questions=(
  "How many people are in this image?"
  "Is anyone not wearing a helmet?"
  "Are there any safety vests?"
  "What's the most common object here?"
  "What colour is the helmet?"
  "What's the capital of France?"
)
i=0
for q in "${questions[@]}"; do
  i=$((i + 1))
  curl -sf -X POST "$API/ask" -F "question=$q" -F "file=@$IMG" | python -m json.tool > "$OUT/ask_$i.json"
  python -c "import json;r=json.load(open('$OUT/ask_$i.json'));print(f'[{r[\"status\"]}] {r[\"question\"]}\n    -> {r[\"answer\"]}')"
done
# hard image: a darkened copy, where a negative answer has to be refused
curl -sf -X POST "$API/ask" -F "question=Are there any safety vests?" -F "file=@$HARD" \
  | python -m json.tool > "$OUT/ask_hard_image.json"
echo "saved to $OUT/"
