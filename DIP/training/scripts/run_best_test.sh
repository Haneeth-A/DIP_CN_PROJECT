#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
BEST_METRICS="${BEST_METRICS:-results/ensembles/search_eval/best_metrics.json}"
MANIFEST="${MANIFEST:-data/manifests/quality_features.csv}"
OUT_DIR="${OUT_DIR:-results/test_final/search_eval}"
SPLIT="${SPLIT:-phase2_test}"

if [ ! -f "$BEST_METRICS" ]; then
  echo "Missing metrics file: $BEST_METRICS"
  exit 1
fi

mapfile -t info < <("$PYTHON_BIN" - "$BEST_METRICS" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    metrics = json.load(f)
for name, weight in zip(metrics["members"], metrics["weights"]):
    run = name[:-len("_eval_tta-full")] if name.endswith("_eval_tta-full") else name
    print(f"checkpoints/{run}/best.pt")
    print(f"{weight:.8f}")
PY
)

checkpoints=()
weights=()
for ((i = 0; i < ${#info[@]}; i += 2)); do
  checkpoint="${info[$i]}"
  weight="${info[$((i + 1))]}"
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
  checkpoints+=("$checkpoint")
  weights+=("$weight")
done

weight_arg="$(IFS=,; echo "${weights[*]}")"
echo "Testing ${#checkpoints[@]} models on $SPLIT"
"$PYTHON_BIN" scripts/ensemble_eval.py \
  --manifest "$MANIFEST" \
  --checkpoints "${checkpoints[@]}" \
  --weights "$weight_arg" \
  --split "$SPLIT" \
  --folder pristine,degraded \
  --tta full \
  --out-dir "$OUT_DIR"
