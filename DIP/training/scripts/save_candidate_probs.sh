#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <run_name> [run_name ...]"
  exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
MANIFEST="${MANIFEST:-data/manifests/quality_features.csv}"
mkdir -p logs results/probs

for run in "$@"; do
  checkpoint="checkpoints/$run/best.pt"
  out_dir="results/probs/${run}_eval_tta-full"
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
  if [ -f "$out_dir/probabilities.npy" ]; then
    echo "Skipping existing probabilities: $out_dir"
    continue
  fi
  echo "Saving TTA probabilities for $run"
  "$PYTHON_BIN" scripts/ensemble_eval.py \
    --manifest "$MANIFEST" \
    --checkpoints "$checkpoint" \
    --split phase2_eval \
    --folder pristine,degraded \
    --tta full \
    --save-probs \
    --out-dir "$out_dir" 2>&1 | tee "logs/${run}_probs.log"
done
