#!/usr/bin/env bash
set -euo pipefail

# Phase B only: evaluate FP32/INT8 on phase2_eval, measure CPU latency, run the
# FP32 fog top-3 TTA-full ensemble, and write the comparison report.
# phase2_test is never used.
#
# Usage (from anywhere):
#   bash models/edge_quant/run_phase_b.sh
#
# Optional overrides:
#   RUN_FOG=0 bash models/edge_quant/run_phase_b.sh
#   RUN_LATENCY=0 bash models/edge_quant/run_phase_b.sh
#   RUN_SMOKE=0 bash models/edge_quant/run_phase_b.sh
#   EVAL_THREADS=8 FOG_BATCH_SIZE=32 bash models/edge_quant/run_phase_b.sh
#   SKIP_EXISTING=1 bash models/edge_quant/run_phase_b.sh
#   LATENCY_RUNS=40 bash models/edge_quant/run_phase_b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

EDGE_DIR="$SCRIPT_DIR"
EVAL_ROOT="${EVAL_ROOT:-$EDGE_DIR/eval/phase2_eval}"
SMOKE_ROOT="${SMOKE_ROOT:-$EDGE_DIR/eval/smoke}"
MANIFEST="DIP/training/data/manifests/quality_features.csv"
DATA_ROOT="DATA"
TRAIN_PY="$PROJECT_ROOT/DIP/training/.venv/bin/python"
TRAINING_ROOT="$PROJECT_ROOT/DIP/training"
ENSEMBLE_EVAL="$TRAINING_ROOT/scripts/ensemble_eval.py"
EVAL_VENV="${EVAL_VENV:-/tmp/edge_quant_eval_venv}"
EVAL_PY="$EVAL_VENV/bin/python"

RUN_FOG="${RUN_FOG:-1}"
RUN_LATENCY="${RUN_LATENCY:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
EVAL_THREADS="${EVAL_THREADS:-4}"
LATENCY_THREADS="${LATENCY_THREADS:-1}"
WARMUP="${WARMUP:-5}"
LATENCY_RUNS="${LATENCY_RUNS:-20}"
FOG_BATCH_SIZE="${FOG_BATCH_SIZE:-64}"
FOG_NUM_WORKERS="${FOG_NUM_WORKERS:-4}"
SMOKE_IMAGES="${SMOKE_IMAGES:-50}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
ONNX_VERSION="${ONNX_VERSION:-1.23.2}"
ORT_VERSION="${ORT_VERSION:-1.23.2}"

FOG_RUNS=(
  "effb0_224_bs64_mixed_focal_reran"
  "effb0_224_bs64_mixed_eff"
  "effb0_224_bs64_mixed"
)

echo "=== Phase B: preflight ==="
echo "project_root=$PROJECT_ROOT"
echo "eval_root=$EVAL_ROOT"
echo "phase2_eval only; phase2_test is not used"

for path in \
  "$EDGE_DIR/model_fp32.onnx" \
  "$EDGE_DIR/model_int8.onnx" \
  "$EDGE_DIR/labels.json" \
  "$EDGE_DIR/preprocessing.json" \
  "$EDGE_DIR/evaluate_onnx.py" \
  "$EDGE_DIR/summarize_phase_b.py" \
  "$MANIFEST" \
  "$DATA_ROOT/pristine/phase2_eval" \
  "$DATA_ROOT/degraded/phase2_eval" \
  "$TRAIN_PY"; do
  if [ ! -e "$path" ]; then
    echo "MISSING REQUIRED PATH: $path"
    exit 1
  fi
done

if [ "$RUN_LATENCY" = "1" ] && [ ! -f "$EDGE_DIR/benchmark_latency.py" ]; then
  echo "MISSING REQUIRED PATH: $EDGE_DIR/benchmark_latency.py"
  exit 1
fi

if [ "$RUN_FOG" = "1" ]; then
  if [ ! -f "$ENSEMBLE_EVAL" ]; then
    echo "MISSING REQUIRED PATH: $ENSEMBLE_EVAL"
    exit 1
  fi
  for run in "${FOG_RUNS[@]}"; do
    ckpt="$TRAINING_ROOT/checkpoints/$run/best.pt"
    if [ ! -f "$ckpt" ]; then
      echo "MISSING FOG CHECKPOINT: $ckpt"
      exit 1
    fi
  done
fi

echo "=== Phase B: tooling setup ==="
echo "[tooling] pinning onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION"
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
if [ ! -x "$EVAL_PY" ]; then
  uv venv --python 3.10 "$EVAL_VENV"
fi
uv pip install --python "$EVAL_PY" \
  "onnx==$ONNX_VERSION" "onnxruntime==$ORT_VERSION" \
  "numpy==2.2.6" "pillow==12.3.0"
"$EVAL_PY" -c \
  "import onnx, onnxruntime, numpy, PIL; print('[tooling] eval venv OK', onnxruntime.__version__)"

mkdir -p "$EVAL_ROOT/edge_fp32" "$EVAL_ROOT/edge_int8" \
  "$EVAL_ROOT/fog_top3_tta_full" "$EVAL_ROOT/latency" \
  "$EVAL_ROOT/comparison"

if [ "$RUN_SMOKE" = "1" ]; then
  echo "=== Phase B: smoke test ($SMOKE_IMAGES phase2_eval images) ==="
  "$EVAL_PY" -u "$EDGE_DIR/evaluate_onnx.py" \
    --model "$EDGE_DIR/model_fp32.onnx" \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --split phase2_eval \
    --folder pristine,degraded \
    --out-dir "$SMOKE_ROOT/edge_fp32" \
    --threads "$LATENCY_THREADS" \
    --limit "$SMOKE_IMAGES" \
    --progress-every 10
  "$EVAL_PY" -u "$EDGE_DIR/evaluate_onnx.py" \
    --model "$EDGE_DIR/model_int8.onnx" \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --split phase2_eval \
    --folder pristine,degraded \
    --out-dir "$SMOKE_ROOT/edge_int8" \
    --threads "$LATENCY_THREADS" \
    --limit "$SMOKE_IMAGES" \
    --progress-every 10
fi

run_edge_eval() {
  local model="$1"
  local name="$2"
  local out="$EVAL_ROOT/$name"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/run_config.json" ]; then
    echo "[eval] skipping existing $name"
    return
  fi
  "$EVAL_PY" -u "$EDGE_DIR/evaluate_onnx.py" \
    --model "$model" \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --split phase2_eval \
    --folder pristine,degraded \
    --out-dir "$out" \
    --threads "$EVAL_THREADS" \
    --limit "$EVAL_LIMIT" \
    --progress-every 500
  if [ ! -f "$out/predictions.csv" ] || [ ! -f "$out/run_config.json" ]; then
    echo "MISSING EVALUATION OUTPUT: $out"
    exit 1
  fi
}

echo "=== Phase B: full FP32 evaluation ==="
run_edge_eval "$EDGE_DIR/model_fp32.onnx" "edge_fp32"

echo "=== Phase B: full INT8 evaluation ==="
run_edge_eval "$EDGE_DIR/model_int8.onnx" "edge_int8"

if [ "$RUN_LATENCY" = "1" ]; then
  echo "=== Phase B: CPU latency (batch 1, $LATENCY_THREADS thread, model-only) ==="
  "$EVAL_PY" -u "$EDGE_DIR/benchmark_latency.py" \
    --model "$EDGE_DIR/model_fp32.onnx" \
    --out-dir "$EVAL_ROOT/latency/edge_fp32" \
    --threads "$LATENCY_THREADS" \
    --warmup "$WARMUP" \
    --runs "$LATENCY_RUNS"
  "$EVAL_PY" -u "$EDGE_DIR/benchmark_latency.py" \
    --model "$EDGE_DIR/model_int8.onnx" \
    --out-dir "$EVAL_ROOT/latency/edge_int8" \
    --threads "$LATENCY_THREADS" \
    --warmup "$WARMUP" \
    --runs "$LATENCY_RUNS"
fi

if [ "$RUN_FOG" = "1" ]; then
  echo "=== Phase B: fog top-3 TTA-full evaluation ==="
  if [ -f "$EVAL_ROOT/fog_top3_tta_full/tau_0/predictions.csv" ] &&
     [ "$SKIP_EXISTING" = "1" ]; then
    echo "[fog] skipping existing fog ensemble"
  else
    rm -rf "$EVAL_ROOT/fog_top3_tta_full"
    mkdir -p "$EVAL_ROOT/fog_top3_tta_full"
    if ! "$TRAIN_PY" -u "$ENSEMBLE_EVAL" \
      --runs "${FOG_RUNS[@]}" \
      --data-root "$DATA_ROOT" \
      --manifest "$MANIFEST" \
      --split phase2_eval \
      --folder pristine,degraded \
      --tta full \
      --tau-grid 0 \
      --batch-size "$FOG_BATCH_SIZE" \
      --num-workers "$FOG_NUM_WORKERS" \
      --save-probs \
      --out-dir "$EVAL_ROOT/fog_top3_tta_full"; then
      echo "WARNING: fog ensemble failed; summary will report edge models only"
    fi
  fi
fi

echo "=== Phase B: summary ==="
SUMMARY_ARGS=(
  --manifest "$MANIFEST"
  --labels "$EDGE_DIR/labels.json"
  --split phase2_eval
  --folder pristine,degraded
  --edge-fp32-dir "$EVAL_ROOT/edge_fp32"
  --edge-int8-dir "$EVAL_ROOT/edge_int8"
  --fog-dir "$EVAL_ROOT/fog_top3_tta_full"
  --latency-dir "$EVAL_ROOT/latency"
  --fp32-model "$EDGE_DIR/model_fp32.onnx"
  --int8-model "$EDGE_DIR/model_int8.onnx"
  --out-dir "$EVAL_ROOT/comparison"
)
if [ "$EVAL_LIMIT" -gt 0 ]; then
  SUMMARY_ARGS+=(--allow-partial)
fi
"$TRAIN_PY" -u "$EDGE_DIR/summarize_phase_b.py" "${SUMMARY_ARGS[@]}"

echo "=== Phase B complete ==="
echo "report=$EVAL_ROOT/comparison/summary.md"
ls -lh "$EVAL_ROOT/comparison"
