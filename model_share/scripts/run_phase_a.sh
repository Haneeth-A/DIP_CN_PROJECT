#!/usr/bin/env bash
set -euo pipefail

# Phase A only: export FP32 ONNX, build calibration IDs, quantize INT8,
# run the 5-image smoke test. Does NOT touch phase2_eval or phase2_test.
#
# Usage (from anywhere):
#   bash models/edge_quant/run_phase_a.sh
#
# Calibration size is automatic:
#   1024 normally, 512 if Cellpose >=300% CPU or 1-min load >=50% of CPUs.
#
# Optional overrides:
#   CALIB_SIZE=512 bash models/edge_quant/run_phase_a.sh
#   AUTO_CALIB=0 bash models/edge_quant/run_phase_a.sh
#   CELLPOSE_HEAVY_PCT=250 LOAD_HEAVY_FRACTION=0.4 bash models/edge_quant/run_phase_a.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

EDGE_DIR="$SCRIPT_DIR"
CKPT="DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt"
FOG1="DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt"
FOG2="DIP/training/checkpoints/effb0_224_bs64_mixed_eff/best.pt"
FOG3="DIP/training/checkpoints/effb0_224_bs64_mixed/best.pt"
MANIFEST="DIP/training/data/manifests/quality_features.csv"
DATA_ROOT="DATA"

CALIB_SIZE_OVERRIDE="${CALIB_SIZE:-}"
CALIB_SIZE=""
CALIB_SEED="${CALIB_SEED:-0}"
OPSET="${OPSET:-17}"
ONNX_VERSION="${ONNX_VERSION:-1.23.2}"
ORT_VERSION="${ORT_VERSION:-1.23.2}"
AUTO_CALIB="${AUTO_CALIB:-1}"
CELLPOSE_HEAVY_PCT="${CELLPOSE_HEAVY_PCT:-300}"
LOAD_HEAVY_FRACTION="${LOAD_HEAVY_FRACTION:-0.5}"
TRAIN_VENV="$PROJECT_ROOT/DIP/training/.venv/bin/python"
TOOL_VENV="/tmp/edge_quant_venv"
ONNX_SHIM="/tmp/edge_quant_onnx"

detect_calib_size() {
  if [ -n "$CALIB_SIZE_OVERRIDE" ]; then
    CALIB_SIZE="$CALIB_SIZE_OVERRIDE"
    echo "[auto-calib] manual override -> CALIB_SIZE=$CALIB_SIZE"
    return
  fi
  if [ "$AUTO_CALIB" != "1" ]; then
    CALIB_SIZE=1024
    echo "[auto-calib] AUTO_CALIB=$AUTO_CALIB -> CALIB_SIZE=$CALIB_SIZE"
    return
  fi

  local nproc load1 cellpose_cpu reason
  nproc="$(nproc)"
  load1="$(awk '{print $1}' /proc/loadavg)"
  cellpose_cpu="$(
    ps -eo pcpu=,comm=,args= | awk '
      tolower($0) ~ /(cellpose|segment_dataset)/ &&
      $0 !~ /awk|grep|opencode|run_phase_a/ {sum += $1}
      END {printf "%.1f", sum+0}
    '
  )"

  if awk -v c="$cellpose_cpu" -v t="$CELLPOSE_HEAVY_PCT" \
      'BEGIN {exit !(c >= t)}'; then
    CALIB_SIZE=512
    reason="cellpose_cpu=${cellpose_cpu}% >= ${CELLPOSE_HEAVY_PCT}%"
  elif awk -v l="$load1" -v n="$nproc" -v f="$LOAD_HEAVY_FRACTION" \
      'BEGIN {exit !(l >= n*f)}'; then
    CALIB_SIZE=512
    reason="load1=${load1} >= ${LOAD_HEAVY_FRACTION}*${nproc}"
  else
    CALIB_SIZE=1024
    reason="cellpose_cpu=${cellpose_cpu}% < ${CELLPOSE_HEAVY_PCT}% and load1=${load1} < ${LOAD_HEAVY_FRACTION}*${nproc}"
  fi

  echo "[auto-calib] nproc=$nproc load1=$load1 cellpose_cpu=${cellpose_cpu}%"
  echo "[auto-calib] thresholds: cellpose>=${CELLPOSE_HEAVY_PCT}%, load>=${LOAD_HEAVY_FRACTION}*nproc"
  echo "[auto-calib] reason: $reason -> CALIB_SIZE=$CALIB_SIZE"
}

echo "=== Phase A: preflight ==="
echo "project_root=$PROJECT_ROOT"
echo "edge_dir=$EDGE_DIR"
echo "calibration_size=auto (set CALIB_SIZE to override) seed=$CALIB_SEED opset=$OPSET"

for path in "$CKPT" "$FOG1" "$FOG2" "$FOG3" "$MANIFEST" "$DATA_ROOT"; do
  if [ ! -e "$path" ]; then
    echo "MISSING REQUIRED PATH: $path"
    exit 1
  fi
done
if [ ! -x "$TRAIN_VENV" ]; then
  echo "MISSING TRAINING VENV PYTHON: $TRAIN_VENV"
  exit 1
fi
for script in quantize.py predict.py smoke_test.py; do
  if [ ! -f "$EDGE_DIR/$script" ]; then
    echo "MISSING SCRIPT: $EDGE_DIR/$script"
    exit 1
  fi
done

echo "=== Phase A: tooling setup ==="
echo "[tooling] pinning onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION"
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
if [ ! -x "$TOOL_VENV/bin/python" ]; then
  uv venv --python 3.10 "$TOOL_VENV"
fi
uv pip install --python "$TOOL_VENV/bin/python" \
  "onnx==$ONNX_VERSION" "onnxruntime==$ORT_VERSION" \
  "numpy==2.2.6" "pillow==12.3.0"
rm -rf "$ONNX_SHIM"
uv pip install --python "$TOOL_VENV/bin/python" \
  --target "$ONNX_SHIM" --no-deps "onnx==$ONNX_VERSION" "ml_dtypes==0.6.0"

# The export runs under the Python 3.10 training venv. Verify the shim ABI
# matches; if not, reinstall with the uv-managed Python 3.10 pip.
if ! PYTHONPATH="$ONNX_SHIM" "$TRAIN_VENV" -c "import onnx" >/dev/null 2>&1; then
  echo "[tooling] onnx shim ABI mismatch; reinstalling with Python 3.10 pip"
  BASE310="$(uv python find 3.10)"
  rm -rf "$ONNX_SHIM"
  "$BASE310" -m pip install --target "$ONNX_SHIM" --no-deps --upgrade \
    "onnx==$ONNX_VERSION" "ml_dtypes==0.6.0"
fi
PYTHONPATH="$ONNX_SHIM" "$TRAIN_VENV" -c \
  "import onnx; print('[tooling] onnx', onnx.__version__, 'OK under training venv')"

echo "=== Stage 1/3: export FP32 ONNX ==="
PYTHONPATH="$ONNX_SHIM" "$TRAIN_VENV" -u "$EDGE_DIR/quantize.py" \
  --stage export \
  --checkpoint "$CKPT" \
  --fog-checkpoints "$FOG1" "$FOG2" "$FOG3" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --out-dir "$EDGE_DIR" \
  --opset "$OPSET"

echo "=== Selecting calibration size ==="
detect_calib_size

echo "=== Stage 2/3: calibrate + quantize INT8 ==="
"$TOOL_VENV/bin/python" -u "$EDGE_DIR/quantize.py" \
  --stage quantize \
  --checkpoint "$CKPT" \
  --fog-checkpoints "$FOG1" "$FOG2" "$FOG3" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --out-dir "$EDGE_DIR" \
  --calib-size "$CALIB_SIZE" \
  --seed "$CALIB_SEED" \
  --calib-moving-average

echo "=== Stage 3/3: smoke test (seed 0, 1 CPU thread) ==="
"$TOOL_VENV/bin/python" -u "$EDGE_DIR/smoke_test.py" \
  --images 5 --seed 0 --threads 1 \
  --calibration-csv "$EDGE_DIR/calibration_ids.csv" \
  --model "$EDGE_DIR/model_int8.onnx"

echo "=== Stage 3/3b: smoke test (seed 1, 1 CPU thread) ==="
"$TOOL_VENV/bin/python" -u "$EDGE_DIR/smoke_test.py" \
  --images 5 --seed 1 --threads 1 \
  --calibration-csv "$EDGE_DIR/calibration_ids.csv" \
  --model "$EDGE_DIR/model_int8.onnx"

echo "=== Phase A complete ==="
ls -lh "$EDGE_DIR"
sha256sum "$EDGE_DIR/model_fp32.onnx" "$EDGE_DIR/model_int8.onnx"
