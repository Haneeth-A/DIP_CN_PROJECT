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
MANIFEST="${MANIFEST:-data/manifests/quality_features_folds.csv}"
RUN_PREFIX="${RUN_PREFIX:-cv_b0_focal_eff}"
EPOCHS="${EPOCHS:-10}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOSS="${LOSS:-focal}"
SAMPLER="${SAMPLER:-effective}"
SEED="${SEED:-42}"
FOLDS=(0 1 2 3 4)

if [ ! -f "$MANIFEST" ]; then
  echo "Missing fold manifest: $MANIFEST"
  echo "Run scripts/make_random_folds.py first."
  exit 1
fi

mkdir -p logs results/probs results/ensembles

is_complete() {
  local run="$1"
  if [ -f "checkpoints/$run/best.pt" ] && [ -f "results/$run/history.csv" ]; then
    local lines
    lines=$(wc -l < "results/$run/history.csv")
    if [ "$lines" -ge "$((EPOCHS + 1))" ]; then
      return 0
    fi
  fi
  return 1
}

for fold in "${FOLDS[@]}"; do
  run="${RUN_PREFIX}_fold${fold}"
  train_folds=()
  for other in "${FOLDS[@]}"; do
    if [ "$other" -ne "$fold" ]; then
      train_folds+=("$other")
    fi
  done
  train_fold_arg="$(IFS=,; echo "${train_folds[*]}")"

  if is_complete "$run"; then
    echo "Skipping complete run: $run"
  else
    echo "Training $run on folds $train_fold_arg, validating on $fold"
    "$PYTHON_BIN" scripts/train.py \
      --manifest "$MANIFEST" \
      --train-split phase1_train,phase2_train \
      --train-folder pristine,degraded \
      --val-split phase1_train,phase2_train \
      --val-folder pristine,degraded \
      --train-fold "$train_fold_arg" \
      --val-fold "$fold" \
      --model efficientnet_b0 \
      --img-size "$IMG_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --loss "$LOSS" \
      --sampler "$SAMPLER" \
      --seed "$SEED" \
      --run-name "$run" 2>&1 | tee "logs/${run}.log"
  fi

  prob_dir="results/probs/${run}_eval_tta-full"
  if [ ! -f "$prob_dir/probabilities.npy" ]; then
    echo "Saving TTA probabilities for $run"
    "$PYTHON_BIN" scripts/ensemble_eval.py \
      --manifest "$MANIFEST" \
      --checkpoints "checkpoints/$run/best.pt" \
      --split phase2_eval \
      --folder pristine,degraded \
      --tta full \
      --save-probs \
      --out-dir "$prob_dir" 2>&1 | tee "logs/${run}_probs.log"
  fi
done

checkpoints=()
for fold in "${FOLDS[@]}"; do
  checkpoints+=("checkpoints/${RUN_PREFIX}_fold${fold}/best.pt")
done

echo "Evaluating 5-fold ensemble"
"$PYTHON_BIN" scripts/ensemble_eval.py \
  --manifest "$MANIFEST" \
  --checkpoints "${checkpoints[@]}" \
  --split phase2_eval \
  --folder pristine,degraded \
  --tta full \
  --save-probs \
  --out-dir "results/ensembles/${RUN_PREFIX}_5fold_tta-full" 2>&1 | tee "logs/${RUN_PREFIX}_5fold_ensemble.log"

echo "Done: results/ensembles/${RUN_PREFIX}_5fold_tta-full"
