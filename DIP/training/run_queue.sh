#!/usr/bin/env bash
cd "/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP/DIP/training" || exit 1
source .venv/bin/activate
export TQDM_DISABLE=1

echo "=== START $(date) ==="

echo "=== 288/64 mixed (15 epochs) $(date) ==="
python scripts/train.py \
  --train-split phase1_train,phase2_train \
  --train-folder pristine,degraded \
  --val-split phase2_eval \
  --val-folder pristine,degraded \
  --img-size 288 --batch-size 64 --lr 5e-4 \
  --epochs 15 --sampler sqrt --loss weighted_ce \
  --run-name effb0_288_bs64_mixed --num-workers 4

echo "=== 224/64 mixed e10 $(date) ==="
python scripts/train.py \
  --train-split phase1_train,phase2_train \
  --train-folder pristine,degraded \
  --val-split phase2_eval \
  --val-folder pristine,degraded \
  --img-size 224 --batch-size 64 --lr 5e-4 \
  --epochs 10 --sampler sqrt --loss weighted_ce \
  --run-name effb0_224_bs64_mixed_e10 --num-workers 4

echo "=== 224/64 mixed focal $(date) ==="
python scripts/train.py \
  --train-split phase1_train,phase2_train \
  --train-folder pristine,degraded \
  --val-split phase2_eval \
  --val-folder pristine,degraded \
  --img-size 224 --batch-size 64 --lr 5e-4 \
  --epochs 10 --sampler sqrt --loss focal \
  --run-name effb0_224_bs64_mixed_focal --num-workers 4

echo "=== 224/64 mixed effective sampler $(date) ==="
python scripts/train.py \
  --train-split phase1_train,phase2_train \
  --train-folder pristine,degraded \
  --val-split phase2_eval \
  --val-folder pristine,degraded \
  --img-size 224 --batch-size 64 --lr 5e-4 \
  --epochs 10 --sampler effective --loss weighted_ce \
  --run-name effb0_224_bs64_mixed_eff --num-workers 4

echo "=== DONE $(date) ==="
