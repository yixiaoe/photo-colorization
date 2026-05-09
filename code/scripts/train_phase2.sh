#!/bin/bash
# Phase 2 training — Su et al. CVPR 2020 (three stages in order)
set -e

DATA_DIR=${1:-"data/train"}
DATASET=${2:-"imagenet_mini"}
NAME_FULL="inst2020_full"
NAME_INST="inst2020_instance"
NAME_FUSE="inst2020_fusion"

cd "$(dirname "$0")/.."

echo "=== Stage 1/3: full ==="
python train.py \
  --method inst2020 \
  --stage full \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_FULL" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 50 \
  --niter_decay 50 \
  --lr 1e-4 \
  --print_freq 100 \
  --save_latest_freq 2000

echo "=== Stage 2/3: instance ==="
python train.py \
  --method inst2020 \
  --stage instance \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_INST" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 50 \
  --niter_decay 50 \
  --lr 1e-4 \
  --print_freq 100 \
  --save_latest_freq 2000

echo "=== Stage 3/3: fusion ==="
python train.py \
  --method inst2020 \
  --stage fusion \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_FUSE" \
  --fineSize 256 \
  --batch_size 8 \
  --niter 30 \
  --niter_decay 30 \
  --lr 1e-4 \
  --print_freq 100 \
  --save_latest_freq 2000

echo "Phase 2 training complete."
