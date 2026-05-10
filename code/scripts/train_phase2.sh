#!/bin/bash
# Phase 2 training — inst_fusion (three stages in order)
set -e

DATA_DIR=${1:-"../data/imagenet_mini/train"}
DATASET=${2:-"imagenet_mini"}
NAME_FULL="inst_fusion_full"
NAME_INST="inst_fusion_instance"
NAME_FUSE="inst_fusion_fusion"

cd "$(dirname "$0")/.."

echo "=== Stage 1/3: full ==="
python train.py \
  --method inst_fusion \
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
  --method inst_fusion \
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
  --method inst_fusion \
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
