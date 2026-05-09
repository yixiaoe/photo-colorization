#!/bin/bash
# Phase 1 training — Zhang et al. 2016 (single stage)
set -e

DATA_DIR=${1:-"data/train"}
DATASET=${2:-"imagenet_mini"}
NAME="zhang2016"

cd "$(dirname "$0")/.."

python train.py \
  --method zhang2016 \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 50 \
  --niter_decay 50 \
  --lr 1e-4 \
  --print_freq 100 \
  --save_latest_freq 2000 \
  --save_epoch_freq 5
