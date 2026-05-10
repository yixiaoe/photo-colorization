#!/bin/bash
# Phase 1 training — cnn_color (single stage)
set -e

DATA_DIR=${1:-"../data/imagenet_mini/train"}
DATASET=${2:-"imagenet_mini"}
NAME=${NAME:-"cnn_color"}
BATCH_SIZE=${BATCH_SIZE:-16}
NITER=${NITER:-50}
NITER_DECAY=${NITER_DECAY:-50}
PRINT_FREQ=${PRINT_FREQ:-100}
SAVE_LATEST_FREQ=${SAVE_LATEST_FREQ:-2000}
SAVE_EPOCH_FREQ=${SAVE_EPOCH_FREQ:-5}
MAX_DATASET_SIZE=${MAX_DATASET_SIZE:-1000000000}

cd "$(dirname "$0")/.."

python train.py \
  --method cnn_color \
  --stage full \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME" \
  --fineSize 256 \
  --batch_size "$BATCH_SIZE" \
  --niter "$NITER" \
  --niter_decay "$NITER_DECAY" \
  --lr 1e-4 \
  --print_freq "$PRINT_FREQ" \
  --save_latest_freq "$SAVE_LATEST_FREQ" \
  --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --max_dataset_size "$MAX_DATASET_SIZE"
