#!/usr/bin/env bash
# Phase 2 stage-full training on ImageNet-Mini.
# Usage:
#   bash scripts/train_phase2_full.sh [imagenet_train_dir] [imagenet_val_dir]

set -e
cd "$(dirname "$0")/.."

IMAGENET_TRAIN=${1:-/root/autodl-tmp/imagenet_mini/train}
IMAGENET_VAL=${2:-/root/autodl-tmp/imagenet_mini/val}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
CKPT_NAME=inst_full

"$PYTHON_BIN" train.py \
    --method inst_fusion \
    --stage full \
    --dataset imagenet_mini \
    --data_dir "$IMAGENET_TRAIN" \
    --val_data_dir "$IMAGENET_VAL" \
    --val_freq 5 \
    --fineSize 256 \
    --batch_size 32 \
    --nThreads 8 \
    --lr 1e-4 \
    --huber_weight 3.0 \
    --niter 30 \
    --niter_decay 30 \
    --max_epochs 100 \
    --grad_clip_norm 5.0 \
    --nan_lr_factor 0.1 \
    --nan_max_retries 3 \
    --early_stop_patience 6 \
    --save_epoch_freq 5 \
    --monitor_dir results/jiandu \
    --monitor_num 50 \
    --monitor_freq 5 \
    --name "$CKPT_NAME" \
    --gpu_ids 0
