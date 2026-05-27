#!/usr/bin/env bash
# Phase 2 stage-full training on AutoDL
# COCO2017 public path: /root/autodl-pub/COCO2017/
# Usage: bash scripts/train_phase2_full.sh

set -e
cd "$(dirname "$0")/.."

COCO_TRAIN=/root/autodl-pub/COCO2017/train2017
COCO_VAL=/root/autodl-pub/COCO2017/val2017
CKPT_NAME=inst_full

python train.py \
    --method inst_fusion \
    --stage full \
    --data_dir "$COCO_TRAIN" \
    --val_data_dir "$COCO_VAL" \
    --val_freq 5 \
    --fineSize 256 \
    --batch_size 32 \
    --nThreads 8 \
    --lr 1e-4 \
    --niter 30 \
    --niter_decay 30 \
    --save_epoch_freq 5 \
    --name "$CKPT_NAME" \
    --gpu_ids 0
