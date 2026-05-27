#!/bin/bash
# Phase 3: CLIP Text-Guided Colorization Training on AutoDL
# COCO2017 public path: /root/autodl-pub/COCO2017/
# Requires: Phase 2 stage-full ckpt at checkpoints/inst_full/best_net_G.pth

set -e
cd "$(dirname "$0")/.."

COCO_TRAIN=/root/autodl-pub/COCO2017/train2017
COCO_VAL=/root/autodl-pub/COCO2017/val2017
COCO_ANN=/root/autodl-pub/COCO2017/annotations

python train_phase3.py \
    --data_dir "$COCO_TRAIN" \
    --caption_file "$COCO_ANN/captions_train2017.json" \
    --val_data_dir "$COCO_VAL" \
    --val_caption_file "$COCO_ANN/captions_val2017.json" \
    --val_freq 5 \
    --full_ckpt checkpoints/inst_full/best_net_G.pth \
    --fineSize 256 \
    --batch_size 16 \
    --nThreads 4 \
    --lr 1e-4 \
    --niter 30 \
    --niter_decay 30 \
    --name text_color \
    --gpu_ids 0 \
    --print_freq 100 \
    --save_epoch_freq 10

