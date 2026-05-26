#!/bin/bash
# Phase 3: CLIP Text-Guided Colorization Training
# Requires: COCO train2017 images + captions_train2017.json + Phase2 stage-full ckpt

set -e

python train_phase3.py \
    --data_dir /root/autodl-tmp/datasets/coco/train2017 \
    --caption_file /root/autodl-tmp/datasets/coco/annotations/captions_train2017.json \
    --full_ckpt checkpoints/inst_full/net_G.pth \
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
