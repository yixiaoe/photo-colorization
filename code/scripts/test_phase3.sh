#!/bin/bash
# Phase 3: Text-Guided Colorization Inference
# Usage: bash scripts/test_phase3.sh "a red car under blue sky"

set -e

PROMPT="${1:-a colorful scene}"

python test_phase3.py \
    --test_img_dir data/test \
    --results_img_dir results/phase3 \
    --full_ckpt checkpoints/inst_full/net_G.pth \
    --fineSize 256 \
    --prompt "$PROMPT" \
    --name text_color \
    --gpu_ids 0
