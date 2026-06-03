#!/usr/bin/env bash
# Phase 3 training launcher.
#
# Usage:
#   bash scripts/train_phase3.sh                     # use defaults
#   bash scripts/train_phase3.sh /path/to/coco/root  # override COCO root
#
# Expects:
#   $COCO_ROOT/train2017
#   $COCO_ROOT/val2017
#   $COCO_ROOT/annotations/instances_{train,val}2017.json
set -euo pipefail

COCO_ROOT="${1:-datasets/coco}"
NAME="${NAME:-phase3_text_color}"
GPU="${GPU:-0}"

JSONL_TRAIN="${JSONL_TRAIN:-data/phase3_color_object_train.jsonl}"
JSONL_VAL="${JSONL_VAL:-data/phase3_color_object_val.jsonl}"
CLIP_CACHE="${CLIP_CACHE:-data/clip_text_cache.pt}"

if [ ! -f "$JSONL_TRAIN" ]; then
  echo "[error] training JSONL not found: $JSONL_TRAIN"
  echo "  Run:  python scripts/build_phase3_jsonl.py \\"
  echo "          --instances_file $COCO_ROOT/annotations/instances_train2017.json \\"
  echo "          --img_dir $COCO_ROOT/train2017 --out_file $JSONL_TRAIN"
  exit 1
fi

python train.py --method text_color \
  --jsonl_file          "$JSONL_TRAIN" \
  --val_jsonl_file      "$JSONL_VAL" \
  --img_dir             "$COCO_ROOT/train2017" \
  --val_img_dir         "$COCO_ROOT/val2017" \
  --instances_file      "$COCO_ROOT/annotations/instances_train2017.json" \
  --val_instances_file  "$COCO_ROOT/annotations/instances_val2017.json" \
  --full_ckpt           checkpoints/inst_fusion_full/80_net_G.pth \
  --inst_ckpt           checkpoints/inst_fusion_instance/25_net_G.pth \
  --fusion_ckpt         checkpoints/inst_fusion_fusion/25_net_G.pth \
  --clip_cache          "$CLIP_CACHE" \
  --fineSize 256 --batch_size 1 --nThreads 0 \
  --lr 1e-4 --beta1 0.5 --huber_weight 3.0 \
  --lambda_inst 1.0 --lambda_rank 0.1 --lambda_outside 0.2 \
  --rank_margin 0.05 --rank_warmup_epoch 5 --rank_warmup_len 5 \
  --niter 30 --niter_decay 30 --save_epoch_freq 5 \
  --name "$NAME" --gpu_ids "$GPU"
