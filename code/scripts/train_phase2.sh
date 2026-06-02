#!/bin/bash
# Phase 2 training — inst_fusion (three stages, RTX 5090, ~5 days total)
#
# All three stages use COCO2017 as the sole dataset.
# Stage 3 requires offline bbox cache (precompute_bbox.py).
#
# Usage (run from code/):
#   bash scripts/train_phase2.sh [coco_img_dir] [coco_ann_file] [bbox_cache]
set -e

COCO_IMG_DIR=${1:-"datasets/coco/train2017"}
COCO_ANN=${2:-"datasets/coco/annotations/instances_train2017.json"}
BBOX_CACHE=${3:-"datasets/bbox_cache_train2017.json"}

NAME_FULL="inst_fusion_full"
NAME_INST="inst_fusion_instance"
NAME_FUSE="inst_fusion_fusion"
CKPT_DIR="./checkpoints"

cd "$(dirname "$0")/.."

# ── Stage 1/3: Full-image backbone (~33h on RTX 5090) ───────────────────────
echo "=== Stage 1/3: full-image backbone (COCO2017 full images) ==="
python train.py \
  --method       inst_fusion \
  --stage        full \
  --dataset      imagenet_mini \
  --data_dir     "$COCO_IMG_DIR" \
  --name         "$NAME_FULL" \
  --fineSize     256 \
  --batch_size   32 \
  --nThreads     8 \
  --niter        40 \
  --niter_decay  40 \
  --lr           3e-5 \
  --lr_policy    lambda \
  --rebalance_gamma 0.5 \
  --print_freq   200 \
  --save_latest_freq 2000 \
  --save_epoch_freq  10

# ── Stage 2/3: Instance network with FiLM (~35h) ────────────────────────────
echo "=== Stage 2/3: instance network (FiLM, COCO GT crops) ==="
python train.py \
  --method       inst_fusion \
  --stage        instance \
  --dataset      imagenet_mini \
  --data_dir     "$COCO_IMG_DIR" \
  --ann_file     "$COCO_ANN" \
  --name         "$NAME_INST" \
  --full_ckpt    "$CKPT_DIR/$NAME_FULL/net_G.pth" \
  --fineSize     256 \
  --batch_size   32 \
  --nThreads     8 \
  --niter        20 \
  --niter_decay  20 \
  --lr           3e-5 \
  --lr_backbone  1e-5 \
  --lr_policy    lambda \
  --num_classes  91 \
  --embed_dim    64 \
  --rebalance_gamma 0.5 \
  --print_freq   200 \
  --save_latest_freq 2000 \
  --save_epoch_freq  5

# ── Stage 3/3: Fusion WeightGenerators (~40h) ───────────────────────────────
echo "=== Stage 3/3: fusion WeightGenerators (COCO + offline bbox) ==="

if [ ! -f "$BBOX_CACHE" ]; then
    echo "[error] Bbox cache not found: $BBOX_CACHE"
    echo "  Run: python scripts/precompute_bbox.py --img_dir $COCO_IMG_DIR --output $BBOX_CACHE"
    exit 1
fi

python train.py \
  --method       inst_fusion \
  --stage        fusion \
  --dataset      imagenet_mini \
  --data_dir     "$COCO_IMG_DIR" \
  --bbox_cache   "$BBOX_CACHE" \
  --name         "$NAME_FUSE" \
  --full_ckpt    "$CKPT_DIR/$NAME_FULL/net_G.pth" \
  --inst_ckpt    "$CKPT_DIR/$NAME_INST/net_G.pth" \
  --fineSize     256 \
  --batch_size   1 \
  --nThreads     4 \
  --niter        15 \
  --niter_decay  15 \
  --lr           2e-5 \
  --lr_policy    lambda \
  --num_classes  91 \
  --embed_dim    64 \
  --box_num      8 \
  --print_freq   2000 \
  --save_latest_freq 5000 \
  --save_epoch_freq  5

echo "=== Phase 2 training complete ==="
