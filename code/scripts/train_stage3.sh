#!/bin/bash
# Stage 3/3: Fusion WeightGenerators training (offline bbox cache version)
#
# Prerequisites:
#   1. Stage 1 done: checkpoints/inst_fusion_full/latest_net_G.pth
#   2. Stage 2 done: checkpoints/inst_fusion_instance/latest_net_G.pth
#   3. Bbox cache:   datasets/bbox_cache_train2017.json
#      (generate with: python scripts/precompute_bbox.py)
#
# Usage (run from code/):
#   bash scripts/train_stage3.sh [coco_img_dir] [bbox_cache]
#
# Defaults:
#   coco_img_dir = datasets/coco/train2017
#   bbox_cache   = datasets/bbox_cache_train2017.json
set -e

COCO_IMG_DIR=${1:-"datasets/coco/train2017"}
BBOX_CACHE=${2:-"datasets/bbox_cache_train2017.json"}
CKPT_DIR="./checkpoints"
NAME_FULL="inst_fusion_full"
NAME_INST="inst_fusion_instance"
NAME_FUSE="inst_fusion_fusion"

cd "$(dirname "$0")/.."

# sanity checks
if [ ! -f "$CKPT_DIR/$NAME_FULL/latest_net_G.pth" ]; then
    echo "[error] Stage 1 checkpoint not found: $CKPT_DIR/$NAME_FULL/latest_net_G.pth"
    exit 1
fi
if [ ! -f "$CKPT_DIR/$NAME_INST/latest_net_G.pth" ]; then
    echo "[error] Stage 2 checkpoint not found: $CKPT_DIR/$NAME_INST/latest_net_G.pth"
    exit 1
fi
if [ ! -f "$BBOX_CACHE" ]; then
    echo "[error] Bbox cache not found: $BBOX_CACHE"
    echo "  Run first: python scripts/precompute_bbox.py --img_dir $COCO_IMG_DIR --output $BBOX_CACHE"
    exit 1
fi

echo "=== Stage 3/3: fusion WeightGenerators ==="
echo "  full  ckpt : $CKPT_DIR/$NAME_FULL/latest_net_G.pth"
echo "  inst  ckpt : $CKPT_DIR/$NAME_INST/latest_net_G.pth"
echo "  bbox cache : $BBOX_CACHE"
echo "  coco  dir  : $COCO_IMG_DIR"

python train.py \
  --method       inst_fusion \
  --stage        fusion \
  --dataset      imagenet_mini \
  --data_dir     "$COCO_IMG_DIR" \
  --bbox_cache   "$BBOX_CACHE" \
  --name         "$NAME_FUSE" \
  --full_ckpt    "$CKPT_DIR/$NAME_FULL/latest_net_G.pth" \
  --inst_ckpt    "$CKPT_DIR/$NAME_INST/latest_net_G.pth" \
  --fineSize     256 \
  --batch_size   1 \
  --nThreads     4 \
  --niter        40 \
  --niter_decay  40 \
  --lr           2e-5 \
  --lr_policy    lambda \
  --num_classes  91 \
  --embed_dim    64 \
  --box_num      8 \
  --max_dataset_size 50000 \
  --print_freq   2000 \
  --save_latest_freq 5000 \
  --save_epoch_freq  10

echo "Stage 3 training complete."
