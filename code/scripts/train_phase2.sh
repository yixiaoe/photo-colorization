#!/bin/bash
# Phase 2 training — inst_fusion (three stages in order)
#
# Usage:
#   bash scripts/train_phase2.sh [imagenet_dir] [coco_img_dir] [coco_ann_file]
#
# Examples:
#   bash scripts/train_phase2.sh ../datasets/imagenet_mini \
#                                ../datasets/coco/train2017 \
#                                ../datasets/coco/annotations/instances_train2017.json
set -e

IMAGENET_DIR=${1:-"../datasets/imagenet_mini"}
COCO_IMG_DIR=${2:-"../datasets/coco/train2017"}
COCO_ANN=${3:-"../datasets/coco/annotations/instances_train2017.json"}

NAME_FULL="inst_fusion_full"
NAME_INST="inst_fusion_instance"
NAME_FUSE="inst_fusion_fusion"
CKPT_DIR="./checkpoints"

cd "$(dirname "$0")/.."

# ── Stage 1/3: Full-image backbone (ImageNet-Mini) ───────────────────────────
echo "=== Stage 1/3: full-image backbone ==="
python train.py \
  --method inst_fusion \
  --stage full \
  --dataset imagenet_mini \
  --data_dir "$IMAGENET_DIR" \
  --name "$NAME_FULL" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 100 \
  --niter_decay 100 \
  --lr 1e-4 \
  --lr_policy lambda \
  --rebalance_gamma 0.5 \
  --print_freq 100 \
  --save_latest_freq 2000 \
  --save_epoch_freq 20

# ── Stage 2/3: Instance network with FiLM (COCO GT crops) ───────────────────
echo "=== Stage 2/3: instance network (FiLM) ==="
python train.py \
  --method inst_fusion \
  --stage instance \
  --dataset imagenet_mini \
  --data_dir "$COCO_IMG_DIR" \
  --ann_file "$COCO_ANN" \
  --name "$NAME_INST" \
  --full_ckpt "$CKPT_DIR/$NAME_FULL/latest_net_G.pth" \
  --fineSize 256 \
  --batch_size 32 \
  --niter 100 \
  --niter_decay 100 \
  --lr 1e-4 \
  --lr_backbone 5e-5 \
  --lr_policy lambda \
  --num_classes 91 \
  --embed_dim 64 \
  --rebalance_gamma 0.5 \
  --print_freq 100 \
  --save_latest_freq 2000 \
  --save_epoch_freq 20

# ── Stage 3/3: Fusion WeightGenerators (COCO full image + online detection) ──
echo "=== Stage 3/3: fusion WeightGenerators ==="
python train.py \
  --method inst_fusion \
  --stage fusion \
  --dataset imagenet_mini \
  --data_dir "$COCO_IMG_DIR" \
  --name "$NAME_FUSE" \
  --full_ckpt "$CKPT_DIR/$NAME_FULL/latest_net_G.pth" \
  --inst_ckpt "$CKPT_DIR/$NAME_INST/latest_net_G.pth" \
  --fineSize 256 \
  --batch_size 1 \
  --niter 20 \
  --niter_decay 20 \
  --lr 2e-5 \
  --lr_policy lambda \
  --num_classes 91 \
  --embed_dim 64 \
  --box_num 8 \
  --print_freq 50 \
  --save_latest_freq 500 \
  --save_epoch_freq 10

echo "Phase 2 training complete."
