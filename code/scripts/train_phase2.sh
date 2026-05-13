#!/bin/bash
# Phase 2 training - inst_fusion (three stages in order)
set -e

DATA_DIR=${1:-"../data/imagenet_mini/train"}
DATASET=${2:-"imagenet_mini"}
PHASE1_NAME=${3:-"cnn_color"}
NAME_FULL="inst_fusion_full"
NAME_INST="inst_fusion_instance"
NAME_FUSE="inst_fusion_fusion"
SMOKE_TEST_ITERS=${SMOKE_TEST_ITERS:-10}

cd "$(dirname "$0")/.."

echo "=== Stage 1/3: full ==="
PHASE1_NAME="$PHASE1_NAME" \
FULL_STAGE_NAME="$NAME_FULL" \
INSTANCE_STAGE_NAME="$NAME_INST" \
SMOKE_TEST_ITERS="$SMOKE_TEST_ITERS" \
python train.py \
  --method inst_fusion \
  --stage full \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_FULL" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 1 \
  --niter_decay 0 \
  --lr 1e-4 \
  --print_freq 1 \
  --save_latest_freq 1000

echo "=== Stage 2/3: instance ==="
PHASE1_NAME="$PHASE1_NAME" \
FULL_STAGE_NAME="$NAME_FULL" \
INSTANCE_STAGE_NAME="$NAME_INST" \
SMOKE_TEST_ITERS="$SMOKE_TEST_ITERS" \
python train.py \
  --method inst_fusion \
  --stage instance \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_INST" \
  --fineSize 256 \
  --batch_size 16 \
  --niter 1 \
  --niter_decay 0 \
  --lr 1e-4 \
  --print_freq 1 \
  --save_latest_freq 1000

echo "=== Stage 3/3: fusion ==="
PHASE1_NAME="$PHASE1_NAME" \
FULL_STAGE_NAME="$NAME_FULL" \
INSTANCE_STAGE_NAME="$NAME_INST" \
SMOKE_TEST_ITERS="$SMOKE_TEST_ITERS" \
python train.py \
  --method inst_fusion \
  --stage fusion \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --name "$NAME_FUSE" \
  --fineSize 256 \
  --batch_size 8 \
  --niter 1 \
  --niter_decay 0 \
  --lr 1e-4 \
  --print_freq 1 \
  --save_latest_freq 1000

echo "Phase 2 training complete."
