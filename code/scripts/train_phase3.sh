#!/bin/bash
# Phase 3: CLIP Text-Guided Colorization Training on AutoDL
# Requires: Phase 2 stage-full ckpt at checkpoints/inst_full/80_net_G.pth

set -e
cd "$(dirname "$0")/.."

COCO_ROOT=${1:-/root/autodl-tmp/coco2017}
COCO_TRAIN="$COCO_ROOT/train2017"
COCO_VAL="$COCO_ROOT/val2017"
COCO_ANN="$COCO_ROOT/annotations"
TRAIN_RECORDS=data/phase3_color_object_no_person_train.jsonl
VAL_RECORDS=data/phase3_color_object_no_person_val.jsonl
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
CLIP_CKPT=${CLIP_CKPT:-checkpoints/clip/open_clip_model.safetensors}

mkdir -p data

if [ ! -f "$TRAIN_RECORDS" ]; then
    "$PYTHON" scripts/build_phase3_color_object_jsonl.py \
        --img_dir "$COCO_TRAIN" \
        --instances_file "$COCO_ANN/instances_train2017.json" \
        --captions_file "$COCO_ANN/captions_train2017.json" \
        --out_file "$TRAIN_RECORDS" \
        --exclude_categories person
fi

if [ ! -f "$VAL_RECORDS" ]; then
    "$PYTHON" scripts/build_phase3_color_object_jsonl.py \
        --img_dir "$COCO_VAL" \
        --instances_file "$COCO_ANN/instances_val2017.json" \
        --captions_file "$COCO_ANN/captions_val2017.json" \
        --out_file "$VAL_RECORDS" \
        --exclude_categories person
fi

"$PYTHON" train_phase3.py \
    --color_object_file "$TRAIN_RECORDS" \
    --val_color_object_file "$VAL_RECORDS" \
    --val_freq 5 \
    --full_ckpt checkpoints/inst_full/80_net_G.pth \
    --clip_arch ViT-B-32-quickgelu \
    --clip_pretrained_path "$CLIP_CKPT" \
    --fineSize 256 \
    --batch_size 16 \
    --nThreads 4 \
    --lr 1e-4 \
    --huber_weight 3.0 \
    --lambda_inst 1.0 \
    --lambda_rank 0.1 \
    --lambda_outside 0.2 \
    --rank_margin 0.05 \
    --niter 30 \
    --niter_decay 30 \
    --name text_color \
    --gpu_ids 0 \
    --print_freq 100 \
    --save_epoch_freq 10
