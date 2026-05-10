#!/bin/bash
# Inference script — supports all method / exemplar combinations
set -e

METHOD=${1:-"cnn_color"}        # cnn_color | inst_fusion
TEST_DIR=${2:-"data/test"}
NAME=${3:-"$METHOD"}
EXEMPLAR=${4:-""}               # pass --exemplar to enable
REF_IMG=${5:-""}

cd "$(dirname "$0")/.."

EXTRA_ARGS=""
if [ "$EXEMPLAR" = "--exemplar" ] && [ -n "$REF_IMG" ]; then
  EXTRA_ARGS="--exemplar --ref_img $REF_IMG"
fi

python test.py \
  --method "$METHOD" \
  --name "$NAME" \
  --test_img_dir "$TEST_DIR" \
  --results_img_dir "results/$METHOD" \
  --which_epoch latest \
  --fineSize 256 \
  $EXTRA_ARGS
