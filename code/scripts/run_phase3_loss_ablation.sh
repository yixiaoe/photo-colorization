#!/usr/bin/env bash
# Run isolated 3-epoch Phase 3 loss ablations.
#
# Usage:
#   bash scripts/run_phase3_loss_ablation.sh rank
#   bash scripts/run_phase3_loss_ablation.sh outside 50
#
# The rank mode keeps lambda_outside at the current baseline 0.2 and tries
# lambda_rank = 10, 50, 100. The outside mode keeps the selected rank fixed
# and tries lambda_outside = 1, 5, 10.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-rank}"
SELECTED_RANK="${2:-}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
TRAIN_RECORDS="${TRAIN_RECORDS:-data/phase3_color_object_no_person_train.jsonl}"
VAL_RECORDS="${VAL_RECORDS:-data/phase3_color_object_no_person_val.jsonl}"
FULL_CKPT="${FULL_CKPT:-checkpoints/inst_full/80_net_G.pth}"
CLIP_CKPT="${CLIP_CKPT:-checkpoints/clip/open_clip_model.safetensors}"
SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NTHREADS="${NTHREADS:-4}"
NPROC="${NPROC:-1}"
GPU_IDS="${GPU_IDS:-0}"
LIMIT="${PROMPT_EVAL_LIMIT:-12}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "[error] required file not found: $1" >&2
        exit 1
    fi
}

require_file "$TRAIN_RECORDS"
require_file "$VAL_RECORDS"
require_file "$FULL_CKPT"
require_file "$CLIP_CKPT"

train_and_eval() {
    local name="$1"
    local lambda_rank="$2"
    local lambda_outside="$3"
    local eval_checkpoints=()
    local epoch
    for ((epoch = 1; epoch <= EPOCHS; epoch++)); do
        eval_checkpoints+=("$epoch")
    done
    eval_checkpoints+=("latest")

    echo "[run] name=$name lambda_rank=$lambda_rank lambda_outside=$lambda_outside seed=$SEED nproc=$NPROC"
    local launcher=("$PYTHON")
    if [ "$NPROC" -gt 1 ]; then
        launcher=("$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$NPROC")
    fi
    "${launcher[@]}" train_phase3.py \
        --color_object_file "$TRAIN_RECORDS" \
        --val_color_object_file "$VAL_RECORDS" \
        --val_freq 1 \
        --full_ckpt "$FULL_CKPT" \
        --clip_arch ViT-B-32-quickgelu \
        --clip_pretrained_path "$CLIP_CKPT" \
        --fineSize 256 \
        --batch_size "$BATCH_SIZE" \
        --nThreads "$NTHREADS" \
        --lr 1e-4 \
        --huber_weight 3.0 \
        --lambda_inst 1.0 \
        --lambda_rank "$lambda_rank" \
        --lambda_outside "$lambda_outside" \
        --rank_margin 0.05 \
        --seed "$SEED" \
        --niter "$EPOCHS" \
        --niter_decay 0 \
        --name "$name" \
        --gpu_ids "$GPU_IDS" \
        --print_freq 100 \
        --save_epoch_freq 1 \
        --save_latest_freq 2000

    "$PYTHON" scripts/eval_phase3_prompt_control.py \
        --records_file "$VAL_RECORDS" \
        --name "$name" \
        --full_ckpt "$FULL_CKPT" \
        --clip_arch ViT-B-32-quickgelu \
        --clip_pretrained_path "$CLIP_CKPT" \
        --checkpoints "${eval_checkpoints[@]}" \
        --limit "$LIMIT" \
        --rank_margin 0.05 \
        --out_dir "results/${name}_prompt_control_eval" \
        --gpu_ids "$GPU_IDS" \
        --save_images
}

case "$MODE" in
    rank)
        for rank in 10 50 100; do
            train_and_eval "text_color_rank${rank}_e${EPOCHS}" "$rank" 0.2
        done
        ;;
    outside)
        if [ -z "$SELECTED_RANK" ]; then
            echo "[error] outside mode requires selected rank, e.g. outside 50" >&2
            exit 1
        fi
        for outside in 1 5 10; do
            train_and_eval \
                "text_color_rank${SELECTED_RANK}_outside${outside}_e${EPOCHS}" \
                "$SELECTED_RANK" "$outside"
        done
        ;;
    *)
        echo "[error] unknown mode: $MODE" >&2
        echo "usage: bash scripts/run_phase3_loss_ablation.sh rank" >&2
        echo "       bash scripts/run_phase3_loss_ablation.sh outside <rank>" >&2
        exit 1
        ;;
esac
