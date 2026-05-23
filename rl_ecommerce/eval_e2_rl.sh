#!/bin/bash
# E2 Action Space Ablation - Evaluation
# Runs inference on test set for each variant's best RL checkpoint, then evaluates.
#
# Usage:
#   bash eval_e2_rl.sh [step]
#   If step is given, uses global_step_{step}; otherwise finds the best checkpoint.

set -e
PYTHON=~/miniconda3/envs/mlx/bin/python

E2_CKPT_BASE=/data/s3-migration/sft_localSearch/e2_rl_checkpoints
SFT_CKPT_BASE=/data/s3-migration/sft_localSearch/e2_checkpoints
DATA_DIR=/home/sunzhen/LocalSearch/data/ecommerce
E2_DATA_DIR=${DATA_DIR}/e2_variants
RESULT_DIR=/home/sunzhen/LocalSearch/data/ecommerce/result/e2
EVAL_SCRIPT=/home/sunzhen/LocalSearch/eval_ecommerce_nocase.py

mkdir -p $RESULT_DIR

STEP=${1:-""}
VARIANTS=("sql_only" "sql_text" "text_only" "sql_image")

# Start embedding server if not already running
if ! curl -s http://127.0.0.1:9123/health > /dev/null 2>&1; then
    echo "[$(date)] Starting embedding server on GPU 5..."
    EMBED_GPU=5 nohup $PYTHON /home/sunzhen/LocalSearch/rl_ecommerce/embedding_server.py > /tmp/e2_embed_server.log 2>&1 &
    EMBED_PID=$!
    for i in $(seq 1 180); do
        if curl -s http://127.0.0.1:9123/health > /dev/null 2>&1; then
            echo "[$(date)] Embedding server ready."
            break
        fi
        sleep 2
    done
else
    EMBED_PID=""
    echo "[$(date)] Embedding server already running."
fi

for VARIANT in "${VARIANTS[@]}"; do
    echo "============================================="
    echo "[$(date)] Evaluating E2 variant: $VARIANT"
    echo "============================================="

    CKPT_DIR="${E2_CKPT_BASE}/${VARIANT}"

    # Find model path
    if [ -n "$STEP" ]; then
        MODEL_DIR="${CKPT_DIR}/global_step_${STEP}"
    else
        # Use the latest checkpoint
        MODEL_DIR=$(ls -d ${CKPT_DIR}/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
        if [ -z "$MODEL_DIR" ]; then
            echo "[$(date)] No RL checkpoint for $VARIANT, using SFT model"
            MODEL_DIR="${SFT_CKPT_BASE}/${VARIANT}/final"
        fi
    fi

    if [ ! -d "$MODEL_DIR" ]; then
        echo "[$(date)] SKIP: model not found: $MODEL_DIR"
        continue
    fi

    echo "[$(date)] Model: $MODEL_DIR"

    PRED_FILE="${RESULT_DIR}/pred_${VARIANT}.json"
    EVAL_FILE="${RESULT_DIR}/eval_${VARIANT}.json"

    # Inference
    export ACTION_SPACE=$VARIANT
    CUDA_VISIBLE_DEVICES=1 $PYTHON /home/sunzhen/LocalSearch/inference_e2.py \
        --model_path "$MODEL_DIR" \
        --test_data "${E2_DATA_DIR}/test_${VARIANT}.parquet" \
        --output "$PRED_FILE" \
        --action_space "$VARIANT" \
        --batch_size 8

    # Evaluation
    $PYTHON $EVAL_SCRIPT \
        --pred "$PRED_FILE" \
        --gt "${DATA_DIR}/test/queries_product_test_step9_gt_v2.jsonl" \
        --output "$EVAL_FILE"

    echo "[$(date)] $VARIANT results:"
    cat "$EVAL_FILE"
    echo ""
done

# Also evaluate SFT-only baselines for comparison
echo "============================================="
echo "[$(date)] Evaluating SFT-only baselines..."
echo "============================================="

for VARIANT in "${VARIANTS[@]}"; do
    SFT_MODEL="${SFT_CKPT_BASE}/${VARIANT}/final"
    PRED_FILE="${RESULT_DIR}/pred_sft_${VARIANT}.json"
    EVAL_FILE="${RESULT_DIR}/eval_sft_${VARIANT}.json"

    if [ -f "$EVAL_FILE" ]; then
        echo "[$(date)] $VARIANT SFT eval already exists, skipping."
        continue
    fi

    export ACTION_SPACE=$VARIANT
    CUDA_VISIBLE_DEVICES=1 $PYTHON /home/sunzhen/LocalSearch/inference_e2.py \
        --model_path "$SFT_MODEL" \
        --test_data "${E2_DATA_DIR}/test_${VARIANT}.parquet" \
        --output "$PRED_FILE" \
        --action_space "$VARIANT" \
        --batch_size 8

    $PYTHON $EVAL_SCRIPT \
        --pred "$PRED_FILE" \
        --gt "${DATA_DIR}/test/queries_product_test_step9_gt_v2.jsonl" \
        --output "$EVAL_FILE"

    echo "[$(date)] $VARIANT SFT-only:"
    cat "$EVAL_FILE"
    echo ""
done

if [ -n "$EMBED_PID" ]; then
    kill $EMBED_PID 2>/dev/null
fi
echo "[$(date)] E2 evaluation complete!"
