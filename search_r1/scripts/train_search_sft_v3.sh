#!/bin/bash
set -e

BASE=/root/autodl-tmp
MODEL=Qwen/Qwen3-4B
SCRIPT=$BASE/scripts/train_sft_v2.py

echo "=== Starting Search-R1 SFT retraining (enable_thinking=True) at $(date) ==="

# Clean old checkpoints
rm -rf $BASE/search_r1/checkpoints/ecommerce_sft_v2
rm -rf $BASE/search_r1/checkpoints/email_sft_v2

# 1. Search-R1 ecommerce
echo "=== [1/2] Search-R1 ecommerce SFT ==="
deepspeed --num_gpus 2 $SCRIPT \
    --model_name $MODEL \
    --data_path $BASE/search_r1/sft_data/ecommerce_search_sft.jsonl \
    --output_dir $BASE/search_r1/checkpoints/ecommerce_sft_v2 \
    --num_epochs 3 \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_length 2048

# 2. Search-R1 email
echo "=== [2/2] Search-R1 email SFT ==="
deepspeed --num_gpus 2 $SCRIPT \
    --model_name $MODEL \
    --data_path $BASE/search_r1/sft_data/email_search_sft.jsonl \
    --output_dir $BASE/search_r1/checkpoints/email_sft_v2 \
    --num_epochs 3 \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_length 2048

echo "=== Both Search-R1 SFT models retrained at $(date) ==="
ls -d $BASE/search_r1/checkpoints/*/final 2>/dev/null
