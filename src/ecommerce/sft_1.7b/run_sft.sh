#!/bin/bash
# Qwen3-1.7B SFT 训练 (单卡 GPU 4)
set -e
cd ~/LocalSearch/ecommerce_1.7b
export CUDA_VISIBLE_DEVICES=4
PYTHON=~/miniconda3/envs/mlx/bin/python

if [ ! -f /home/sunzhen/LocalSearch/ecommerce/train.jsonl ]; then
    echo "[$(date)] ERROR: train.jsonl not found!"
    exit 1
fi

echo "[$(date)] Starting SFT training on GPU 4 (single card)"
$PYTHON train_sft.py
echo "[$(date)] SFT training finished."
