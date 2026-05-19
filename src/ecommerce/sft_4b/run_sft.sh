#!/bin/bash
# ── LocalSearch-R1 SFT 训练启动脚本 ──
# GPU 1,4,5,6 四卡训练
# 用法: nohup bash run.sh > train.log 2>&1 &

set -e
cd ~/sft_localSearch

# ── GPU 配置 ──
export CUDA_VISIBLE_DEVICES=1,4,5,6
NPROC=4
MASTER_PORT=29501

PYTHON=~/miniconda3/envs/mlx/bin/python
LOGFILE=~/sft_localSearch/train.log

# ── 检查数据 ──
if [ ! -f train.jsonl ]; then
    echo "[$(date)] Preparing data..."
    $PYTHON prepare_data.py
else
    echo "[$(date)] train.jsonl exists ($(wc -l < train.jsonl) lines), skipping."
fi

# ── 检查 checkpoint ──
LATEST_CKPT=$(ls -d output/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "[$(date)] Found checkpoint: $LATEST_CKPT — will resume."
else
    echo "[$(date)] No checkpoint — training from scratch."
fi

# ── 启动训练 ──
echo "[$(date)] Starting training on GPUs: $CUDA_VISIBLE_DEVICES ($NPROC processes)"
$PYTHON -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    train.py

echo "[$(date)] Training finished."
