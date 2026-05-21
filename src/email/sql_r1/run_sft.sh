#!/bin/bash
# Email SQL-R1: Build DB + Generate SFT data + SFT Training
# Run on server 102
set -ex
cd /home/sunzhen/SQL_R1_email

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
NPROC=6
MASTER_PORT=29502
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=~/miniconda3/envs/mlx/bin/python

echo "[$(date)] Step 1: Build SQLite DB"
$PYTHON build_email_db.py --split train --data-dir /home/sunzhen/LocalSearch/data/email --output /home/sunzhen/LocalSearch/data/email/train/emails_train.db

echo "[$(date)] Step 2: Generate SFT data"
$PYTHON gen_sft_data.py

echo "[$(date)] Step 3: Prepare RL data"
$PYTHON prepare_data.py

echo "[$(date)] Step 4: Start SFT training"
$PYTHON -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    train_sft.py

echo "[$(date)] SFT training finished."
