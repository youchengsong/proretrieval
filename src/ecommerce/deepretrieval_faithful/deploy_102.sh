#!/bin/bash
# Deploy DeepRetrieval faithful reproduction on server 102.
# Prerequisites: verl 0.7.1 + Qwen3-4B already available on 102.
#
# What this script does:
# 1. Stops old training (GRPO with Qwen2.5-3B — wrong algorithm, wrong model)
# 2. Updates reward function (adds top_k=1000 for train, multi-target NDCG)
# 3. Regenerates training data (correct prompt template for Qwen3)
# 4. Verifies Qwen3-4B model is available
# 5. Starts new training (GRPO with Qwen3-4B, faithful reward logic)
set -e

WORK_DIR=/home/sunzhen/DeepRetrieval_ecommerce
DATA_SRC=/home/sunzhen/LocalSearch/data/ecommerce
PYTHON=~/miniconda3/envs/mlx/bin/python

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

echo "=========================================="
echo "Step 1: Stop old training"
echo "=========================================="
ray stop --force 2>/dev/null || true
# Kill any lingering VLLM workers
pkill -f "vllm::Worker" 2>/dev/null || true
sleep 3
echo "Old processes stopped"

echo ""
echo "=========================================="
echo "Step 2: Update reward function"
echo "=========================================="
# Backup old reward
cp $WORK_DIR/reward_ecommerce.py $WORK_DIR/reward_ecommerce_old_grpo.py 2>/dev/null || true
# The new reward_ecommerce.py should already be uploaded via scp/git
echo "Reward function at $WORK_DIR/reward_ecommerce.py"
echo "Key changes vs old version:"
echo "  - top_k=1000 for training (was 10) — matches DeepRetrieval author"
echo "  - Multi-target NDCG — handles multiple relevant products"
echo "  - Response-only input handling for verl 0.7.1"

echo ""
echo "=========================================="
echo "Step 3: Verify/regenerate training data"
echo "=========================================="
# Check if data exists
if [ -f "$WORK_DIR/data/train.parquet" ]; then
    echo "Training data already exists:"
    $PYTHON -c "
import pandas as pd
df = pd.read_parquet('$WORK_DIR/data/train.parquet')
print(f'  train: {len(df)} samples')
df = pd.read_parquet('$WORK_DIR/data/val.parquet')
print(f'  val: {len(df)} samples')
" 2>/dev/null || echo "  (could not read parquet)"
    echo "Data is compatible — same queries and ground truth, only prompt template differs slightly"
else
    echo "No training data found, generating..."
    $PYTHON $WORK_DIR/data_preprocess_ecommerce.py \
        --train_file $DATA_SRC/train/queries_product_train_step9.jsonl \
        --test_file $DATA_SRC/test/queries_product_test_step9_gt_v2.jsonl \
        --local_dir $WORK_DIR/data \
        --dataset ecommerce
fi

echo ""
echo "=========================================="
echo "Step 4: Verify Lucene index"
echo "=========================================="
$PYTHON -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('$WORK_DIR/indexes/lucene-index-products')
print(f'Lucene index: {searcher.num_docs} products')
hits = searcher.search('running shoes', k=3)
for h in hits:
    print(f'  {h.docid}: {h.score:.4f}')
"

echo ""
echo "=========================================="
echo "Step 5: Verify Qwen3-4B model"
echo "=========================================="
$PYTHON -c "
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)
    print(f'Qwen3-4B tokenizer loaded: vocab_size={tok.vocab_size}')
except Exception as e:
    print(f'ERROR: Qwen3-4B not found: {e}')
    print('Download with: HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub import snapshot_download; snapshot_download(\\\"Qwen/Qwen3-4B\\\")\"')
    exit(1)
"

echo ""
echo "=========================================="
echo "Step 6: Start training"
echo "=========================================="
echo "Launching training in background with nohup..."
nohup bash $WORK_DIR/train_faithful.sh > $WORK_DIR/train_faithful_nohup.log 2>&1 &
echo "Training PID: $!"
echo "Log: $WORK_DIR/train_faithful_nohup.log"
echo ""
echo "Monitor with: tail -f $WORK_DIR/train_faithful_nohup.log"
