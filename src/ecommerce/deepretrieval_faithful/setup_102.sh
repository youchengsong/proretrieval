#!/bin/bash
# Complete setup for DeepRetrieval faithful reproduction on server 102.
# Run this script on 102 to set up everything from scratch.
set -e

WORK_DIR=/home/sunzhen/DeepRetrieval
DATA_DIR=/home/sunzhen/LocalSearch/data/ecommerce
CONDA_ENV=deepretrieval

echo "=========================================="
echo "Step 0: Prerequisites check"
echo "=========================================="
echo "Java version:"
java -version 2>&1 || echo "ERROR: Java not found! Install: sudo apt install openjdk-21-jdk"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

echo ""
echo "=========================================="
echo "Step 1: Clone DeepRetrieval repo"
echo "=========================================="
if [ -d "$WORK_DIR" ]; then
    echo "DeepRetrieval already exists at $WORK_DIR, pulling latest..."
    cd $WORK_DIR && git pull
else
    git clone https://github.com/pat-jj/DeepRetrieval.git $WORK_DIR
fi
cd $WORK_DIR

mkdir -p exp_log

echo ""
echo "=========================================="
echo "Step 2: Create conda environment"
echo "=========================================="
# Create env if not exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Creating conda env: $CONDA_ENV"
    conda create -n $CONDA_ENV python=3.10 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo ""
echo "=========================================="
echo "Step 3: Install dependencies"
echo "=========================================="
# Install PyTorch first
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (upgraded for Qwen3 support)
# Author uses vllm<=0.6.3 but Qwen3-4B needs newer vLLM
pip install vllm==0.8.3

# Install transformers (upgraded for Qwen3 support)
# Author uses transformers<4.48 but Qwen3-4B needs >=4.51
pip install transformers>=4.51

# Install flash-attn
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install ray accelerate codetiming datasets dill hydra-core numpy pandas pybind11 wandb
pip install tensordict<0.6
pip install pyserini

# Install the verl package from the repo (author's fork)
cd $WORK_DIR/code
pip install -e .

echo ""
echo "=========================================="
echo "Step 4: Download Qwen3-4B model"
echo "=========================================="
MODEL_DIR=$HOME/.cache/huggingface/hub
if [ ! -d "$MODEL_DIR/models--Qwen--Qwen3-4B" ]; then
    echo "Downloading Qwen3-4B..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', cache_dir='$MODEL_DIR')
print('Download complete')
" || echo "If HuggingFace is blocked, use: HF_ENDPOINT=https://hf-mirror.com python -c ..."
else
    echo "Qwen3-4B already cached"
fi

echo ""
echo "=========================================="
echo "Step 5: Copy ecommerce data files"
echo "=========================================="
# Copy our custom files into the DeepRetrieval repo structure
ECOM_LUCENE=$WORK_DIR/code/src/Lucene/ecommerce
ECOM_REWARD=$WORK_DIR/code/verl/utils/reward_score
ECOM_PREPROCESS=$WORK_DIR/code/data_preprocess
ECOM_SCRIPTS=$WORK_DIR/code/scripts/train

mkdir -p $ECOM_LUCENE
mkdir -p code/data/local_index_search/ecommerce/jsonl_docs
mkdir -p code/data/local_index_search/ecommerce/sparse

echo "Copying search module, data preprocessing, reward function, and training script..."
# These files should already be placed by the user or via scp

echo ""
echo "=========================================="
echo "Step 6: Convert products to JSONL"
echo "=========================================="
cd $WORK_DIR/code
python src/Lucene/ecommerce/1_convert_format.py \
    --input $DATA_DIR/test/products_test.json \
    --output_dir code/data/local_index_search/ecommerce/jsonl_docs

echo ""
echo "=========================================="
echo "Step 7: Build Pyserini BM25 index"
echo "=========================================="
bash src/Lucene/ecommerce/2_build_database.sh

echo ""
echo "=========================================="
echo "Step 8: Prepare training data"
echo "=========================================="
python data_preprocess/ecommerce.py \
    --train_file $DATA_DIR/train/queries_product_train_step9.jsonl \
    --test_file $DATA_DIR/test/queries_product_test_step9_gt_v2.jsonl

echo ""
echo "=========================================="
echo "Step 9: Patch main_ppo.py"
echo "=========================================="
# Add ecommerce to _select_rm_score_fn in main_ppo.py
if ! grep -q "ecommerce" verl/trainer/main_ppo.py; then
    echo "Patching main_ppo.py to add ecommerce reward function..."
    # Insert before the 'else: raise NotImplementedError' line
    sed -i "/else:/i\\
    elif 'ecommerce' in data_source:\\
        from verl.utils.reward_score import ecommerce\\
        return ecommerce.compute_score" verl/trainer/main_ppo.py
    echo "Patched successfully"
else
    echo "main_ppo.py already patched"
fi

echo ""
echo "=========================================="
echo "Step 10: Verify setup"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import transformers
print(f'Transformers: {transformers.__version__}')

import vllm
print(f'vLLM: {vllm.__version__}')

from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('code/data/local_index_search/ecommerce/pyserini_index')
print(f'Pyserini index: {searcher.num_docs} documents')

print('All checks passed!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To start training:"
echo "  conda activate $CONDA_ENV"
echo "  cd $WORK_DIR/code"
echo "  bash scripts/train/ecommerce.sh"
