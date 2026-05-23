#!/bin/bash
# Upload E2 RL files to 102 server and generate RL data.
# Run from local machine (macOS).
#
# Usage:
#   bash rl_ecommerce/setup_e2_rl.sh

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SSH_TARGET="sunzhen@58.34.13.47"
SSH_PORT=10087
REMOTE_BASE="/home/sunzhen/LocalSearch"

echo "[setup] Uploading E2 RL files to 102..."

# 1. Reward function
scp -P $SSH_PORT \
    "$REPO_DIR/rl_ecommerce/reward_dsl_e2.py" \
    ${SSH_TARGET}:${REMOTE_BASE}/rl_ecommerce/reward_dsl_e2.py

# 2. Embedding server
scp -P $SSH_PORT \
    "$REPO_DIR/rl_ecommerce/embedding_server.py" \
    ${SSH_TARGET}:${REMOTE_BASE}/rl_ecommerce/embedding_server.py

# 3. RL training script
scp -P $SSH_PORT \
    "$REPO_DIR/rl_ecommerce/run_e2_rl.sh" \
    ${SSH_TARGET}:${REMOTE_BASE}/rl_ecommerce/run_e2_rl.sh

# 4. RL data generation script
scp -P $SSH_PORT \
    "$REPO_DIR/src/ecommerce/gen_e2_variants.py" \
    ${SSH_TARGET}:${REMOTE_BASE}/src/ecommerce/gen_e2_variants.py

# 5. Inference script
scp -P $SSH_PORT \
    "$REPO_DIR/src/ecommerce/inference_e2.py" \
    ${SSH_TARGET}:${REMOTE_BASE}/inference_e2.py

# 6. Eval script
scp -P $SSH_PORT \
    "$REPO_DIR/rl_ecommerce/eval_e2_rl.sh" \
    ${SSH_TARGET}:${REMOTE_BASE}/rl_ecommerce/eval_e2_rl.sh

echo "[setup] Files uploaded. Now generate RL data on 102..."
echo "[setup] Run on 102:"
echo "  ~/miniconda3/envs/mlx/bin/python ~/LocalSearch/src/ecommerce/gen_e2_variants.py"
echo "  # then after SFT finishes:"
echo "  nohup bash ~/LocalSearch/rl_ecommerce/run_e2_rl.sh > /tmp/e2_rl.log 2>&1 &"
