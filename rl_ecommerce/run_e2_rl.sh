#!/bin/bash
# E2 Action Space Ablation - RL (GRPO) Training
# Runs 4 variants sequentially: sql_only -> sql_text -> text_only -> sql_image
# Hardware: 102 server (7x A6000 48GB)
#   GPU 0: occupied (OCR)
#   GPU 5: embedding server (~32GB for Qwen3-Embedding-8B + Qwen3-VL-Embedding-8B)
#   GPUs 1,2,3,4: RL training (GRPO via verl)
#
# Prerequisites:
#   1. E2 SFT models at: /data/s3-migration/sft_localSearch/e2_checkpoints/{variant}/final
#   2. RL data parquets at: /home/sunzhen/LocalSearch/data/ecommerce/e2_variants/
#   3. Embedding models: Qwen3-Embedding-8B and Qwen3-VL-Embedding-8B
#   4. Product feature parquets in data/ecommerce/{train,test}/
#
# Usage:
#   nohup bash run_e2_rl.sh > /tmp/e2_rl.log 2>&1 &

set -e
PYTHON=~/miniconda3/envs/mlx/bin/python

export WANDB_API_KEY="45bbc2ae38c9c6ae72dc3ea81333cd64881629b3"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# ---------- paths ----------
DATA_DIR=/home/sunzhen/LocalSearch/data/ecommerce
E2_DATA_DIR=${DATA_DIR}/e2_variants
E2_CKPT_BASE=/data/s3-migration/sft_localSearch/e2_rl_checkpoints
SFT_CKPT_BASE=/data/s3-migration/sft_localSearch/e2_checkpoints
REWARD_SCRIPT=/home/sunzhen/LocalSearch/rl_ecommerce/reward_dsl_e2.py
EMBED_SERVER_SCRIPT=/home/sunzhen/LocalSearch/rl_ecommerce/embedding_server.py

cd ~/LocalSearch/rl_ecommerce

# ---------- Step 0: preflight checks ----------
echo "[$(date)] Preflight checks..."

for V in sql_only sql_text text_only sql_image; do
    SFT_DIR="${SFT_CKPT_BASE}/${V}/final"
    if [ ! -d "$SFT_DIR" ]; then
        echo "ERROR: SFT model missing: $SFT_DIR"
        exit 1
    fi
done

if [ ! -f "${E2_DATA_DIR}/train_sql_only.parquet" ]; then
    echo "[$(date)] E2 RL data not found, generating..."
    $PYTHON /home/sunzhen/LocalSearch/src/ecommerce/gen_e2_variants.py
fi

for V in sql_only sql_text text_only sql_image; do
    if [ ! -f "${E2_DATA_DIR}/train_${V}.parquet" ]; then
        echo "ERROR: RL data missing: ${E2_DATA_DIR}/train_${V}.parquet"
        exit 1
    fi
done

if [ ! -f "$REWARD_SCRIPT" ]; then
    echo "ERROR: Reward script missing: $REWARD_SCRIPT"
    exit 1
fi
echo "[$(date)] All preflight checks passed."

# ---------- Step 1: start embedding server on GPU 5 ----------
ray stop --force 2>/dev/null
pkill -f embedding_server.py 2>/dev/null
sleep 2

echo "[$(date)] Starting embedding server on GPU 5..."
EMBED_GPU=5 nohup $PYTHON $EMBED_SERVER_SCRIPT > /tmp/e2_embed_server.log 2>&1 &
EMBED_PID=$!

set +x
echo "[$(date)] Waiting for embedding server (pid=$EMBED_PID)..."
for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:9123/health > /dev/null 2>&1; then
        echo "[$(date)] Embedding server ready."
        break
    fi
    if ! kill -0 $EMBED_PID 2>/dev/null; then
        echo "[$(date)] ERROR: Embedding server died! Check /tmp/e2_embed_server.log"
        exit 1
    fi
    if [ $i -eq 180 ]; then
        echo "[$(date)] ERROR: Embedding server timeout after 360s"
        exit 1
    fi
    sleep 2
done
set -x

# ---------- Step 2: train 4 variants sequentially ----------
VARIANTS=("sql_only" "sql_text" "text_only" "sql_image")

for VARIANT in "${VARIANTS[@]}"; do
    echo "============================================="
    echo "[$(date)] E2 RL: starting variant=$VARIANT"
    echo "============================================="

    SFT_MODEL="${SFT_CKPT_BASE}/${VARIANT}/final"
    CKPT_DIR="${E2_CKPT_BASE}/${VARIANT}"
    TRAIN_DATA="${E2_DATA_DIR}/train_${VARIANT}.parquet"
    TEST_DATA="${E2_DATA_DIR}/test_${VARIANT}.parquet"

    mkdir -p "$CKPT_DIR"

    # Clean up ray between variants
    ray stop --force 2>/dev/null
    pkill -f "VLLM::Worker" 2>/dev/null
    sleep 3

    export CUDA_VISIBLE_DEVICES=1,2,3,4
    export ACTION_SPACE=$VARIANT

    $PYTHON -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA \
        data.val_files=$TEST_DATA \
        data.train_batch_size=32 \
        data.val_batch_size=4 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=False \
        "data.truncation='left'" \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$SFT_MODEL \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        "trainer.logger=['console','wandb']" \
        trainer.project_name='LocalSearch-R1-E2' \
        trainer.experiment_name="grpo_e2_${VARIANT}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=1 \
        trainer.default_local_dir=$CKPT_DIR \
        custom_reward_function.path=$REWARD_SCRIPT \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False

    echo "[$(date)] E2 RL: finished variant=$VARIANT"
done

# ---------- cleanup ----------
echo "[$(date)] All 4 variants done. Stopping embedding server..."
kill $EMBED_PID 2>/dev/null
wait $EMBED_PID 2>/dev/null
echo "[$(date)] E2 RL training complete!"
