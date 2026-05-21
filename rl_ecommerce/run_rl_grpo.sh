#!/bin/bash
# Qwen3-4B RL GRPO 训练
# GPU 3,4,5,6 训练 | GPU 2 Embedding server
set -x
export WANDB_API_KEY="45bbc2ae38c9c6ae72dc3ea81333cd64881629b3"
cd ~/LocalSearch/rl_ecommerce

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 vLLM 内存池不兼容，已移除

PYTHON=~/miniconda3/envs/mlx/bin/python
SFT_MODEL=/data/s3-migration/sft_localSearch/output/final
CKPT_DIR=/data/s3-migration/rl_ecommerce/checkpoints

mkdir -p $CKPT_DIR
ray stop --force 2>/dev/null
pkill -f "VLLM::Worker" 2>/dev/null
sleep 2

# Embedding server 健康检查
if ! curl -s http://127.0.0.1:9123/health > /dev/null 2>&1; then
    echo "[$(date)] ERROR: Embedding server not reachable"
    exit 1
fi
echo "[$(date)] Embedding server OK"

export CUDA_VISIBLE_DEVICES=3,4,5,6

$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/sunzhen/LocalSearch/data/ecommerce/train/product_rl_train_filtered.parquet \
    data.val_files=/home/sunzhen/LocalSearch/data/ecommerce/test/product_rl_test_filtered.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=4 \
    data.max_prompt_length=1200 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    "data.truncation='left'" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    trainer.project_name='LocalSearch-R1' \
    trainer.experiment_name='grpo_qwen3-4b_ecommerce' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=$CKPT_DIR \
    custom_reward_function.path=/home/sunzhen/LocalSearch/rl_ecommerce/reward_dsl.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False

echo "[$(date)] Training finished."
