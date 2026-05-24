#!/bin/bash
# Qwen3-8B GRPO ecommerce — 4× Blackwell RTX PRO 6000 (97GB each)
# Based on v6 (stable on 2-GPU), adapted for 4-GPU
set -x

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/rl_8b

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_memory_monitor_refresh_ms=0

# NCCL: SHM for same-node, disable P2P and IB (Blackwell requirement)
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1
export TORCH_COMPILE_DISABLE=1

/root/miniconda3/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/data/ecommerce/product_rl_train_filtered.parquet \
    data.val_files=/root/autodl-tmp/data/ecommerce/product_rl_train_filtered.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=1200 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    'data.truncation='\''left'\''' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/root/autodl-tmp/sft_8b/ecommerce/output_continue/checkpoint-624 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=LocalSearch-R1 \
    trainer.experiment_name=grpo_qwen3-8b_ecom_2ep \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=9999 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=/root/autodl-tmp/rl_8b/ecommerce/checkpoints_2ep \
    custom_reward_function.path=/root/autodl-tmp/rl_8b/reward_dsl_ecommerce.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "TRAINING_SUCCESS"
else
    echo "TRAINING_FAILED exit_code=$exit_code"
    exit $exit_code
fi
