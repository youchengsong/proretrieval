#!/bin/bash
# Restart script: Search-R1 full pipeline + SQL-R1 RL resume
# All SFT steps completed. RL steps need to run.
set -x

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR
RESTART_LOG=$LOG_DIR/restart_progress.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $RESTART_LOG
}

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

SCRIPTS_DIR=/root/autodl-tmp/scripts

log "============================================"
log "  RESTART PIPELINE"
log "  1. Search-R1 (SFT+RL ecom, SFT+RL email)"
log "  2. SQL-R1 RL ecom + RL email"
log "============================================"

# Clean up failed Search-R1 checkpoints
rm -rf /root/autodl-tmp/search_r1/checkpoints/search_r1_sft_ecommerce/checkpoint-* 2>/dev/null
log "Cleaned up failed Search-R1 SFT checkpoints"

# ══════════════════════════════════════════
# PART 1: Search-R1 Full Pipeline
# ══════════════════════════════════════════

# ── Search-R1 SFT ecommerce ──
log "Step 1/6: Search-R1 SFT ecommerce..."
SFT_OUTPUT=/root/autodl-tmp/search_r1/checkpoints/search_r1_sft_ecommerce

python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29502 \
    $SCRIPTS_DIR/train_sft.py \
    --model_name Qwen/Qwen3-4B \
    --data_path /root/autodl-tmp/search_r1/sft_data/ecommerce_search_sft.jsonl \
    --output_dir $SFT_OUTPUT \
    --num_epochs 3 \
    --max_length 2048 \
    --learning_rate 2e-5 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee -a $LOG_DIR/search_r1_sft_ecommerce_r2.log

if [ ! -d "$SFT_OUTPUT/final" ]; then
    log "ERROR: Search-R1 SFT ecommerce failed!"
    exit 1
fi
log "Step 1/6 DONE: Search-R1 SFT ecommerce."

# ── Search-R1 RL ecommerce ──
log "Step 2/6: Search-R1 RL ecommerce..."
RL_OUTPUT=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_ecommerce

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_OUTPUT/final \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='Search-R1-Baseline' \
    trainer.experiment_name='search_r1_grpo_ecommerce' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=$RL_OUTPUT \
    custom_reward_function.path=$SCRIPTS_DIR/reward_search_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/search_r1_rl_ecommerce_r2.log

log "Step 2/6 DONE: Search-R1 RL ecommerce."

# ── Search-R1 SFT email ──
log "Step 3/6: Search-R1 SFT email..."
SFT_EMAIL=/root/autodl-tmp/search_r1/checkpoints/search_r1_sft_email

ray stop --force 2>/dev/null
sleep 2

python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29502 \
    $SCRIPTS_DIR/train_sft.py \
    --model_name Qwen/Qwen3-4B \
    --data_path /root/autodl-tmp/search_r1/sft_data/email_search_sft.jsonl \
    --output_dir $SFT_EMAIL \
    --num_epochs 3 \
    --max_length 2048 \
    --learning_rate 2e-5 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee -a $LOG_DIR/search_r1_sft_email_r2.log

log "Step 3/6 DONE: Search-R1 SFT email."

# ── Search-R1 RL email ──
log "Step 4/6: Search-R1 RL email..."
RL_EMAIL=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_email

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/search_r1/rl_data/email_search_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_EMAIL/final \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='Search-R1-Baseline' \
    trainer.experiment_name='search_r1_grpo_email' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=$RL_EMAIL \
    custom_reward_function.path=$SCRIPTS_DIR/reward_search_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/search_r1_rl_email_r2.log

log "Step 4/6 DONE: Search-R1 RL email."

# ══════════════════════════════════════════
# PART 2: SQL-R1 RL (SFT already done)
# ══════════════════════════════════════════

# ── SQL-R1 RL ecommerce ──
log "Step 5/6: SQL-R1 RL ecommerce..."
SQL_SFT_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_ecommerce/final
SQL_RL_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_ecommerce

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/sql_r1/rl_data/ecommerce_sql_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SQL_SFT_ECOM \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='SQL-R1-Baseline' \
    trainer.experiment_name='sql_r1_grpo_ecommerce' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=8 \
    trainer.default_local_dir=$SQL_RL_ECOM \
    custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/sql_r1_rl_ecommerce_r2.log

log "Step 5/6 DONE: SQL-R1 RL ecommerce."

# ── SQL-R1 RL email ──
log "Step 6/6: SQL-R1 RL email..."
SQL_SFT_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_email/final
SQL_RL_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_email

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/sql_r1/rl_data/email_sql_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SQL_SFT_EMAIL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='SQL-R1-Baseline' \
    trainer.experiment_name='sql_r1_grpo_email' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=8 \
    trainer.default_local_dir=$SQL_RL_EMAIL \
    custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/sql_r1_rl_email_r2.log

log "Step 6/6 DONE: SQL-R1 RL email."

log "============================================"
log "  RESTART PIPELINE COMPLETE"
log "============================================"
