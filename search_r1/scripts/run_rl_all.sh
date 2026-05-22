#!/bin/bash
# Fixed RL script: runs all 4 RL GRPO steps with data.val_files set
# Fixes: FileNotFoundError on default val_files path (gsm8k/test.parquet)
set -x

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR
RL_LOG=$LOG_DIR/rl_all_progress.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $RL_LOG
}

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

SCRIPTS_DIR=/root/autodl-tmp/scripts

log "============================================"
log "  RL-ALL: 4 GRPO steps with val_files fix"
log "============================================"

# ── Step 1: Search-R1 RL ecommerce ──
SFT_SEARCH_ECOM=/root/autodl-tmp/search_r1/checkpoints/search_r1_sft_ecommerce/final
RL_SEARCH_ECOM=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_ecommerce
TRAIN_SEARCH_ECOM=/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_train.parquet

if [ -d "$SFT_SEARCH_ECOM" ]; then
    log "Step 1/4: Search-R1 RL ecommerce"
    ray stop --force 2>/dev/null
    sleep 2

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_SEARCH_ECOM \
        data.val_files=$TRAIN_SEARCH_ECOM \
        data.train_batch_size=32 \
        data.val_batch_size=4 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$SFT_SEARCH_ECOM \
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
        trainer.default_local_dir=$RL_SEARCH_ECOM \
        custom_reward_function.path=$SCRIPTS_DIR/reward_search_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/search_r1_rl_ecommerce_r3.log

    log "Step 1/4 DONE: Search-R1 RL ecommerce."
else
    log "SKIP Step 1/4: $SFT_SEARCH_ECOM not found"
fi

# ── Step 2: Search-R1 RL email ──
SFT_SEARCH_EMAIL=/root/autodl-tmp/search_r1/checkpoints/search_r1_sft_email/final
RL_SEARCH_EMAIL=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_email
TRAIN_SEARCH_EMAIL=/root/autodl-tmp/search_r1/rl_data/email_search_rl_train.parquet

if [ -d "$SFT_SEARCH_EMAIL" ]; then
    log "Step 2/4: Search-R1 RL email"
    ray stop --force 2>/dev/null
    sleep 2

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_SEARCH_EMAIL \
        data.val_files=$TRAIN_SEARCH_EMAIL \
        data.train_batch_size=32 \
        data.val_batch_size=4 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$SFT_SEARCH_EMAIL \
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
        trainer.default_local_dir=$RL_SEARCH_EMAIL \
        custom_reward_function.path=$SCRIPTS_DIR/reward_search_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/search_r1_rl_email_r3.log

    log "Step 2/4 DONE: Search-R1 RL email."
else
    log "SKIP Step 2/4: $SFT_SEARCH_EMAIL not found"
fi

# ── Step 3: SQL-R1 RL ecommerce ──
SFT_SQL_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_ecommerce/final
RL_SQL_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_ecommerce
TRAIN_SQL_ECOM=/root/autodl-tmp/sql_r1/rl_data/ecommerce_sql_rl_train.parquet

if [ -d "$SFT_SQL_ECOM" ]; then
    log "Step 3/4: SQL-R1 RL ecommerce"
    ray stop --force 2>/dev/null
    sleep 2

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_SQL_ECOM \
        data.val_files=$TRAIN_SQL_ECOM \
        data.train_batch_size=32 \
        data.val_batch_size=4 \
        data.max_prompt_length=2048 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$SFT_SQL_ECOM \
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
        trainer.default_local_dir=$RL_SQL_ECOM \
        custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/sql_r1_rl_ecommerce_r3.log

    log "Step 3/4 DONE: SQL-R1 RL ecommerce."
else
    log "SKIP Step 3/4: $SFT_SQL_ECOM not found"
fi

# ── Step 4: SQL-R1 RL email ──
SFT_SQL_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_email/final
RL_SQL_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_email
TRAIN_SQL_EMAIL=/root/autodl-tmp/sql_r1/rl_data/email_sql_rl_train.parquet

if [ -d "$SFT_SQL_EMAIL" ]; then
    log "Step 4/4: SQL-R1 RL email"
    ray stop --force 2>/dev/null
    sleep 2

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_SQL_EMAIL \
        data.val_files=$TRAIN_SQL_EMAIL \
        data.train_batch_size=32 \
        data.val_batch_size=4 \
        data.max_prompt_length=2048 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$SFT_SQL_EMAIL \
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
        trainer.default_local_dir=$RL_SQL_EMAIL \
        custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/sql_r1_rl_email_r3.log

    log "Step 4/4 DONE: SQL-R1 RL email."
else
    log "SKIP Step 4/4: $SFT_SQL_EMAIL not found"
fi

log "============================================"
log "  RL-ALL COMPLETE"
log "============================================"
log "Checkpoints:"
ls -la /root/autodl-tmp/search_r1/checkpoints/*/final/ 2>/dev/null | tee -a $RL_LOG
ls -la /root/autodl-tmp/sql_r1/checkpoints/*/final/ 2>/dev/null | tee -a $RL_LOG
