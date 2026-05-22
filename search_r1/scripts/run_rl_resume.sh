#!/bin/bash
# Resume script: runs all RL steps that failed due to Hydra config bug,
# plus Search-R1 SFT+RL pipeline
set -x

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR
RESUME_LOG=$LOG_DIR/resume_progress.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $RESUME_LOG
}

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

SCRIPTS_DIR=/root/autodl-tmp/scripts

log "============================================"
log "  RESUME PIPELINE: RL steps + Search-R1"
log "============================================"

# ── SQL-R1 RL ecommerce ──
SFT_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_ecommerce/final
RL_ECOM=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_ecommerce

if [ -d "$SFT_ECOM" ]; then
    log "Step 1: SQL-R1 RL ecommerce (from SFT checkpoint)"
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
        actor_rollout_ref.model.path=$SFT_ECOM \
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
        trainer.default_local_dir=$RL_ECOM \
        custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/sql_r1_rl_ecommerce_resume.log

    log "Step 1 DONE: SQL-R1 RL ecommerce."
else
    log "SKIP Step 1: SFT ecommerce checkpoint not found at $SFT_ECOM"
fi

# ── SQL-R1 RL email ──
SFT_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_sft_email/final
RL_EMAIL=/root/autodl-tmp/sql_r1/checkpoints/sql_r1_rl_email

if [ -d "$SFT_EMAIL" ]; then
    log "Step 2: SQL-R1 RL email (from SFT checkpoint)"
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
        actor_rollout_ref.model.path=$SFT_EMAIL \
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
        trainer.default_local_dir=$RL_EMAIL \
        custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
        custom_reward_function.name=compute_score \
        trainer.val_before_train=False \
        2>&1 | tee -a $LOG_DIR/sql_r1_rl_email_resume.log

    log "Step 2 DONE: SQL-R1 RL email."
else
    log "SKIP Step 2: SFT email checkpoint not found at $SFT_EMAIL"
fi

# Search-R1 should already be handled by the original pipeline with fixed scripts
log "NOTE: Search-R1 should have been run by the original pipeline. If not, run:"
log "  bash /root/autodl-tmp/scripts/run_search_r1_pipeline.sh"

log "============================================"
log "  RESUME PIPELINE COMPLETE"
log "============================================"

log "Checkpoints:"
ls -la /root/autodl-tmp/sql_r1/checkpoints/*/final/ 2>/dev/null | tee -a $RESUME_LOG
ls -la /root/autodl-tmp/search_r1/checkpoints/*/final/ 2>/dev/null | tee -a $RESUME_LOG
