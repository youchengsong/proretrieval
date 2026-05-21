#!/bin/bash
set -e

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_DIR/search_rl_email_progress.log
}

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SCRIPTS_DIR=/root/autodl-tmp/scripts

# ── Email RL (with online BM25 retrieval via verl 0.7.1 multi-turn) ──

SFT_CKPT=/root/autodl-tmp/search_r1/checkpoints/email_sft_v2/final
TRAIN_DATA=/root/autodl-tmp/search_r1/rl_data/email_search_rl_train_fixed.parquet
OUTPUT=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_email_v3
REWARD_SCRIPT=$SCRIPTS_DIR/reward_search_r1_v3.py
TOOL_CONFIG=$SCRIPTS_DIR/search_tool_config.yaml

log "============================================"
log "  Search-R1 RL: Email GRPO"
log "  verl 0.7.1 multi-turn + BM25 retrieval"
log "  rollout.n=4 for proper GRPO grouping"
log "============================================"

# Ensure retrieval server is running
if ! curl -s http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
    log "ERROR: Retrieval server not running at http://127.0.0.1:8000"
    exit 1
fi
log "Retrieval server is up"

# Clean old checkpoints
if [ -d "$OUTPUT" ]; then
    log "Removing old email RL checkpoints to start fresh"
    rm -rf $OUTPUT
fi

ray stop --force 2>/dev/null || true
sleep 3

log "Starting RL training with rollout.n=4"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TRAIN_DATA \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_CKPT \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.rollout.multi_turn.format=search_r1 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='Search-R1-Baseline' \
    trainer.experiment_name='search_r1_grpo_email' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=9999 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir=$OUTPUT \
    custom_reward_function.path=$REWARD_SCRIPT \
    custom_reward_function.name=compute_score \
    2>&1 | tee -a $LOG_DIR/search_r1_rl_email.log

log "Email RL DONE"
