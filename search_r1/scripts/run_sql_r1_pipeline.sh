#!/bin/bash
# SQL-R1 Pipeline: SFT -> RL for ecommerce scenario
# Run inside screen session on Blackwell server
set -x

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR
PROGRESS_LOG=$LOG_DIR/sql_r1_progress.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $PROGRESS_LOG
}

log "========== SQL-R1 Pipeline Starting =========="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "CUDA devices: $CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

WORK_DIR=/root/autodl-tmp/sql_r1
SCRIPTS_DIR=/root/autodl-tmp/scripts
MODEL_NAME="Qwen/Qwen3-4B"

# ── Step 0: Download model if needed ──
log "Step 0: Checking model..."
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True); print('Model cached')" 2>/dev/null
if [ $? -ne 0 ]; then
    log "Downloading $MODEL_NAME..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('$MODEL_NAME', torch_dtype=torch.bfloat16, trust_remote_code=True)
print('Model download complete.')
" 2>&1 | tee -a $PROGRESS_LOG
fi

# ── Step 1: Prepare data ──
log "Step 1: Preparing SQL-R1 data..."
python3 $SCRIPTS_DIR/prepare_sql_r1_data.py 2>&1 | tee -a $PROGRESS_LOG
if [ $? -ne 0 ]; then
    log "ERROR: Data preparation failed!"
    exit 1
fi
log "Step 1 DONE: Data prepared."

# ── Step 2: SFT Training (ecommerce) ──
log "Step 2: Starting SFT training (ecommerce)..."
SFT_OUTPUT=$WORK_DIR/checkpoints/sql_r1_sft_ecommerce

python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    $SCRIPTS_DIR/train_sft.py \
    --model_name $MODEL_NAME \
    --data_path $WORK_DIR/sft_data/ecommerce_sql_sft.jsonl \
    --output_dir $SFT_OUTPUT \
    --num_epochs 3 \
    --max_length 2048 \
    --learning_rate 2e-5 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee -a $LOG_DIR/sql_r1_sft_ecommerce.log

if [ ! -d "$SFT_OUTPUT/final" ]; then
    log "ERROR: SFT training failed - no final checkpoint!"
    exit 1
fi
log "Step 2 DONE: SFT ecommerce complete. Model at $SFT_OUTPUT/final"

# ── Step 3: RL Training (ecommerce) ──
log "Step 3: Starting RL GRPO training (ecommerce)..."
RL_OUTPUT=$WORK_DIR/checkpoints/sql_r1_rl_ecommerce

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$WORK_DIR/rl_data/ecommerce_sql_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_OUTPUT/final \
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
    trainer.default_local_dir=$RL_OUTPUT \
    custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/sql_r1_rl_ecommerce.log

log "Step 3 DONE: RL ecommerce complete."

# ── Step 4: SFT Training (email) ──
log "Step 4: Starting SFT training (email)..."
SFT_EMAIL_OUTPUT=$WORK_DIR/checkpoints/sql_r1_sft_email

ray stop --force 2>/dev/null
sleep 2

python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    $SCRIPTS_DIR/train_sft.py \
    --model_name $MODEL_NAME \
    --data_path $WORK_DIR/sft_data/email_sql_sft.jsonl \
    --output_dir $SFT_EMAIL_OUTPUT \
    --num_epochs 3 \
    --max_length 2048 \
    --learning_rate 2e-5 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee -a $LOG_DIR/sql_r1_sft_email.log

log "Step 4 DONE: SFT email complete."

# ── Step 5: RL Training (email) ──
log "Step 5: Starting RL GRPO training (email)..."
RL_EMAIL_OUTPUT=$WORK_DIR/checkpoints/sql_r1_rl_email

ray stop --force 2>/dev/null
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$WORK_DIR/rl_data/email_sql_rl_train.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_EMAIL_OUTPUT/final \
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
    trainer.default_local_dir=$RL_EMAIL_OUTPUT \
    custom_reward_function.path=$SCRIPTS_DIR/reward_sql_r1.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    2>&1 | tee -a $LOG_DIR/sql_r1_rl_email.log

log "Step 5 DONE: RL email complete."
log "========== SQL-R1 Pipeline Complete =========="
