#!/bin/bash
# DeepRetrieval faithful reproduction: GRPO + Qwen3-4B + BM25
# Adapted from rl_ecommerce/run_rl_grpo.sh
# 4× A6000, 1 epoch
set -x

export WANDB_API_KEY="45bbc2ae38c9c6ae72dc3ea81333cd64881629b3"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

PYTHON=~/miniconda3/envs/mlx/bin/python
WORK_DIR=~/LocalSearch/deepretrieval
DATA_DIR=$WORK_DIR/data
REWARD_SCRIPT=$WORK_DIR/reward_deepretrieval.py
CKPT_DIR=$WORK_DIR/checkpoints

mkdir -p $CKPT_DIR

cd $WORK_DIR

ray stop --force 2>/dev/null
sleep 2

# Verify Lucene index
$PYTHON -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('$WORK_DIR/index')
print(f'Lucene index OK: {s.num_docs} docs')
" || { echo "ERROR: Lucene index not found. Run setup_deepretrieval.sh first."; exit 1; }

export CUDA_VISIBLE_DEVICES=3,4,5,6

$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    "data.truncation='error'" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name='LocalSearch-R1' \
    trainer.experiment_name='deepretrieval_grpo_qwen3_4b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$CKPT_DIR \
    custom_reward_function.path=$REWARD_SCRIPT \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False

echo "[$(date)] Training finished."
