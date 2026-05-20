#!/bin/bash
set -x
export WANDB_API_KEY="45bbc2ae38c9c6ae72dc3ea81333cd64881629b3"
cd /home/sunzhen/DeepRetrieval_ecommerce
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
# Removed PYTORCH_CUDA_ALLOC_CONF to avoid vLLM engine core init failure
# Force vLLM v0 engine to avoid multiprocessing executor issues
export VLLM_USE_V1=0

PYTHON=~/miniconda3/envs/mlx/bin/python
BASE_MODEL=/data/LLM_Models/Qwen3-4B
DATA_DIR=/home/sunzhen/DeepRetrieval_ecommerce/data
REWARD_SCRIPT=/home/sunzhen/DeepRetrieval_ecommerce/reward_ecommerce.py
CKPT_DIR=/home/sunzhen/DeepRetrieval_ecommerce/checkpoints_v7
mkdir -p $CKPT_DIR

ray stop --force 2>/dev/null
sleep 3

$PYTHON -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=${DATA_DIR}/train.parquet \
 data.val_files=${DATA_DIR}/val.parquet \
 data.train_batch_size=192 \
 data.val_batch_size=48 \
 data.max_prompt_length=512 \
 data.max_response_length=1024 \
 data.filter_overlong_prompts=False \
 "data.truncation='error'" \
 data.return_raw_chat=True \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=192 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.use_dynamic_bsz=False \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.enable_chunked_prefill=True \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.free_cache_engine=True \
 actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
 actor_rollout_ref.rollout.val_kwargs.n=1 \
 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.ref.fsdp_config.param_offload=False \
 algorithm.use_kl_in_reward=False \
 trainer.critic_warmup=0 \
 "trainer.logger=['console','wandb']" \
 trainer.project_name='LocalSearch-R1' \
 trainer.experiment_name='deepretrieval_ecommerce_qwen3_4b_v7' \
 trainer.n_gpus_per_node=6 \
 trainer.nnodes=1 \
 trainer.save_freq=50 \
 trainer.test_freq=100 \
 trainer.total_epochs=1 \
 trainer.default_local_dir=$CKPT_DIR \
 custom_reward_function.path=$REWARD_SCRIPT \
 custom_reward_function.name=compute_score \
 trainer.val_before_train=False
