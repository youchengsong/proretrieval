#!/bin/bash
set -x
export WANDB_API_KEY="45bbc2ae38c9c6ae72dc3ea81333cd64881629b3"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 抑制冗余告警
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/LocalSearch/rl_ecommerce

# 清理之前的 ray 和 embedding server
ray stop --force 2>/dev/null
pkill -f embedding_server.py 2>/dev/null
sleep 2

# 启动 embedding server（GPU 2，端口 9123）
echo "[run] Starting embedding server on GPU 2..."
EMBED_GPU=2 ~/miniconda3/envs/mlx/bin/python embedding_server.py &
EMBED_PID=$!

# 等待 server 就绪（静默轮询）
set +x
echo "[run] Waiting for embedding server..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:9123/health > /dev/null 2>&1; then
        echo "[run] Embedding server ready (pid=$EMBED_PID)"
        break
    fi
    if ! kill -0 $EMBED_PID 2>/dev/null; then
        echo "[run] ERROR: Embedding server died!"
        exit 1
    fi
    sleep 2
done
set -x

# 训练用 GPU 4,5（embedding 在 GPU 2，GPU 3 被其他用户占用）
export CUDA_VISIBLE_DEVICES=4,5

~/miniconda3/envs/mlx/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/sunzhen/LocalSearch/data/ecommerce/train/product_rl_train_filtered.parquet \
    data.val_files=/home/sunzhen/LocalSearch/data/ecommerce/test/product_rl_test_filtered.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    "data.truncation='error'" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/s3-migration/rl_ecommerce/checkpoints/global_step_1200_merged \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','tensorboard','wandb']" \
    trainer.project_name='LocalSearch-R1' \
    trainer.experiment_name='grpo_qwen3-4b_ecommerce' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/data/s3-migration/rl_ecommerce/checkpoints_v2 \
    custom_reward_function.path=/home/sunzhen/LocalSearch/rl_ecommerce/reward_dsl.py \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False \
    trainer.resume_mode=disable

# 训练结束，关闭 embedding server
echo "[run] Training finished, stopping embedding server..."
kill $EMBED_PID 2>/dev/null
wait $EMBED_PID 2>/dev/null
