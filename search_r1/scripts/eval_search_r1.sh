#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

CKPT=/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_ecommerce_v3/global_step_800/actor
TEST_DATA=/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_test.parquet
TOOL_CONFIG=/root/autodl-tmp/scripts/search_tool_config.yaml
OUTPUT_DIR=/root/autodl-tmp/search_r1/eval_results

mkdir -p $OUTPUT_DIR

echo '[INFO] Checking retrieval server...'
curl -s http://127.0.0.1:8000/retrieve -X POST -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1 || { echo 'ERROR: Retrieval server not running'; exit 1; }
echo '[INFO] Retrieval server OK'

echo '[INFO] Starting inference with step_800 checkpoint...'
python3 -m verl.trainer.main_ppo     algorithm.adv_estimator=grpo     data.train_files=$TEST_DATA     data.val_files=$TEST_DATA     data.train_batch_size=16     data.max_prompt_length=4096     data.max_response_length=500     data.return_raw_chat=True     actor_rollout_ref.model.path=$CKPT     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.temperature=0.0     actor_rollout_ref.rollout.top_k=1     actor_rollout_ref.rollout.do_sample=False     actor_rollout_ref.rollout.n=1     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.rollout.gpu_memory_utilization=0.7     actor_rollout_ref.rollout.multi_turn.enable=True     actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG     actor_rollout_ref.rollout.multi_turn.format=search_r1     actor_rollout_ref.rollout.multi_turn.max_user_turns=3     actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4     actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1     actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500     actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle     actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent     trainer.val_before_train=True     trainer.total_epochs=1     trainer.total_training_steps=0     trainer.n_gpus_per_node=1     trainer.nnodes=1     "trainer.logger=['console']"     trainer.project_name=Search-R1-Eval     trainer.experiment_name=search_r1_eval_step800     trainer.default_local_dir=$OUTPUT_DIR     custom_reward_function.path=/root/autodl-tmp/scripts/reward_search_r1_v3.py     custom_reward_function.name=compute_score     2>&1 | tee /root/autodl-tmp/logs/search_r1_eval_step800.log

echo '[INFO] Eval done'
