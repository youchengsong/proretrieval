# Search-R1 / SQL-R1 RL 训练环境配置说明书

> 基于 2× NVIDIA RTX PRO 6000 Blackwell (96GB) 实测，踩坑后总结

## 一、基础环境

```
Python:        3.12.3 (miniconda3)
CUDA:          12.x (需支持 Blackwell / compute capability 12.0)
OS:            Ubuntu 22.04
```

## 二、关键包版本（严格匹配）

```bash
# 核心四件套 — 版本必须对齐，否则各种兼容性问题
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install vllm==0.17.1
pip install verl==0.7.1
pip install transformers==5.8.1

# RL/训练依赖
pip install ray==2.55.1
pip install deepspeed==0.19.0
pip install accelerate==1.13.0
pip install flash-attn==2.8.3
pip install peft==0.19.1

# 数据处理
pip install pandas==3.0.3
pip install pyarrow==24.0.0
pip install sentencepiece==0.2.1
pip install tiktoken==0.13.0
pip install datasets

# 可选（SFT 数据生成用）
pip install rank_bm25
```

## 三、环境变量（写入 ~/.bashrc）

```bash
# 中国镜像（AutoDL 必须）
export HF_ENDPOINT=https://hf-mirror.com

# vllm 设置
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1

# 训练设置
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
```

## 四、模型下载

```bash
# Qwen3-4B（SFT 和 RL 的 base model）
# 如果用 HF 镜像：
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/autodl-tmp/models/Qwen3-4B
```

## 五、⚠️ 必须知道的坑

### 坑 1: Qwen3 tokenizer 会吞标签（最严重）

`tokenizer.apply_chat_template()` 会把 assistant 消息中的 `<search>` `<information>` 标签当成 tool call 自动删除。

**后果**: SFT 数据中搜索格式被完全删除，模型学不会搜索。573 tokens 缩到 189 tokens，你都不会察觉。

**解决**: 手动拼 chat template，绝对不用 `apply_chat_template()`：

```python
def build_chat_text(messages):
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    return "".join(parts)
```

### 坑 2: enable_thinking=False 注入空 think 块

推理时 `enable_thinking=False` + `add_generation_prompt=True` 会自动注入 `<think>\n\n</think>\n\n`，导致模型跳过思考。

**解决**: 推理也用手动 template，不依赖 `add_generation_prompt`。

### 坑 3: SFT label 必须做 masking

默认 `labels = input_ids.copy()` 会在 system+user 上也算 loss，稀释学习信号。

**解决**: 找到 `<|im_start|>assistant\n` 的 token 位置，之前全设 -100：

```python
ASSISTANT_HEADER = "<|im_start|>assistant\n"
assistant_header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

idx = find_sublist(input_ids, assistant_header_ids)
if idx >= 0:
    labels[:idx + len(assistant_header_ids)] = [-100] * (idx + len(assistant_header_ids))
```

### 坑 4: TRL 与 transformers 5.8.1 不兼容

SFTTrainer 参数名全变了（max_length→max_seq_length 等）。

**解决**: 不用 TRL，直接用 `transformers.Trainer`。

### 坑 5: RL reward 函数的 ground_truth 格式

RL parquet 里 `ground_truth` 是 `{"target": ["ID1", ...]}` (dict)，但可能也有 list 或 string 格式。

**解决**: reward 函数里同时处理三种格式：

```python
if isinstance(ground_truth, list):
    target_ids = ground_truth
elif isinstance(ground_truth, dict):
    target_ids = ground_truth.get("target", [])
elif isinstance(ground_truth, str):
    parsed = json.loads(ground_truth)
    # 再判断 list/dict...
```

### 坑 6: verl validation 会卡死训练

`test_freq=200` + `val_files=训练集(20000条)` + `val_batch_size=4` = 每次 validation 跑 5000 batch，耗时数小时。

**解决**: 设 `trainer.test_freq=9999`，跳过中间 validation。训完后单独评估。

### 坑 7: verl checkpoint 是 FSDP 分片格式

保存的 checkpoint 不能直接用 `AutoModel.from_pretrained()` 加载，需要合并 DTensor 分片。

**解决**: 用合并脚本（见 `scripts/merge_fsdp_ckpt.py`），从 `model_world_size_N_rank_*.pt` 提取 `_local_tensor`，按 placement（Replicate/Shard）合并。

### 坑 8: OOM 相关

- SFT: batch_size=32 可能 OOM（特别是长序列），降到 16 + gradient_accumulation_steps=2
- RL: `gpu_memory_utilization=0.5`（给 FSDP 留空间），`param_offload=True` + `optimizer_offload=True`

## 六、SFT 训练命令

```bash
# DeepSpeed ZeRO-2 配置（ds_config.json）
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    },
    "gradient_checkpointing": true,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}

# 启动
deepspeed --num_gpus 2 scripts/train_sft_v2.py \
    --model_name_or_path /path/to/Qwen3-4B \
    --train_file /path/to/sft_data.jsonl \
    --output_dir /path/to/checkpoint \
    --max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --deepspeed ds_config.json
```

## 七、RL (GRPO) 训练命令

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_PARQUET \
    data.val_files=$TRAIN_PARQUET \
    data.train_batch_size=32 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_CHECKPOINT \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name='Search-R1-Baseline' \
    trainer.experiment_name='your_experiment_name' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=9999 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir=$OUTPUT_DIR \
    custom_reward_function.path=$REWARD_SCRIPT \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=False
```

## 八、RL 数据格式 (parquet)

| 列名 | 类型 | 说明 |
|------|------|------|
| data_source | str | 数据来源标识 |
| prompt | numpy array shape (1,) | 包含 1 条 user message dict |
| ability | str | 任务类型 |
| reward_model | str (JSON) | `{"style":"rule","ground_truth":{"target":["ID1","ID2"]}}` |
| extra_info | dict | 额外信息 |

## 九、训练速度参考（2× RTX PRO 6000 96GB）

| 阶段 | 速度 | 总时间 |
|------|------|--------|
| SFT (18K 样本, 3 epochs) | ~3.8s/step | ~50 min |
| RL (1000 steps, batch=32) | ~60s/step | ~16 hr |

## 十、检查训练是否正常的快速命令

```bash
# 看 RL 进度
grep 'Training Progress' /path/to/log | tail -1

# 看 reward 趋势
grep -oP 'critic/score/mean:\K[0-9.]+' /path/to/log | tail -10

# 看 checkpoint
ls -d /path/to/output/global_step_*

# 看 GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```
