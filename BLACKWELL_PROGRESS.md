# Blackwell 服务器 8B RL 训练进度

> 服务器: ssh -p 39581 root@connect.westb.seetacloud.com
> 硬件: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97.9GB each)
> 工作目录: /root/autodl-tmp/
> CPU: 50 核心, 内存: 240 GB

## 当前任务：Qwen3-8B GRPO 电商

### 状态：训练中（Step 100/625 per epoch）

- 启动时间: 2026-05-21 04:49 UTC
- 脚本: `run_grpo_ecommerce_v6.sh` (v6 = 最终稳定版)
- Screen: `train_grpo_v6`
- 日志: `/root/autodl-tmp/rl_8b/train_grpo_ecom_8b_v6.log`
- Checkpoint: `/root/autodl-tmp/rl_8b/ecommerce/checkpoints/global_step_100`

### Score 趋势 (critic/score/mean)

| Step | Score | Step | Score |
|------|-------|------|-------|
| 1 | 0.556 | 6 | 0.573 |
| 2 | 0.516 | 7 | 0.642 |
| 3 | 0.483 | 8 | 0.667 |
| 4 | 0.644 | 9 | 0.627 |
| 5 | 0.553 | 10 | 0.635 |

Step 11-100 持续攀升，峰值 **0.830**（step ~83），平均 0.73-0.75。
对比 4B GRPO 在 step 300 才达到 0.835，8B 学习速度显著更快。

### 训练参数

- Base model: `/root/autodl-tmp/sft_8b/ecommerce/output/final` (8B SFT)
- Batch size: 32, micro_batch: 2/GPU, n=4 rollouts
- LR: 1e-6, KL loss: 0.01 (low_var_kl)
- FSDP: param_offload=False, **optimizer_offload=False** (关键修复)
- vLLM: gpu_memory_utilization=0.25, TP=2, max_model_len=2048
- Attention: SDPA (Flash Attention 2 在 Blackwell 上不兼容)
- Model dtype: bf16
- Save freq: 50, Test freq: 100, Total epochs: 5
- GPU 峰值显存: ~75GB / 98GB

### Blackwell 适配修复历程

在 Blackwell GPU 上训练 8B 模型经历了 6 个版本的迭代：

1. **v1-v2**: Flash Attention 2 + float32 导致 CUDA crash → 改用 SDPA + bf16
2. **v3**: param_offload=True 导致模型加载异常 → 关闭 param_offload
3. **v4**: NCCL Socket transport 在 step 19 崩溃 → 改用 SHM transport
4. **v5**: NCCL collective timeout 在 step 15 → **根因: optimizer_offload=True 导致 CPU 优化器步骤在两个 rank 间不同步**
5. **v6 (当前)**: optimizer_offload=False + NCCL heartbeat timeout 3600s → **稳定运行 100+ steps**

关键环境变量:
```bash
export NCCL_P2P_DISABLE=1     # Blackwell 不支持 SymmMemCommunicator
export NCCL_IB_DISABLE=1      # 无 InfiniBand
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export VLLM_USE_V1=0          # vLLM v1 engine 不兼容 verl
```

### 已知问题

1. **Validation 极慢**: test_freq=100 触发 validation，val_set=train_set=20000 条，val_batch_size=4 → ~5000 batch → ~7 小时/次。每 100 步浪费 7 小时。建议后续 test_freq 设为 625+。
2. **磁盘紧张**: 550GB 总量，每个 checkpoint 46GB。需及时删除旧 checkpoint，只保留最新一个。

### 训练速度

- Step time: ~108s/step (稳态)
- 每 epoch (625 steps): ~19 小时训练 + ~42 小时 validation (6 次)
- 如关闭/减少 validation: ~19 小时/epoch

## 待训练任务

| 任务 | 状态 | Base Model |
|------|------|------------|
| 8B GRPO 电商 | 训练中 (step 100) | sft_8b/ecommerce |
| 8B DAPO 电商 | 待训练 | sft_8b/ecommerce |
| 8B GRPO 邮件 | 待训练 | sft_8b/email |
| 8B DAPO 邮件 | 待训练 | sft_8b/email |

## 关键路径

- RL 8B 训练: `/root/autodl-tmp/rl_8b/`
- SFT 8B 模型: `/root/autodl-tmp/sft_8b/{ecommerce,email}/output/final/`
- Base 模型: `/root/autodl-tmp/models/{Qwen3-4B,Qwen3-8B,Qwen3-Embedding-8B}`
- 数据: `/root/autodl-tmp/data/{ecommerce,email}/`
- Embedding server: port 9123 (screen `embed`), text-only (Qwen3-Embedding-8B)
