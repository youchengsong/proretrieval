# Blackwell 服务器训练进度

> 服务器: ssh -p 34907 root@connect.westb.seetacloud.com (密码: JUmCDt0QqyKf)
> 硬件: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97.9GB each)
> 工作目录: /root/autodl-tmp/
> CPU: 50 核心, 内存: 240 GB
> 磁盘: 350GB (截至 2026-05-21 清理后使用 99G/350G)

---

## 当前任务 1：Search-R1 邮件 RL 训练

### 状态：从 step 200 恢复训练中 (→1000)

- 启动时间: 2026-05-21 11:34 UTC (恢复)
- 脚本: `run_search_rl_email_resume.sh`
- 日志: `/root/autodl-tmp/logs/email_resume_nohup.log`
- Checkpoint: `search_r1_rl_email_v3/global_step_200` (47G, 完整)
- 磁盘: 252G 可用 (清理了 254G 旧 checkpoint)

### Reward 趋势 (step 1-399, 磁盘满前)

| 区间 | Reward | 备注 |
|------|--------|------|
| step 1-50 | 0.047 | 初始值很低 |
| step 100 | ~0.10 | 稳步上升 |
| step 200 | ~0.15 | checkpoint 保存 |
| step 250 | ~0.25 | 峰值 |
| step 350-399 | 0.17-0.20 | 磁盘满崩溃前 |

### 关键修复: verl agent loop stop string

邮件 SFT 模型在 `</search>` 后继续幻觉生成 `<information>` 内容，填满 500 token response_length。verl 的 termination check 在 tool call extraction 之前执行，导致 tool call 永远无法提取。

**解决方案**: 在 `tool_agent_loop.py` 中为 search_r1 格式注入 `</search>` 作为 vLLM stop string。
详见: `search_r1/patches/patch_agent_loop_stop_string.py`

### 磁盘清理记录 (2026-05-21)

| 删除项 | 大小 | 原因 |
|--------|------|------|
| search_r1_rl_email_v3/global_step_400 | 39G | 磁盘满导致损坏 |
| email_sft_v2/checkpoint-789 | 61G | 中间 checkpoint，有 final |
| ecommerce_sft_v2/checkpoint-867 | 61G | 中间 checkpoint，有 final |
| search_r1_rl_ecommerce_v3 | 76G | 已有 merged 模型 |
| search_r1_sft_ecommerce + email | 16.6G | 旧版 SFT v1 |
| 合计 | ~254G | |

保留:
- `email_sft_v2/final` (8.3G) — email RL base model
- `search_r1_rl_email_v3/global_step_200` (47G) — 恢复训练 checkpoint
- `rl_ecom_step200_merged` (7.6G) — ecommerce 评测模型
- `search_r1_step800_merged` (7.6G) — ecommerce 评测模型
- `ecommerce_sft_v2/final` (8.3G) — ecommerce SFT base

### 训练参数 (4B Search-R1)

- Base model: `email_sft_v2/final` (Qwen3-4B SFT)
- Batch: 16, micro_batch: 2/GPU, rollout.n=4
- LR: 1e-6, KL: 0.001 (low_var_kl), entropy: 0.001
- FSDP: param_offload=True, optimizer_offload=True
- vLLM: gpu_memory_utilization=0.5, TP=1
- Multi-turn: max_user_turns=3, max_assistant_turns=4
- Total: 1000 steps, save_freq=200
- resume_mode=auto (从 global_step_200 恢复)

---

## 已完成：Search-R1 电商 RL + 评估

### 训练

- 脚本: `run_search_rl_v3_fixed.sh`
- 1000 steps, Qwen3-4B GRPO
- Checkpoints: step_200 (merged: `rl_ecom_step200_merged`), step_800 (merged: `search_r1_step800_merged`)

### 评估结果 (step_800, N=2773)

| 指标 | 值 | 说明 |
|------|-----|------|
| Raw EM | 0.18% (5/2773) | 模型输出 "Doc N" 格式 |
| **Post-processed EM** | **45.87%** (1272/2773) | "Doc N" → product_id mapping |
| No answer | 0 | 所有样本都有 answer |
| Answer types | doc_mapped=1666, raw=1331 | 60% 为 "Doc N" 格式 |

> 远低于 DSL 方法 (GRPO=80.9%)，因为 BM25 text-only 无法利用结构化/图像特征。也低于 DeepRetrieval (63.8%)，因为 Boolean query 比纯 text search 更精准。

---

## 当前任务 2：Qwen3-8B GRPO 电商

### 状态：训练中（Step 100/625 per epoch）

- 启动时间: 2026-05-21 04:49 UTC
- 脚本: `run_grpo_ecommerce_v6.sh` (v6 = 最终稳定版)
- Screen: `train_grpo_v6`
- 日志: `/root/autodl-tmp/rl_8b/train_grpo_ecom_8b_v6.log`

### Score 趋势

Step 11-100 持续攀升，峰值 **0.830**（step ~83），平均 0.73-0.75。
对比 4B GRPO 在 step 300 才达到 0.835，8B 学习速度显著更快。

### 训练参数 (8B)

- Base model: `/root/autodl-tmp/sft_8b/ecommerce/output/final`
- Batch: 32, micro_batch: 2/GPU, n=4 rollouts
- LR: 1e-6, KL: 0.01 (low_var_kl)
- FSDP: param_offload=False, **optimizer_offload=False** (关键修复)
- vLLM: gpu_memory_utilization=0.25, TP=2, max_model_len=2048
- Attention: SDPA (Flash Attention 2 在 Blackwell 上不兼容)

### Blackwell 适配修复历程

1. **v1-v2**: Flash Attention 2 + float32 → CUDA crash → 改用 SDPA + bf16
2. **v3**: param_offload=True → 模型加载异常 → 关闭 param_offload
3. **v4**: NCCL Socket transport → step 19 崩溃 → 改用 SHM transport
4. **v5**: NCCL collective timeout → **根因: optimizer_offload=True 导致 CPU 优化器步骤不同步**
5. **v6 (当前)**: optimizer_offload=False + NCCL heartbeat timeout 3600s → **稳定 100+ steps**

```bash
export NCCL_P2P_DISABLE=1     # Blackwell 不支持 SymmMemCommunicator
export NCCL_IB_DISABLE=1      # 无 InfiniBand
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export VLLM_USE_V1=0          # vLLM v1 engine 不兼容 verl
```

---

## 关键路径

### Search-R1 (4B)
- 脚本: `/root/autodl-tmp/scripts/`
- SFT 模型: `/root/autodl-tmp/search_r1/checkpoints/{ecommerce,email}_sft_v2/final/`
- RL checkpoint: `/root/autodl-tmp/search_r1/checkpoints/search_r1_rl_email_v3/`
- Merged 模型: `/root/autodl-tmp/search_r1/checkpoints/{rl_ecom_step200_merged,search_r1_step800_merged}/`
- RL 数据: `/root/autodl-tmp/search_r1/rl_data/`
- 评估结果: `/root/autodl-tmp/search_r1/eval_results/`
- BM25 retrieval server: `http://127.0.0.1:8000/retrieve`

### 8B RL (DSL)
- 训练: `/root/autodl-tmp/rl_8b/`
- SFT 模型: `/root/autodl-tmp/sft_8b/{ecommerce,email}/output/final/`
- Base 模型: `/root/autodl-tmp/models/{Qwen3-4B,Qwen3-8B,Qwen3-Embedding-8B}`

### 数据
- 电商: `/root/autodl-tmp/data/ecommerce/`
- 邮件: `/root/autodl-tmp/data/email/`
- Embedding server: port 9123 (screen `embed`)

## verl patches 记录

| Patch | 文件 | 说明 |
|-------|------|------|
| SearchR1ToolParser | `tool_parser.py` | 注册 search_r1 格式，解析 `<search>` 标签 |
| search_r1 response format | `tool_agent_loop.py` | 将 search tool 响应包装为 `<information>` |
| stop string fix | `tool_agent_loop.py` | 为 search_r1 注入 `</search>` vLLM stop string |
