# Search-R1 Baseline (B3)

Search-R1 是一个基于 verl 0.7.1 的多轮 RL 搜索基线，使用 `<search>query</search>` / `<information>results</information>` 格式进行 BM25 在线检索。

## 架构

```
User Query → Model generates <think>...</think><search>query</search>
           → BM25 retrieval server returns top-3 results wrapped in <information>...</information>
           → Model reads results, generates <think>...</think><answer>id</answer>
           → Reward: EM(answer, ground_truth_ids)
```

- **Model**: Qwen3-4B (SFT → GRPO)
- **RL Framework**: verl 0.7.1 + vLLM + FSDP
- **Search Tool**: pyserini BM25 via HTTP retrieval server
- **Reward**: Exact Match on document IDs

## 结果

### 电商 (Ecommerce, N=2773)

| Checkpoint | Raw EM | Post-processed EM | 备注 |
|---|---|---|---|
| step_800 (merged) | 0.18% | **45.87%** | "Doc N" → product_id mapping |

> Raw EM 极低是因为模型输出 "Doc 1" / "Doc 2" 而非 product_id。Post-processing 将 "Doc N" 映射回 BM25 返回的 product_id 后 EM=45.87%。
> 相比我们的 DSL 方法 (GRPO step300 Hit@1=80.9%)，Search-R1 差距显著 (34.9pp)，说明 BM25 text-only 检索无法覆盖结构化和图像模态。

### 邮件 (Email)

| 状态 | 训练进度 | Reward 趋势 |
|---|---|---|
| 🔄 训练中 | step 200→1000 (从 checkpoint 恢复) | 0.047→0.25 (step 250 peak) |

> 邮件场景 RL 训练从 SFT checkpoint 启动，reward 持续上升中。
> 关键修复: 需要在 verl agent loop 中注入 `</search>` 作为 vLLM stop string，否则模型在 `</search>` 后继续生成，填满 response_length 导致 tool call 无法提取。

## 目录结构

```
search_r1/
├── README.md
├── scripts/
│   ├── run_search_rl_ecommerce.sh     # 电商 RL 训练 (GRPO, verl 0.7.1)
│   ├── run_search_rl_email.sh         # 邮件 RL 训练
│   ├── run_search_rl_email_resume.sh  # 邮件 RL 恢复训练 (resume_mode=auto)
│   ├── reward_search_r1.py            # Reward 函数 (EM check)
│   ├── search_tool_config.yaml        # BM25 search tool schema for verl
│   ├── eval_search_r1.sh              # verl-based 评估脚本
│   ├── eval_search_r1_vllm.py         # vLLM 独立评估 (multi-turn)
│   ├── eval_search_r1_postprocess.py  # 带 "Doc N" → product_id 后处理的评估
│   ├── merge_fsdp_ckpt.py             # FSDP DTensor → HuggingFace 合并
│   ├── convert_corpus_to_jsonl.py     # parquet → pyserini JSONL 转换
│   ├── train_search_sft.sh            # SFT 训练 (ecommerce + email)
│   ├── train_sft.py                   # SFT 训练脚本 (deepspeed)
│   ├── setup_blackwell.sh             # Blackwell 环境初始化 (BM25 index + verl)
│   └── launch_all.sh                  # 一键启动 (retrieval server + RL)
└── patches/
    ├── patch_verl_search_r1.py        # 为 verl 添加 search_r1 tool parser
    └── patch_agent_loop_stop_string.py # 修复 agent loop stop string bug
```

## verl Patches

Search-R1 需要对 verl 0.7.1 做两个 patch:

### 1. SearchR1ToolParser (`patch_verl_search_r1.py`)
在 `tool_parser.py` 中注册 `search_r1` 格式解析器，将 `<search>query</search>` 转换为 `FunctionCall(name="search", arguments={"query_list": ["query"]})`。同时在 `tool_agent_loop.py` 中将 search tool 响应包装为 `<information>...</information>` 格式。

### 2. Stop String Fix (`patch_agent_loop_stop_string.py`)
在 `tool_agent_loop.py` 的 `_handle_generating_state` 方法中，为 search_r1 格式注入 `</search>` 作为 vLLM stop string。

**背景**: 邮件 SFT 模型在生成 `</search>` 后会继续幻觉生成 `<information>` 内容，填满 500 token 的 response_length。verl 的 termination check (`len(response_mask) >= response_length`) 在 tool call extraction 之前执行，导致 tool call 永远无法被提取，training reward 恒为 0。添加 stop string 后模型在 `</search>` 处停止，tool call 正确提取，reward 恢复正常。

## Blackwell 服务器训练参数

```
GPU: 2x RTX PRO 6000 (96GB each)
Batch: 16, micro_batch: 2/GPU, rollout.n=4
LR: 1e-6, KL: 0.001 (low_var_kl), entropy: 0.001
FSDP: param_offload=True, optimizer_offload=True
vLLM: gpu_memory_utilization=0.5, TP=1
Multi-turn: max_user_turns=3, max_assistant_turns=4, max_tool_response_length=500
Save: every 200 steps, total 1000 steps
```
