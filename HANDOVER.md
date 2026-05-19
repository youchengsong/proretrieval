# LocalSearch-R1 项目交接文档

> 最后更新：2026-05-19
> 目标：EMNLP 2026 Long Paper（ARR May cycle，截止 **2026-05-25**）

---

## 1. 项目概述

**核心思路**：让 LLM 通过 RL（GRPO/DAPO）学会生成可执行的 DSL（融合 SQL 结构化过滤 + 文本向量检索 + 图像向量检索），动态编排多模态检索路径。范式名称为 **Retrieval Program Synthesis**。

- **两个场景**：电商（Amazon ESCI + Review，结构化+文本+图像）、邮件（Enron Email，结构化+文本）
- **Base model**：Qwen3-4B / Qwen3-8B
- **RL 算法**：GRPO / DAPO
- **训练框架**：verl 0.7.1 + vLLM + FSDP

---

## 2. 服务器连接

### 2.1 服务器 102（内网，7×A6000-48GB，主训练服务器）

```bash
ssh sunzhen@10.60.200.102
# 密码: sunzhen
```

- **GPU**: 7× NVIDIA RTX A6000 (48GB each)
- **注意**: GPU 0 被 shaohuwang 的 OCR Flask 服务占用 (38GB)，训练/推理用 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`
- **磁盘**: 3.7TB 总量，当前 98% 使用（仅 83GB 剩余）⚠️
- **Python**: `~/miniconda3/envs/mlx/bin/python`
- **Conda 环境**: `mlx`（主环境）, `transfer`

**关键目录**:
```
~/DeepRetrieval_ecommerce/           # DeepRetrieval B3 baseline 训练
  ├── data/                          # train.parquet, val.parquet, test.parquet (18483条训练)
  ├── indexes/                       # Lucene BM25 索引
  ├── checkpoints_v7/global_step_96/ # Epoch 1 最终模型 (merged to actor/huggingface/)
  ├── checkpoints_v8/global_step_96/ # Epoch 2 最终模型 (merged to actor/huggingface/)
  ├── eval_results/                  # 所有评估结果 JSON
  ├── reward_ecommerce.py            # Reward 函数 (Pyserini BM25 + NDCG)
  ├── 5_eval_checkpoint.py           # 评估脚本 (vLLM 推理)
  ├── merge_fsdp_ckpt.py             # FSDP 分片合并脚本
  └── 4_train*.sh                    # 各版本训练脚本
```

**expect 自动化命令模板**（从本地 Mac 执行）:
```bash
expect -c 'spawn ssh -o StrictHostKeyChecking=no sunzhen@10.60.200.102 {<COMMAND>}; expect "password:"; send "sunzhen\r"; expect eof' 2>&1
```

### 2.2 ai4sci 服务器（1×RTX 4090，推理/评估服务器）

```bash
ssh -o "ProxyCommand=cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no sunzhen@ai4sci.claude-code-manager.com
# 密码: 12345678
```

- **GPU**: 1× RTX 4090 (24GB)
- **用途**: 推理、评估、B1/B2/B3 baseline 运行
- **需要 cloudflared**: 本地已安装在 `/Users/trl/.local/bin/cloudflared`

**关键目录**:
```
~/LocalSearch/                        # 主工作目录
  ├── data/ecommerce/test/           # 电商测试数据
  │   ├── products_test.json         # 3000 条产品
  │   ├── queries_product_test_step9_gt_v2.jsonl  # 3000 条 GT 查询
  │   ├── product_text_features_test.parquet      # 预计算文本特征 (3000×4096)
  │   └── product_image_features_test.parquet     # 预计算图像特征 (3000×4096)
  ├── data/ecommerce/result/         # 所有电商实验结果 (predictions_*.json, eval_*.json, b*_*.json)
  ├── data/email/test/               # 邮件测试数据
  ├── data/email/result/             # 邮件实验结果
  ├── rl_ecommerce/                  # 电商 RL 训练代码
  ├── rl_email/                      # 邮件 RL 训练代码
  ├── inference_closedapi.py         # 商业模型推理脚本
  ├── eval_ecommerce.py              # 电商评估脚本
  ├── eval_ecommerce_nocase.py       # 电商评估（不区分大小写）
  └── eval_email.py                  # 邮件评估脚本
~/models/                            # 模型权重
  ├── Qwen3-Embedding-8B             # 文本 embedding
  └── Qwen3-VL-Embedding-8B         # 图像 embedding（权重不完整）
```

**expect 自动化命令模板**:
```bash
cat > /tmp/ssh_cmd.exp << 'EXPEOF'
spawn ssh -o "ProxyCommand=cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no sunzhen@ai4sci.claude-code-manager.com {<COMMAND>}
expect "password:"
send "12345678\r"
expect eof
EXPEOF
expect /tmp/ssh_cmd.exp 2>&1
```

### 2.3 seetacloud（无 GPU，仅存储/CPU）

```bash
ssh -p 42056 root@connect.westd.seetacloud.com
# 密码: 4rBL4oA/dSl6
```

- **Python**: `/root/autodl-tmp/mlx/bin/python`
- **目录结构**: `/root/autodl-tmp/LocalSearch/`
- **当前无 GPU 可用**

### 2.4 服务器间连接关系

```
本地 Mac ──直连──→ 服务器 102 (10.60.200.102)
本地 Mac ──cloudflared──→ ai4sci
ai4sci ✗→ 102（不同子网，timeout）
102 ✗→ ai4sci（未测试）
```

---

## 3. GitHub 远程仓库

- **仓库**: `git@github.com:zjw49246/LocalSearch-R1.git`（私有）
- **认证**: GitHub Classic Token `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`，账户 `nbdss`
- **本地 clone**: `/Users/trl/Documents/LocalSearch-R1/`

### 仓库结构

```
LocalSearch-R1/
├── CLAUDE.md                  # Claude Code 上下文（API keys、服务器信息等）
├── PROGRESS.md                # 📊 实验进展总表（最重要的文档）
├── HANDOVER.md                # 本交接文档
├── docs/                      # 设计文档
│   ├── LocalSearch-R1_最终工作计划.md
│   ├── LocalSearch-R1_Baselines完整体系.md
│   ├── LocalSearch-R1_数据集构建说明.md
│   ├── LocalSearch-R1_Query构造.md
│   ├── LocalSearch-R1_描述生成方案.md
│   ├── training_baselines_reproduction.md
│   └── 01-04_*.md             # 早期 idea 分析和评审文档
├── results/
│   ├── ecommerce/             # 电商所有 predictions + eval 结果
│   └── email/                 # 邮件 eval 结果
├── src/
│   ├── ecommerce/             # 电商相关脚本
│   │   ├── b1_baselines.py    # B1 纯检索 baselines
│   │   ├── b2_*.py            # B2 LLM-Enhanced baselines (HyDE, IRCoT, RankGPT)
│   │   ├── b3_*.py            # B3 RL-based baselines (DeepRetrieval, SQL-R1)
│   │   ├── b4_graphrag.py     # B4 GraphRAG
│   │   ├── b5_fewshot_api.py  # B5 Few-shot DSL
│   │   ├── deepretrieval/     # DeepRetrieval 完整训练流程
│   │   ├── deepretrieval_faithful/  # DeepRetrieval 忠实复现版
│   │   ├── sft_1.7b/          # SFT 1.7B 训练脚本
│   │   ├── sft_4b/            # SFT 4B 训练脚本
│   │   ├── pipeline_product.py  # DSL 执行引擎
│   │   └── gen_*.py           # 数据生成脚本
│   ├── email/                 # 邮件相关脚本
│   ├── eval_ecommerce.py      # 电商评估
│   ├── eval_email.py          # 邮件评估
│   ├── inference_closedapi.py # 商业模型推理
│   └── inference_*.py         # 各场景推理脚本
├── retrieve_agent/            # RL 训练代码（verl-based）
│   ├── run_agent.sh           # GRPO 训练入口
│   ├── dapo/                  # DAPO 训练代码
│   ├── retrieve_env.py        # 检索环境（DSL 执行）
│   ├── retrieve_verify.py     # 结果验证
│   └── rl-data-gen/           # 训练数据生成
├── figures/                   # 训练曲线等
└── tools/                     # 辅助工具
```

### 重要文档

| 文档 | 位置 | 内容 |
|------|------|------|
| **PROGRESS.md** | 根目录 | 📊 所有实验结果总表、完成状态、关键结论 |
| **CLAUDE.md** | 根目录 | API 凭证、服务器信息、数据格式说明 |
| **Baselines 完整体系** | docs/ | B1-B5 baseline 分类、设计思路 |
| **最终工作计划** | docs/ | 完整实验规划 |
| **数据集构建说明** | docs/ | 电商/邮件数据集格式和构建方法 |
| **training_baselines_reproduction** | docs/ | 训练类 baseline 复现指南 |

---

## 4. 实验结果汇总

### 4.1 电商场景 (E-commerce, N=3000)

| 类别 | 方法 | Hit@1 | Hit@3 | MRR | 备注 |
|------|------|-------|-------|-----|------|
| **Ours (RL)** | GRPO step300 (4B) | **0.809** | **0.854** | **0.835** | 最佳 |
| | DAPO step420 (4B) | 0.808 | 0.852 | 0.833 | |
| **SFT** | SFT 4B | 0.680 | 0.721 | 0.705 | |
| **商业模型** | GPT-5.5 | 0.667 | 0.704 | 0.689 | |
| | Claude Opus 4.7 | 0.642 | 0.665 | 0.655 | |
| | GPT-5.4 | 0.581 | 0.591 | 0.587 | |
| **B1 纯检索** | BGE-Reranker | 0.593 | 0.719 | 0.664 | N=2737 |
| | BM25 | 0.548 | 0.685 | 0.630 | N=2737 |
| **B2 LLM增强** | RankGPT | 0.742 | 0.772 | 0.759 | N=300 |
| **B3 RL检索** | DeepRetrieval ep2 | 0.638 | 0.729 | 0.697 | N=2737, 已饱和 |
| | SQL-R1 (zero-shot) | 0.617 | 0.637 | 0.628 | N=300, GPT-5.4 |

### 4.2 邮件场景 (Email, N=3000)

| 方法 | Hit@1 | Hit@3 | MRR |
|------|-------|-------|-----|
| DAPO step843 | **0.909** | **0.910** | **0.910** |
| GRPO step843 | 0.906 | 0.908 | 0.907 |
| SFT | 0.891 | 0.893 | 0.892 |
| GPT-5.5 | 0.855 | 0.864 | 0.861 |
| GPT-5.4 | 0.854 | 0.864 | 0.859 |
| Opus 4.7 | 0.830 | 0.837 | 0.834 |

### 4.3 关键结论

1. **RL >> 商业大模型**：电商 0.809 vs GPT-5.5 0.667 (+14.2pp)
2. **RL >> 纯检索**：0.809 vs BGE-Reranker 0.593 (+21.6pp)
3. **RL > SFT**：电商 +12.8pp，邮件 +1.8pp
4. **Action space 是关键**：DeepRetrieval(BM25 Boolean)=0.638, SQL-R1(SQL)=0.617, Ours(DSL)=0.809
5. **DeepRetrieval 已完全饱和**：Epoch 2 reward 无提升，eval 仅 +0.3pp

---

## 5. API 凭证

### GPT-5.4 (Azure)
```
Endpoint: https://algorithm.cognitiveservices.azure.com/
API Key: AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX
Deployment: gpt-5.4
API Version: 2024-12-01-preview
```

### GPT-5.5 (Azure)
```
Endpoint: https://zhouj-mp1gbpux-eastus2.cognitiveservices.azure.com/
API Key: 0PGxfkUTBoEmHcj3mheyc5Ep6qQuibrATMi7aHMFJqyyMkSdZDMPJQQJ99CEACHYHv6XJ3w3AAAAACOGhsiT
Deployment: gpt-5.5
API Version: 2024-12-01-preview
```

### Claude Opus 4.7 (代理)
```
Base URL: http://47.94.22.126/v1
API Key: tum_ywUTn-A_TYQ9Q2xkLCxpicD6c6VVC_Y5gHL0VLMbEyk
Model: anthropic/claude-opus-4.7
格式: OpenAI 兼容 (/v1/chat/completions)
```

### Claude Opus 4.6 (GCP Vertex AI)
```
项目: claude-project-489006
服务账号: claude@claude-project-489006.iam.gserviceaccount.com
```

### Gemini (代理)
```
Base URL: https://token.claude-code-manager.com/v1
API Key: tum_VWYxGAiOLi0t0FJCjvDpAjibhNMui7BVJRWLucx83Q0
```

---

## 6. 数据格式

### 电商产品库 (`products_test.json`, 3000条)
```json
{
    "product_id": "B0xxx",
    "brand": "Nike",
    "color": "Black",
    "image_url": "https://...",
    "category": "Shoes",
    "price": 99.99,
    "average_rating": 4.5,
    "rating_number": 1234,
    "text": "product description..."
}
```

### GT 查询 (`queries_product_test_step9_gt_v2.jsonl`, 3000条)
```json
{
    "query_id": "q_1",
    "natural_language_query": "...",
    "gt_document_ids": ["B0xxx"],
    "level": 1,        // 1=简单, 2=中等, 3=复杂
    "modalities": ["structured", "text", "image"],
    "leaves": [...],
    "dsl": "..."        // ground truth DSL
}
```

### 训练数据 (DeepRetrieval `train.parquet`, 18483条)
```python
{
    "data_source": "ecommerce_product_sparse_train",
    "prompt": [
        {"role": "system", "content": "...Boolean query prompt..."},
        {"role": "user", "content": "...query text..."}
    ],
    "ability": "product_retrieval",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["B0xxx", "B0yyy", ...]}
    },
    "extra_info": {"index": 0, "query_id": "q_1", "split": "train"}
}
```

---

## 7. 关键操作指南

### 7.1 在 102 上跑评估
```bash
cd ~/DeepRetrieval_ecommerce
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/mlx/bin/python 5_eval_checkpoint.py \
    --checkpoint checkpoints_v8/global_step_96/actor/huggingface \
    --gpu-mem 0.3
```

### 7.2 合并 FSDP 分片
```bash
cd ~/DeepRetrieval_ecommerce
~/miniconda3/envs/mlx/bin/python merge_fsdp_ckpt.py checkpoints_v8/global_step_96
# 输出到 checkpoints_v8/global_step_96/actor/huggingface/
```

### 7.3 在 ai4sci 上跑电商评估
```bash
cd ~/LocalSearch
python eval_ecommerce_nocase.py --pred data/ecommerce/result/predictions_xxx.json \
    --gt data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl
```

### 7.4 在 ai4sci 上跑商业模型推理
```bash
cd ~/LocalSearch
python inference_closedapi.py --model gpt54 --scenario ecommerce --limit 3000
```

### 7.5 GRPO 训练（以 DeepRetrieval 为例）
```bash
# 在 102 上
cd ~/DeepRetrieval_ecommerce
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 nohup bash 4_train.sh > train.log 2>&1 &
# 监控: tail -f train.log | grep "critic/score/mean"
```

### 7.6 清理磁盘空间（102 经常满）
```bash
# 删除 FSDP 分片和 optimizer（保留 huggingface/ 即可）
rm checkpoints_vX/global_step_Y/actor/model_world_size_*.pt
rm checkpoints_vX/global_step_Y/actor/optim_world_size_*.pt
rm checkpoints_vX/global_step_Y/actor/extra_state_world_size_*.pt
# 每个 checkpoint 可省 ~40GB
```

---

## 8. 当前工作状态

### 已完成 ✅

- 电商+邮件 SFT/GRPO/DAPO 训练+评估
- 商业模型推理（GPT-5.4/5.5, Opus 4.6/4.7, Gemini）双场景
- B1 纯检索 baselines (BM25, ColQwen, RRF, BGE-M3, BGE-Reranker, Dense Image)
- B2 LLM-Enhanced baselines (HyDE, IRCoT, RankGPT, SQR) N=300
- B3 DeepRetrieval (Qwen3-4B GRPO, 2 epochs, 已饱和 Hit@1=0.638)
- B3 SQL-R1 zero-shot (GPT-5.4, Hit@1=0.617) N=300
- E3 Complexity Breakdown (by Level L1/L2/L3)
- E5 Modality Ablation (by struct/text/image)
- SQL-R1 详细复现调研（见 `/Users/trl/Downloads/SQL-R1_复现调研报告.md`）

### 未完成 ❌（按优先级）

| 实验 | 优先级 | 说明 |
|------|--------|------|
| **B5 Few-shot DSL 重跑** | 🔴 高 | 旧结果因 prompt schema 错误已删除 |
| **B3 SQL-R1 RL 复现** | 🟡 中 | 已有 zero-shot 结果，RL 可留 camera-ready |
| **B4 GraphRAG** | 🟡 中 | 构图+API调用 |
| **Qwen3-8B 全系列** | 中 | 需多 GPU |
| **B2 扩到 N=3000** | 中 | API 成本 |
| **E2 Action Space Ablation** | 低 | 4变体训练，之前卡在 vLLM 初始化 |

---

## 9. 已知坑点

1. **verl `use_dynamic_bsz=True` 会静默覆盖 micro_batch 设置**，必须显式设 `False`
2. **verl 不支持从 epoch 边界恢复训练**（total_epochs=2 从 step_96 开始会直接退出），workaround: 用 merged 模型作新 base + total_epochs=1 + 新 checkpoint 目录
3. **102 磁盘经常满**（98%），checkpoint 每个 47-63GB，训练前先清理
4. **GPU 0 被占**，所有操作用 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`
5. **ai4sci 通过 cloudflared 连接**，不能直接 SSH，需要代理命令
6. **ai4sci 和 102 不在同一子网**，不能互相 SSH
7. **Qwen3-VL-Embedding-8B 模型权重不完整**（language_model 随机初始化），Dense Image 结果极差 (Hit@1=0.063) 仅供参考
8. **Few-shot baseline 旧结果无效**——之前的 prompt schema 有 bug，结果已删除，需修复重跑

---

## 10. DeepRetrieval 训练详情（102 服务器）

### 训练版本历史

| 版本 | 目录 | Base Model | 说明 |
|------|------|-----------|------|
| v7 (Epoch 1) | checkpoints_v7 | Qwen3-4B | 96 steps, Hit@1=0.635 |
| v8 (Epoch 2) | checkpoints_v8 | v7/step_96 merged | 48 steps (resume from 48), Hit@1=0.638 |

### 训练参数
```
batch_size=192, lr=1e-6, kl_coef=0.001, rollout_n=4
temperature=0.6, max_prompt=512, max_response=1024
6 GPU (A6000), FSDP, gradient checkpointing
```

### Reward 函数
- Format: +0.1（正确 JSON+think+answer 格式）/ -2（格式错误）
- Answer: NDCG@1000（Pyserini BM25 检索 → 与 gt targets 比较）
- 组合: format_correct 且 answer>0 时 total = format + answer，否则取较低值

---

## 11. 下一步 SQL-R1 RL 复现方案

如果要训练 SQL-R1 style baseline（纯 SQL WHERE 子句 + GRPO），方案：

1. **训练数据**: 直接复用 DeepRetrieval 的 `train.parquet`（18483条），只改 system prompt
2. **Prompt 改为**: 让模型生成 SQL WHERE 子句（而非 Boolean query）
3. **Reward 改为**: 在 SQLite 内存数据库上执行 SQL，检查是否命中 gt_product_ids
4. **SFT 冷启动**: 用 GPT-5.4 对训练集 query 生成 WHERE 子句，筛选能命中 gt 的作为 SFT 数据
5. **Base model**: Qwen3-4B（与我们主实验一致，保证公平）
6. **训练**: 在 102 上用 6×A6000，配置与 DeepRetrieval 基本一致

详细调研报告见: `/Users/trl/Downloads/SQL-R1_复现调研报告.md`
