# LocalSearch-R1 项目上下文

## 项目概述

LocalSearch-R1 是一篇面向 EMNLP 2026 Long Paper 的研究项目（ARR May cycle，截止 2026-05-25）。

核心思路：让 LLM 通过 RL 学会生成可执行的 DSL（融合 SQL 结构化过滤 + 文本向量检索 + 图像向量检索），动态编排多模态检索路径。范式名称为 **Retrieval Program Synthesis**。

两个场景：
- 电商（Amazon ESCI + Review，多模态：结构化 + 文本 + 图像）
- 邮件（Enron Email，结构化 + 文本，无图像）

Base model: Qwen3-4B / Qwen3-8B，RL 算法: GRPO / DAPO。

## 远程仓库

- GitHub 私有仓库：`https://github.com/zjw49246/LocalSearch-R1`
- 本地 clone 路径：`/home/ubuntu/LocalSearch-R1/`
- 认证：GitHub Classic Token（`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`），存于 `~/.git-credentials`，账户 `nbdss`

## 远程服务器

### 服务器1：ai4sci（4090，通过 cloudflared 跳板）
连接命令（需要 expect 自动输密码，因为没有 sshpass）：
```bash
# 单条命令执行模板：
cat > /tmp/ssh_cmd.exp << 'EXPEOF'
spawn ssh -o "ProxyCommand=cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no sunzhen@ai4sci.claude-code-manager.com {<COMMAND>}
expect "password:"
send "12345678\r"
expect eof
EXPEOF
expect /tmp/ssh_cmd.exp 2>&1
```

目录结构：
- `~/LocalSearch/` — 主工作目录
- `~/models/` — 模型权重（Qwen3-Embedding-8B, Qwen3-VL-Embedding-8B）
- GPU: 1× RTX 4090
- Python 环境有 sentence_transformers, transformers；缺 rank_bm25, FlagEmbedding

关键路径：
- 数据：`~/LocalSearch/data/ecommerce/` 和 `~/LocalSearch/data/email/`
- 电商结果：`~/LocalSearch/data/ecommerce/result/*.json`
- 邮件结果：`~/LocalSearch/data/email/result/*.json`
- RL 训练（电商）：`~/LocalSearch/rl_ecommerce/`
- RL 训练（邮件）：`~/LocalSearch/rl_email/`
- 推理/评估代码：`~/LocalSearch/inference_closedapi.py`、`~/LocalSearch/eval_ecommerce.py`、`~/LocalSearch/eval_ecommerce_nocase.py`、`~/LocalSearch/eval_email.py`

### 服务器2：seetacloud（无GPU，仅存储/CPU）
```bash
# 密码: 4rBL4oA/dSl6
ssh -p 42056 root@connect.westd.seetacloud.com
```
- Python: `/root/autodl-tmp/mlx/bin/python`
- 目录结构与 ai4sci 基本一致：`/root/autodl-tmp/LocalSearch/`
- 模型权重在 `/root/autodl-tmp/models/`
- **当前无GPU可用**（torch.cuda.is_available()=False）

### 服务器3：内网 10.60.200.102
```bash
# 用户: sunzhen，密码: sunzhen
ssh sunzhen@10.60.200.102
```
- **注意：从当前机器（AWS）无法直接访问**（内网地址，Connection timed out）
- 需要从 ai4sci 或其他同网段机器跳转，或用户自行在本地连接

## 当前实验结果（截至 2026-05-21）

### 电商 (N=3000)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate | 备注 |
|------|-------|-------|-----|-----------|------|
| Qwen3-4B Base (zero-shot) | 0.117 | 0.118 | 0.118 | 16.9% | |
| SFT 1.7B | 0.680 | 0.720 | 0.706 | 99.6% | |
| SFT 4B | 0.680 | 0.721 | 0.705 | 99.7% | |
| GRPO step200 (4B) | 0.806 | 0.853 | 0.833 | 99.7% | |
| GRPO step300 (4B) | 0.809 | 0.854 | 0.835 | 99.7% | 最佳 GRPO |
| GRPO step550 (4B) | 0.798 | 0.844 | 0.825 | 99.8% | 开始过拟合 |
| GRPO step1200 (4B) | 0.003 | 0.003 | 0.003 | 99.6% | 崩溃 |
| DAPO step420 (4B) | 0.808 | 0.852 | 0.833 | 99.6% | |
| **GRPO step100 (8B)** | — | — | — | — | 训练中，score_mean=0.83 (step83)，预计超越4B |
| GPT-5.4 (Azure) | 0.581 | 0.591 | 0.587 | 100% | model=gpt-5.4-2026-03-05 |
| GPT-5.4 thinking | 0.680 | 0.723 | 0.704 | 100% | 仅 300 条，reasoning_effort=medium |
| Claude Opus 4.6 | 0.586 | 0.596 | 0.592 | 99.9% | 通过 GCP Vertex AI |
| Claude Opus 4.7 | 0.642 | 0.665 | 0.655 | 100% | 通过代理，N=2998；L1=0.669, L2=0.618, L3=0.598; image=0.349 |
| Gemini 3.1 Pro | 0.587 | 0.597 | 0.593 | 100% | |
| GPT-5.5 (Azure) | 0.667 | 0.704 | 0.689 | 100% | N=2998；L1=0.666, L2=0.680, L3=0.642; image=0.481 |

### B3 SQL-R1 Baseline (电商, N=2737)

| 方法 | Hit@1 | Hit@3 | MRR | Format OK | Exec OK | 备注 |
|------|-------|-------|-----|-----------|---------|------|
| SQL-R1 RL step115 (4B) | 0.780 | 0.797 | 0.790 | 99.9% | 82.5% | 102服务器，结果文件 b2_sql_r1_rl.json |

### B1 纯检索 Baselines (电商, N=2737)

| 方法 | Hit@1 | Hit@3 | MRR | Recall | 备注 |
|------|-------|-------|-----|--------|------|
| BM25 | 0.548 | 0.685 | 0.630 | 0.397 | CPU, 14s |
| ColQwen (Dense Text) | 0.497 | 0.604 | 0.569 | 0.360 | Qwen3-Embedding-8B fp16, 19s |
| RRF (BM25+ColQwen) | 0.555 | 0.688 | 0.633 | — | k=60 |
| BGE-M3 | 0.548 | 0.678 | 0.628 | 0.402 | 31s |
| BGE-Reranker (on RRF) | 0.593 | 0.719 | 0.664 | 0.410 | bge-reranker-v2-m3, 334s |
| Dense Image (Qwen3-VL) | 0.063 | 0.128 | 0.113 | 0.009 | 模型权重不完整，仅供参考 |

注：N=2737 因 263 条 query 无 gt_document_ids。BGE-Reranker 是最佳纯检索 baseline，但远低于 RL (Hit@1=0.593 vs 0.809)。

### 邮件 (N=3000, Qwen3-4B)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate |
|------|-------|-------|-----|-----------|
| Base (zero-shot) | 0.243 | 0.244 | 0.244 | 24.8% |
| SFT | 0.891 | 0.893 | 0.892 | 95.9% |
| GRPO step843 | 0.906 | 0.908 | 0.907 | 99.3% |
| DAPO step843 | 0.909 | 0.910 | 0.910 | 99.3% |
| DeepRetrieval step100 | 0.745 | 0.851 | 0.805 | 99.9% |

注：DeepRetrieval 的 N=2569（431 条 query 无 gt_document_ids）。DeepRetrieval 使用 Boolean query + BM25，远低于我们的 DSL 方法（Hit@1=0.745 vs 0.909），差距比电商场景更大。

### B3 SQL-R1 Baseline (邮件, N=2872)

| 方法 | Hit@1 | Hit@3 | MRR | Format OK | Exec OK | 备注 |
|------|-------|-------|-----|-----------|---------|------|
| SQL-R1 SFT (4B) | 0.546 | 0.584 | 0.565 | 94.8% | 63.4% | 102服务器 |
| SQL-R1 RL step100 (4B) | 0.541 | 0.578 | 0.561 | — | 61.9% | RL未提升，OOM于step121 |

注：邮件 SQL-R1 远低于 DSL 方法（0.54 vs 0.91），根因是 SQL LIKE 无法做语义匹配，邮件以自由文本为主（Subject/Message），31% 查询因过多 LIKE+AND 条件返回空结果。电商 SQL-R1（0.78）表现更好，因结构化字段（brand/color/price）适合精确匹配。

### B2 LLM 增强检索 Baselines (电商, N=268, GPT-5.4)

| 方法 | Hit@1 | Hit@3 | MRR | 备注 |
|------|-------|-------|-----|------|
| RankGPT | 0.743 | 0.772 | 0.759 | LLM listwise 重排 (on RRF top-100) |
| IRCoT | 0.563 | 0.675 | 0.631 | 多轮检索+推理 (2轮 RRF) |
| HyDE | 0.511 | 0.646 | 0.588 | 假设文档生成+向量检索 |

注：N=268（子集，非全量 3000）。RankGPT 效果最好但仍远低于 RL (0.743 vs 0.809)。

### B3 受限结构化查询 Baselines (电商)

| 方法 | Hit@1 | Hit@3 | MRR | N | 备注 |
|------|-------|-------|-----|---|------|
| DeepRetrieval (Boolean+BM25) | 0.638 | 0.729 | 0.697 | 2737 | GRPO epoch2, full eval |
| LangChain SQR | 0.590 | 0.690 | 0.650 | 268 | GPT-5.4 结构化过滤+向量 |
| SQL-R1 (zero-shot) | 0.617 | 0.637 | 0.628 | 300 | GPT-5.4 纯 SQL, exec=69.3% |
| Search-R1 (BM25 multi-turn) | 0.459 (post-EM) | — | — | 2773 | verl GRPO step800 |

### B5 Few-shot DSL (电商, N=300)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate | 备注 |
|------|-------|-------|-----|-----------|------|
| GPT-5.5 few-shot | 0.253 | 0.280 | 0.268 | 48.0% | format=288/300 |
| Opus 4.7 few-shot | 0.250 | 0.270 | 0.262 | 50.3% | format=300/300 |

注：Few-shot DSL exec rate 极低（~50%），大量 SQL 语法错误，说明 few-shot 不足以学会 DSL。

### 邮件商业模型 (N=3000)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate |
|------|-------|-------|-----|-----------|
| GPT-5.5 | 0.855 | 0.864 | 0.861 | 99.9% |
| GPT-5.4 | 0.854 | 0.864 | 0.859 | 99.9% |
| Opus 4.7 | 0.830 | 0.837 | 0.834 | 96.6% |

注：邮件场景商业模型表现明显好于电商（GPT-5.5 邮件 0.855 vs 电商 0.667），但仍低于 RL (DAPO=0.909)。

### 关键结论

1. RL 显著优于所有商业大模型：DAPO 电商 Hit@1=0.808 vs GPT-5.5=0.667 vs Opus 4.7=0.642
2. RL > SFT：电商 +12.8pp，邮件 +1.8pp
3. RL >> 纯检索 baselines：GRPO Hit@1=0.809 vs BGE-Reranker=0.593 vs BM25=0.548
4. RL >> B2 LLM 增强：GRPO 0.809 vs RankGPT=0.743 vs IRCoT=0.563
5. RL >> B3 受限结构化：GRPO 0.809 vs DeepRetrieval=0.638 vs SQR=0.590
6. Few-shot DSL 完全失败：exec rate ~50%，Hit@1 ~0.25，说明 DSL 需要 RL 训练
7. GRPO 在电商上 step300 后开始过拟合，step1200 完全崩溃
8. DAPO 比 GRPO 稍好（邮件场景更明显）
9. GPT-5.5 > Opus 4.7 > Opus 4.6 ≈ GPT-5.4 ≈ Gemini 3.1 Pro
10. 邮件场景商业模型远好于电商（GPT-5.5: 0.855 vs 0.667），但 RL 差距仍显著

### 尚未完成

**已完成 Baselines：**
- ✅ B1: BM25, ColQwen, RRF, BGE-M3, BGE-Reranker, Dense Image (2026-05-12)
- ✅ B2: HyDE, IRCoT, RankGPT (2026-05-13, N=268)
- ✅ B3: DeepRetrieval 电商+邮件, LangChain SQR, SQL-R1, Search-R1 电商 (2026-05-13~20)
- ✅ B5 Few-shot: GPT-5.5, Opus 4.7 (2026-05-13, N=300)
- ✅ 邮件商业模型: GPT-5.4, GPT-5.5, Opus 4.7 (2026-05-16~18)
- ✅ 分析实验: E3 Complexity Breakdown, E5 Modality Ablation (2026-05-20)

**进行中：**
- 🔄 Qwen3-8B GRPO 电商：训练中（step 100/625，score_mean 峰值 0.83，Blackwell 2×RTX PRO 6000）
- 🔄 Search-R1 邮件 RL：从 step 200 恢复训练中（→1000）

**待完成：**
- Qwen3-8B GRPO 邮件 / DAPO 电商+邮件：待 8B GRPO 电商完成后顺序训练
- B4 GraphRAG：代码完成，KG cache 116 条，需继续跑
- B2 全量评估：当前 N=268，需扩展到 N=3000
- B3 SQR/SQL-R1 全量评估：当前 N=268/300
- B5 Few-shot Gemini：代码完成，API 调用失败
- Action Space Ablation (E2)：数据准备好，未跑训练

### 数据格式说明

- **产品库**：`products_test.json`，3000条，字段：product_id, brand, color, image_url, category, price, average_rating, rating_number, text
- **GT 查询**：`queries_product_test_step9_gt_v2.jsonl`，3000条，字段：query_id, natural_language_query, gt_document_ids, level(1/2/3), modalities, leaves, dsl
- **预计算特征**：`product_text_features_test.parquet`（3000×4096，Qwen3-Embedding-8B）、`product_image_features_test.parquet`（3000×4096，Qwen3-VL-Embedding-8B）
- **文本描述**：`text_descriptions_test.jsonl`（product_id, text_pos, text_neg）
- **图像描述**：`image_descriptions_test.jsonl`（product_id, image_pos, image_neg）
- **eval 脚本**：`eval_ecommerce.py` / `eval_ecommerce_nocase.py`，接受 `--pred`（DSL格式 JSON）和 `--gt`，输出 Hit@1/3, MRR, Recall, 按 Level 和 Modality 分层
- B1 baseline 不生成 DSL，需自写评估（直接对比 ranked IDs vs gt_document_ids）

## API 凭证

### GPT (Azure OpenAI)
**GPT-5.4:**
- Endpoint: `https://algorithm.cognitiveservices.azure.com/`
- API Key: `AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX`
- Deployment: `gpt-5.4`

**GPT-5.5:**
- Endpoint: `https://zhouj-mp1gbpux-eastus2.cognitiveservices.azure.com/`
- API Key: `0PGxfkUTBoEmHcj3mheyc5Ep6qQuibrATMi7aHMFJqyyMkSdZDMPJQQJ99CEACHYHv6XJ3w3AAAAACOGhsiT`
- Deployment: `gpt-5.5`
- API Version: `2024-12-01-preview`

### Claude (通过 GCP Vertex AI)
- 项目: `claude-project-489006`
- 服务账号: `claude@claude-project-489006.iam.gserviceaccount.com`
- 服务账号 JSON 密钥已提供（private_key_id: facecf34...）

### Claude Opus 4.7 (通过代理)
- 代理 Base URL: `http://47.94.22.126/v1`
- API Key: `tum_ywUTn-A_TYQ9Q2xkLCxpicD6c6VVC_Y5gHL0VLMbEyk`
- 模型名: `anthropic/claude-opus-4.7`
- 调用方式: OpenAI 兼容格式（`/v1/chat/completions`）或 Anthropic 格式（`/v1/messages`）
- Thinking: `enabled` 模式被代理拦截无效；`adaptive` 模式可用但对 DSL 任务几乎不触发（10条仅1条L3产生10个reasoning tokens）
- 定价: Input $5/MTok, Output $25/MTok；平均每条 prompt≈1351 tokens, completion≈106 tokens
- 电商 3000 条实际花费约 $28

### Gemini (通过代理)
- 代理 Base URL: `https://token.claude-code-manager.com/v1`
- API Key: `tum_VWYxGAiOLi0t0FJCjvDpAjibhNMui7BVJRWLucx83Q0`

## 文档位置

- 最终工作计划：`/home/ubuntu/LocalSearch-R1/docs/LocalSearch-R1_最终工作计划.md`
- Baselines 完整体系：`/home/ubuntu/LocalSearch-R1/docs/LocalSearch-R1_Baselines完整体系.md`
- 其他文档：`/home/ubuntu/LocalSearch-R1/docs/` 下
