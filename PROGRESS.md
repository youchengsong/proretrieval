# LocalSearch-R1 实验进展

> 最后更新：2026-05-12
> 目标：EMNLP 2026 Long Paper（ARR May cycle，截止 2026-05-25）

## 项目简介

核心思路：让 LLM 通过 RL（GRPO/DAPO）学会生成可执行的 DSL（融合 SQL 结构化过滤 + 文本向量检索 + 图像向量检索），动态编排多模态检索路径。范式名称为 **Retrieval Program Synthesis**。

- 两个场景：电商（结构化+文本+图像）、邮件（结构化+文本）
- Base model：Qwen3-4B / Qwen3-8B
- RL 算法：GRPO / DAPO

---

## 实验结果汇总

### 电商场景 (E-commerce)

#### 主实验：DSL 生成方法 (N=3000)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate | 备注 |
|------|-------|-------|-----|-----------|------|
| **RL Methods (Ours)** | | | | | |
| GRPO step300 (4B) | **0.809** | **0.854** | **0.835** | 99.7% | 最佳 GRPO |
| DAPO step420 (4B) | 0.808 | 0.852 | 0.833 | 99.6% | |
| GRPO step200 (4B) | 0.806 | 0.853 | 0.833 | 99.7% | |
| GRPO step550 (4B) | 0.798 | 0.844 | 0.825 | 99.8% | 开始过拟合 |
| GRPO step1200 (4B) | 0.003 | 0.003 | 0.003 | 99.6% | 崩溃 |
| **SFT Methods** | | | | | |
| SFT 4B | 0.680 | 0.721 | 0.705 | 99.7% | |
| SFT 1.7B | 0.680 | 0.720 | 0.706 | 99.6% | |
| **Commercial LLMs (zero-shot DSL)** | | | | | |
| GPT-5.5 (Azure) | 0.667 | 0.704 | 0.689 | 100% | L1=0.666, L2=0.680, L3=0.642; image=0.481 |
| GPT-5.4 thinking | 0.680 | 0.723 | 0.704 | 100% | 仅 300 条，reasoning_effort=medium |
| Claude Opus 4.7 | 0.642 | 0.665 | 0.655 | 100% | L1=0.669, L2=0.618, L3=0.598; image=0.349 |
| Gemini 3.1 Pro | 0.587 | 0.597 | 0.593 | 100% | |
| Claude Opus 4.6 | 0.586 | 0.596 | 0.592 | 99.9% | |
| GPT-5.4 (Azure) | 0.581 | 0.591 | 0.587 | 100% | |
| **Base (no training)** | | | | | |
| Qwen3-4B Base (zero-shot) | 0.117 | 0.118 | 0.118 | 16.9% | |

#### B1 纯检索 Baselines (N=2737)

| 方法 | Hit@1 | Hit@3 | MRR | Recall | 备注 |
|------|-------|-------|-----|--------|------|
| BGE-Reranker (on RRF) | 0.593 | 0.719 | 0.664 | 0.410 | 最佳纯检索 baseline |
| RRF (BM25+ColQwen) | 0.555 | 0.688 | 0.633 | — | k=60 |
| BM25 | 0.548 | 0.685 | 0.630 | 0.397 | |
| BGE-M3 | 0.548 | 0.678 | 0.628 | 0.402 | |
| ColQwen (Dense Text) | 0.497 | 0.604 | 0.569 | 0.360 | Qwen3-Embedding-8B |
| Dense Image (Qwen3-VL) | 0.063 | 0.128 | 0.113 | 0.009 | 模型权重不完整，仅供参考 |

> N=2737 因 263 条 query 无 gt_document_ids

### 邮件场景 (Email, N=3000, Qwen3-4B)

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate |
|------|-------|-------|-----|-----------|
| DAPO step843 | **0.909** | **0.910** | **0.910** | 99.3% |
| GRPO step843 | 0.906 | 0.908 | 0.907 | 99.3% |
| SFT | 0.891 | 0.893 | 0.892 | 95.9% |
| Base (zero-shot) | 0.243 | 0.244 | 0.244 | 24.8% |

---

## 关键结论

1. **RL >> 商业大模型**：GRPO 电商 Hit@1=0.809 vs GPT-5.5=0.667 (+14.2pp) vs Opus 4.7=0.642 (+16.7pp)
2. **RL >> 纯检索**：GRPO Hit@1=0.809 vs BGE-Reranker=0.593 (+21.6pp) vs BM25=0.548 (+26.1pp)
3. **RL > SFT**：电商 +12.8pp，邮件 +1.8pp
4. **GRPO 过拟合**：电商 step300 后开始下降，step1200 完全崩溃
5. **DAPO 更稳定**：邮件场景 DAPO 略优于 GRPO
6. **商业模型排序**：GPT-5.5 > Opus 4.7 > Opus 4.6 ≈ GPT-5.4 ≈ Gemini 3.1 Pro
7. **纯检索排序**：BGE-Reranker > RRF ≈ BM25 ≈ BGE-M3 > ColQwen

---

## 实验完成状态

### 已完成 ✅

| 实验 | 场景 | 模型 | 状态 |
|------|------|------|------|
| Base zero-shot | 电商+邮件 | Qwen3-4B | ✅ |
| SFT | 电商+邮件 | Qwen3-4B, 1.7B | ✅ |
| GRPO 训练+推理 | 电商+邮件 | Qwen3-4B | ✅ |
| DAPO 训练+推理 | 电商+邮件 | Qwen3-4B | ✅ |
| 商业模型 (电商) | 电商 | GPT-5.4/5.5, Opus 4.6/4.7, Gemini | ✅ |
| B1 纯检索 Baselines | 电商 | BM25, ColQwen, RRF, BGE-M3, Reranker, Dense Image | ✅ (2026-05-12) |

### 未完成 ❌

| 实验 | 场景 | 难度 | 依赖 | 优先级 |
|------|------|------|------|--------|
| **商业模型 (邮件)** | 邮件 | 低 | API调用 | 高 — 补齐表格 |
| **Few-shot ICL** | 电商+邮件 | 低 | Qwen3-4B 推理 | 高 — 论文需要 |
| **E3 Complexity Breakdown** | 电商+邮件 | 低 | 已有结果按Level分层 | 高 — 已有数据 |
| **E5 Modality Ablation** | 电商 | 低 | 已有结果按Modality分层 | 高 — 已有数据 |
| **E2 Action Space Ablation** | 电商 | 中 | 需训练/推理变体模型 | 中 |
| **Qwen3-8B 全系列** | 电商+邮件 | 高 | 需多GPU训练 | 中 — 规模实验 |
| **B2 (HyDE, IRCoT, RankGPT, Self-RAG)** | 电商 | 中 | API调用+代码 | 中 |
| **B3 (DeepRetrieval, SQL-R1)** | 电商 | 高 | 训练/复现 | 低 |
| **B4 (GraphRAG)** | 电商 | 高 | 搭建复杂 | 低 |

---

## 代码与数据位置

### 远程服务器 (ai4sci, RTX 4090)
- 连接：`ssh -o "ProxyCommand=cloudflared access ssh --hostname %h" sunzhen@ai4sci.claude-code-manager.com`，密码 `12345678`
- 数据：`~/LocalSearch/data/ecommerce/test/`, `~/LocalSearch/data/email/test/`
- 结果：`~/LocalSearch/data/ecommerce/result/`, `~/LocalSearch/data/email/result/`
- 模型：`~/models/Qwen3-Embedding-8B`, `~/models/Qwen3-VL-Embedding-8B`
- RL 训练：`~/LocalSearch/rl_ecommerce/`, `~/LocalSearch/rl_email/`
- 已装包：sentence_transformers, transformers, rank_bm25, FlagEmbedding

### GitHub 仓库
- `results/ecommerce/` — 电商所有预测和评估结果 (predictions_*.json, eval_*.json, b1_*.json)
- `results/email/` — 邮件评估结果
- `src/ecommerce/` — 电商相关脚本 (b1_baselines.py, pipeline, feature gen)
- `src/eval_*.py` — 评估脚本
- `src/inference_*.py` — 推理脚本
- `docs/` — 计划文档、baseline 体系文档

### API 凭证
- GPT-5.4/5.5: Azure OpenAI（详见 CLAUDE.md）
- Claude Opus 4.6: GCP Vertex AI
- Claude Opus 4.7: 代理 `http://47.94.22.126/v1`
- Gemini: 代理 `https://token.claude-code-manager.com/v1`

---

## 时间线

| 日期 | 里程碑 |
|------|--------|
| ~4月中旬 | 电商+邮件 SFT/GRPO/DAPO 训练完成 |
| ~4月下旬 | 商业模型 (电商) GPT-5.4, Opus 4.6, Gemini 完成 |
| 2026-05-12 | Opus 4.7, GPT-5.5 电商 3000 条完成 |
| 2026-05-12 | B1 纯检索 baselines 电商全部完成 |
| **2026-05-25** | **ARR May 截止** |
