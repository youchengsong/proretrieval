# LocalSearch-R1 实验进展

> 最后更新：2026-05-13
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

#### B2 LLM-Enhanced 检索 Baselines (N=300)

| 方法 | Hit@1 | Hit@3 | MRR | 备注 |
|------|-------|-------|-----|------|
| RankGPT (BM25 + GPT-5.4 rerank) | **0.742** | **0.772** | **0.759** | 最佳 B2 baseline |
| SQR (LangChain Self-Query) | 0.590 | 0.690 | 0.650 | GPT-5.4 结构化过滤 + text retrieval |
| IRCoT (2-round retrieval) | 0.563 | 0.675 | 0.631 | GPT-5.4 refine + Qwen3-Embedding |
| HyDE | 0.511 | 0.646 | 0.588 | GPT-5.4 hypothetical doc + Qwen3-Embedding |

> N=300 为节省 API 成本，使用 API 的 baseline 仅跑 300 条

#### B5 Few-shot DSL

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate | N | 状态 |
|------|-------|-------|-----|-----------|---|------|
| GPT-5.5 few-shot | — | — | — | — | — | ❌ 待重跑（旧结果因 prompt schema 错误已删除） |
| Opus 4.7 few-shot | — | — | — | — | — | ❌ 待重跑 |
| Qwen3-4B few-shot ICL | — | — | — | — | — | ❌ 未做 |
| Qwen3-8B few-shot ICL | — | — | — | — | — | ❌ 未做 |

#### E3 Complexity Breakdown (by Level, N=3000)

| 方法 | L1 Hit@1 | L1 MRR | L2 Hit@1 | L2 MRR | L3 Hit@1 | L3 MRR |
|------|----------|--------|----------|--------|----------|--------|
| GRPO step300 | **0.864** | **0.888** | 0.797 | 0.821 | **0.720** | **0.752** |
| DAPO step420 | 0.859 | 0.885 | **0.800** | **0.823** | 0.717 | 0.749 |
| SFT 4B | 0.752 | 0.765 | 0.644 | 0.673 | 0.602 | 0.641 |
| GPT-5.5 | 0.666 | 0.678 | 0.680 | 0.706 | 0.642 | 0.676 |
| Opus 4.7 | 0.669 | 0.678 | 0.618 | 0.643 | 0.598 | 0.632 |
| GPT-5.4 | 0.561 | 0.564 | 0.611 | 0.615 | 0.563 | 0.575 |
| Opus 4.6 | 0.570 | 0.574 | 0.608 | 0.612 | 0.573 | 0.587 |
| Gemini 3.1 Pro | 0.551 | 0.559 | 0.603 | 0.607 | 0.572 | 0.578 |
| Base (zero-shot) | 0.218 | 0.219 | 0.071 | 0.073 | 0.000 | 0.000 |
| **B1 Baselines** | | | | | | |
| BGE-Reranker | 0.455 | 0.549 | 0.691 | 0.741 | 0.703 | 0.767 |
| RRF | 0.442 | 0.534 | 0.627 | 0.693 | 0.665 | 0.735 |
| BM25 | 0.435 | 0.538 | 0.623 | 0.687 | 0.652 | 0.723 |
| BGE-M3 | 0.424 | 0.519 | 0.636 | 0.699 | 0.648 | 0.727 |

> RL 方法在各级别全面领先；B1 纯检索 baseline 呈现相反趋势(L3>L2>L1)，因为复杂查询包含更多特征词匹配更好；RL 在 L1 优势最大(0.864 vs 0.455)

#### E5 Modality Ablation (by Modality, N=3000)

| 方法 | Struct Hit@1 | Struct MRR | Text Hit@1 | Text MRR | Image Hit@1 | Image MRR |
|------|-------------|-----------|-----------|---------|-------------|----------|
| GRPO step300 | **0.843** | **0.861** | **0.690** | **0.726** | **0.716** | **0.761** |
| DAPO step420 | 0.843 | 0.861 | 0.690 | 0.726 | 0.711 | 0.756 |
| SFT 4B | 0.743 | 0.765 | 0.634 | 0.671 | 0.320 | 0.359 |
| GPT-5.5 | 0.724 | 0.744 | 0.607 | 0.640 | 0.481 | 0.517 |
| Opus 4.7 | 0.703 | 0.723 | 0.572 | 0.603 | 0.349 | 0.380 |
| GPT-5.4 | 0.690 | 0.695 | 0.400 | 0.407 | 0.391 | 0.398 |
| Opus 4.6 | 0.696 | 0.702 | 0.403 | 0.412 | 0.394 | 0.400 |
| Gemini 3.1 Pro | 0.684 | 0.689 | 0.401 | 0.407 | 0.400 | 0.403 |
| **B1 Baselines** | | | | | | |
| BGE-Reranker | 0.558 | 0.631 | 0.779 | 0.828 | 0.789 | 0.848 |
| RRF | 0.518 | 0.596 | 0.758 | 0.815 | 0.753 | 0.820 |
| BM25 | 0.516 | 0.598 | 0.761 | 0.816 | 0.724 | 0.794 |
| BGE-M3 | 0.510 | 0.591 | 0.759 | 0.813 | 0.720 | 0.796 |

> RL 在 structured 上优势最大(0.843 vs Reranker 0.558)，因为 SQL 条件精准过滤结构化字段；B1 在 text/image 上更强(Reranker text 0.779 > SFT 0.634)但仍低于 RL(0.690)

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
8. **LLM-Enhanced 检索**：RankGPT(0.742) > SQR(0.590) > IRCoT(0.563) > HyDE(0.511)，RankGPT 远超其他但仍低于 RL(0.809)
9. **RankGPT 分析**：仅做 reranking 无法生成新候选，但利用 GPT 理解力大幅提升排序质量

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
| B2 LLM-Enhanced Baselines | 电商 | HyDE, IRCoT, RankGPT, SQR (N=300) | ✅ (2026-05-13) |
| E3 Complexity Breakdown | 电商 | 13 methods, by Level (L1/L2/L3) | ✅ (2026-05-13) |
| E5 Modality Ablation | 电商 | 13 methods, by Modality (struct/text/image) | ✅ (2026-05-13) |
| B5 Few-shot DSL | 电商 | — | ❌ 旧结果因 prompt schema 错误已删除，待修复重跑 |

### 进行中 🔄

| 实验 | 场景 | 状态 | 备注 |
|------|------|------|------|
| B3 DeepRetrieval | 电商 | 🔄 102 服务器训练中 | GRPO v6, step ~90/384, 6×A6000, ~25h 总计 |
| 商业模型 (邮件) | 邮件 | 🔄 ai4sci 运行中 | GPT-5.5, GPT-5.4, Opus 4.7 三个并行，N=3000 |

### 已暂停 ⏸️

| 实验 | 场景 | 状态 | 备注 |
|------|------|------|------|
| E2 Action Space Ablation | 电商 | ⏸️ 已停止 | 4 个变体 (sql_only/sql_text/text_only/sql_image) 均未完成。sql_only 在 102 GPU2-5 训练时卡在 vLLM 初始化 >1h，已 kill。待 baselines 全部完成后再恢复 |
| E3/E5 细分分析 | — | ⏸️ | 已完成电商场景；邮件场景待补 |

### 未完成 ❌（按当前优先级排序）

| 实验 | 场景 | 难度 | 依赖 | 优先级 |
|------|------|------|------|--------|
| **B5 Few-shot DSL 重跑** | 电商 | 低 | 修复 prompt schema | **🔴 高** — 旧结果因 bug 已删除 |
| **B3 SQL-R1** | 电商 | 高 | RL训练复现 | **🔴 高** — P1 baseline |
| **B4 GraphRAG** | 电商 | 高 | 构图+API调用 | **🟡 中** — P1 baseline |
| **Qwen3-8B 全系列** | 电商+邮件 | 高 | 需多GPU训练 | 中 — 规模实验 |
| **Qwen3-4B/8B few-shot ICL** | 电商+邮件 | 低 | 本地推理 | 中 |
| **B2 扩到 N=3000** | 电商 | 低 | API 调用 | 中 — 当前仅 N=300 |
| E2 Action Space Ablation | 电商 | 中 | GPU训练 | 低 — baselines 完成后再做 |

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
| 2026-05-13 | B2 LLM-Enhanced baselines 完成 (HyDE, IRCoT, RankGPT, SQR) |
| 2026-05-13 | E2 Action Space Ablation 开始训练 (102 服务器 GPU2-5) |
| **2026-05-25** | **ARR May 截止** |
