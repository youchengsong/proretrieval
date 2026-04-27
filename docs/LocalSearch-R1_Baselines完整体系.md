# LocalSearch-R1 Baseline 完整体系

> 配套文档：`LocalSearch-R1_最终工作计划.md`（§六 Baselines、§八 Experiments）
>
> 本文目的：以"封住每一类 reviewer 质疑"为组织原则，重新梳理需要跑的 baseline。
> 计划原版只列了 5 个 group（Single / Hybrid / Commercial LLM / OSS w/o RL / Ours），覆盖不全；
> 本文补齐传统 IR、NL2SQL、同 DSL 不同范式、同 RL 不同动作空间四条对比线。

---

## 一、为什么要重新组织 Baseline

EMNLP 主会 2025 录取率仅 22%。reviewer 攻击 baseline 的方式不是"你跑了多少"，
而是"是否每一个独立质疑都有方法专门反驳"。Baseline 完整性 > 数量。

`docs/04_acceptance_probability_and_improvements.md` 已经把"缺少 hybrid baseline"列为
**致命风险 #3**；本文把这一风险扩展为五大类 reviewer 质疑，每类配一条 baseline 线。

---

## 二、五类 Reviewer 质疑 → 五条 Baseline 线

### Q1：传统 IR 早就够用，为什么需要 LLM？

**对手**：纯检索方法（无 LLM 生成）。本类 baseline 的目的是证明
"复杂 logic + 多模态 query 上传统 IR 天花板远低于 GT"。

| ID | Baseline | 类型 | 必要性 | 说明 |
|----|----------|------|--------|------|
| A1 | **BM25** | 稀疏 | P0 | 经典稀疏 baseline，必跑 |
| A2 | **ColQwen / BGE / E5** | 稠密单向量 | P0 | 现代稠密单路 |
| A3 | **ColBERT v2** | 稠密多向量 (late interaction) | P1 | 比单向量强；电商邮件都适用 |
| A4 | **CLIP** | 图像稠密（仅电商） | P0 | 图像通道单跑 |
| A5 | **BM25 + Dense RRF (k=60)** | 工业级 hybrid | P0 | 计划已列，最低 hybrid 门槛 |
| A6 | **First-stage + Cross-encoder reranker**（BGE-reranker / mxbai-rerank） | 工业 RAG 标配 | P0 | 不报会被问"为什么不和 production pipeline 比" |
| A7 | **HyDE** (Gao 2022) | query 改写 | P0 | 几乎免费但 reviewer 必看 |
| A8 | **Query2doc** | query 扩写 | P2 | A7 的 LLM-augmented 替代品，二选一 |

> 结论：**A1/A2/A4/A5/A6/A7 必须有**。A1+A2+A5 单独不够，必须加 A6 reranker
> 才算"工业级强 baseline"。

---

### Q2：结构化部分用 NL2SQL 不就够了，为什么要 DSL？

**对手**：纯 text-to-SQL（不带向量编排）。本类 baseline 的目的是证明
"含语义条件的 query 上 NL2SQL 拿不到 GT"，因此 SQL+向量才是必需。

| ID | Baseline | 类型 | 必要性 | 说明 |
|----|----------|------|--------|------|
| B1 | **GPT-4o / Gemini / Claude zero-shot SQL** | LLM prompt → 纯 SQL | P0 | 让模型只生成 SQL（不带向量），看 Hit@1 损失多少 |
| B2 | **GPT-4o few-shot SQL**（3-5 ICL） | Few-shot prompting | P0 | 配合 B1 |
| B3 | **DIN-SQL / DAIL-SQL / MAC-SQL** | 经典 NL2SQL pipeline | P1 | 学术 NL2SQL 强 baseline |
| B4 | **C3 / CHESS / RESDSQL** | 学术 NL2SQL SOTA | P2 | 时间充裕再做 |
| B5 | **SQL-R1** (arXiv 2025) | RL-trained NL2SQL | **P0** | **最直接 RL 同动作空间竞品**；不跑会被认为"刻意回避" |
| B6 | **CodeS / SQLNet** | SFT-based NL2SQL | P2 | 时间充裕再做 |

> 关键论证：**"纯 SQL 拿不到含语义条件的 GT"** 是 LocalSearch-R1 最强的 motivation，
> 但口头说不算，必须有 B1+B5 的实测数字。

---

### Q3：LangChain Self-querying RAG 已经有 filter+vector，新在哪？

**对手**：filter+vector 固定逻辑组合。

| ID | Baseline | 类型 | 必要性 | 说明 |
|----|----------|------|--------|------|
| C1 | **LangChain Self-querying RAG** | filter+vector，固定 AND | P0 | 计划已列 |
| C2 | **LlamaIndex Auto-Retriever** | 类似 C1 | P2 | 与 C1 重复，可省 |
| C3 | **GraphRAG / KG-RAG** | KG+向量 | P2 | 不同方向，可省 |

> 结论：**C1 必跑**，证明"固定 AND 在含 OR/NOT/嵌套的 query 上失败"。

---

### Q4：和最新 RL-RAG 比怎么样？

**对手**：同时代 RL-based 检索方法。

| ID | Baseline | 类型 | 必要性 | 说明 |
|----|----------|------|--------|------|
| D1 | **Search-R1** (COLM 2025) | RL+文本 query+多轮 | P1 | 可用 §E2 ablation 替代真复现 |
| D2 | **DeepRetrieval** (COLM 2025) | RL+boolean/SQL，**最直接 RL 竞品** | **P0** | 04 文档已警告"必须准确描述"；只放 ablation 不够 |
| D3 | **s3** (EMNLP 2025) | RL+searcher/generator 解耦 | P1 | D 类挑 1 个跑足矣 |
| D4 | **R1-Searcher** (EMNLP 2025 Findings) | 两阶段 RL | P2 | 与 D3 重复 |
| D5 | **R3-RAG** (EMNLP 2025 Findings) | RL+多轮 | P2 | 与 D3 重复 |
| D6 | **MCTS-RAG** (Yale, EMNLP 2025 Findings) | MCTS+RAG（非 RL） | P2 | 不同范式，可作 future work 提及 |

> 结论：**D2 必须真复现**（在我们数据上跑），D1 可用 ablation 替代，D3 中挑一个。

---

### Q5：同 DSL 设计，你的 RL 真比 SFT/Prompt 好？

**对手**：同 DSL 设计、不同训练范式。这条线最锋利地回答"RL 是否必要"。

| ID | Baseline | 类型 | 必要性 | 说明 |
|----|----------|------|--------|------|
| E1 | **Text2VectorSQL** (arXiv) | 同 DSL，SFT | **P0** | **最直接 DSL 竞品**；同 DSL，SFT vs RL，回答 RL 必要性 |
| E2 | **SUQL** (Stanford NAACL 2024) | SQL+语义 primitive，prompting | P1 | 先驱设计，prompt-based 复现成本低 |
| E3 | **Qwen3-4B/8B + DSL few-shot ICL** | 自家开源同尺寸 prompt baseline | **P0** | 计划里漏了；不跑就缺"开源 ICL vs RL"对比 |
| E4 | **Qwen3-4B + DSL SFT** | 同 DSL SFT | ✅ 已有 | 4090/Blackwell 上 SFT-4B Hit@1=0.762 (电商) / 0.891 (邮件) |
| E5 | **Qwen3-4B + DSL Zero-shot** | 自家 base model | ✅ 已有 | 已跑：电商 0.121 / 邮件 0.243 |

> 关键论证：**"同 DSL，RL 比 SFT 强多少"**（E4 vs G）+ **"同 DSL，RL 比 ICL 强多少"**（E3 vs G）
> 是论文最强的"RL 必要性"证据。

---

## 三、计划原版已有但状态不全的项

| 计划项 | 实际情况 | 需要 |
|---------|---------|------|
| GPT-4o zero-shot DSL | 计划写"已有"，**实际只有 GPT-5.4 / Gemini / Claude，没有 GPT-4o** | 跑一次（API 便宜） |
| 邮件场景 commercial LLM | **完全没跑**（只电商有） | GPT-5.4 / Gemini / Claude 邮件版各一次 |
| Qwen3-8B 系列 | **完全没跑**（base + GRPO + DAPO） | 训练 + 推理；最贵 |

---

## 四、Baseline 全景表（合到 §E1 主表）

| 类别 | 子类 | 代表方法 | 回答的质疑 |
|------|------|---------|-----------|
| **A. 传统 IR** | A1 稀疏 | BM25 | Q1 |
| | A2 稠密单向量 | ColQwen / BGE | Q1 |
| | A3 稠密多向量 | ColBERT v2 | Q1 |
| | A4 图像 | CLIP | Q1 |
| | A5 工业 hybrid | BM25+Dense RRF | Q1 |
| | A6 重排 | First-stage + BGE-reranker | Q1 |
| | A7 query 改写 | HyDE | Q1 |
| **B. NL2SQL** | B1/B2 prompting | GPT-4o zero/few-shot SQL | Q2 |
| | B3 学术 SOTA | DIN-SQL / DAIL-SQL | Q2 |
| | B5 RL | SQL-R1 | Q2 |
| **C. Filter+Vector** | C1 prompting | LangChain Self-query | Q3 |
| **D. RL-RAG** | D1 文本 query | Search-R1 (or §E2 ablation) | Q4 |
| | D2 boolean | DeepRetrieval | Q4 |
| | D3 多轮 | s3 / R1-Searcher | Q4 |
| **E. 同 DSL 不同范式** | E1 SFT | Text2VectorSQL | Q5 |
| | E2 prompting | SUQL / Qwen3 few-shot | Q5 |
| | E3 SFT (own) | Qwen3-4B SFT (✅ 已有) | Q5 |
| | E5 Zero-shot (own) | Qwen3-4B base (✅ 已有) | Q5 |
| **F. Commercial LLM** | F1 zero-shot | GPT-4o / Gemini / Claude | 综合上限 |
| | F2 few-shot | 同上 + ICL | 综合上限 |
| | F3 thinking | GPT-5.4 thinking / o1 | reasoning 模型上限 |
| **G. Ours** | G1 GRPO 4B/8B | LocalSearch-R1 | — |
| | G2 DAPO 4B/8B | LocalSearch-R1 (主结果) | — |

---

## 五、按"每个质疑都有专门 baseline"的最小封口集

按 ROI 排，以 EMNLP **主会**为目标的底线：

### 🔴 P0（不做就要被拒）
1. **Text2VectorSQL**（E1）—— 同 DSL，SFT vs RL，回答 Q5（RL 是否必要）
2. **DeepRetrieval（在你数据上跑）**（D2）—— 同 RL，纯 boolean/SQL，回答 Q4 + 印证向量必要性
3. **SQL-R1**（B5）—— 同 RL，纯 SQL，回答 Q2 + 印证向量必要性
4. **LangChain Self-querying RAG**（C1）—— 回答 Q3
5. **HyDE**（A7）—— 回答 Q1 (query 改写)
6. **BM25 + Dense RRF**（A5）—— 回答 Q1 (hybrid)
7. **Cross-encoder reranker**（A6）—— 回答 Q1 (production)
8. **邮件场景 commercial LLM** 补齐（F1）
9. **GPT-4o zero/few-shot DSL** 补齐（F1/F2，计划遗漏）
10. **Qwen3-4B + DSL few-shot ICL**（E3）—— 自家开源公平对比

### 🟡 P1（main paper 加分）
11. BM25 / ColQwen / CLIP single-mode（A1/A2/A4）
12. ColBERT v2（A3）
13. DIN-SQL / DAIL-SQL（B3）
14. Search-R1 真复现（D1，可用 ablation 代替）
15. s3 或 R1-Searcher 挑一个（D3）
16. SUQL prompt-based（E2）

### 🟢 P2（时间够再补）
17. Query2doc（A8）
18. C3 / CHESS / RESDSQL（B4）
19. CodeS / SQLNet（B6）
20. R3-RAG / MCTS-RAG（D5/D6）
21. Qwen3-8B 系列（训练成本高）

---

## 六、三档实施方案

| 档次 | 时间 | 必跑（最少） | 目标 |
|------|------|-------------|------|
| **保底（Findings）** | 2 周 | A1/A2/A4/A5/A7 + B1/B2 + C1 + E3 + F1 + G | 主表 ~10 行，覆盖 Q1/Q3/Q5 |
| **冲 Main** | 4 周 | + A3/A6 + B5 (SQL-R1) + D2 (DeepRetrieval) + E1 (Text2VectorSQL) + F2/F3 | 覆盖 Q1-Q5 全部，每类 ≥1 个 |
| **完整版** | 6+ 周 | + B3 + D1/D3 + E2 + 8B 系列 + 更多 | 真正鲁棒；时间紧建议砍 |

---

## 七、复现路径与开源资源

### A 类（传统 IR）

| 方法 | 推荐实现 | 备注 |
|------|---------|------|
| BM25 | `rank_bm25` (PyPI) 或 Pyserini | Pyserini 同时支持 BM25+Dense Hybrid |
| ColQwen | 已有特征：`product_text_features_*.parquet` 复用 | — |
| ColBERT v2 | `colbert-ai` (Stanford) | 训练好的 ColBERTv2.0 ckpt 公开 |
| CLIP | OpenAI CLIP / SigLIP（已有图像特征） | — |
| RRF | 自己实现（10 行） | k=60 经典值 |
| Reranker | `BAAI/bge-reranker-large` 或 `mixedbread-ai/mxbai-rerank-large-v1` | HuggingFace 直接 inference |
| HyDE | LangChain `HypotheticalDocumentEmbedder` | 用 GPT-4o 生成假设文档 |

### B 类（NL2SQL）

| 方法 | 推荐实现 |
|------|---------|
| GPT-4o/Gemini SQL | 改 prompt 让模型只生成 SQL（不带 vector_query_list） |
| SQL-R1 | https://github.com/MJinXiang/SQL-R1（开源） |
| DIN-SQL | https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting |
| DAIL-SQL | https://github.com/BeachWang/DAIL-SQL |

### C 类

| 方法 | 推荐实现 |
|------|---------|
| LangChain Self-query | LangChain SelfQueryRetriever；适配我们的 schema |

### D 类（RL-RAG）

| 方法 | 推荐实现 |
|------|---------|
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 |
| DeepRetrieval | https://github.com/pat-jj/DeepRetrieval |
| s3 | https://github.com/pat-jj/s3 |
| R1-Searcher | https://github.com/RUCAIBox/R1-Searcher |

### E 类（同 DSL 不同范式）

| 方法 | 推荐实现 |
|------|---------|
| Text2VectorSQL | 论文给出代码，参考其 SFT 配方在我们数据上跑 |
| SUQL | https://github.com/stanford-oval/suql（prompt-based，最快） |
| Qwen3-4B few-shot ICL | 自己实现：用 3-5 个 DSL 示例 + 当前 query |

---

## 八、与原计划 §六 的对应关系

原计划 5 个 Group 在新体系中的位置：

| 原 Group | 新分类 | 状态 |
|---------|-------|------|
| Group 1: Single-mode (BM25 / ColQwen / CLIP) | A1/A2/A4 | 计划列出但**未跑** |
| Group 2: Hybrid (RRF, Self-query) | A5 / C1 | 计划列出但**未跑** |
| Group 3: Commercial LLM | F1/F2/F3 | 电商部分已跑，邮件未跑；GPT-4o zero/few-shot 全缺 |
| Group 4: OSS w/o RL (Qwen3-4B/8B base) | E5 | 4B 已跑，8B 缺 |
| Group 5: Ours (GRPO/DAPO 4B/8B) | G | 4B 已跑，8B 缺 |
| **新增 A3/A6/A7（传统 IR 强 baseline）** | A3/A6/A7 | 计划缺，需加 |
| **新增 B 类 NL2SQL** | B1/B2/B5 | 计划缺，需加 |
| **新增 D2 DeepRetrieval（真复现）** | D2 | 计划只在 ablation，需补真复现 |
| **新增 E1 Text2VectorSQL** | E1 | 计划"考虑加入"，需落地 |
| **新增 E3 Qwen3 few-shot ICL** | E3 | 计划完全没列 |

---

## 九、与 §八 实验设计的对应

- **E1 Main Results**：本文档的全景表，按 A-G 分组替换原 5 个 Group
- **E2 Action Space Ablation**：仍按原计划（text query / SQL only / boolean / DSL 全套），但
  注意此 ablation **不能替代** D2/B5 的真复现 baseline
- **E3 Complexity Breakdown**：按 Level 1/2/3 + 含 OR/NOT 细分，所有 baseline 都要分层报
- **E5 Modality Ablation**：电商专用，与本文档独立
- **E6 Orchestration Behavior**：定性分析，与本文档独立

---

## 十、风险与应对

| 风险 | 应对 |
|------|------|
| Text2VectorSQL/DeepRetrieval/SQL-R1 复现工作量大 | 每篇 2-3 天，集中 1 周搞定；用论文公开 ckpt 优先 |
| 无 SQL benchmark schema 适配 | 我们的 email/电商 schema 和 BIRD/Spider 类似，重点适配 prompt |
| RL baselines 跑 SOTA 数据集 ≠ 跑我们数据集 | 用我们的 train set 重训他们的方法（fair comparison）|
| Reranker 选错 | 用 BGE-reranker-large 默认，简单稳 |
| HyDE 生成错误假设文档 | 用 GPT-4o + temperature=0.3 + 3 个 candidates |

---

**文档维护**：本文与 `LocalSearch-R1_最终工作计划.md` 配套使用。
当 baseline 跑完、状态变化时同步更新本文表格的"必要性"列。
