# LocalSearch-R1 Baseline 完整体系

> 配套：`LocalSearch-R1_最终工作计划.md` §一 范式对比、§六 Baselines、§八 Experiments
>
> 论文 §4 主表按范式（action space + 正交轴）组织，本文按 **9 个范式** 重组 baseline（计划范式表 + NL2SQL + 三个正交轴范式）。
>
> 计划 §六 五个 Group 的 baseline **全部保留**；其余仅保留对实验 solidness 必需的项，alternative/兜底候选不列。

---

## 一、范式分类轴

本文采用 **双轴范式体系**：

- **主轴（output complexity）**：P0-P5，沿"输出结构化程度 / 检索通道丰富度 / 逻辑组合表达力"递增
- **正交轴**：P6-P8，分别在"时间轴（轮数）/ 流水线轴（阶段）/ 知识源轴（KG）"上扩展，可与主轴范式叠加

### 主轴范式（output complexity，P0-P5）

| ID | 范式 | 模型输出 | 检索通道 | 逻辑组合 | 代表 |
|----|------|---------|---------|---------|------|
| **P0** | No Generation | —（无 LLM 生成） | 单/固定多路 | 无/固定加权 | BM25, ColQwen, RRF |
| **P1** | Query Rewriting | NL query | 固定单路 | 无 | HyDE, Search-R1 |
| **P2** | Boolean Query Gen | boolean 表达式 | 固定单路 | 有限 AND/OR | DeepRetrieval |
| **P3** | Self-Querying | filter + vector query | 固定双路 | 固定 AND | LangChain SQR |
| **P4** | NL2SQL（补） | pure SQL | 结构化单路 | 完整 SQL 但无向量 | SQL-R1 |
| **P5** | **Retrieval Program Synthesis** | **可执行 DSL** | **动态多路多模态** | **任意组合** | **Ours** |

### 正交轴范式（与主轴正交，可叠加，P6-P8）

| ID | 正交轴 | 范式 | 模型输出 | 与主轴关系 | 代表 |
|----|--------|------|---------|-----------|------|
| **P6** | 时间轴（single-pass → multi-pass）| Multi-step Iterative Agent | NL queries + 动作序列（多轮）| 把任意主轴范式包成多轮循环 | Self-RAG, IRCoT, R1-Searcher |
| **P7** | 流水线轴（first-stage → +rerank）| LLM Reranking（二阶段）| reranked list of top-K | 套在任意主轴范式 first-stage 之后 | RankGPT, RankVicuna |
| **P8** | 知识源轴（corpus → +KG）| KG-augmented Retrieval | KG path + vec query | 把语料预先构成 KG 再混合检索 | GraphRAG, KG-RAG |

> **正交性说明**：P6/P7/P8 不是 P0-P5 的替代，而是叠加维度。例如 "Self-RAG + dense retrieval" = P6 ∘ P0，"RankGPT + RRF" = P7 ∘ P0，"GraphRAG" = P8 ∘ P1。本文将每个正交轴各取 1-2 个代表纳入，论证 P5 在主轴的优势同时不被正交轴反超。

主轴 = 范式 P0-P5；副轴 = 训练方式（zero-shot / few-shot ICL / SFT / RL）；正交轴 = P6-P8。

---

## 二、9 个范式下保留的 Baseline

### P0 No Generation

| 方法 | 来源 | 必要性 |
|------|------|--------|
| BM25 | 计划 | P0 |
| ColQwen | 计划 | P0 |
| CLIP（仅电商）| 计划 | P0 |
| BM25 + Dense RRF（k=60）| 计划 | P0 |
| + BGE-reranker（叠加在 RRF 之上，cross-encoder rerank）| 加 | P0（production 标杆，HF 直接 inference；与 P7 LLM rerank 正交）|
| **现代多模态 embedder：BGE-M3 / SigLIP（电商）** | 加 | P0（CLIP 已偏旧，reviewer 必问 newer multi-modal embedder；HF 直接 inference）|

### P1 Query Rewriting

| 方法 | 来源 | 必要性 |
|------|------|--------|
| HyDE | 加 | P0（prompt-based，几乎免费，reviewer 必看）|

> Search-R1 / s3 等 RL Query Rewriting 由 §E2 Action Space Ablation 的 "text query" 行覆盖，不再单跑。

### P2 Boolean Query Generation

| 方法 | 来源 | 必要性 |
|------|------|--------|
| **DeepRetrieval（在我们数据上真复现）** | 加 | **P0**（04 文档列为致命风险；E2 内部 ablation 不能替代）|

### P3 Self-Querying

| 方法 | 来源 | 必要性 |
|------|------|--------|
| LangChain SelfQueryRetriever | 计划（"Self-querying RAG"） | P0 |

### P4 NL2SQL

| 方法 | 来源 | 必要性 |
|------|------|--------|
| SQL-R1 | 加 | P1（加分；E2 内部 "SQL only" 行已提供同 RL 控制对照，外部 SOTA 加分但非必需）|

### P5 Retrieval Program Synthesis（Ours 所在）

按训练方式分四档，副轴上证明 **RL 必要性**：

| 训练 | 方法 | 来源 | 状态 |
|------|------|------|------|
| Zero-shot | Qwen3-4B base + DSL | 计划 | ✅ 已跑（电商 0.121 / 邮件 0.243）|
| Zero-shot | Qwen3-8B base + DSL | 计划 | 缺 |
| Zero-shot | GPT-4o zero-shot DSL | 计划 | **缺**（计划标"已有"实际未跑）|
| Few-shot ICL | GPT-4o few-shot DSL | 计划 | 缺 |
| Few-shot ICL | Gemini few-shot DSL | 计划 | 缺 |
| Few-shot ICL | **Qwen3-4B/8B + DSL few-shot** | 加 | **P0**（开源同尺寸 ICL 公平对比，计划遗漏）|
| SFT | Qwen3-4B SFT（自家） | ✅ | 电商 0.762 / 邮件 0.891 |
| SFT | **Text2VectorSQL** | 加 | **P0**（同 DSL SFT vs RL，Q5 RL 必要性核心论证）|
| **RL（Ours）** | Qwen3-4B + GRPO | 计划 | ✅ |
| **RL（Ours）** | Qwen3-4B + DAPO | 计划 | ✅ |
| **RL（Ours）** | Qwen3-8B + GRPO | 计划 | **缺** |
| **RL（Ours）** | **Qwen3-8B + DAPO（主结果）** | 计划 | **缺** |

### P6 Multi-step Iterative Agent（时间轴）

| 方法 | 训练 | 来源 | 必要性 |
|------|------|------|--------|
| **IRCoT**（iterative CoT + retrieve）| prompt-based | 加 | **P0**（多轮范式最便宜代表，无训练成本）|
| **Self-RAG**（self-reflection + retrieve）| SFT | 加 | P1（多轮 SFT 代表，加分；公开 ckpt 可直接跑）|
| R1-Searcher | RL multi-turn | 加 | P1（多轮 RL 代表，加分；可与 P5 RL 在轮数轴上对比）|
| ReAct-Retrieval | prompt-based | 加 | P2（与 IRCoT 同 prompt-based 多轮，二选一即可）|

> **对比意义**：P5 在 single-pass 内一次生成完整 DSL；P6 通过多轮交互逐步逼近答案。两者各有 trade-off（latency / cost vs flexibility），E2 ablation 可量化"多轮带来的额外信息增益是否被 P5 的结构化输出覆盖"。

### P7 LLM Reranking（流水线轴）

| 方法 | 训练 | 来源 | 必要性 |
|------|------|------|--------|
| **RankGPT-4o on top of RRF** | prompt-based | 加 | **P0**（LLM rerank 最便宜代表，必看；与 BGE-reranker cross-encoder 形成对照）|
| RankVicuna | SFT | 加 | P1（开源 LLM rerank SFT 代表，加分）|
| MonoT5 | SFT | 加 | P2（pre-LLM 时代 cross-encoder rerank，可作历史对照）|

> **对比意义**：P7 是 second-stage 增强，可叠加在任意 first-stage 上。需要论证：**P5 first-stage 已超过 P0 + P7 cascade**（即 P5 端到端 ≥ "弱 first-stage + LLM rerank 救场"）。同时报 P5 + P7 的进一步增益（如 +2-3pp 则正交补强；如无增益则说明 P5 已饱和）。

### P8 KG-augmented Retrieval（知识源轴）

| 方法 | 训练 | 来源 | 必要性 |
|------|------|------|--------|
| **GraphRAG**（Microsoft）| prompt-based pipeline | 加 | **P1**（构图开销大但范式独特，加分项；时间充裕必跑）|
| KG-RAG | prompt-based | 加 | P2（与 GraphRAG 重复，二选一）|

> **对比意义**：P8 把语料预先构成 KG 再走图遍历 + 向量混合，与 P5 的"字段约束 + 向量"是不同的结构化方式。**预期结果**：P5 在结构化字段密集场景（电商 brand/price、邮件 From/Date）胜出；P8 在实体关系密集场景胜出。本文不主张 P5 全面 dominance，而是定位为 **"字段-密集" vs "关系-密集"两种结构化范式的互补**。

---

## 三、论证链

```
跨主轴（P0-P5，action space 价值）：
  P5 > P0   需要 LLM 生成
  P5 > P1   需要结构化输出           [E2 ablation 内部覆盖]
  P5 > P2   需要多模态+完整逻辑       [DeepRetrieval vs Ours, 同 RL]
  P5 > P3   需要任意逻辑组合          [LangChain SQR vs Ours]
  P5 > P4   需要向量原语             [E2 "SQL only" 内部 + SQL-R1 外部加分]

跨正交轴（P6-P8，叠加价值与互补性）：
  P5 ≥ P6   single-pass DSL 已捕获多轮交互的等价信息  [IRCoT/Self-RAG vs Ours]
  P5 ≥ P7   first-stage P5 已超过 dense + LLM rerank cascade  [RankGPT on RRF vs Ours]
  P5 ⊥ P8   不同结构化轴（字段 vs 图边）；字段-密集场景 P5 胜，关系-密集场景 P8 胜  [GraphRAG vs Ours]
  P5+P7 增益   正交补强：P5 first-stage + RankGPT rerank 是否进一步提升

P5 内训练价值（RL 必要性）：
  RL > SFT       [Text2VectorSQL vs Ours, 同 DSL 不同训练]
  RL > ICL       [Qwen3-4B/8B few-shot DSL vs Ours, 同 base 同 DSL]
  RL > zero-shot [Qwen3 base / GPT-4o vs Ours]
```

每条比较只动一个轴，控制变量明确。"≥" 表示主轴 dominance，"⊥" 表示正交互补不主张全面胜出。

---

## 四、计划"已有"但实际未跑的项

| 项 | 状态 | 范式 |
|----|------|------|
| GPT-4o zero-shot DSL | 计划标"已有"，实际未跑 | P5 zero-shot |
| 邮件场景 commercial LLM | **完全没跑**（只电商有）| P5 zero/few-shot |
| Qwen3-8B 系列（base + GRPO + DAPO）| **完全没跑** | P5 zero-shot + RL |
| Qwen3-4B/8B + DSL few-shot ICL | 计划遗漏 | P5 ICL |
| **正交轴范式（P6/P7/P8）** | **完全没跑**（原计划排除，本次纳入）| P6/P7/P8 |

---

## 五、最小封口集

**🔴 P0（不跑要被拒）**

主轴：
1. **DeepRetrieval 真复现**（P2 RL）
2. **Text2VectorSQL**（P5 SFT，同 DSL 范式）
3. LangChain SelfQueryRetriever（P3）
4. HyDE（P1 prompt）
5. BM25 + Dense RRF（P0）
6. + BGE-reranker（P0 production，cross-encoder）
7. **Qwen3-4B/8B + DSL few-shot ICL**（P5 ICL，计划遗漏）
8. GPT-4o zero/few-shot DSL（计划遗漏）
9. 邮件 commercial LLM（计划遗漏）
10. **Qwen3-8B + GRPO/DAPO**（P5 RL 主结果，最贵）

正交轴：
11. **IRCoT**（P6 多轮 prompt-based 代表）
12. **RankGPT-4o on top of RRF**（P7 LLM rerank 代表）

**🟡 P1（加分）**

主轴：
- SQL-R1（P4 RL）
- Single-mode 单跑：BM25 / ColQwen / CLIP（简单稳，可作下界）

正交轴：
- **Self-RAG**（P6 多轮 SFT 代表）
- **R1-Searcher**（P6 多轮 RL 代表，与 P5 RL 在"轮数轴"上对比）
- RankVicuna（P7 LLM rerank SFT 代表）
- **GraphRAG**（P8 KG 范式代表）
- **P5 + P7 cascade**（RankGPT 叠加在 P5 之上，论证正交补强或饱和）

---

## 六、三档实施方案

| 档次 | 时间 | 必跑（增量）|
|------|------|------------|
| 保底（Findings）| 3 周 | 计划 5 个 Group + HyDE + reranker + Qwen3 ICL + 计划遗漏补齐 + **IRCoT + RankGPT-4o**（P6/P7 prompt-based 代表）|
| 冲 Main | 5 周 | 保底 + DeepRetrieval + Text2VectorSQL + **Self-RAG + GraphRAG**（P6 SFT + P8 KG）|
| 完整版 | 7+ 周 | 上述 + SQL-R1 + Search-R1 真复现 + **R1-Searcher + RankVicuna + P5+P7 cascade**（正交轴全套）|

> 与上一版相比：保底/冲 Main/完整版各延 1 周，覆盖正交轴 P6/P7/P8。

---

## 七、复现资源

主轴：

| 方法 | 资源 |
|------|------|
| BM25 | `rank_bm25` 或 Pyserini |
| RRF | 自实现 10 行，k=60 |
| BGE-reranker | `BAAI/bge-reranker-large`（HF，cross-encoder）|
| BGE-M3 / SigLIP | `BAAI/bge-m3` / `google/siglip-large-patch16-384`（HF）|
| HyDE | LangChain `HypotheticalDocumentEmbedder` + GPT-4o，temp=0.3，3 candidates |
| LangChain SQR | LangChain SelfQueryRetriever，适配 schema |
| DeepRetrieval | https://github.com/pat-jj/DeepRetrieval |
| SQL-R1 | https://github.com/MJinXiang/SQL-R1 |
| Text2VectorSQL | 论文公开代码，按其 SFT 配方在我们数据上重训 |
| Qwen3 few-shot ICL | 自实现：3-5 个 DSL 示例 + 当前 query |

正交轴：

| 方法 | 资源 |
|------|------|
| **IRCoT**（P6） | https://github.com/StonyBrookNLP/ircot ；可用 GPT-4o / Qwen3-8B 作 reasoning model |
| **Self-RAG**（P6） | https://github.com/AkariAsai/self-rag ；公开 7B/13B SFT ckpt |
| **R1-Searcher**（P6） | https://github.com/RUCAIBox/R1-Searcher ；公开 RL ckpt |
| **RankGPT**（P7） | https://github.com/sunnweiwei/RankGPT ；prompt-based，GPT-4o backbone |
| **RankVicuna**（P7） | https://github.com/castorini/rank_llm ；公开 7B SFT ckpt |
| **MonoT5**（P7） | `castorini/monot5-3b-msmarco-10k`（HF）|
| **GraphRAG**（P8） | https://github.com/microsoft/graphrag ；构图开销 ~corpus 大小 × 0.5-1 |

---

## 八、对计划文件 §一 范式表的补充建议

`最终工作计划.md` §一 范式对比表缺 NL2SQL 与三个正交轴范式，建议补：

```
| NL2SQL | pure SQL | 结构化单路 | 完整 SQL 但无向量 | DIN-SQL, SQL-R1 |
| Multi-step Agent | NL + 动作序列 | 多轮动态 | reasoning-based | Self-RAG, IRCoT, R1-Searcher |
| LLM Reranking | reranked list | 第二阶段 | implicit | RankGPT, RankVicuna |
| KG-augmented | KG path + vec | KG + vec | graph traversal | GraphRAG |
```

否则 P4/P6/P7/P8 baseline 在 §2 Related Work / §4 Method 写作时无对手范式名可挂。

---

## 九、与 §八 实验设计的对应

- **E1 Main Results**：用本文 §二 的 9 范式分组替换计划 §六 的 5 个 Group。主轴 P0-P5 横排比，正交轴 P6-P8 单独一组（标注"orthogonal axis"）。
- **E2 Action Space Ablation**：固定 P5 的 base + RL，改 action space——内部覆盖 P1/P2/P4 三个主轴范式作控制对照。**不能替代 DeepRetrieval 真复现**（自家训练 ≠ 原论文 SOTA）。新增 **multi-round ablation**（P5 single-pass vs P5 强行包成多轮）来与 P6 对比。
- **E3 Complexity Breakdown**：所有 baseline 按 Level 1/2/3 分层；预期 Level 3 时 P5 相对 P3/P4 优势最大（OR/NOT/嵌套）。**公平性约定**：Level 1（单一字段/单一语义）是各范式的公平比较层，性能应接近；Level 2/3（多字段组合 / OR/NOT/嵌套）是 P5 的优势展示层，P0-P4 在此层崩盘是预期结果而非 unfair setup —— §写作时需明示该约定，否则 reviewer 会质疑"为什么不在 Level 1 也报数"。
- **新增 E_orthogonal Cascade Ablation**：报 P5 + P7（叠加 LLM rerank）和 P5 vs P8（KG-augmented）的 head-to-head；P6 的多轮成本 / 收益曲线（轮数 vs Recall@K）。
- **E4-E11**：与本文档独立（reward / modality / orchestration / scaling / error / latency / cost）。

---

## 十、风险与应对

| 风险 | 应对 |
|------|------|
| Text2VectorSQL / DeepRetrieval 复现工作量 | 用公开 ckpt，2-3 天/篇 |
| **DeepRetrieval 多模态适配**（原方法纯文本布尔，需扩展到图像 + 结构化字段）| **提前 2 周开跑留 buffer**；兜底：跑不动则改用 §E2 内部 "boolean only" ablation 替代，并在 §Limitations 显式说明 "DeepRetrieval 原设计为 text-only，多模态扩展非平凡" |
| 跑 SOTA 数据集 ≠ 跑我们数据集 | 用我们 train set 重训他们的方法（fair comparison）|
| 8B 训练成本高，时间不够 | 兜底：报 4B 全套 + 8B zero-shot/ICL，把 8B+RL 移 P1 |
| 跨范式被指模型规模不公平 | P0-P5 各范式尽量保持模型规模在 4B-8B 区间；规模消融见 §E8 |
| **正交轴范式不公平**（P6 多轮、P8 KG 都自带额外 compute）| 报 latency / token cost 双指标（§E9-E10）；不要求 P5 在 compute 上 dominance，但要求 **per-token efficiency** 上 dominance |
| **GraphRAG 构图开销**（10K corpus ~ 数小时 LLM 调用）| 限制构图用 GPT-4o-mini 降本；如果时间不够，仅在电商域跑 GraphRAG，邮件域留 §Limitations 说明 |
| **P6 IRCoT/Self-RAG 与 P5 不同 base model** | 用同尺寸 backbone（Qwen3-4B 或 Llama-3-8B）替换原论文 backbone；如不可行，在表格脚注标注 backbone 差异 |

---

## 附录 A：已删除的 Baseline 候选与原因（更新版）

下表列出原 Q1-Q5 reviewer 质疑体系中曾纳入、本次重组按"**不在 `LocalSearch-R1_最终工作计划.md` §六 里 + 对实验 solidness 非必需**"标准砍掉的候选 baseline。**注：原"排除三类范式（multi-step / LLM rerank / KG）"已在本次重组纳入 P6/P7/P8，从删除列表中移除。**

| 删除项 | 原所属类 | 删除原因 |
|--------|---------|---------|
| BGE / E5 | A2 稠密单向量 | ColQwen 已代表稠密单向量 baseline，重复 |
| ColBERT v2 | A3 稠密多向量 | late-interaction 替代品，可作时间充裕的加分项 |
| Query2doc | A8 query 改写 | HyDE 已代表 LLM-aug query rewriting，二选一 |
| GPT-4o / Gemini / Claude zero-shot SQL | B1 NL2SQL prompt | P4 范式由 SQL-R1 + §E2 内部 "SQL only" 行覆盖 |
| GPT-4o few-shot SQL | B2 NL2SQL ICL | 同上 |
| DIN-SQL / DAIL-SQL / MAC-SQL | B3 NL2SQL 学术 baseline | 同上 |
| C3 / CHESS / RESDSQL | B4 NL2SQL SOTA | 同上 |
| CodeS / SQLNet | B6 NL2SQL SFT | 同上 |
| LlamaIndex Auto-Retriever | C2 Self-querying | 与 LangChain SelfQueryRetriever 重复 |
| KG-RAG | ~~C3 KG-augmented~~ → 与 GraphRAG 二选一 | 与 P8 GraphRAG 重复，二选一 |
| Search-R1（真复现）| D1 RL Query Rewriting | 由 §E2 Action Space Ablation 的 "text query" 行内部覆盖，外部真复现非必需 |
| s3 | D3 RL searcher/generator 解耦 | 与 Search-R1 同 P1 范式（RL Query Rewriting），二选一 |
| R3-RAG | D5 RL 多轮 | P6 已有 R1-Searcher 代表 RL 多轮，重复 |
| MCTS-RAG | D6 MCTS+RAG | 非 RL 范式（MCTS 规划），与方向不一致 |
| ReAct-Retrieval | P6 prompt-based 多轮 | 与 IRCoT 重复（同为 prompt-based 多轮），二选一 |
| SUQL | E2 同 DSL prompt-based | Text2VectorSQL 已代表"同 DSL 不同训练"对照，重复 |
| GPT-5.4 thinking / o1 | F3 reasoning 模型 | 不在最终工作计划 §六，reasoning 模型上限非必需 |
| Claude zero / few-shot DSL | F1/F2 commercial LLM | 计划 §六 Group 3 仅列 GPT-4o + Gemini，不含 Claude |

> **本次重组的纳入变化**：
> - **新纳入**：IRCoT、Self-RAG、R1-Searcher（P6）；RankGPT、RankVicuna、MonoT5（P7）；GraphRAG（P8）
> - **保留删除**：与新纳入项重复或同范式的（KG-RAG、ReAct-Retrieval、R3-RAG、s3、Search-R1）
>
> 本附录仅记录候选 baseline 的删除决策，便于审稿/Rebuttal 阶段快速回溯"为什么没跑 X"。

---

**文档维护**：本文与 `LocalSearch-R1_最终工作计划.md` 配套使用。baseline 跑完时同步更新 §二 状态列；§一 范式表更新（如补 NL2SQL / 正交轴范式）时同步本文 §一。
