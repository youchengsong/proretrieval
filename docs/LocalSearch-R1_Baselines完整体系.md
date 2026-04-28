# LocalSearch-R1 Baseline 完整体系

> 配套：`LocalSearch-R1_最终工作计划.md` §一 范式对比、§六 Baselines、§八 Experiments
>
> 论文 §4 主表按范式（action space）组织，本文按 **6 个范式** 重组 baseline（计划范式表 + NL2SQL 补充）。
>
> 计划 §六 五个 Group 的 baseline **全部保留**；其余仅保留对实验 solidness 必需的项，alternative/兜底候选不列。

---

## 一、范式分类轴

计划 §一 定义了 5 个范式，本文补 **NL2SQL** 为独立范式：

| ID | 范式 | 模型输出 | 检索通道 | 逻辑组合 | 代表 |
|----|------|---------|---------|---------|------|
| **P0** | No Generation | —（无 LLM 生成） | 单/固定多路 | 无/固定加权 | BM25, ColQwen, RRF |
| **P1** | Query Rewriting | NL query | 固定单路 | 无 | HyDE, Search-R1 |
| **P2** | Boolean Query Gen | boolean 表达式 | 固定单路 | 有限 AND/OR | DeepRetrieval |
| **P3** | Self-Querying | filter + vector query | 固定双路 | 固定 AND | LangChain SQR |
| **P4** | NL2SQL（补） | pure SQL | 结构化单路 | 完整 SQL 但无向量 | SQL-R1 |
| **P5** | **Retrieval Program Synthesis** | **可执行 DSL** | **动态多路多模态** | **任意组合** | **Ours** |

主轴 = 范式 P0-P5；副轴 = 训练方式（zero-shot / few-shot ICL / SFT / RL）。

### Scope 排除声明

本文范式仅覆盖 **single-pass retrieval generation**（一次 NL→检索动作生成）。以下范式不在 baseline 体系内，论文 §1 Related Work / §2 Scope 须显式声明：

| 排除范式 | 代表方法 | 排除理由 |
|---------|---------|---------|
| Multi-step / iterative agent retrieval | Self-RAG, IRCoT, ReAct-Retrieval | 多轮检索-反思循环，与本文 single-pass setting 正交；可作为正交工作在 §Future Work 提及 |
| LLM-based reranking | RankGPT, RankVicuna, MonoT5 | 第二阶段重排，本文 focus on first-stage retrieval；BGE-reranker 已代表 cross-encoder 类 production 重排 |
| KG / Graph augmented retrieval | GraphRAG, KG-RAG | 依赖外部知识图谱构建，与本文"结构化字段+向量混合"范式不同 |

---

## 二、6 个范式下保留的 Baseline

### P0 No Generation

| 方法 | 来源 | 必要性 |
|------|------|--------|
| BM25 | 计划 | P0 |
| ColQwen | 计划 | P0 |
| CLIP（仅电商）| 计划 | P0 |
| BM25 + Dense RRF（k=60）| 计划 | P0 |
| + BGE-reranker（叠加在 RRF 之上）| 加 | P0（production 标杆，HF 直接 inference；first-stage 后 cross-encoder 重排，LLM reranking 见 scope 排除）|
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

---

## 三、论证链

```
跨范式（action space 价值）：
  P5 > P0   需要 LLM 生成
  P5 > P1   需要结构化输出           [E2 ablation 内部覆盖]
  P5 > P2   需要多模态+完整逻辑       [DeepRetrieval vs Ours, 同 RL]
  P5 > P3   需要任意逻辑组合          [LangChain SQR vs Ours]
  P5 > P4   需要向量原语             [E2 "SQL only" 内部 + SQL-R1 外部加分]

P5 内训练价值（RL 必要性）：
  RL > SFT       [Text2VectorSQL vs Ours, 同 DSL 不同训练]
  RL > ICL       [Qwen3-4B/8B few-shot DSL vs Ours, 同 base 同 DSL]
  RL > zero-shot [Qwen3 base / GPT-4o vs Ours]
```

每条比较只动一个轴，控制变量明确。

---

## 四、计划"已有"但实际未跑的项

| 项 | 状态 | 范式 |
|----|------|------|
| GPT-4o zero-shot DSL | 计划标"已有"，实际未跑 | P5 zero-shot |
| 邮件场景 commercial LLM | **完全没跑**（只电商有）| P5 zero/few-shot |
| Qwen3-8B 系列（base + GRPO + DAPO）| **完全没跑** | P5 zero-shot + RL |
| Qwen3-4B/8B + DSL few-shot ICL | 计划遗漏 | P5 ICL |

---

## 五、最小封口集

**🔴 P0（不跑要被拒）**

1. **DeepRetrieval 真复现**（P2 RL）
2. **Text2VectorSQL**（P5 SFT，同 DSL 范式）
3. LangChain SelfQueryRetriever（P3）
4. HyDE（P1 prompt）
5. BM25 + Dense RRF（P0）
6. + BGE-reranker（P0 production）
7. **Qwen3-4B/8B + DSL few-shot ICL**（P5 ICL，计划遗漏）
8. GPT-4o zero/few-shot DSL（计划遗漏）
9. 邮件 commercial LLM（计划遗漏）
10. **Qwen3-8B + GRPO/DAPO**（P5 RL 主结果，最贵）

**🟡 P1（加分）**

- SQL-R1（P4 RL）
- Single-mode 单跑：BM25 / ColQwen / CLIP（简单稳，可作下界）

---

## 六、三档实施方案

| 档次 | 时间 | 必跑（增量）|
|------|------|------------|
| 保底（Findings）| 2 周 | 计划 5 个 Group + HyDE + reranker + Qwen3 ICL + 计划遗漏补齐 |
| 冲 Main | 4 周 | 保底 + DeepRetrieval + Text2VectorSQL |
| 完整版 | 6+ 周 | 上述 + SQL-R1 + Search-R1 真复现 |

---

## 七、复现资源

| 方法 | 资源 |
|------|------|
| BM25 | `rank_bm25` 或 Pyserini |
| RRF | 自实现 10 行，k=60 |
| BGE-reranker | `BAAI/bge-reranker-large`（HF）|
| HyDE | LangChain `HypotheticalDocumentEmbedder` + GPT-4o，temp=0.3，3 candidates |
| LangChain SQR | LangChain SelfQueryRetriever，适配 schema |
| DeepRetrieval | https://github.com/pat-jj/DeepRetrieval |
| SQL-R1 | https://github.com/MJinXiang/SQL-R1 |
| Text2VectorSQL | 论文公开代码，按其 SFT 配方在我们数据上重训 |
| Qwen3 few-shot ICL | 自实现：3-5 个 DSL 示例 + 当前 query |

---

## 八、对计划文件 §一 范式表的补充建议

`最终工作计划.md` §一 范式对比表缺 NL2SQL，建议补：

```
| NL2SQL | pure SQL | 结构化单路 | 完整 SQL 但无向量 | DIN-SQL, SQL-R1 |
```

否则 P4 baseline 在 §2 Related Work / §4 Method 写作时无对手范式名可挂。

---

## 九、与 §八 实验设计的对应

- **E1 Main Results**：用本文 §二 的 6 范式分组替换计划 §六 的 5 个 Group。
- **E2 Action Space Ablation**：固定 P5 的 base + RL，改 action space——内部覆盖 P1/P2/P4 三个范式作控制对照。**不能替代 DeepRetrieval 真复现**（自家训练 ≠ 原论文 SOTA）。
- **E3 Complexity Breakdown**：所有 baseline 按 Level 1/2/3 分层；预期 Level 3 时 P5 相对 P3/P4 优势最大（OR/NOT/嵌套）。**公平性约定**：Level 1（单一字段/单一语义）是各范式的公平比较层，性能应接近；Level 2/3（多字段组合 / OR/NOT/嵌套）是 P5 的优势展示层，P0-P4 在此层崩盘是预期结果而非 unfair setup —— §写作时需明示该约定，否则 reviewer 会质疑"为什么不在 Level 1 也报数"。
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

---

## 附录 A：已删除的 Baseline 候选与原因

下表列出原 Q1-Q5 reviewer 质疑体系中曾纳入、本次重组按"**不在 `LocalSearch-R1_最终工作计划.md` §六 里 + 对实验 solidness 非必需**"标准砍掉的候选 baseline。如时间充裕或新增 reviewer 关注点，可重新纳入。

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
| GraphRAG / KG-RAG | C3 KG-augmented | 不同方向（KG），与本文"结构化+向量混合"范式不一致 |
| Search-R1（真复现）| D1 RL Query Rewriting | 由 §E2 Action Space Ablation 的 "text query" 行内部覆盖，外部真复现非必需 |
| s3 | D3 RL searcher/generator 解耦 | 与 Search-R1 同 P1 范式（RL Query Rewriting），二选一 |
| R1-Searcher | D4 RL 两阶段 | 同 P1 范式，与 s3 重复 |
| R3-RAG | D5 RL 多轮 | 同 P1 范式，重复 |
| MCTS-RAG | D6 MCTS+RAG | 非 RL 范式（MCTS 规划），与方向不一致 |
| SUQL | E2 同 DSL prompt-based | Text2VectorSQL 已代表"同 DSL 不同训练"对照，重复 |
| GPT-5.4 thinking / o1 | F3 reasoning 模型 | 不在最终工作计划 §六，reasoning 模型上限非必需 |
| Claude zero / few-shot DSL | F1/F2 commercial LLM | 计划 §六 Group 3 仅列 GPT-4o + Gemini，不含 Claude |

> **注**：原 Q1-Q5 reviewer 质疑映射在主体已降级为可选标注；本附录仅记录候选 baseline 的删除决策，便于审稿/Rebuttal 阶段快速回溯"为什么没跑 X"。

---

**文档维护**：本文与 `LocalSearch-R1_最终工作计划.md` 配套使用。baseline 跑完时同步更新 §二 状态列；§一 范式表更新（如补 NL2SQL）时同步本文 §一。
