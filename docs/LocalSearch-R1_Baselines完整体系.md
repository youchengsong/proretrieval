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

---

## 二、6 个范式下保留的 Baseline

### P0 No Generation

| 方法 | 来源 | 必要性 |
|------|------|--------|
| BM25 | 计划 | P0 |
| ColQwen | 计划 | P0 |
| CLIP（仅电商）| 计划 | P0 |
| BM25 + Dense RRF（k=60）| 计划 | P0 |
| + BGE-reranker（叠加在 RRF 之上）| 加 | P0（production 标杆，HF 直接 inference）|

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
- **E3 Complexity Breakdown**：所有 baseline 按 Level 1/2/3 分层；预期 Level 3 时 P5 相对 P3/P4 优势最大（OR/NOT/嵌套）。
- **E4-E11**：与本文档独立（reward / modality / orchestration / scaling / error / latency / cost）。

---

## 十、风险与应对

| 风险 | 应对 |
|------|------|
| Text2VectorSQL / DeepRetrieval 复现工作量 | 用公开 ckpt，2-3 天/篇 |
| 跑 SOTA 数据集 ≠ 跑我们数据集 | 用我们 train set 重训他们的方法（fair comparison）|
| 8B 训练成本高，时间不够 | 兜底：报 4B 全套 + 8B zero-shot/ICL，把 8B+RL 移 P1 |
| 跨范式被指模型规模不公平 | P0-P5 各范式尽量保持模型规模在 4B-8B 区间；规模消融见 §E8 |

---

**文档维护**：本文与 `LocalSearch-R1_最终工作计划.md` 配套使用。baseline 跑完时同步更新 §二 状态列；§一 范式表更新（如补 NL2SQL）时同步本文 §一。
