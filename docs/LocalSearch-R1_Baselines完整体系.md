# LocalSearch-R1 Baseline 完整体系

> 配套：`LocalSearch-R1_最终工作计划.md` §一 范式对比、§六 Baselines、§八 Experiments
>
> 论文 §4 主表按 **5 类范式**（B1-B5）组织：4 类 baseline + Ours，对应"action space + 知识源"两个核心轴。
>
> 计划 §六 五个 Group 的 baseline **全部保留**；其余仅保留对实验 solidness 必需的项。

---

## 一、范式分类轴

本文采用 **5 类范式（B1-B5）**：

| ID | 范式 | 核心特征 | 模型输出 | 与 Ours 关系 | 代表方法 |
|----|------|---------|---------|------------|---------|
| **B1** | Pure Retrieval | 无 LLM 生成 | — | dominance | BM25, ColQwen, BGE-M3, RRF, BGE-reranker |
| **B2** | LLM-augmented Retrieval | LLM 输出 NL / 列表（不生成结构化查询）| NL query / reranked list / 多轮动作 | dominance | HyDE, IRCoT, Search-R1, Self-RAG, RankGPT |
| **B3** | Constrained Structured Query | LLM 生成受限语法的结构化查询 | boolean / filter+vec / SQL | dominance | DeepRetrieval, LangChain SQR, SQL-R1 |
| **B4** | Knowledge-Graph Retrieval | KG 路径 + 向量混合 | KG path + vec query | **互补（⊥）** | GraphRAG |
| **B5** | **Retrieval Program Synthesis** | **可执行 DSL（任意逻辑 + 多模态 + 向量原语）** | **可执行 DSL** | **Ours** | **Qwen3-4B/8B + DSL + RL** |

**两个核心轴**：
- **主轴（action space 表达力）**：B1 → B2 → B3 → B5，沿"输出结构化程度 / 检索通道丰富度 / 逻辑组合表达力"递增
- **正交轴（知识源结构化方式）**：B4 单列，与 B5 是"图边结构化 vs 字段结构化"的互补关系

副轴（仅作用于 B5 内部）：训练方式 = zero-shot / few-shot ICL / SFT / RL。

> **B2 的合并依据**：HyDE（query 改写）、IRCoT/Self-RAG（多轮交互）、RankGPT（LLM 重排）三者表面形式不同，但**共同特征是 LLM 输出 NL 或操作列表，不生成结构化查询**。这是与 B3/B5 的核心分界。

> **B4 单列的依据**：GraphRAG 把语料预先构成 KG 再走图遍历 + 向量混合，结构化轴是"图边/实体关系"而非"字段约束"。本文不主张 B5 全面胜出 B4，而是定位为 **"字段-密集" vs "关系-密集"两种结构化范式的互补**。

---

## 二、各范式保留的 Baseline

### B1 Pure Retrieval

| 方法 | 来源 | 必要性 |
|------|------|--------|
| BM25 | 计划 | P0 |
| ColQwen | 计划 | P0 |
| CLIP（仅电商）| 计划 | P0 |
| BM25 + Dense RRF（k=60）| 计划 | P0 |
| + BGE-reranker（cross-encoder rerank）| 加 | P0（production 标杆，HF 直接 inference）|
| **现代多模态 embedder：BGE-M3 / SigLIP（电商）** | 加 | P0（CLIP 已偏旧，reviewer 必问 newer multi-modal embedder）|

### B2 LLM-augmented Retrieval

按子形式分三档：

| 子形式 | 方法 | 训练 | 来源 | 必要性 |
|--------|------|------|------|--------|
| Query 改写 | **HyDE** | prompt-based | 加 | **P0**（prompt-based 几乎免费，reviewer 必看）|
| 多轮交互 | **IRCoT** | prompt-based | 加 | **P0**（多轮范式最便宜代表）|
| 多轮交互 | Self-RAG | SFT | 加 | P1（多轮 SFT 代表，公开 ckpt 可直接跑）|
| 多轮交互 | **Search-R1** | RL multi-turn | 加 | **P0**（多轮 RL + query rewriting 代表；与 B5 的核心区别是多轮 vs 单轮，E2 内部 ablation 只控制 action space 不控制轮数，无法替代真复现）|
| 多轮交互 | R1-Searcher | RL multi-turn | 加 | P2（多轮 RL 代表，与 B5 RL 在"轮数轴"上对比）|
| 多轮交互 | HiPRAG | RL + process reward | 加 | P2（层次化 process reward 训练 agentic RAG，与 B5 复合奖励设计思路类似，时间充裕可跑）|
| LLM 重排 | **RankGPT-4o on top of RRF** | prompt-based | 加 | **P0**（LLM rerank 最便宜代表，与 B1 BGE-reranker 形成对照）|
| LLM 重排 | RankVicuna | SFT | 加 | P1（开源 LLM rerank SFT 代表）|

> s3 等 RL Query Rewriting 由 §E2 Action Space Ablation 的 "text query" 行覆盖，不再单跑。Search-R1 因其**多轮交互**特性（检索→推理→再搜索循环）与 B5 的单轮 DSL 生成形成核心对比维度，E2 仅控制 action space 无法覆盖轮数差异，因此**恢复为独立 P0 baseline**。

### B3 Constrained Structured Query

| 方法 | 范式细分 | 训练 | 来源 | 必要性 |
|------|----------|------|------|--------|
| **DeepRetrieval（在我们数据上真复现）** | boolean query | RL | 加 | **P0**（04 文档列为致命风险；E2 内部 ablation 不能替代）|
| LangChain SelfQueryRetriever | filter + vector | prompt-based | 计划 | P0 |
| SQL-R1 | pure SQL | RL | 加 | P1（加分；E2 内部 "SQL only" 行已提供同 RL 控制对照，外部 SOTA 加分但非必需）|

### B4 Knowledge-Graph Retrieval

| 方法 | 训练 | 来源 | 必要性 |
|------|------|------|--------|
| **GraphRAG**（Microsoft）| prompt-based pipeline | 加 | **P1**（构图开销大但范式独特，加分项；时间充裕必跑）|

> **对比策略**：报 head-to-head（电商域 vs 邮件域），**预期结果**：B5 在结构化字段密集场景胜出，B4 在实体关系密集场景胜出。本文定位为 **互补范式**，不主张 dominance。

### B5 Retrieval Program Synthesis（Ours）

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
| SFT | **Text2VectorSQL** | 加 | **P0**（同 DSL SFT vs RL，RL 必要性核心论证）|
| **RL（Ours）** | Qwen3-4B + GRPO | 计划 | ✅ |
| **RL（Ours）** | Qwen3-4B + DAPO | 计划 | ✅ |
| **RL（Ours）** | Qwen3-8B + GRPO | 计划 | **缺** |
| **RL（Ours）** | **Qwen3-8B + DAPO（主结果）** | 计划 | **缺** |

---

## 三、论证链

```
跨范式（action space + 知识源价值）：
  B5 > B1   需要 LLM 生成
  B5 > B2   需要结构化输出（一次性生成 vs 反复 LLM 调用）  [HyDE/IRCoT/RankGPT vs Ours]
  B5 > B3   需要任意逻辑 + 多模态 + 向量原语                [DeepRetrieval/LangChain SQR/SQL-R1 vs Ours]
  B5 ⊥ B4   不同结构化轴（字段 vs 图边），互补而非 dominance  [GraphRAG vs Ours]

正交叠加（B5 + 其他范式）：
  B5 + B1 (BGE-reranker)   first-stage Ours + cross-encoder rerank，预期小幅增益
  B5 + B2 (RankGPT)        first-stage Ours + LLM rerank，验证正交补强或饱和

B5 内训练价值（RL 必要性）：
  RL > SFT       [Text2VectorSQL vs Ours, 同 DSL 不同训练]
  RL > ICL       [Qwen3-4B/8B few-shot DSL vs Ours, 同 base 同 DSL]
  RL > zero-shot [Qwen3 base / GPT-4o vs Ours]
```

每条比较只动一个轴，控制变量明确。"⊥" 表示正交互补不主张全面胜出。

---

## 四、计划"已有"但实际未跑的项

| 项 | 状态 | 范式 |
|----|------|------|
| GPT-4o zero-shot DSL | 计划标"已有"，实际未跑 | B5 zero-shot |
| 邮件场景 commercial LLM | **完全没跑**（只电商有）| B5 zero/few-shot |
| Qwen3-8B 系列（base + GRPO + DAPO）| **完全没跑** | B5 zero-shot + RL |
| Qwen3-4B/8B + DSL few-shot ICL | 计划遗漏 | B5 ICL |
| **B2 多轮交互 + LLM 重排** | **完全没跑**（原计划排除，本次纳入）| B2 |
| **B4 KG-augmented (GraphRAG)** | **完全没跑**（原计划排除，本次纳入）| B4 |

---

## 五、最小封口集

**🔴 P0（不跑要被拒）**

B1 Pure Retrieval：
1. BM25 + Dense RRF（B1）
2. + BGE-reranker（B1，production cross-encoder）
3. BGE-M3 / SigLIP（B1，现代多模态 embedder）

B2 LLM-augmented：
4. HyDE（B2 query 改写）
5. **IRCoT**（B2 多轮 prompt-based）
6. **Search-R1 真复现**（B2 多轮 RL，单轮 vs 多轮核心对比）
7. **RankGPT-4o on top of RRF**（B2 LLM rerank）

B3 Structured Query：
8. **DeepRetrieval 真复现**（B3 boolean RL）
9. LangChain SelfQueryRetriever（B3 filter+vec）

B5 Ours：
10. **Text2VectorSQL**（B5 SFT，同 DSL 范式）
11. **Qwen3-4B/8B + DSL few-shot ICL**（B5 ICL，计划遗漏）
12. GPT-4o zero/few-shot DSL（计划遗漏）
13. 邮件 commercial LLM（计划遗漏）
14. **Qwen3-8B + GRPO/DAPO**（B5 RL 主结果，最贵）

**🟡 P1（加分）**

- SQL-R1（B3 NL2SQL RL）
- Self-RAG（B2 多轮 SFT）
- **GraphRAG**（B4 KG 范式代表，互补论证）
- **B5 + RankGPT cascade**（正交补强验证）
- Single-mode 单跑：BM25 / ColQwen / CLIP（简单稳，可作下界）

**🟢 P2（时间充裕）**

- R1-Searcher（B2 多轮 RL）
- RankVicuna（B2 LLM rerank SFT）
- HiPRAG（B2 多轮 RL + 层次化 process reward）

---

## 六、三档实施方案

| 档次 | 时间 | 必跑（增量）|
|------|------|------------|
| 保底（Findings）| 3 周 | B1 全套 + HyDE + **IRCoT + Search-R1 + RankGPT-4o** + LangChain SQR + B5 ICL + B5 计划遗漏补齐 |
| 冲 Main | 5 周 | 保底 + DeepRetrieval + Text2VectorSQL + **Self-RAG + GraphRAG** |
| 完整版 | 7+ 周 | 上述 + SQL-R1 + R1-Searcher + RankVicuna + **B5 + RankGPT cascade** |

---

## 七、复现资源

B1 Pure Retrieval：

| 方法 | 资源 |
|------|------|
| BM25 | `rank_bm25` 或 Pyserini |
| RRF | 自实现 10 行，k=60 |
| BGE-reranker | `BAAI/bge-reranker-large`（HF，cross-encoder）|
| BGE-M3 / SigLIP | `BAAI/bge-m3` / `google/siglip-large-patch16-384`（HF）|

B2 LLM-augmented：

| 方法 | 资源 |
|------|------|
| HyDE | LangChain `HypotheticalDocumentEmbedder` + GPT-4o，temp=0.3，3 candidates |
| **IRCoT** | https://github.com/StonyBrookNLP/ircot |
| **Search-R1** | https://github.com/PeterGriffinJin/Search-R1 |
| Self-RAG | https://github.com/AkariAsai/self-rag （公开 7B/13B SFT ckpt）|
| R1-Searcher | https://github.com/RUCAIBox/R1-Searcher （公开 RL ckpt）|
| HiPRAG | https://github.com/qualidea1217/HiPRAG |
| **RankGPT** | https://github.com/sunnweiwei/RankGPT （prompt-based，GPT-4o backbone）|
| RankVicuna | https://github.com/castorini/rank_llm （公开 7B SFT ckpt）|

B3 Structured Query：

| 方法 | 资源 |
|------|------|
| LangChain SQR | LangChain SelfQueryRetriever，适配 schema |
| DeepRetrieval | https://github.com/pat-jj/DeepRetrieval |
| SQL-R1 | https://github.com/MJinXiang/SQL-R1 |

B4 Knowledge-Graph：

| 方法 | 资源 |
|------|------|
| **GraphRAG** | https://github.com/microsoft/graphrag （构图开销 ~corpus × 0.5-1 LLM 调用）|

B5 Ours：

| 方法 | 资源 |
|------|------|
| Text2VectorSQL | 论文公开代码，按其 SFT 配方在我们数据上重训 |
| Qwen3 few-shot ICL | 自实现：3-5 个 DSL 示例 + 当前 query |

---

## 八、对计划文件 §一 范式表的补充建议

`最终工作计划.md` §一 范式对比表建议简化为本文 5 类（B1-B5），核心新增：

```
| LLM-augmented Retrieval | NL / list / 多轮 | 不生成结构化查询 | HyDE, IRCoT, RankGPT |
| Constrained Structured Query | boolean / filter+vec / SQL | 受限语法 | DeepRetrieval, LangChain SQR, SQL-R1 |
| KG-augmented | KG path + vec | 图边结构化 | GraphRAG |
| Retrieval Program Synthesis | DSL | 任意逻辑 + 多模态 + 向量 | **Ours** |
```

否则 B2/B3/B4 baseline 在 §2 Related Work / §4 Method 写作时无对手范式名可挂。

---

## 九、与 §八 实验设计的对应

- **E1 Main Results**：用本文 §二 的 5 类范式（B1-B5）作主表分组。每类 1-3 个代表方法，主表 ~12-15 行。
- **E2 Action Space Ablation**：固定 B5 的 base + RL，改 action space——内部覆盖 B2/B3 三个范式作控制对照。**不能替代 DeepRetrieval / IRCoT / Search-R1 真复现**（自家训练 ≠ 原论文 SOTA；且 E2 仅控制 action space，不控制**单轮 vs 多轮**，无法覆盖 Search-R1 的多轮检索-推理循环）。新增 **multi-round ablation**（B5 single-pass vs B5 强行包成多轮）来与 B2 多轮对比。
- **E3 Complexity Breakdown**：所有 baseline 按 Level 1/2/3 分层；预期 Level 3 时 B5 相对 B3 优势最大（OR/NOT/嵌套）。**公平性约定**：Level 1（单一字段/单一语义）是各范式的公平比较层，性能应接近；Level 2/3（多字段组合 / OR/NOT/嵌套）是 B5 的优势展示层，B1-B3 在此层崩盘是预期结果而非 unfair setup —— §写作时需明示该约定，否则 reviewer 会质疑"为什么不在 Level 1 也报数"。
- **新增 E_cascade Ablation**：报 B5 + B1 (BGE-reranker)、B5 + B2 (RankGPT) 的正交补强或饱和；B5 vs B4 (GraphRAG) 的 head-to-head（按"字段-密集"vs"关系-密集"分组报数）；B2 多轮的成本/收益曲线（轮数 vs Recall@K）。
- **E4-E11**：与本文档独立（reward / modality / orchestration / scaling / error / latency / cost）。

---

## 十、风险与应对

| 风险 | 应对 |
|------|------|
| Text2VectorSQL / DeepRetrieval 复现工作量 | 用公开 ckpt，2-3 天/篇 |
| **DeepRetrieval 多模态适配**（原方法纯文本布尔，需扩展到图像 + 结构化字段）| **提前 2 周开跑留 buffer**；兜底：跑不动则改用 §E2 内部 "boolean only" ablation 替代，并在 §Limitations 显式说明 "DeepRetrieval 原设计为 text-only，多模态扩展非平凡" |
| 跑 SOTA 数据集 ≠ 跑我们数据集 | 用我们 train set 重训他们的方法（fair comparison）|
| 8B 训练成本高，时间不够 | 兜底：报 4B 全套 + 8B zero-shot/ICL，把 8B+RL 移 P1 |
| 跨范式被指模型规模不公平 | B1-B5 各范式尽量保持模型规模在 4B-8B 区间；规模消融见 §E8 |
| **B2 多轮、B4 KG 自带额外 compute 不公平** | 报 latency / token cost 双指标（§E9-E10）；不要求 B5 在 compute 上 dominance，但要求 **per-token efficiency** 上 dominance |
| **GraphRAG 构图开销**（10K corpus ~ 数小时 LLM 调用）| 限制构图用 GPT-4o-mini 降本；如果时间不够，仅在电商域跑 GraphRAG，邮件域留 §Limitations 说明 |
| **B2 IRCoT/Self-RAG 与 B5 不同 base model** | 用同尺寸 backbone（Qwen3-4B 或 Llama-3-8B）替换原论文 backbone；如不可行，在表格脚注标注 backbone 差异 |

---

## 附录 A：已删除的 Baseline 候选与原因

下表列出曾纳入、本次按"**不在 `LocalSearch-R1_最终工作计划.md` §六 里 + 对实验 solidness 非必需 + 与已纳入方法重复**"标准砍掉的候选 baseline。

| 删除项 | 原所属类 | 删除原因 |
|--------|---------|---------|
| BGE / E5 | 稠密单向量 | ColQwen / BGE-M3 已代表稠密单向量 baseline，重复 |
| ColBERT v2 | 稠密多向量 | late-interaction 替代品，可作时间充裕的加分项 |
| Query2doc | query 改写 | HyDE 已代表 LLM-aug query rewriting，二选一 |
| GPT-4o / Gemini / Claude zero-shot SQL | NL2SQL prompt | B3 范式由 SQL-R1 + §E2 内部 "SQL only" 行覆盖 |
| GPT-4o few-shot SQL | NL2SQL ICL | 同上 |
| DIN-SQL / DAIL-SQL / MAC-SQL | NL2SQL 学术 baseline | 同上 |
| C3 / CHESS / RESDSQL | NL2SQL SOTA | 同上 |
| CodeS / SQLNet | NL2SQL SFT | 同上 |
| LlamaIndex Auto-Retriever | Self-querying | 与 LangChain SelfQueryRetriever 重复 |
| KG-RAG | KG-augmented | 与 GraphRAG 二选一，B4 已选 GraphRAG |
| s3 | RL searcher/generator 解耦 | Search-R1 已恢复为 P0；s3 与其同范式，二选一 |
| R3-RAG | RL 多轮 | B2 已有 R1-Searcher 代表 RL 多轮，重复 |
| MCTS-RAG | MCTS+RAG | 非 RL 范式（MCTS 规划），与方向不一致 |
| ReAct-Retrieval | prompt-based 多轮 | 与 IRCoT 重复（同为 prompt-based 多轮），二选一 |
| MonoT5 | cross-encoder rerank | BGE-reranker 已代表 cross-encoder 重排，重复 |
| SUQL | 同 DSL prompt-based | Text2VectorSQL 已代表"同 DSL 不同训练"对照，重复 |
| GPT-5.4 thinking / o1 | reasoning 模型 | 不在最终工作计划 §六，reasoning 模型上限非必需 |
| Claude zero / few-shot DSL | commercial LLM | 计划 §六 Group 3 仅列 GPT-4o + Gemini，不含 Claude |

> 本附录仅记录候选 baseline 的删除决策，便于审稿/Rebuttal 阶段快速回溯"为什么没跑 X"。

---

**文档维护**：本文与 `LocalSearch-R1_最终工作计划.md` 配套使用。baseline 跑完时同步更新 §二 状态列；§一 范式表更新时同步本文 §一。
