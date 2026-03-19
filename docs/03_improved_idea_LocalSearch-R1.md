# LocalSearch-R1: 改进版论文方案

> 从 DSL-R1 到 LocalSearch-R1：从"查询生成"到"检索编排"的范式升级

## 改版核心策略

**原版问题本质**：Novelty 定位为 "统一 DSL"，但 reviewers 认为 structured+unstructured 融合是工业常见做法。

**改版核心转变**：
- 原版 framing: "我们设计了一个统一 DSL" → 方法层面的贡献
- 新版 framing: "我们将 LLM 从查询生成者升级为检索编排者" → 范式层面的贡献
- 原版标题: DSL-R1 → 新版标题: **LocalSearch-R1**
- 从 short paper → **long paper**（支撑更完整的论证和实验）

---

## Part 1: Introduction 中文草稿

### 第一段：问题背景——异构数据的混合检索需求

现实世界中的文档检索任务往往同时涉及结构化约束与语义匹配，且两者之间存在复杂的逻辑组合关系。以企业邮件检索为例，用户可能提出如下查询："找到上周市场部发来的关于Q3预算的邮件，但排除已归档的"。这一查询同时包含时间范围过滤（上周）、发件人属性匹配（市场部）、内容语义理解（Q3预算）以及否定条件（排除已归档），四个约束之间构成交集关系。更复杂的情况下，用户可能需要并集（"找到张三或李四发来的报告"）、嵌套逻辑（"找到标记为星标或未读的、且来自财务部的、包含合同关键词的邮件"）等组合。类似的混合检索需求广泛存在于科学文献管理、个人文档搜索、企业知识库等场景中。

这一问题在本地部署场景下尤为突出。出于隐私保护、数据合规、成本控制和离线可用性等考虑，越来越多的应用需要在用户本地设备上完成检索，而非依赖云端服务。本地环境中的文档信息天然分散在异构存储中——结构化元数据存储在关系型数据库或文件系统索引中，非结构化内容依赖向量索引进行语义检索——不存在一个统一的搜索引擎能够在这两类信息上执行任意逻辑组合。这使得本地混合检索不仅需要理解查询意图，还需要自主编排多条异构检索路径并规划它们之间的逻辑关系。

### 第二段：现有方法的局限——表达能力与编排能力的双重缺失

现有的检索方法在面对上述需求时存在两层不足：**表达能力不足**和**编排能力不足**。

在**表达能力**层面，传统的稀疏检索（如BM25）和稠密检索（如基于CLIP或DPR的方案）各自仅能处理一种数据类型，无法在单次查询中兼顾结构化过滤与语义匹配。工业界广泛使用的混合检索管线（如Reciprocal Rank Fusion）虽然融合了多路检索的结果，但其本质操作仅是"合并并重排"——它只能表达多路结果的并集加权，无法表达交集（同时满足结构化条件和语义条件）、否定（排除满足某条件的结果）或嵌套逻辑。Self-querying RAG框架（如LangChain的实现）支持将查询分解为metadata filter加语义检索，但其过滤能力局限于简单的键值匹配，不支持OR、NOT等灵活的逻辑组合。换言之，现有方法的检索编排逻辑是**固定且受限的**，无法根据查询意图动态选择和组合不同的检索操作。

在**编排能力**层面，近期强化学习驱动的检索查询生成取得了显著进展。Search-R1 (Jin et al., 2025) 训练LLM在推理过程中自主调用搜索引擎，DeepRetrieval (Jiang et al., 2025) 通过RL优化查询改写使3B模型在文献检索上超越GPT-4o。这些工作有力地证明了RL在提升检索质量方面的潜力。然而，它们共享一个关键的设计前提：**存在一个功能完备的检索后端来执行搜索**。在这些框架中，LLM扮演的是"查询生成者"（query generator）——它优化的是"问什么"，而"怎么找"完全由外部搜索引擎负责。因此，它们的动作空间是文本查询字符串，不具备在结构化与非结构化检索路径之间进行逻辑编排的能力。

当面对需要在多条异构检索路径之间执行复杂逻辑组合的混合检索任务时——尤其是在缺乏统一搜索引擎的本地环境中——LLM的角色需要从"查询生成者"升级为**"检索编排者"（retrieval orchestrator）**。它不仅需要理解"问什么"，更需要规划"怎么找"：哪些条件应通过SQL执行精确过滤，哪些语义意图应通过向量检索进行模糊匹配，这些路径之间应取交集、并集还是差集，以及是否需要嵌套组合。这在强化学习的形式化层面构成了一个本质不同的问题——模型的动作空间从一个文本查询字符串扩展为一个**支持任意逻辑组合的可执行程序**。

### 第三段：我们的方法

基于上述分析，我们提出 **LocalSearch-R1**，一个通过强化学习训练轻量级模型编排混合检索流程的智能体框架。我们的核心洞察是：如果将DSL设计为SQL语法与向量检索原语的统一体，那么SQL本身的完备逻辑表达能力（交、并、补及任意嵌套）就自然成为检索编排的逻辑骨架，而向量检索作为SQL中的子查询参与逻辑组合。这将"编排异构检索路径"的复杂决策转化为一个可学习的程序生成问题，同时继承了SQL在逻辑表达上的完备性。

具体而言，框架包含三个组件：

**混合DSL检索智能体。** 我们设计了一种将SQL操作符与向量检索原语嵌入同一可执行程序的DSL。给定自然语言查询，智能体生成一个DSL程序，其中向量检索以占位符形式嵌入SQL语句中（如 `message_id IN <vector_0>`），在执行时先由向量索引返回候选集，再注入SQL引擎中与结构化条件进行逻辑组合。由于SQL提供了完整的集合运算语义，该DSL能够表达任意复杂的检索编排逻辑：通过AND实现结构化条件与语义条件的交集过滤，通过OR实现多路检索的并集召回，通过NOT实现结果排除，通过子查询和嵌套实现多层次的检索逻辑组合。这一表达能力远超基于固定融合策略的混合检索管线，也超越了现有RL检索方法的文本查询动作空间。

**面向本地文档场景的检索基准。** 我们基于企业邮件和科学论文两个领域构建了大规模检索基准。每个检索实例都同时涉及结构化元数据约束和非结构化内容匹配，且涵盖不同复杂度的逻辑组合（从简单的单条件过滤到多条件嵌套），确保对混合检索编排能力的全面评估。企业邮件场景是本地混合检索的典型代表——邮件天然包含丰富的结构化字段（发件人、时间、标签、已读状态）和非结构化内容（正文、附件），且出于隐私考虑通常存储在本地客户端中。

**复合奖励驱动的强化学习。** 生成可执行的DSL程序比生成文本查询面临更严格的层次化约束：程序必须先通过语法检查（格式正确），再被执行引擎成功执行（SQL和向量检索均无错误），最终返回准确的检索结果。为此，我们设计了由格式奖励（S_f）、执行奖励（S_e）、结果奖励（S_r）和长度奖励（S_l）四个分量组成的复合奖励函数，显式建模了这一层次化质量要求：语法正确性 → 执行可靠性 → 检索精度 → 响应效率。这一设计与Search-R1和DeepRetrieval采用的单一结果匹配奖励形成对比——对于可执行程序生成任务，仅靠最终结果的对错信号不足以引导模型学会生成合法且高效的程序。我们采用GRPO和DAPO两种优化算法，其中DAPO在稳定性和效率之间取得了最佳平衡。

### 第四段：核心发现与贡献

在两个检索基准上的实验表明，LocalSearch-R1让8B参数的本地模型在Hit@1指标上超越GPT-4o达9个百分点，同时将推理延迟降低45%。进一步的分析揭示了三个关键发现：
1. DSL的hybrid action space（SQL + 向量检索）显著优于单一的文本查询或纯SQL的action space，验证了检索编排相对于查询生成的优势
2. 在逻辑复杂度较高的查询上（涉及多条件嵌套或否定逻辑），DSL相对于固定融合策略的优势更为显著，证实了SQL逻辑表达能力的价值
3. RL训练后，模型学会了根据查询意图动态调整结构化过滤与语义匹配的编排策略，在不同类型的查询上展现出差异化的检索路径选择行为

**贡献：**

**(1) 提出检索编排的新范式。** 我们将混合检索问题从"查询生成"重新定义为"检索流程编排"，设计了融合SQL逻辑表达能力与向量检索语义匹配能力的DSL作为编排语言。该DSL支持对结构化与非结构化检索路径进行任意逻辑组合（交、并、补及嵌套），其表达能力显著超越现有的固定融合管线和文本查询动作空间，填补了RL在混合检索编排方面的空白。

**(2) 设计面向可执行程序生成的层次化奖励训练框架。** 我们针对DSL程序"可解析 → 可执行 → 结果正确 → 响应高效"的层次化约束设计了四维复合奖励函数，并通过GRPO和DAPO优化。相比单一的结果匹配奖励，该设计为可执行程序的生成提供了更丰富的学习信号，可推广至其他需要生成可执行代码的RL任务。

**(3) 构建本地混合检索基准并验证小模型编排能力。** 我们构建了涵盖不同逻辑复杂度的大规模混合检索基准，证明8B本地模型经RL训练后可超越GPT-4o，验证了轻量级模型进行复杂检索编排的可行性。我们将开源代码、数据和模型以促进后续研究。

---

## Part 2: Related Work 重写方案

### 2.1 Hybrid Retrieval（新增，重点讨论表达能力）

覆盖内容：
- Structured + unstructured 检索的工业实践
- BM25 + dense retrieval 融合：RRF (Cormack et al., 2009)、linear combination
  - → 局限：只能做并集加权重排，不支持交集/否定/嵌套
- Self-querying RAG：LangChain、LlamaIndex 的 metadata filtering
  - → 局限：键值匹配，不支持 OR/NOT/嵌套
- 引入表达能力对比表
- 指出核心 gap：现有方法的编排逻辑是**固定且表达受限的**

### 2.2 RL for Retrieval Query Generation（最关键新增）

覆盖内容：
- **Search-R1**：RL 训练 LLM 与搜索引擎交互，multi-turn reasoning + search
- **DeepRetrieval**：RL 优化查询改写，3B 模型 beat GPT-4o，COLM'25
- **ReSearch**：类似 Search-R1 的 reasoning + search 框架
- **s3**：只训练 searcher 不训练 generator

明确指出共同假设：都依赖现成检索后端，LLM 只做 query generation

与我们的两层区别：
- (a) action space：text query vs executable program with full logical expressiveness
- (b) 角色：query generator vs retrieval orchestrator

### 2.3 DSLs in AI Systems（精简保留）

- 程序合成、数据分析中的 DSL 应用
- 与 RL 结合的 DSL 优化
- Text-to-SQL：强调我们的 DSL 不是纯 SQL，而是扩展了 vector search 原语

### 2.4 On-device and Privacy-preserving Retrieval（新增）

- Google AI Edge RAG SDK、EmbeddingGemma
- Edge LLM 部署：Gemma 3n、MobileLLM 等
- Federated RAG
- 指出这些工作是工程实现，检索能力局限于简单的向量相似度匹配，缺乏对复合查询编排的学术优化

---

## Part 3: Method Section 修改方案

### 3.1 Overview（添加系统架构图）

```
用户查询 → LLM（本地8B模型）→ DSL程序 → SQL引擎 + 向量索引 → 结果融合 → 返回
```

强调：完全本地的 pipeline，没有任何数据离开设备

### 3.2 DSL Agent Design（重点改写，拆成4个小节）

#### 3.2.1 DSL 语法设计

保留原有的 SQL + vector query list 的 JSON 结构描述。新增 DSL 的逻辑表达能力分析：

**DSL 编排模式表：**

| 编排模式 | DSL 表达 | 语义 |
|---------|---------|------|
| 结构化∩语义 | `WHERE is_starred=1 AND id IN <vector_0>` | 必须同时满足结构化条件和语义匹配 |
| 结构化∪语义 | `WHERE is_starred=1 OR id IN <vector_0>` | 满足任一条件即可召回 |
| 否定排除 | `WHERE id IN <vector_0> AND NOT is_archived=1` | 语义匹配但排除特定结构化条件 |
| 多路语义交集 | `WHERE id IN <vector_0> AND id IN <vector_1>` | 必须同时匹配两个语义条件 |
| 多路语义并集 | `WHERE id IN <vector_0> OR id IN <vector_1>` | 匹配任一语义条件即可 |
| 嵌套逻辑 | `WHERE (is_starred=1 OR is_read=0) AND date>='...' AND id IN <vector_0>` | 多层嵌套组合 |
| 纯结构化 | `WHERE sender='alice' AND date>='...'` | 仅结构化过滤，无需语义 |
| 纯语义 | `WHERE id IN <vector_0>` | 仅语义匹配，无结构化条件 |

**与现有方法的表达能力对比表：**

| 编排能力 | RRF | Self-querying RAG | Text Query (Search-R1) | DSL (Ours) |
|---------|-----|------------------|----------------------|------------|
| 结构化过滤 | ✗ | ✓ (键值匹配) | ✗ | ✓ (完整SQL) |
| 语义检索 | ✓ | ✓ | ✓ | ✓ |
| 交集 (AND) | ✗ | ✓ (隐式) | ✗ | ✓ |
| 并集 (OR) | 仅结果合并 | ✗ | ✗ | ✓ |
| 否定 (NOT) | ✗ | ✗ | ✗ | ✓ |
| 嵌套逻辑 | ✗ | ✗ | ✗ | ✓ |
| 多路向量检索组合 | ✗ | ✗ | ✗ | ✓ |
| 动态路径选择 | ✗ | ✗ | ✗ | ✓ (RL学习) |
| 可学习的编排策略 | ✗ | ✗ | ✗ | ✓ (RL优化) |

#### 3.2.2 执行机制（必须添加，回应 Reviewer 13wi）

**伪代码：**
```python
function execute_dsl(dsl_program):
    # Step 1: 执行所有向量检索
    for each vector_query in dsl_program.vector_query_list:
        candidate_ids = vector_index.search(vector_query, top_k=K)
        replace <vector_i> placeholder in SQL with candidate_ids

    # Step 2: 执行SQL（此时SQL中已注入向量检索结果）
    results = sql_engine.execute(dsl_program.sql)

    # 关键insight：逻辑组合（AND/OR/NOT）完全由SQL语义决定
    # 不需要额外的fusion策略——SQL本身就是fusion逻辑
    return results
```

#### 3.2.3 Action Space 对比（新增，支撑核心论点）

```
Search-R1 输出:  "quantum computing applications in 2024"
                 → 一个文本字符串，交给外部搜索引擎

DeepRetrieval 输出:  "quantum computing" OR "quantum applications" ...
                     → 一个改写/增强后的文本查询

LocalSearch-R1 输出:
{
  "sql": "SELECT DISTINCT message_id FROM email
          WHERE is_draft = 1
          AND draft_modified_date >= date('now', '-7 day')
          AND is_starred = 1
          AND message_id IN <vector_0>",
  "vector_query_list": [
    {"email_content": "budget"}
  ]
}
→ 一个可执行程序，包含结构化过滤+语义检索+逻辑组合
```

#### 3.2.4 Prompt Template（新增，回应所有 reviewer 的可复现性要求）

主体放 appendix，method 里给精简版，包括：
- 任务描述
- Schema 定义
- 输出格式要求
- DSL 语法说明
- 示例

### 3.3 Data Preparation（重写）

- 用标准 IR notation 重写（q 表示 query，y 表示 DSL program）
- 添加数据集构造流程图
- **新增：按逻辑复杂度分层统计**

| 逻辑复杂度 | Train | Eval | 说明 |
|-----------|-------|------|------|
| 简单 (1个条件) | ... | ... | 纯结构化或纯语义 |
| 中等 (2-3个AND) | ... | ... | 结构化∩语义 |
| 复杂 (含OR/NOT/嵌套) | ... | ... | 多路逻辑组合 |

### 3.4 Reinforcement Learning（精简 + 细节增补）

GRPO 和 DAPO 的公式精简，详细推导移到 appendix。

**Reward function 详细计算（必须添加）：**
```
S_f: binary, 1 if output contains valid <query>...</query> tags, else 0
S_e: binary, 1 if DSL can be parsed and executed without error, else 0
S_r: continuous, Hit@1 score against gold reference
S_l: continuous, -α * max(0, len(output) - threshold) / threshold
R = S_f + S_e + S_r + S_l （直接相加，equal weight）
```

Equal weight 合理性：层次化结构天然形成隐式优先级——S_f=0 ⇒ S_e=0 ⇒ S_r=0

**与 Search-R1/DeepRetrieval 的 reward 对比：**
- 它们：单一的 exact match 或 recall 作为 reward
- 我们：四维复合 reward，因为可执行程序需要额外的格式和执行约束

---

## Part 4: Experiments Section 修改方案

### 必须新增的实验（高优先级）

| 实验 | 目的 | 工作量 |
|------|------|--------|
| RRF baseline (BM25 + ColQwen) | 堵住"没有 hybrid baseline"的口 | 1-2天 |
| GPT-4o few-shot DSL generation | 证明 RL 必要性 | 1-2天 |
| Self-querying RAG baseline (LangChain) | 对比工业实践 | 2-3天 |
| **Action space ablation** | **支撑核心论点的最关键实验** | 2-3天 |
| Model size scaling (1.5B → 4B → 8B → 14B) | 小模型能力边界 | 3-5天 |

### 建议新增的实验（中优先级）

| 实验 | 目的 | 工作量 |
|------|------|--------|
| 按逻辑复杂度分组的 Performance breakdown | 证明 DSL 在复杂 query 上优势更大 | 1-2天 |
| Orchestration 行为分析 | 展示 RL 训练后的差异化编排策略 | 1-2天 |
| Latency breakdown | DSL 生成 vs SQL 执行 vs vector search | 半天 |
| Error analysis | syntax error / execution error / wrong result 分布 | 1天 |
| DSL 逻辑操作符使用统计 | 统计 AND/OR/NOT 的分布 | 半天 |

### Table 1 (Main Results) 重新组织

```
Category                | Model                    | Hit@1 | Hit@3 | MRR  | Lat.
========================|==========================|=======|=======|======|=====
Single-mode Retrieval   | BM25                     | 42.1  | 64.7  | 53.2 | 18
                        | ColQwen                  | 55.3  | 78.9  | 65.4 | 35
------------------------|--------------------------|-------|-------|------|-----
Hybrid Baselines (新增) | BM25 + ColQwen RRF       | ...   | ...   | ...  | ...
                        | Self-querying RAG         | ...   | ...   | ...  | ...
                        | GPT-4o few-shot DSL       | ...   | ...   | ...  | ...
------------------------|--------------------------|-------|-------|------|-----
LLM (w/o RL)           | Qwen3-4B                 | 58.3  | 80.1  | 69.2 | 95
                        | Qwen3-8B                 | 62.1  | 83.6  | 72.5 | 140
                        | GPT-4o                   | 68.9  | 88.7  | 78.8 | 30
------------------------|--------------------------|-------|-------|------|-----
LocalSearch-R1 (Ours)   | Qwen3-4B + GRPO          | 69.5  | 88.9  | 78.6 | 55
                        | Qwen3-4B + DAPO          | 71.7  | 90.6  | 80.9 | 52
                        | Qwen3-8B + GRPO          | 75.8  | 92.9  | 84.6 | 82
                        | Qwen3-8B + DAPO          | 77.9  | 94.2  | 86.1 | 78
```

### 新增 Table: Action Space Ablation（最关键）

```
Action Space                        | Hit@1 | Hit@3 | MRR  | 说明
text query (→ vector retrieval)     | ...   | ...   | ...  | 类Search-R1
SQL only (→ structured DB)          | ...   | ...   | ...  | 类Text-to-SQL
vector query only (→ vector index)  | ...   | ...   | ...  | 类DeepRetrieval
DSL: SQL + vector (Ours)            | 77.9  | 94.2  | 86.1 | 完整编排能力
```
所有行使用同一 base model (Qwen3-8B) + 同一 RL 算法 (DAPO)，只改变 action space。
**直接证明：hybrid DSL 的编排能力 > 任何单一路径。**

### 新增 Table: Performance by Query Logical Complexity

```
Query Complexity    | RRF   | Self-query RAG | GPT-4o | LocalSearch-R1 | Δ vs RRF
简单 (1条件)        | ...   | ...            | ...    | ...            | ...
中等 (2-3 AND)      | ...   | ...            | ...    | ...            | ...
复杂 (OR/NOT/嵌套)  | ...   | ...            | ...    | ...            | ...
```
**预期结果**：简单 query 差距不大，复杂 query 上 DSL 优势显著拉开。

### 新增 Table: Model Size Scaling

```
Model Size | Hit@1 (pretrained) | Hit@1 (+ DAPO) | Δ      | 超过GPT-4o?
1.5B       | ...                | ...            | ...    | ✗
4B         | 58.3               | 71.7           | +13.4  | ✓
8B         | 62.1               | 77.9           | +15.8  | ✓
14B        | ...                | ...            | ...    | ✓
GPT-4o     | 68.9               | N/A            | N/A    | —
```

### 新增 Figure: Orchestration Behavior Analysis

分组柱状图或热力图：
- X轴：query 类型（structure-dominated / content-dominated / balanced / complex-logic）
- Y轴：模型生成的 DSL 中各操作类型的使用比例
  - 纯 SQL 条件的比例
  - 向量检索的比例
  - AND/OR/NOT 使用频率
- 对比：训练前（pretrained）vs 训练后（+ DAPO）

**预期发现**：
- 训练后模型学会了对 structure-dominated query 减少不必要的向量检索
- 对 content-dominated query 增加向量检索的权重
- 对 complex-logic query 学会使用 OR/NOT 等高级操作

### 新增 Figure: DSL 逻辑操作符使用统计

统计训练后模型在 evaluation set 上生成的 DSL 中：
- AND/OR/NOT 的使用频率
- 嵌套条件的使用频率
- 多路向量检索（>1个 vector query）的使用频率
- 以及这些操作在正确检索 vs 错误检索中的分布差异

---

## Part 5: 其他 Section 修改方案

### Abstract 重写

核心 message：
> "现有RL检索方法将LLM视为查询生成者，依赖外部搜索引擎执行检索。我们将LLM升级为检索编排者，通过生成融合SQL逻辑与向量检索的可执行DSL程序，实现对异构检索路径的动态逻辑组合。"

保留 8B beat GPT-4o 和 +12.3% 的数字，强调 DSL 支持任意逻辑组合（AND/OR/NOT/嵌套）。

### Conclusion

围绕 "from query generation to retrieval orchestration" 总结。

Future work：
- Multi-turn retrieval（对话式渐进检索）
- Multi-agent 协作（query 理解 agent + 编排 agent 分离）
- Schema-agnostic DSL（自动发现数据 schema）
- 真正的 on-device deployment 验证
- 更多模态（图片、音频内容的检索）

### Limitations

诚实讨论：
1. DSL schema 需要预定义，面对全新数据格式需要人工设计 DSL 模板
2. 当前实验在 GPU 服务器上运行，未做真正的 edge device 部署验证
3. 单轮检索，不支持多轮对话式 refinement
4. 只测了两个 domain（email + arxiv），更多 domain 的泛化性待验证
5. DSL 的表达能力理论上支持任意逻辑组合，但模型是否能稳定生成所有类型的复杂组合仍需进一步验证

### Appendix 必须添加

- 完整的 prompt template
- Reward function 详细公式和代码片段
- 更多 DSL case samples（至少 6-8 个），覆盖：
  - 简单 AND 查询
  - 含 OR 的查询
  - 含 NOT 的查询
  - 嵌套逻辑查询
  - 多路向量检索查询
  - 纯结构化查询
  - 纯语义查询
- GRPO/DAPO 的详细公式推导
- 训练曲线图（reward vs training step）
- 数据集完整 schema 定义

---

## Part 6: 核心论证链条

```
问题：混合检索需要在结构化和非结构化路径间进行复杂逻辑组合
                    ↓
现有方法不足：RRF只能做并集重排，Search-R1/DeepRetrieval只生成text query
                    ↓
我们的解法：DSL = SQL逻辑骨架 + 向量检索原语 → 任意逻辑组合
                    ↓
训练方法：RL + 四维复合reward（因为可执行程序需要层次化约束）
                    ↓
实验验证（三层）：
  Layer 1: 整体结果 — 8B模型 beat GPT-4o（Table 1）
  Layer 2: 为什么好 — hybrid action space > 单一action space（Action Space Ablation）
  Layer 3: 好在哪里 — 复杂逻辑query上优势更大（Complexity Breakdown）
                    ↓
深度分析：模型确实学会了动态编排（Orchestration Behavior Analysis）
```

每一层实验都在回答上一层自然引出的问题。

---

## Part 7: 实施时间表（建议）

| 阶段 | 内容 | 时间 |
|------|------|------|
| Week 1 | RRF baseline + Self-querying RAG baseline 实验 | 5天 |
| Week 2 | Action space ablation + GPT-4o few-shot 实验 | 5天 |
| Week 2-3 | Model size scaling 实验 | 5天 |
| Week 3 | Complexity breakdown + Orchestration analysis | 3天 |
| Week 3-4 | Error analysis + DSL 操作符统计 | 2天 |
| Week 4-5 | Introduction + Related Work 重写 | 5天 |
| Week 5-6 | Method Section 改写 + Appendix 补充 | 5天 |
| Week 6 | Experiments Section 重写 + 全文打磨 | 5天 |
| **Total** | | **~5-6 weeks** |

---

## Part 8: 文献调研与 EMNLP 2026 可行性评估

> 调研时间：2026-03-19

### 8.1 EMNLP 2026 关键时间线

| 节点 | 日期 |
|------|------|
| ARR 提交截止（May cycle，最后一轮） | **2026年5月25日** |
| 审稿完成 | 2026年7月6日 |
| Author response | 2026年7月7-13日 |
| EMNLP commitment deadline | **2026年8月2日** |
| 录取通知 | **2026年8月20日** |
| 会议 | 2026年10月24-29日，布达佩斯 |

**注意事项：**
- 所有作者必须注册为潜在 reviewer/AC，未完成注册自动 desk reject
- 从 2026年3月起只接受 LaTeX，Word 模板 desk reject
- 从今天（3月19日）到5月25日截止，约 **9 周半**

---

### 8.2 完整文献调研结果

#### RL 驱动的检索查询生成（最密切竞品）

| 论文 | 发表 | 核心方法 | 与本文的关键差异 |
|------|------|---------|----------------|
| **Search-R1** (Jin et al.) | COLM 2025 | RL 训练 LLM 在推理链中插入文本搜索调用；retrieved token masking 稳定训练 | 单一检索后端，action space 为文本查询字符串 |
| **DeepRetrieval** (Jiang et al.) | COLM 2025 | RL 训练 LLM 生成 **结构化查询**（boolean/SQL），3B 模型 11/13 数据集超越 GPT-4o | 单引擎查询优化（不编排多后端）；boolean 表达能力有限，不支持完整嵌套逻辑 |
| **ReSearch** (Chen et al.) | arXiv 2503 | RL 训练"在推理中搜索"，无需 SFT warmup；ReCall 扩展到任意工具调用 | 文本查询，依赖外部搜索引擎 |
| **R1-Searcher** (Song et al.) | EMNLP 2025 Findings | 两阶段 RL：先学搜索，再优化搜索质量 | 文本查询，OpenWeb 搜索场景 |
| **s3** (Jiang et al.) | EMNLP 2025 | 解耦 searcher/generator；"Gain Beyond RAG" 奖励；仅 2.4k 训练样本 | 文本查询；searcher 独立训练但不生成结构化 DSL |
| **R3-RAG** (Du et al.) | EMNLP 2025 Findings | 两阶段：cold-start 学 interleaved 模式 + RL 优化 | 文本查询，open-domain QA |
| **Search-o1** (Li et al.) | EMNLP 2025 | Prompting-based，非 RL；Reason-in-Documents 模块去噪 | 不训练，依赖 prompting |
| **MCTS-RAG** (Yale NLP) | EMNLP 2025 Findings | MCTS 做检索规划；7B 匹配 GPT-4o | Prompting-based + MCTS，不生成 DSL |

#### 混合检索 / 统一查询语言（DSL 相关）

| 论文 | 发表 | 核心方法 | 与本文的关键差异 |
|------|------|---------|----------------|
| **Text2VectorSQL** (Wang et al.) | arXiv 2506 | 将 NL 翻译为 VectorSQL（vector 操作 + SQL 统一），47% 优于 text-to-SQL | 监督学习（非 RL）；post-filtering 问题（向量搜索→SQL 过滤会丢弃>30% 正确结果，LocalSearch-R1 的 SQL-primary 设计更优） |
| **SUQL** (Stanford) | NAACL 2024 | SQL 中嵌入语义原语 `answer()` 和 `contains()`，扩展 SQL 表达能力 | 监督学习，面向对话系统而非检索任务 |
| **A-RAG** (arXiv 2602) | arXiv 2026.2 | 暴露 keyword+semantic+chunk 三种检索工具给 agent 自主选择；94.5% HotpotQA | Prompting-based agent，不用 RL 训练；工具粒度为整体查询策略，非细粒度逻辑组合 |
| **RRF / Hybrid Pipeline** | 工业实践 | BM25 + 向量检索，Reciprocal Rank Fusion 合并结果 | 固定融合策略，只能做并集加权，不支持交集/否定/嵌套逻辑 |
| **Self-querying RAG** | LangChain | NL → metadata filter + 语义检索 | 键值匹配，不支持 OR/NOT，不可学习 |

#### 其他相关工作

| 论文 | 发表 | 核心方法 |
|------|------|---------|
| **UR2** | arXiv 2508 | difficulty-aware curriculum training，仅困难问题触发检索 |
| **ZeroSearch** | arXiv 2025 | 在隐空间模拟检索，消除 API 开销 |
| **StepSearch** | arXiv 2025 | 步骤级奖励，惩罚冗余查询 |
| **IRCoT** | ACL 2023 | Chain-of-thought + 检索交替，奠基性工作 |
| **DeepSeek-R1** | arXiv 2501 | 纯 RL 激发推理能力，Search-R1 等工作的基础 |

---

### 8.3 核心 Gap 分析

**现有工作能力对比矩阵：**

| 能力 | Search-R1 | DeepRetrieval | s3 | Text2VectorSQL | A-RAG | **LocalSearch-R1** |
|------|-----------|--------------|----|----|-------|---|
| RL 训练 | ✓ | ✓ | ✓ | ✗ | ✗ | **✓** |
| 结构化查询输出 | ✗ (NL) | ✓ (boolean/SQL-like) | ✗ (NL) | ✓ (VectorSQL) | ✗ (策略选择) | **✓ (完整DSL)** |
| 混合检索（多异构后端） | ✗ | ✗ | ✗ | ✓ | ✓ (多工具) | **✓** |
| 完整逻辑编排 (AND/OR/NOT/嵌套) | ✗ | 有限 | ✗ | 部分 | ✗ | **✓** |
| 动态编排策略（可学习） | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

**结论：没有现有论文同时具备这五项能力。LocalSearch-R1 的 novelty 定位成立。**

---

### 8.4 Related Work 必须覆盖的关键论文（更新）

在 Part 2（Related Work）已有基础上，需新增或重点讨论：

**必须正面对比（不能回避）：**
1. **DeepRetrieval** — 与本文最接近的 RL + 结构化查询生成，必须明确指出其局限：单引擎、不支持向量检索组合、boolean 表达能力不完整
2. **Text2VectorSQL** — 最接近的 DSL 设计，必须指出：监督学习（非 RL）+ post-filtering 问题（丢失 >30% 正确结果）
3. **A-RAG** — 最接近的多工具检索 agent，必须指出：Prompting-based，不用 RL 训练，粒度为整体策略选择而非细粒度逻辑组合

**需要引用的重要工作：**
- s3（"Gain Beyond RAG" 奖励思路）
- MCTS-RAG（同类 EMNLP 2025 工作，证明检索规划方向的活跃度）
- SUQL（SQL + 语义原语的先驱设计）

---

### 8.5 可借鉴的技术细节

| 技术 | 来源 | 如何借鉴 |
|------|------|---------|
| **Retrieved token masking** | Search-R1 | 确保 RL 梯度只更新模型生成的 DSL token，不更新检索结果 token |
| **"Gain Beyond RAG" 奖励** | s3 | 可在 S_r 的基础上引入相对提升奖励，对比 naive RAG baseline |
| **Difficulty-aware curriculum** | UR2, ZeroSearch | 从简单 query（纯结构化/纯语义）渐进到复杂逻辑组合（AND/OR/NOT/嵌套） |
| **VectorSQL DSL 设计参考** | Text2VectorSQL | 对比验证 LocalSearch-R1 "SQL-primary" 设计优于 "vector-primary + SQL post-filter" |
| **解耦 searcher/generator** | s3 | DSL 生成 agent 可独立训练，与 downstream LLM 解耦 |

---

### 8.6 EMNLP 2026 可行性评估

**总体评估：录取概率 40-50%（Long Paper）**

#### 优势因素

1. **Gap 真实存在**：RL + 完整混合检索编排的交叉点确实没有人做，Novelty 可以成立
2. **场景定位清晰**：本地隐私场景有真实需求，与 open-domain QA 有差异化
3. **实验结果强**：8B 模型 beat GPT-4o，数字有吸引力
4. **EMNLP 2025 有先例**：Search-R1、MCTS-RAG、R3-RAG、s3 都在同方向被录取，说明 EMNLP 接受此类工作
5. **时间可行**：从今天到5月25日约9周半，实验计划约5-6周，紧凑但可行

#### 风险因素

1. **领域非常拥挤**：RL + 检索方向 2025 年论文爆发（Search-R1, DeepRetrieval, R3-RAG, R1-Searcher 等），EMNLP 2026 reviewer 对 novelty 要求会更高
2. **DeepRetrieval 是强劲竞品**：同是 RL + 结构化查询，必须在 Related Work 和实验中清晰区分
3. **本地场景的必要性需论证**：为什么不能用云端搜索引擎？为什么必须本地？需要更清晰的 motivation
4. **实验基准是自建的**：自建 benchmark 会被质疑 cherry-picking，需要充分的数据集统计和难度分层说明
5. **规模偏小**：目前只有 email + arxiv 两个域，泛化性存疑

#### 提高录取率的关键动作

1. **Action space ablation 是最关键实验**：必须证明 DSL > 纯文本查询（对标 Search-R1）且 DSL > 纯 SQL（对标 DeepRetrieval）且 DSL > Text2VectorSQL 的设计
2. **在 Introduction 中明确 DeepRetrieval 的局限**：直接写"DeepRetrieval 生成 boolean/SQL-like 查询，但不支持向量检索组合，且受限于单一检索后端"
3. **新增 Text2VectorSQL 作为强 baseline**：把它加入主实验对比表，而不只是在 Related Work 中讨论
4. **Complexity breakdown 实验**：证明 DSL 在逻辑复杂查询上的优势比简单查询更显著，这是 "为什么需要完整逻辑编排" 最直接的证明
5. **修正 DeepRetrieval 描述**（当前方案中有误）：方案中写"DeepRetrieval 输出改写/增强后的文本查询"——实际上 DeepRetrieval 已经输出结构化 boolean/SQL 查询，需要修正以避免 reviewer 认为作者没有认真读相关文献
