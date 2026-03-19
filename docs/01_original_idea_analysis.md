# DSL-R1 原始论文详细分析

> 基于论文 PDF 和 OpenReview 审稿记录的完整分析

## 1. 论文基本信息

- **标题**: DSL-R1: From SQL to DSL for Training Retrieval Agents across Structured and Unstructured Data with Reinforcement Learning
- **投稿**: ACL ARR 2026 January (Short Paper), Submission #3060
- **评分**: Overall Assessment 2.5 (Borderline Findings)
- **审稿人**: XRX8 (2分), 6rE4 (2分), 3avs (4分), 13wi (2.5分)
- **作者**: Yunhai Hu, Junwei Zhou, Yumo Cao, Yitao Long, Yiwei Xu, Qiyi Jiang, Weiyao Wang, Xiaoyu Cao, Zhen Sun, Yiran Zou, nan du
- **提交日期**: 2026年1月3日（修改：2月10日）
- **Research Area**: AI/LLM Agents
- **Keywords**: DSL, Retrieval, Agentic RL

## 2. 核心 Idea

### 2.1 问题定义

现实世界的检索任务需要同时处理**结构化元数据**（如日期、发件人、标签、已读状态）和**非结构化内容**（如邮件正文、附件文本）。现有系统将两者隔离：
- **SQL/SPARQL**：擅长结构化精确过滤，无法处理语义匹配
- **向量检索**（DPR、CLIP、ColQwen）：擅长语义匹配，无法执行逻辑约束

**核心挑战**：如何在一个统一框架中桥接结构化推理与语义匹配？

### 2.2 提出的解决方案

**DSL-R1 框架**由三个组件构成：

#### (1) DSL-Based Retrieval Agent

设计了一种领域特定语言（DSL），将 SQL 逻辑操作符与向量检索原语统一在一个可执行程序中。DSL 程序包含两部分：
- **SQL-like clause**：处理结构化元数据条件，使用 `<vector_x>` 占位符引用向量检索结果
- **vector_query_list**：指定语义检索查询，对应各占位符

**执行流程**（根据作者在 Reviewer 13wi 回复中的详细说明）：
1. 向量检索模块先执行 `vector_query_list` 中的查询，返回候选 ID 集合（如 `{msg_042, msg_107, msg_215, msg_318, msg_421}`）
2. 将候选 ID 注入 SQL 语句中的 `<vector_x>` 占位符（替换为 `IN ('msg_042','msg_107',...)`）
3. SQL 引擎执行最终查询，结构化条件与语义候选在 SQL 层面进行逻辑组合（AND/OR/NOT 由 SQL 原生语义处理）

**关键设计 insight**：不需要额外的 fusion 策略，SQL 本身就是 fusion 逻辑。

**DSL 示例（论文中的3个 case）**：

Case 1 - Budget Drafts（结构化 ∩ 语义）：
```json
{
  "sql": "SELECT DISTINCT message_id FROM email WHERE is_draft = 1 AND draft_modified_date >= date('now', '-7 day') AND is_starred = 1 AND message_id IN <vector_0>",
  "vector_query_list": [
    {"email_content": "budget"}
  ]
}
```

Case 2 - Important Folders（JSON 字段处理）：
```json
{
  "sql": "SELECT DISTINCT json_extract(json_each.value, '$.id') AS folder_id FROM email, json_each(email.folder_labels) WHERE json_extract(json_each.value, '$.id') IN <vector_0>",
  "vector_query_list": [
    {"folder_labels": "important"}
  ]
}
```

Case 3 - Contract Attachments（多路向量检索）：
```json
{
  "sql": "SELECT DISTINCT json_extract(json_each.value, '$.id') AS attachment_id FROM email, json_each(email.attachment_list) WHERE message_id IN <vector_0> AND json_extract(json_each.value, '$.id') IN <vector_1>",
  "vector_query_list": [
    {"subject": "contract"},
    {"attachment_list": "contract"}
  ]
}
```

**作者回复中展示的复杂示例**（OR + 多路语义交集）：

> 查询："Find unread or starred emails from last month where the body discusses the merger and the attachments contain financial reports"

```json
{
  "sql": "SELECT DISTINCT message_id FROM email WHERE (is_read = 0 OR is_starred = 1) AND received_date >= date('now', '-30 day') AND message_id IN <vector_0> AND message_id IN <vector_1>",
  "vector_query_list": [
    {"email_content": "merger discussion"},
    {"attachment_list": "financial report"}
  ]
}
```

此程序展示了：结构化字段上的析取（unread OR starred）、日期范围约束、以及两路独立语义检索（正文 AND 附件）的交集。作者指出"A rule-based partitioning system would not generate OR conditions across structured fields, nor would it dispatch two separate semantic queries to different content fields and AND them together."

#### (2) Data Generation Component

基于企业邮件和科学论文两个领域构建检索基准：

**Industrial Email Dataset**（Table 3 统计）：

| 指标 | Train | Eval | Total |
|------|-------|------|-------|
| Corpus Size | 2,000 | 500 | 2,500 |
| Avg. Doc Length (tokens) | 154.2 | 156.8 | 154.7 |
| Structured Keys (k_str) | 1.8 | 2.1 | 2.0 |
| Unstructured Keys (k_uns) | 3.1 | 3.0 | 3.1 |
| Total Complexity (k_total) | 4.9 | 5.1 | 5.0 |
| Structure-Dominated (%) | 30.0 | 28.5 | 29.7 |
| Content-Dominated (%) | 45.0 | 46.5 | 45.3 |
| Balanced (%) | 25.0 | 25.0 | 25.0 |

**字段定义**（Table 4）：

| 类别 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 结构化 | account_email | TEXT | 账户邮箱 |
| 结构化 | received_date | TEXT | 接收时间戳 |
| 结构化 | is_draft | INTEGER | 是否草稿 |
| 结构化 | draft_created_date | TEXT | 草稿创建时间 |
| 结构化 | draft_modified_date | TEXT | 草稿修改时间 |
| 结构化 | is_read | INTEGER | 是否已读 |
| 结构化 | is_starred | INTEGER | 是否加星 |
| 结构化 | is_archived | INTEGER | 是否归档 |
| 结构化 | thread_msg_count | INTEGER | 线程消息数 |
| 非结构化 | sender_email | TEXT | 发件人邮箱 |
| 非结构化 | sender_name | TEXT | 发件人名称 |
| 非结构化 | recipient_list | TEXT (JSON) | 收件人列表 |
| 非结构化 | cc_list | TEXT (JSON) | 抄送列表 |
| 非结构化 | bcc_list | TEXT (JSON) | 密送列表 |
| 非结构化 | folder_labels | TEXT (JSON) | 文件夹标签 |
| 非结构化 | attachment_list | TEXT (JSON) | 附件列表 |
| 非结构化 | subject | TEXT | 主题 |
| 非结构化 | content | TEXT | 正文 |

- 采用 STaRK 框架 (Wu et al., 2024) 生成检索实例
- 数据构造：选金标文档 → 基于元数据重叠采样负例(Stage 1) → 基于内容相似度精炼(Stage 2) → 生成 NL 查询和对应 DSL

**ArxivQA**：
- 评估使用 ViDoRe benchmark (Faysse et al., 2024) 中的 ArXivQA 子集
- 训练使用原始 ArXivQA dataset (Li et al., 2024)，避免过拟合
- 包含结构化元数据（domain, category, author, publish time）和视觉丰富文档

#### (3) Reinforcement Learning Mechanism

**训练基础设施**：
- verl 框架 (Sheng et al., 2024)
- ColQwen2.5 (Chen et al., 2024) 作为 RAG embedding backbone
- 4 × NVIDIA H100 GPUs
- 10 epochs, 学习率 1×10⁻⁶
- 推理和延迟测量在单 NVIDIA A100 GPU 上进行

**GRPO 目标函数**：
$$\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\min(r_i A_i, \text{clip}(r_i)A_i) + \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

**DAPO 改进**：token 级 advantage 归一化 + 非对称 clipping bounds (ε_l, ε_h)

**四维复合奖励函数**（来自作者回复的精确定义）：

| 奖励分量 | 类型 | 精确计算方式 |
|---------|------|-----------|
| S_f (Format Reward) | Binary | 1 if output contains valid DSL tags, 0 otherwise |
| S_e (Execution Reward) | Binary | 1 if DSL can be parsed and executed without error, 0 otherwise |
| S_r (Result Reward) | Continuous | Hit@1 score against gold reference |
| S_l (Length Reward) | Continuous | Penalty proportional to output length exceeding a threshold |

**总奖励**：R = S_f + S_e + S_r + S_l（等权求和）

**等权合理性**（作者回复原话）："We chose equal weighting because the rewards have an implicit hierarchical dependency: if format is incorrect (Sf=0), execution necessarily fails (Se=0), which in turn means the result cannot be correct (Sr=0). This natural ordering makes explicit weighting unnecessary."

### 2.3 System Prompt（来自作者回复，完整公开）

```python
system_prompt = """
# Role
You are an AI retrieval engineer that converts natural language queries into SQL
with vector retrieval placeholders.

# Goal
Given a user query, generate:
1. SQL with <vector_x> placeholders
2. Corresponding vector_query_list

# Database schema (single-table query only)

Table: email
- message_id TEXT
- account_email TEXT (structured)
- received_date TEXT (structured)
- is_read INTEGER (structured)
- is_starred INTEGER (structured)
- is_archived INTEGER (structured)
- is_draft INTEGER (structured)
- subject TEXT (vector search required)
- sender_name TEXT (vector search required)
- recipient_list TEXT (JSON, vector search required)
- cc_list TEXT (JSON, vector search required)
- email_content TEXT (vector search required)
- folder_labels TEXT (JSON, vector search required)
- attachment_list TEXT (JSON, vector search required)

Table: person
- person_id TEXT
- contact_name TEXT (vector search required)
- contact_email TEXT (vector search required)
- relationship TEXT (vector search required)
- email_count INTEGER (structured)

# Rules
- Unstructured fields MUST use vector search via <vector_0> ... <vector_9>
- Structured fields use normal SQL filters
- message_id match: message_id IN <vector_x>
- person_id match: person_id IN <vector_x>
- JSON fields use json_each() and json_extract()

# Output format
Return ONLY JSON:
{
  "sql": "...",
  "vector_query_list": [
    {"field": "query text"}
  ]
}

# Example
Query: find starred emails in 2025 with subject about budget
Output:
{
  "sql": "SELECT message_id FROM email WHERE is_starred=1 AND received_date LIKE '2025%' AND message_id IN <vector_0>",
  "vector_query_list": [
    {"subject": "budget"}
  ]
}
"""
```

### 2.4 核心论点

1. DSL 统一了 SQL 逻辑推理与向量语义检索
2. RL 通过执行反馈和检索质量奖励优化 DSL 生成
3. 小模型（4B/8B）经 RL 训练后可超越 GPT-4o

## 3. 实验结果

### 3.1 主要结果 (Table 1)

| Category | Model | Email Hit@1 | Email Hit@3 | Email MRR | Email Lat. | ArxivQA NDCG@5 | ArxivQA Lat. |
|----------|-------|-------------|-------------|-----------|------------|----------------|--------------|
| Traditional | BM25 | 42.1 | 64.7 | 53.2 | 18 | 31.6 | 25 |
| Traditional | ColQwen | 55.3 | 78.9 | 65.4 | 35 | 73.9 | 42 |
| LLM Based | Qwen3-4B | 58.3 | 80.1 | 69.2 | 95 | 55.4 | 105 |
| LLM Based | Qwen3-8B | 62.1 | 83.6 | 72.5 | 140 | 61.2 | 155 |
| LLM Based | GPT-4o | 68.9 | 88.7 | 78.8 | **30** | 70.5 | **35** |
| DSL-R1 | Qwen3-4B + GRPO | 69.5 | 88.9 | 78.6 | 55 | 74.8 | 62 |
| DSL-R1 | Qwen3-4B + DAPO | 71.7 | 90.6 | 80.9 | 52 | 75.1 | 58 |
| DSL-R1 | Qwen3-8B + GRPO | 75.8 | 92.9 | 84.6 | 82 | 77.4 | 90 |
| DSL-R1 | **Qwen3-8B + DAPO** | **77.9*** | **94.2*** | **86.1** | 78 | **81.2*** | 85 |

关键发现：
- 8B + DAPO 在 Hit@1 上超越 GPT-4o 达 **9 个百分点**（77.9 vs 68.9）
- RL 训练将延迟降低约 **45%**（140ms → 78ms）
- DAPO 整体优于 GRPO（在两种模型规模上一致）
- 在 ArxivQA 上跨域泛化有效（81.2 vs GPT-4o 的 70.5，+10.7）

### 3.2 消融实验 (Table 2)

| Reward Function | Accuracy (%) |
|-----------------|-------------|
| Qwen3-8B (pretrained) | 62.1 |
| S_f + S_e + S_r + S_l | **77.9** |
| - w/o S_f (Format Reward) | 75.3 (↓2.6) |
| - w/o S_e (Execution Reward) | 75.6 (↓2.3) |
| - w/o S_r (Result Reward) | 77.0 (↓0.9) |
| - w/o S_l (Length Reward) | 75.9 (↓2.0) |

论文还提到定性消融发现：
- 去除 SQL filter → 精度下降（丢失属性约束）
- 去除 vector retrieval → 召回下降（改写查询失效）
- 去除反馈信号 → 收敛变慢、跨域泛化减弱

## 4. 论文结构评估

### 4.1 优势
- 框架设计清晰（Figure 1 概念图简洁）
- 实验结果强劲（+15.8 Hit@1 improvement over pretrained）
- 四维 reward function 设计有参考价值
- Benchmark 有社区价值（STaRK 框架构造，结构化+非结构化混合）
- 跨域泛化验证（Email → ArxivQA）

### 4.2 不足
- **Short Paper 限制**：关键细节放不下（prompt template、fusion 机制、reward 计算细节）
- **Novelty framing 不清晰**：定位为 "unified DSL"，但工业界已有类似做法
- **Related Work 不完整**：未讨论 Search-R1、DeepRetrieval、RRF、Self-querying RAG
- **Baseline 不充分**：缺少 hybrid retrieval baseline
- **分析深度不够**：没有 error analysis、complexity breakdown、行为分析
- **GRPO/DAPO 公式占了太多正文篇幅**（应移到附录）
- **Reproducibility 差**：3/4 reviewer 给 2 分（最低档）

## 5. 与现有工作的关系

### 5.1 论文中已引用的工作
- Seq2SQL (Zhong et al., 2017): 仅处理结构化数据的 RL 方法
- ColPali/ColQwen (Faysse/Chen et al., 2024): 多模态检索
- STaRK (Wu et al., 2024): 数据集构造框架
- DPR, ANCE, CLIP, BLIP-2: 检索基础方法
- ReAct (Yao et al., 2023): LLM Agent 框架

### 5.2 论文中完全缺失的关键相关工作
- **Search-R1** (Jin et al., 2025): RL 训练 LLM 调用搜索引擎，COLM 2025
- **DeepRetrieval** (Jiang et al., 2025): RL 优化查询改写，3B 超越 GPT-4o，COLM 2025
- **RRF** (Reciprocal Rank Fusion): 工业界标准混合检索方法
- **Self-querying RAG** (LangChain): metadata filter + 语义检索
- **Text2VectorSQL** (Wang et al.): SQL + 向量操作的统一查询语言
- **SUQL** (Stanford, NAACL 2024): SQL 中嵌入语义原语
- **A-RAG**: 多工具检索 agent
- **s3, R1-Searcher, R3-RAG, Search-o1, MCTS-RAG**: 同方向 EMNLP 2025 工作

## 6. 核心问题诊断

| 问题 | 严重程度 | 涉及 Reviewer | 可修复性 |
|------|---------|-------------|---------|
| Novelty 定位不清 | 致命 | XRX8, 6rE4, 3avs | 可通过重新 framing 解决 |
| 缺少 hybrid baseline | 致命 | XRX8, 6rE4, AC | 需要跑新实验 |
| Prompt template 未公开 | 严重 | 6rE4, 13wi, AC | 已在回复中补充，需加入正文 |
| Reward function 细节不足 | 中等 | 13wi, AC | 已在回复中补充，需加入正文 |
| 融合机制描述模糊 | 中等 | 13wi | 已在回复中补充，需加入正文 |
| 缺少深度分析 | 严重 | XRX8, 3avs, AC | 需要新实验和分析 |
| 写作规范性 | 轻微 | XRX8 | 改写即可 |
| 泛化性/schema 依赖 | 中等 | 6rE4 | 需在 Limitations 中诚实讨论 |
| GRPO/DAPO 占篇幅过多 | 轻微 | 13wi | 移到附录 |
| Related Work 不完整 | 严重 | 全部 | 重写 Related Work |

## 7. 代码仓库状态

- **GitLab 地址**: `https://gitlab.thinredline.cn/zhoujunwei/agent_training_rl/-/tree/main/retrieve_agent`
- **访问状态**: 私有仓库，当前 SSH key 已过期，无法远程访问
- **框架**: verl (HybridFlow) 用于 RL 训练
- **Embedding**: ColQwen2.5
