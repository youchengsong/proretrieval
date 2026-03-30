# LocalSearch-R1 完整工作计划

> EMNLP 2026 Long Paper | ARR May cycle 截止 2026-05-25 | 从 2026-03-23 起共 9 周

---

## 一、论文定位

### 一句话

> 现有 RL 检索方法让 LLM 决定"问什么"，我们让 LLM 决定"怎么找"——生成可执行的多模态检索程序，动态编排结构化过滤、文本语义、图像语义三条异构路径及其逻辑关系。

### 范式对比

| 范式 | 模型输出 | 检索通道 | 逻辑组合 | 代表 |
|------|---------|---------|---------|------|
| Query Rewriting | NL query | 固定单路 | 无 | HyDE, Search-R1 |
| Boolean Query Gen | boolean 表达式 | 固定单路 | 有限 AND/OR | DeepRetrieval |
| Self-Querying | filter + query | 固定双路 | 固定 AND | LangChain |
| Rank Fusion | 多路独立结果 | 固定多路 | 固定加权 | RRF |
| **Retrieval Program Synthesis** | **可执行程序** | **动态多路多模态** | **任意组合** | **Ours** |

### 三条贡献

1. **Retrieval Program Synthesis 范式。** DSL 融合 SQL + 文本向量 + 图像向量，支持 AND/OR/NOT 及其任意组合，模型通过 RL 学会动态编排。
2. **层次化复合奖励 RL 框架。** 四维 reward（格式→执行→结果→效率）建模程序生成的层次化约束。
3. **多模态混合检索基准 + 小模型编排能力验证。** 叶子数量驱动的自动构造 pipeline，8B 模型 beat GPT-4o。

---

## 二、数据集

### 2.1 场景与数据来源

| 场景 | 模态 | 角色 | 公开数据来源 |
|------|------|------|------------|
| 电商产品 | 结构化 + 文本 + **图像** | 主场景，完整多模态编排 | Amazon ESCI + Amazon Review |
| 企业邮件 | 结构化 + 文本（无图像）| 第二场景，证明泛化 | Enron Email Dataset |

#### 电商数据

| 数据集 | 内容 | 获取 |
|--------|------|------|
| Amazon ESCI | product_id, title, brand, category, price + 产品图 URL | GitHub 公开 |
| Amazon Review | 用户评论原文 | UCSD 公开 |

处理流程：
- 从 ESCI 提取产品基础信息和图片
- 从 Amazon Review 对每个产品聚合评论，用 LLM 生成 review_summary（3-5 句概括优缺点）
- 补充 rating, num_reviews, in_stock, release_date 等结构化字段（从原始数据提取或合理合成）
- 产品图用 CLIP ViT-L/14 编码建图像向量索引

#### 邮件数据

| 数据集 | 内容 | 获取 |
|--------|------|------|
| Enron Email | ~500K 封真实企业邮件，含 thread 结构、发件人、时间、主题、正文 | 公开 |

处理流程：
- 从 Enron 原始数据提取结构化字段（sender, date, is_read 等，部分需合理推断或合成）
- 正文和主题作为文本字段
- 无图像字段

### 2.2 Schema

#### 电商

```
Table: product
-- 结构化（SQL 过滤）
product_id, brand, category, price, rating, num_reviews, in_stock, release_date

-- 文本（<text_x> 占位符）
title, description, review_summary

-- 图像（<image_x> 占位符）
product_image
```

#### 邮件

```
Table: email
-- 结构化（SQL 过滤）
message_id, received_date, is_draft, is_read, is_starred, is_archived, thread_msg_count

-- 文本（<text_x> 占位符）
sender_name, recipient_list, folder_labels, subject, content, attachment_list
```

### 2.3 检索后端

| 通道 | 后端 | 说明 |
|------|------|------|
| 结构化 | SQLite | SQL WHERE clause |
| 文本语义 | ColQwen2.5 | text → text embedding |
| 图像语义 | CLIP ViT-L/14 | text → image embedding（电商专用）|

### 2.4 数据规模

| | 电商 | 邮件 |
|--|------|------|
| 语料库 | ~5,000 产品 | ~2,500 邮件 |
| Train | 4,000 | 2,000 |
| Eval | 1,000 | 500 |

---

## 三、Query 构造 Pipeline（九步法）

### 3.1 流程总览

```
Step 1: 随机叶子数量 N → 确定复杂度
Step 2: 随机分组 → 括号结构 + 操作符
Step 3: 确定 gold document 数量并采样
Step 4: 每组内随机选字段
Step 5: 每个叶子决定肯定/否定
Step 6: 从 gold 提取值填入叶子
Step 7: 编译成 DSL
Step 8: 确定完整 GT（结构化超集 或 全量兜底）
Step 9: 生成自然语言 + 记录元数据
```

### 3.2 Step 1：叶子数量

| N | 概率 | Level |
|---|------|-------|
| 1 | 30% | Level 1 |
| 2 | 20% | Level 2 |
| 3 | 20% | Level 2 |
| 4 | 15% | Level 3 |
| 5 | 15% | Level 3 |

### 3.3 Step 2：随机分组

**2a. 切割：** N-1 个间隙，每个以概率 p 成为切割点，分成 K 组。

| Level | p |
|-------|---|
| Level 2 | 30% |
| Level 3 | 40% |

**2b. 硬币：** 决定两层操作符。

| 硬币 | 组内 | 组间 | 模式 |
|------|------|------|------|
| 正面 50% | AND | OR | (A AND B) OR (C AND D) |
| 反面 50% | OR | AND | (A OR B) AND (C OR D) |

K=1 时组内操作符即全局操作符，随机 AND 或 OR。N=1 跳过。

覆盖范围：无嵌套 + 所有两层嵌套。

### 3.4 Step 3：Gold document

| 组间操作符 | Gold 数量 |
|-----------|----------|
| OR | K 个（每组一个不同 gold）|
| AND | 1 个（共享）|

组间 OR 建议同大类内采样。

### 3.5 Step 4：选字段

按权重从字段池不重复选取。结构化字段权重更高。

**电商字段权重：**

| 字段 | 模态 | 权重 |
|------|------|------|
| brand | 结构化 | 2 |
| category | 结构化 | 2 |
| price | 结构化 | 2 |
| rating | 结构化 | 2 |
| num_reviews | 结构化 | 1 |
| in_stock | 结构化 | 1 |
| release_date | 结构化 | 1 |
| title | 文本 | 1 |
| description | 文本 | 2 |
| review_summary | 文本 | 2 |
| product_image | 图像 | 2 |

组间 OR 字段可重复，组间 AND 字段不重复。

### 3.6 Step 5：肯定/否定

| Level | 肯定 | 否定 |
|-------|------|------|
| Level 1 | 70% | 30% |
| Level 2 | 60% | 40% |
| Level 3 | 50% | 50% |

无强制约束，全否定组允许出现。

### 3.7 Step 6：填充值

**肯定叶子（值来自本组 gold）：**

| 类型 | 方式 |
|------|------|
| 结构化-类别型 | 等值：`brand = 'Sony'` |
| 结构化-数值型 | 随机比较符 + 阈值（gold 满足）|
| 结构化-布尔型 | 直接用 gold 值 |
| 文本 | LLM 从 gold 文本提取 2-5 个语义词 |
| 图像 | VLM 从 gold 图片生成视觉描述 |

**否定叶子（生成 gold 不满足的条件）：**

| 类型 | 方式 |
|------|------|
| 结构化 | 取 gold 不具备的值 |
| 文本 | LLM 生成 gold 文本中不存在的特征 |
| 图像 | LLM 生成与 gold 外观不同的描述 |

### 3.8 Step 7：编译

叶子编译：结构化→SQL条件，文本→`id IN/NOT IN <text_N>`，图像→`id IN/NOT IN <image_N>`。

组装：K=1 用组内操作符平铺；K>1 组间 OR 时 `(组1) OR (组2)`，组间 AND 时 `(组1) AND (组2)`。单叶子组不加括号。

### 3.9 Step 8：确定完整 GT

构造时的 gold 不是唯一 GT，需执行 DSL 找全部满足条件的文档。

**超集提取规则：**

| 结构 | 规则 | 结果 |
|------|------|------|
| K=1 组内 AND | 删向量条件，保留结构化 | 超集 |
| K=1 组内 OR | 无法安全提取 | 全量执行 |
| K>1 组间 OR 组内 AND | 每组删向量保结构化，OR 连接。某组变空则全量 | 超集或全量 |
| K>1 组间 AND 组内 OR | 去掉含向量的组，保留纯结构化组。全去掉则全量 | 超集或全量 |

外 AND 内 OR 不能直接删向量条件（会变严格），需整组去掉。

**执行：** 超集有效时先跑超集 SQL 缩小候选集再执行完整 DSL；退化为全量时直接全量执行。

**验证：** GT 非空 + GT < 语料库 50%。Gold 不一定在 GT 中（文本/图像 NOT 的向量噪音可能排除 gold，这是正常现象）。

**成本：** ~85% 超集有效 (<10ms/条)，~15% 全量 (~200ms/条)。6000 条总计约 4 分钟。

### 3.10 Step 9：自然语言 + 元数据

LLM 将 DSL → 口语化查询。记录完整元数据：

```json
{
  "query_id": "q_00001",
  "natural_language_query": "...",
  "gold_dsl": {"sql": "...", "retrieval_list": [...]},
  "gold_document_ids": ["P001", "P002"],
  "gt_document_ids": ["P001", "P002", "P006", "P011"],
  "level": 3,
  "num_leaves": 4,
  "num_groups": 2,
  "group_inner_op": "OR",
  "group_outer_op": "AND",
  "num_not_leaves": 1,
  "modalities_used": ["structured", "text", "image"],
  "logic_pattern": "text AND (struct OR struct OR NOT image)",
  "gt_discovery_method": "structural_superset",
  "gt_size": 4
}
```

评估使用 `gt_document_ids`。

---

## 四、DSL 设计

### 4.1 格式

```json
{
  "sql": "SELECT DISTINCT {id} FROM {table} WHERE {conditions}",
  "retrieval_list": [
    {"type": "text|image", "field": "字段名", "query": "检索文本"}
  ]
}
```

### 4.2 执行

```
1. 遍历 retrieval_list，按 type 分发: text → ColQwen, image → CLIP
2. 各路返回 top-K 候选 ID
3. 注入 SQL 占位符
4. SQL 引擎执行（AND/OR/NOT 由 SQL 原生处理）
```

SQL 即 fusion，不需额外融合策略。

### 4.3 表达能力对比

| 能力 | RRF | Self-querying | Search-R1 | DeepRetrieval | **Ours** |
|------|-----|--------------|-----------|---------------|----------|
| 结构化过滤 | ✗ | 简单键值 | ✗ | 有限 | **完整 SQL** |
| 文本语义 | ✓ | ✓ | ✓ | ✗ | **✓** |
| 图像语义 | ✗ | ✗ | ✗ | ✗ | **✓** |
| AND/OR/NOT | 仅合并 | 仅 AND | ✗ | 有限 | **✓** |
| 嵌套 | ✗ | ✗ | ✗ | ✗ | **✓** |
| 跨模态逻辑组合 | ✗ | ✗ | ✗ | ✗ | **✓** |
| 动态模态选择 | ✗ | ✗ | ✗ | ✗ | **✓ (RL)** |

---

## 五、训练

### 5.1 配置

| 项 | 值 |
|----|-----|
| RL 框架 | verl (HybridFlow) |
| Base model | Qwen3-4B, Qwen3-8B |
| 文本 embedding | Qwen3 embedding 8b |
| 图像 embedding | Qwen3VL embedding 8b |
| RL 算法 | GRPO, DAPO |
| 硬件 | 4 × H100 |
| Epochs | 10, lr=1e-6 |
| 推理 | 单 A100 |

### 5.2 四维 Reward

| 分量 | 类型 | 说明 |
|------|------|------|
| S_f | Binary | DSL 格式合法 |
| S_e | Binary | 可解析可执行 |
| S_r | Continuous | Hit@1（多 GT 时任一命中即得分）|
| S_l | Continuous | 长度惩罚 |

R = S_f + S_e + S_r + S_l。等权，层次化隐式优先级。对模态透明。

---

## 六、Baselines

### 6.1 完整列表

**Group 1: Single-mode Retrieval**

| 方法 | 说明 |
|------|------|
| BM25 | 稀疏检索 |
| ColQwen | 稠密向量检索 |
| CLIP（仅电商）| 图像向量检索 |

**Group 2: Hybrid Baselines（核心新增）**

| 方法 | 实现 | 工作量 |
|------|------|--------|
| BM25 + ColQwen RRF | 两路独立跑，RRF(k=60) 融合 | 1-2 天 |
| Self-querying RAG | GPT-4o 提取 filter + query，带 filter 向量检索 | 2-3 天 |

**Group 3: Commercial LLM（回应 Reviewer XRX8 + AC）**

| 方法 | 实现 | 工作量 |
|------|------|--------|
| GPT-4o zero-shot DSL | 直接生成 DSL | 已有 |
| GPT-4o few-shot DSL | 3-5 个示例 ICL | 1 天 |
| Gemini few-shot DSL | 同上，换 API | 1 天 |

**Group 4: Open-source LLM w/o RL**

| 方法 | 说明 |
|------|------|
| Qwen3-4B | 基座直接生成 DSL |
| Qwen3-8B | 基座直接生成 DSL |

**Group 5: LocalSearch-R1 (Ours)**

| 方法 | 说明 |
|------|------|
| Qwen3-4B + GRPO | RL 训练 |
| Qwen3-4B + DAPO | RL 训练 |
| Qwen3-8B + GRPO | RL 训练 |
| Qwen3-8B + DAPO | RL 训练（主结果）|

### 6.2 每个 Baseline 回答的问题

| Baseline | 问题 |
|----------|------|
| BM25 / ColQwen / CLIP | 单路够不够？ |
| RRF | 工程方案够不够？ |
| Self-querying RAG | 简单 AND 够不够？ |
| GPT-4o / Gemini few-shot | 大模型 ICL 够不够？RL 是否必要？ |
| Qwen w/o RL | RL 带来了什么？ |
| Action Space Ablation | 完整 DSL 比单一路径好多少？ |

---

## 七、评估

### 7.1 候选池

每条 query 固定 500 个文档的候选池，所有方法在同一候选池上评估：

```
GT 文档
+ 结构化重叠 negatives（共享部分结构化属性）
+ 文本语义近邻 negatives（ColQwen 近邻）
+ 图像视觉近邻 negatives（CLIP 近邻，仅电商）
+ 随机 negatives（补足到 500）
```

### 7.2 指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Hit@1 | GT 中任一文档在 top-1 | 主表 |
| Hit@3 | GT 中任一文档在 top-3 | 主表 |
| MRR | 第一个 GT 文档排名的倒数 | 主表 |
| Recall@k | top-k 中 GT 文档的比例 | 分析（多 GT 覆盖率）|
| nDCG@5 | 归一化折扣累积增益 | 分析 |
| Latency | ms/query | 主表 |

### 7.3 Cost-effectiveness 分析

报告每种方法的 cost/query：

```
GPT-4o API:  ~$0.003/query + 网络延迟 500ms+
Gemini API:  类似
LocalSearch-R1 本地: ~$0/query + 延迟 78ms
```

即使商业模型性能略强，本地模型在频繁检索场景（邮件/电商）下成本优势显著。性能-成本散点图展示 Pareto frontier。

---

## 八、实验设计

### 8.1 总表

| # | 实验 | 目的 | 优先级 |
|---|------|------|--------|
| E1 | Main Results | 整体对比 | P0 |
| E2 | Action Space Ablation | hybrid > 单路 | P0 |
| E3 | Complexity Breakdown | 复杂 query 优势 | P0 |
| E4 | Reward Ablation | 四维 reward 贡献 | 已有 |
| E5 | Modality Ablation（电商）| 图像路价值 | P0 |
| E6 | Orchestration Behavior | 动态编排证据 | P1 |
| E7 | Modality Selection（电商）| 何时用图像路 | P1 |
| E8 | Model Scaling | 最小有效规模 | P1 |
| E9 | Error Analysis | 错误分类 | P2 |
| E10 | Latency Breakdown | 延迟分析 | P2 |
| E11 | Cost Analysis | 成本效益 | P2 |

### 8.2 E1: Main Results

```
Category              | Model                  | Email       | E-commerce
                      |                        | Hit@1  MRR  | Hit@1  MRR
======================|========================|=============|============
Single-mode           | BM25                   | TBD         | TBD
                      | ColQwen                | TBD         | TBD
                      | CLIP                   | —           | TBD
----------------------|------------------------|-------------|------------
Hybrid                | BM25+ColQwen RRF       | TBD         | TBD
                      | Self-querying RAG      | TBD         | TBD
----------------------|------------------------|-------------|------------
Commercial LLM        | GPT-4o zero-shot       | TBD         | TBD
                      | GPT-4o few-shot        | TBD         | TBD
                      | Gemini few-shot        | TBD         | TBD
----------------------|------------------------|-------------|------------
Open-source w/o RL    | Qwen3-4B               | TBD         | TBD
                      | Qwen3-8B               | TBD         | TBD
----------------------|------------------------|-------------|------------
LocalSearch-R1        | Qwen3-4B + GRPO        | TBD         | TBD
                      | Qwen3-4B + DAPO        | TBD         | TBD
                      | Qwen3-8B + GRPO        | TBD         | TBD
                      | Qwen3-8B + DAPO        | TBD         | TBD
```

### 8.3 E2: Action Space Ablation

Qwen3-8B + DAPO，只改 action space：

| Action Space | 说明 |
|-------------|------|
| text query → vector retrieval | 类 Search-R1 |
| SQL only → structured DB | 类 Text-to-SQL |
| boolean → single retriever | 类 DeepRetrieval |
| DSL: SQL + text | 无图像路 |
| DSL: SQL + text + image（仅电商）| 完整多模态 |

### 8.4 E3: Complexity Breakdown

按 Level 1/2/3 分层报性能。可进一步按含 OR / 含 NOT / 含 OR+NOT 细分。

预期：Level 1 各方法差距小，Level 3 DSL-R1 显著领先。

### 8.5 E5: Modality Ablation（电商）

| 配置 | 说明 |
|------|------|
| text only | 只有文本向量路 |
| image only | 只有图像向量路 |
| text + image fixed AND | 固定双路 AND |
| text + image fixed OR | 固定双路 OR |
| text + image dynamic (Ours) | 模型动态决定 |

预期：dynamic > fixed > single。

### 8.6 E7: Modality Selection Behavior（电商）

按 query 类型统计模型使用图像路的比例：

| Query 类型 | 预期使用图像路比例 |
|-----------|------------------|
| 含视觉描述 | 80%+ |
| 纯功能/属性 | <20% |
| 混合 | 40-60% |

### 8.7 E8: Model Scaling

| Model | Hit@1 (pretrained) | Hit@1 (+ DAPO) | Beat GPT-4o? |
|-------|-------------------|----------------|--------------|
| 1.5B | TBD | TBD | TBD |
| 4B | TBD | TBD | TBD |
| 8B | TBD | TBD | TBD |
| 14B | TBD | TBD | TBD |

---

## 九、Related Work

| Section | 核心内容 | 必须准确描述的论文 |
|---------|---------|-----------------|
| §2.1 Hybrid Retrieval | RRF, Self-querying RAG, 工业实践 | 指出编排逻辑固定受限 |
| §2.2 RL for Retrieval | Search-R1, DeepRetrieval, s3, R1-Searcher, R3-RAG, MCTS-RAG | **DeepRetrieval 输出结构化 boolean/SQL query，必须准确描述** |
| §2.3 DSLs in AI | SUQL, Text2VectorSQL, 程序合成 | SUQL 是先驱设计，Text2VectorSQL 是 SFT 非 RL |
| §2.4 On-device Retrieval | Edge RAG SDK, Edge LLM | 工程实现缺乏编排优化 |

---

## 十、论文结构

| Section | 页数 |
|---------|------|
| Abstract | 0.3 |
| §1 Introduction（四段式） | 1.5 |
| §2 Related Work（四个 subsection） | 1.5 |
| §3 Method（DSL 设计 + 执行 + 数据构造 + RL）| 3.0 |
| §4 Experiments（主表 + 6 个分析实验）| 3.0 |
| §5 Conclusion + Future Work | 0.5 |
| §6 Limitations | 0.3 |
| References | 1.0 |
| Appendix | 3-4 |

正文 ~10 页 + Appendix 3-4 页。

### Appendix 内容

- 完整 Prompt Template（电商 + 邮件）
- Reward Function 详细公式
- GRPO/DAPO 详细推导
- 更多 DSL Case Samples（覆盖各种逻辑模式）
- 数据集完整统计表
- Query 构造 Pipeline 细节
- 训练曲线图
- Cost-effectiveness 详细计算

### Limitations（诚实讨论）

1. DSL schema 需预定义（但 email/电商等场景天然有 well-defined schema）
2. 当前支持两层嵌套，更深嵌套语法上可行但未在训练数据中覆盖
3. 两个 domain（email + 电商），更多领域泛化待验证
4. 图像路依赖 CLIP 的 text-to-image 对齐质量
5. 单轮检索，不支持多轮对话式 refinement

---

## 十一、时间表

### Phase 1：数据与基础设施（Week 1-3，→ 4月13日）

| 任务 | 截止 |
|------|------|
| 获取 Amazon ESCI + Review，构建电商语料库 | 3月26日 |
| 获取 Enron Email，构建邮件语料库 | 3月28日 |
| ColQwen 文本索引 + CLIP 图像索引构建 | 3月30日 |
| 实现九步 Query 构造 Pipeline | 4月4日 |
| 生成电商 train(4000) + eval(1000) | 4月6日 |
| 生成邮件 train(2000) + eval(500) | 4月9日 |
| 数据质量检查 + 统计 | 4月11日 |

### Phase 2：训练与核心实验（Week 3-6，→ 5月4日）

| 任务 | 截止 |
|------|------|
| 电商 RL 训练 (4B+8B, GRPO+DAPO) | 4月16日 |
| 邮件 RL 训练 | 4月19日 |
| E1: Main Results 全部 baseline | 4月24日 |
| E2: Action Space Ablation | 4月27日 |
| E3: Complexity Breakdown | 4月29日 |
| E5: Modality Ablation | 5月1日 |
| E6+E7: 行为分析 | 5月3日 |

### Phase 3：补充实验 + 写作（Week 6-8，→ 5月18日）

| 任务 | 截止 |
|------|------|
| E8: Model Scaling (1.5B, 14B) | 5月6日 |
| E9-E11: Error / Latency / Cost | 5月8日 |
| Introduction + Related Work | 5月12日 |
| Method + Experiments | 5月18日 |

### Phase 4：打磨与提交（Week 8-9，→ 5月25日）

| 任务 | 截止 |
|------|------|
| Conclusion + Limitations + Appendix | 5月20日 |
| 全文打磨 + 图表 | 5月22日 |
| 共同作者审阅 + 修改 | 5月24日 |
| 提交 | **5月25日** |

---

## 十二、风险与应对

| 风险 | 应对 |
|------|------|
| Action Space Ablation 差距不大 | 按 query 类型细分，找 DSL 优势最大的子集 |
| 图像路增益小 | 聚焦视觉 query 子集；强调 dynamic > fixed |
| RRF 比预期强 | 聚焦复杂 query 优势；承认简单 query RRF 够用 |
| 商业模型强于我们 | 从 local 角度解释：成本、延迟、隐私优势；报性能-成本 Pareto |
| CLIP text-to-image 精度差 | 换 SigLIP；调大 top_k |
| 数据构造时间超预期 | 先 1000 条小规模验证 pipeline |
| 被认为 Text-to-SQL 扩展 | Action Space Ablation 证明向量原语增量价值 |
| DeepRetrieval 描述不准确 | 仔细读原文 |
| 时间不够 | 砍 P2 实验；Model Scaling 只做两个点 |

### 保底策略

最小可行论文（冲 Findings）：E1 + E2 + E3 + E4(已有)。

完整论文（冲 Main）：上述 + E5 + E6/E7 + E8。

---

## 十三、概率评估

| 目标 | 概率 |
|------|------|
| Main（乐观） | 60-65% |
| Main（基准） | 55% |
| Main（悲观） | 40-45% |
| **Findings** | **75-85%** |
