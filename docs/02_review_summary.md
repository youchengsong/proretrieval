# 审稿意见详细汇总与回应分析

> ACL ARR 2026 January Submission #3060

## 评分总览

| Reviewer | Overall | Soundness | Excitement | Confidence | Reproducibility |
|----------|---------|-----------|------------|------------|-----------------|
| XRX8     | 2       | 3         | 2          | 4          | 2               |
| 6rE4     | 2       | 3         | 2.5        | 4          | 2               |
| 3avs     | 4       | 4         | 3.5        | 4          | 3               |
| 13wi     | 2.5     | 2.5       | 2.5        | 3          | 2               |
| **Avg**  | **2.5** | **3.1**   | **2.6**    | **3.75**   | **2.25**        |

Area Chair 评价：Borderline Findings

## 各 Reviewer 详细意见

### Reviewer XRX8 — "需要走得更远"

**核心关切**：Novelty 不足。认为 structured+unstructured 查询合并在工业界已是常见做法（rule-based partitioning + fixed merge）。

**关键原话**：
> "The paper would be novel if it went a step further. For example, if it worked on some of the proposed future work or proposed a truly agentic retrieval system that can study the shape of the corpora and understand how to better plan the query for realizing maximum recall."

**具体问题**：
1. 未对比 Gemini/GPT 等商业模型
2. Section 3.3 写作不规范，缺少标准 IR 符号
3. "Hit@1/3" 在 abstract 中不清楚

**作者回应要点**：
- 展示了 DSL 可以表达 OR/NOT 等复杂逻辑组合，超越固定 rule-based 分割
- 承诺添加 Gemini 对比和标准符号
- 承诺改写 Section 3.3

---

### Reviewer 6rE4 — 最严厉，4个维度批评

**Weakness 1: Limited Generalizability**
> "The proposed framework relies heavily on a rigid 'SQL-plus-Vector' DSL structure, which assumes a pristine dichotomy between structured metadata and unstructured content that is often absent in real-world scenarios."

**Weakness 2: Lack of Transparency**
> "The paper fails to disclose the specific system prompts and instruction templates used to guide the LLMs in generating the DSL, which critically hinders reproducibility."

**Weakness 3: Weak Baselines**
> "The authors neglect established 'strong' hybrid baselines, such as combining BM25 and dense retrieval, and fail to compare against specialized Text-to-SQL models or existing self-querying RAG frameworks."

**Weakness 4: Limited Technical Novelty**
> "Combining vector retrieval with symbolic metadata filtering is a well-established engineering practice... The core contribution is effectively limited to applying existing reinforcement learning algorithms to this specific domain."

**作者回应要点**：
- Schema 预定义对邮件/文档管理等实际场景是合理假设，ArxivQA 跨域验证了泛化性
- 承诺在修订版中公开完整 prompt template 和 reward 计算细节
- 承诺添加 RRF、Self-querying RAG baseline
- 强调 DSL 的 learnable composition（可学习的逻辑组合）是与工业实践的核心区别

---

### Reviewer 3avs — 最友好，给了4分

**优点认可**：
> "The performance is impressive... DSL-R1 achieves a +12.3% improvement in Hit@1/3. Moreover, the framework (Figure 1) is conceptually clean."

**批评**：
- Novelty 有限，主要是使用 GRPO 做 RL
- 需要讨论与 Seq2SQL 和 ColPali 的关系
- 需要更多性能分析

**作者回应要点**：
- 三方面超越 GRPO 应用：(1) DSL 设计本身统一了逻辑和语义，(2) 四维 reward 针对可执行程序，(3) 小模型超越 GPT-4o 的实证发现
- Seq2SQL 仅处理结构化数据，ColPali 在表示层面操作，我们的 DSL 在程序层面操作
- 承诺添加 error analysis、complexity breakdown、case study

---

### Reviewer 13wi — 简短但精准的技术质疑

**4个具体问题**：
1. Prompt template 不清楚
2. SQL 和 embedding retrieval 的融合机制不清楚
3. 四个 reward function 的计算方式和权重不清楚
4. GRPO/DAPO 详细公式占太多篇幅，建议移到附录

**作者回应**：
- 详细解释了融合机制（vector search → candidate IDs → inject into SQL → execute）
- 公开了完整的 system prompt
- 明确了 reward 的计算方式和 equal weight 的理由
- 同意将 GRPO/DAPO 公式移到附录

---

## 共性问题提炼

| 问题类别 | 提到的 Reviewer | 严重程度 |
|---------|---------------|---------|
| Novelty 定位不清 | XRX8, 6rE4, 3avs | ⭐⭐⭐⭐⭐ |
| 缺少 hybrid baseline | XRX8, 6rE4 | ⭐⭐⭐⭐⭐ |
| Prompt template 未公开 | 6rE4, 13wi | ⭐⭐⭐⭐ |
| Reward function 细节不足 | 13wi | ⭐⭐⭐ |
| 融合机制描述模糊 | 13wi | ⭐⭐⭐ |
| 缺少深度分析 | XRX8, 3avs | ⭐⭐⭐⭐ |
| 写作规范性 | XRX8 | ⭐⭐ |
| 泛化性/schema 依赖 | 6rE4 | ⭐⭐⭐ |
| GRPO/DAPO 占篇幅过多 | 13wi | ⭐⭐ |

## 修订方向指引

根据审稿意见，修订版需要解决的核心问题按优先级排序：

1. **重新定位 Novelty**：从 "unified DSL" 升级为 "retrieval orchestration"（检索编排），强调 DSL 的逻辑表达能力（AND/OR/NOT/嵌套）和可学习的编排策略
2. **补充 hybrid baseline**：RRF、Self-querying RAG、GPT-4o few-shot DSL
3. **添加深度分析**：action space ablation、complexity breakdown、orchestration behavior analysis
4. **公开方法细节**：prompt template、reward 计算公式、融合机制伪代码
5. **重写 Related Work**：覆盖 Search-R1、DeepRetrieval、hybrid retrieval 实践
6. **改善写作**：标准符号、流程图、规范化表述
