# 审稿意见详细汇总与回应分析

> ACL ARR 2026 January Submission #3060
> OpenReview: `https://openreview.net/forum?id=AJpdfW2kpC`

## 评分总览

| Reviewer | Overall | Soundness | Excitement | Confidence | Reproducibility |
|----------|---------|-----------|------------|------------|-----------------|
| XRX8     | 2       | 3         | 2          | 4          | 2               |
| 6rE4     | 2       | 3         | 2.5        | 4          | 2               |
| 3avs     | 4       | 4         | 3.5        | 4          | 3               |
| 13wi     | 2.5     | 2.5       | 2.5        | 3          | 2               |
| **Avg**  | **2.5** | **3.1**   | **2.6**    | **3.75**   | **2.25**        |

Area Chair (xyFs) 评价：Borderline Findings (2.5)

---

## Meta Review (Area Chair xyFs)

**Summary of Reasons To Publish**:
1. 解决了实际问题：如何有效结合结构化和非结构化数据源进行检索
2. RL reward 设计有效，优化了 executability + result quality + retrieval utility
3. 性能强劲（但 reviewer 指出需要更多 commercially-available 模型对比）
4. 新 benchmark 对未来研究有价值

**Summary of Suggested Revisions**:
1. 公开 prompt templates、reward function 细节、query execution mechanism
2. 增加更强的 baselines（包括商业模型）
3. 添加 error analysis / case studies
4. 进一步明确 novelty 和贡献
5. 成本分析（cost breakdown and cost-effectiveness）

---

## 各 Reviewer 详细意见

### Reviewer XRX8 — "需要走得更远" (Overall: 2)

**Paper Summary**: 认可了 DSL 定义（SQL filters + semantic search sub-queries 的组合）、RL 框架、benchmark 和 reward function 的设计。

**Summary Of Strengths**:
- Benchmark 有社区价值（如果能开源）
- Reward function 设计 thoughtful，可作为有用参考

**Summary Of Weaknesses**:
- **Novelty 不足**（核心问题）：合并结构化和非结构化查询结果在工业界已是常见做法（rule-based partitioning + fixed merge）。DSL 生成本身也不新。

  **关键原话**：
  > "The paper would be novel if it went a step further. For example, if it worked on some of the proposed future work or proposed a truly agentic retrieval system that can study the shape of the corpora and understand how to better plan the query for realizing maximum recall."

- **Baseline 不合适**：不确定是否使用了合适的评估基线。现代模型应该已经能够通过 few-shot 生成 DSL 格式。建议对比 Gemini/GPT 等商业模型。
- **写作质量**：Section 3.3（Data Preparation）写得不够规范，缺少标准 IR 符号（如用 q 表示 query），公式不好关联。Abstract 中 "Hit@1/3" 含义不清。

**Reproducibility**: 2（最低档）
**Overall Assessment**: 2 = Resubmit next cycle

**作者回应核心论点**:
- DSL 可以表达 OR/NOT 等复杂逻辑组合，超越固定 rule-based 分割
- 展示了复杂 DSL 示例证明表达能力
- 承诺添加 Gemini 对比和标准符号

---

### Reviewer 6rE4 — 最严厉，四维度批评 (Overall: 2)

**Summary Of Strengths**:
- **Robustness via Execution-Aware Optimization**: 执行反馈集成到 RL loop 中，有效对齐模型输出与实际检索效用
- **Operational Efficiency**: 长度惩罚成功将延迟降低 45%，验证了框架可满足实际部署的严格运行时约束

**Summary Of Weaknesses**:

**Weakness 1: Limited Generalizability and Scope**
> "The proposed framework relies heavily on a rigid 'SQL-plus-Vector' DSL structure, which assumes a pristine dichotomy between structured metadata and unstructured content that is often absent in real-world scenarios."

指出框架不适合：缺乏清晰 schema 的域（如 noisy web crawls）、需要超越简单属性过滤的复杂推理场景。

**Weakness 2: Lack of Transparency and Reproducibility**
> "The paper fails to disclose the specific system prompts and instruction templates used to guide the LLMs in generating the DSL, which critically hinders reproducibility."

**Weakness 3: Weak Baselines and Lack of Comparisons**
> "The authors neglect established 'strong' hybrid baselines, such as combining BM25 and dense retrieval, and fail to compare against specialized Text-to-SQL models or existing self-querying RAG frameworks."

指出只对比了 single-mode baselines 和 general-purpose LLMs，无法评估相对于工业级 hybrid search pipelines 的真实增量贡献。

**Weakness 4: Limited Technical Novelty and Incremental Contribution**
> "Combining vector retrieval with symbolic metadata filtering is a well-established engineering practice in modern information retrieval systems rather than a new architectural invention. The core contribution is effectively limited to applying existing reinforcement learning algorithms to this specific domain."

认为性能提升可能来自 domain-specific 训练而非方法论突破。

**Reproducibility**: 2 | **Datasets**: 1 (No usable datasets) | **Software**: 1 (No usable software)

**作者回应核心论点**:
1. Schema 预定义对邮件/文档管理等实际场景是合理假设，ArxivQA 跨域验证了泛化性
2. 承诺公开完整 prompt template 和 reward 计算细节
3. 承诺添加 RRF、Self-querying RAG baseline
4. 强调 DSL 的 **learnable composition**（可学习的逻辑组合）是与工业实践的核心区别
5. 展示了 OR + 多路语义交集的复杂 DSL 示例，证明"a fixed fusion pipeline cannot express such combinations"

---

### Reviewer 3avs — 最友好 (Overall: 4)

**Summary Of Strengths**:
> "The performance is impressive. In particular, evaluations on a large-scale industrial email benchmark demonstrate that DSL-R1 achieves a +12.3% improvement in Hit@1/3. Moreover, the framework (Figure 1) is conceptually clean."

**Summary Of Weaknesses**:
- Novelty 有限，主要是使用 GRPO 做 RL。虽然有些变化，但贡献较有限。
- 类似 idea 已在以下工作中探索：
  - Seq2SQL (Zhong et al., 2017): 用 RL 从自然语言生成结构化查询
  - ColPali (Faysse et al., 2024): 视觉语言模型的高效文档检索
- 需要更多分析说明性能提升来源

**Reproducibility**: 3 | **Soundness**: 4 (Strong)

**作者回应核心论点**:
- 三方面超越 GRPO 应用：(1) DSL 设计本身统一了逻辑和语义，(2) 四维 reward 针对可执行程序，(3) 小模型超越 GPT-4o 的实证发现
- Seq2SQL 仅处理结构化数据，ColPali 在表示层面操作，我们的 DSL 在程序层面操作
- 承诺添加：(a) error analysis，(b) complexity breakdown，(c) case studies

---

### Reviewer 13wi — 简短但精准的技术质疑 (Overall: 2.5)

**Summary Of Strengths**:
1. 整体 idea 合理。结构化和非结构化数据各有优劣，结合两者自然可以取长补短
2. 在两个 benchmark 上取得的准确率看起来很有前景

**Summary Of Weaknesses**:
1. **Prompt template 不清楚**
2. **SQL 和 embedding retrieval 的融合机制不清楚**
3. **四个 reward function 的计算方式和权重不清楚**
4. **GRPO/DAPO 详细公式占太多篇幅**，建议移到附录

**作者回应**（最详细的技术回复）:

**On fusion mechanism**: 用具体例子详细说明了两步执行流程：
1. Vector search → candidate IDs (e.g., `{msg_042, msg_107, msg_215, msg_318, msg_421}`)
2. Replace placeholder → execute SQL with injected IDs
3. "The logical combination between structured conditions and semantic matches is handled natively by SQL semantics (AND/OR/NOT), requiring no separate fusion strategy or score interpolation."

**On reward computation**: 给出了精确的表格定义（见原始论文分析文档）

**On GRPO/DAPO space**: 同意移到附录，正文保留 key intuition 和 differences

---

## 共性问题提炼

| 问题类别 | 提到的 Reviewer | 严重程度 | 改版中的解决方案 |
|---------|---------------|---------|---------------|
| Novelty 定位不清 | XRX8, 6rE4, 3avs | 致命 | 重新 framing 为 "retrieval orchestration" |
| 缺少 hybrid baseline | XRX8, 6rE4, AC | 致命 | 新增 RRF, Self-querying RAG, GPT-4o few-shot |
| Prompt template 未公开 | 6rE4, 13wi, AC | 严重 | 正文精简版 + appendix 完整版 |
| 缺少深度分析 | XRX8, 3avs, AC | 严重 | Action space ablation, complexity breakdown, orchestration analysis |
| Reward function 细节不足 | 13wi, AC | 中等 | 详细计算公式 + 代码片段 |
| 融合机制描述模糊 | 13wi | 中等 | 伪代码 + 执行示例 |
| 写作规范性 | XRX8 | 轻微 | 标准 IR notation, 流程图 |
| 泛化性/schema 依赖 | 6rE4 | 中等 | Limitations 中诚实讨论 |
| GRPO/DAPO 占篇幅过多 | 13wi | 轻微 | 移到 appendix |
| Related Work 不完整 | 隐含（全部） | 严重 | 新增 4 个 subsection，覆盖 RL retrieval + hybrid retrieval |
| 成本分析缺失 | AC | 中等 | 新增 cost-effectiveness 讨论 |

## 关键启示

1. **Reviewer XRX8 的建议实际上指明了方向**："a truly agentic retrieval system that can study the shape of the corpora and understand how to better plan the query" — 这正是 "retrieval orchestrator" 的核心 idea
2. **Reviewer 6rE4 的批评最致命但也最可修复**：4 个 weakness 中，transparency 和 baselines 是纯工作量问题，generalizability 可以通过 honest limitations 缓解，novelty 可以通过 reframing 解决
3. **Reviewer 3avs 已经认可了结果的强度**，只需要更多分析来 explain why
4. **Reviewer 13wi 的问题全部是 presentation 层面**，技术本身没有被质疑

**底线判断**：这篇论文的核心技术和实验结果是 solid 的，主要问题在 framing、presentation 和 baseline completeness 上。从 short paper 扩展为 long paper，配合 reframing + 新实验，有较好的改版前景。
