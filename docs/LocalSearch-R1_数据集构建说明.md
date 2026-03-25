# LocalSearch-R1 数据集构建说明

> 本文档说明数据集的构建策略，以及如何回应 reviewer 对"自建数据集"的质疑。

---

## 数据集概览

| 数据集 | 语料来源 | Query/GT | 规模 |
|--------|----------|----------|------|
| 电商（Ecommerce）| Amazon ESCI（公开）| 自动构造 | 13,334 商品，10,000 queries |
| 邮件（Email）| Enron Email Corpus（公开）| 自动构造 | TBD |

---

## 为什么不用现有 benchmark？

### 调研过的公开邮件检索数据集

| 数据集 | 问题 |
|--------|------|
| TREC Enterprise Track 2005/2006（W3C 语料）| Query 仅 50~100 条，数量太少；且 W3C 语料需联系 NIST 申请，获取有门槛 |
| Avocado Research Email Collection | 无官方 query/GT；需签署双重协议 |
| TREC 2007 Spam Corpus | 分类任务，不是检索任务 |
| Enron（裸数据）| 语料公开易得，但无官方检索 query/GT |

**结论**：目前没有任何公开 benchmark 覆盖"结构化字段 + 语义检索"组合的 DSL 检索任务，因此需要自建。

---

## 如何回应 Reviewer 质疑

### 质疑：「实验只在自建数据集上进行，缺乏客观性」

**回应策略（可直接用于 rebuttal）**：

**1. 语料库是公开的**

> *"Our corpora are drawn from publicly available datasets: the Amazon ESCI dataset for e-commerce and the Enron Email Corpus for enterprise email. Both are widely used in the research community. The query construction pipeline, all code, and the resulting datasets will be fully released to ensure reproducibility."*

**2. 这个任务本身没有现成 benchmark**

> *"To the best of our knowledge, no existing benchmark covers structured-semantic hybrid retrieval with executable DSL programs combining SQL and vector retrieval. Constructing evaluation datasets is therefore necessary, as is common practice when introducing a new task (e.g., HotpotQA, WebQA, BEIR)."*

**3. GT 是确定性计算得到的，不是主观标注**

> *"Ground truth is determined by executing DSL queries against the corpus, making annotation deterministic and reproducible rather than subject to human judgment variability. This is more objective than crowdsourced relevance judgments."*

**4. 两个独立域，不是单一自建集**

> *"We evaluate on two independent domains—e-commerce and enterprise email—to demonstrate cross-domain generalization. The two domains differ substantially in vocabulary, query structure, and field semantics."*

**5. 所有 baseline 在同一数据集上公平对比**

> *"All baseline methods are evaluated on the same datasets under identical conditions. No dataset-specific tuning was applied to favor our approach."*

---

## 数据集构建的可信度保障

| 保障措施 | 说明 |
|----------|------|
| **语料公开** | Enron、Amazon ESCI 均为学界公认公开数据集 |
| **Pipeline 透明** | 9步构造流程完整公开，参数全部记录在文档中 |
| **GT 确定性** | 执行 DSL 程序得到 GT，非人工打标，可复现 |
| **代码开源** | 全部构造脚本将随论文发布 |
| **双域验证** | 电商 + 邮件两个独立域，防止过拟合单一场景 |
| **人工抽查（可选）**| 可从 query 中随机抽取 100 条进行人工验证 GT 质量，报告 Precision@GT |

---

## 参考：类似做法的论文

- **HotpotQA**（Yang et al., 2018）：基于 Wikipedia 自动构造多跳问答数据集
- **BEIR**（Thakur et al., 2021）：从多个公开语料自动构造零样本检索 benchmark
- **WebQA**（Chang et al., 2022）：基于公开网页自动构造多模态问答数据集

这些工作均采用"公开语料 + 自动构造 query/GT"的方式，被顶会广泛接受。
