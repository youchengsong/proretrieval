# 中稿概率评估与改进空间调研

> 基于 2026 年 3 月最新竞品分析，针对 EMNLP 2026 ARR May cycle

---

## 1. 竞争格局（2026年3月最新）

| 竞品论文 | RL? | 结构化查询? | 向量检索? | 会议 |
|---------|-----|-----------|---------|------|
| Search-R1 (Jin et al.) | ✓ | ✗ (纯文本) | ✗ | COLM 2025 |
| DeepRetrieval (Jiang et al.) | ✓ | ✓ (boolean/SQL-like) | ✗ | COLM 2025 |
| s3 (Jiang et al.) | ✓ | ✗ | ✗ | EMNLP 2025 Main |
| R3-RAG | ✓ | ✗ | ✗ | EMNLP 2025 Findings |
| MCTS-RAG (Yale NLP) | ✗ (MCTS) | ✗ | ✗ | EMNLP 2025 Findings |
| R1-Searcher (RUCAIBox) | ✓ | ✗ | ✗ | arXiv preprint |
| Text2VectorSQL | ✗ (SFT/prompting) | ✓ (SQL) | ✓ | arXiv preprint |
| SUQL (Stanford) | ✗ (prompting) | ✓ (SQL+语义) | ✗ | NAACL 2024 Findings |
| SQL-R1 | ✓ | ✓ (SQL) | ✗ | arXiv preprint |
| **LocalSearch-R1** | **✓** | **✓ (完整SQL)** | **✓** | — |

**核心结论：没有现有论文同时具备 RL + SQL + 向量检索的统一编排。这个 gap 是真实存在的。**

---

## 2. EMNLP 录取率趋势

| 年份 | 投稿数 | 主会录取率 | Findings 录取率 | 合计 |
|------|--------|-----------|---------------|------|
| 2023 | 4,909 | ~41% | — | ~41% |
| 2024 | 6,395 | ~37% | — | ~37% |
| 2025 | 8,172 | **~22%** | ~17% | ~39% |

2025 年主会录取率暴跌至 22%，竞争显著加剧。

---

## 3. 中稿概率评估

### 场景1：按 03 文档改进方案全部执行

**主会概率：25-30%**
**Findings 概率：35-45%**
**主会 + Findings 综合录取概率：~50-55%**

### 场景2：只做部分改进（缺少 Action Space Ablation 或 Complexity Breakdown）

**主会概率：10-15%**
**Findings 概率：25-30%**

Action space ablation 是整个论证链条的支柱，缺了它 novelty claim 就是空谈。

---

## 4. 致命风险（任一项可直接导致拒稿）

1. **Action space ablation 结果不理想** — 如果 DSL (SQL+vector) 只比 text query 或 pure SQL 好 1-2 个点，reviewer 会认为 DSL 的额外复杂度不值得

2. **对 DeepRetrieval 描述不准确** — DeepRetrieval 输出的是结构化 boolean/SQL 查询，不是纯文本。如果论文暗示它只输出文本，会被 reviewer 视为不诚实，直接差评

3. **缺少 hybrid baseline** — 这是上一轮 4 个 reviewer 中 3 个的共识，不补就是自杀

---

## 5. 最能提升概率的动作（按 ROI 排序）

| 优先级 | 动作 | 预期概率提升 | 工作量 |
|--------|------|-----------|--------|
| **P0** | Action space ablation（同模型同算法，只改动作空间）| +10-15% | 2-3天 |
| **P0** | 补 RRF + Self-querying RAG baseline | +8-10% | 1-2天 |
| **P0** | GPT-4o few-shot DSL generation baseline | +5% | 1天 |
| **P1** | 按逻辑复杂度分层的 performance breakdown | +5-8% | 1-2天 |
| **P1** | Orchestration 行为分析（训练前后对比） | +3-5% | 1-2天 |
| **P1** | 补充 Text2VectorSQL 作为 baseline | +3-5% | 1-2天 |
| **P2** | Error analysis + case study | +3% | 1天 |
| **P2** | Model size scaling (1.5B/14B) | +2% | 3-5天 |

---

## 6. 进一步改进空间

### 6.1 Retrieved Token Masking（来自 Search-R1）

Search-R1 提出的 retrieved token masking 技术已被证明可以稳定 RL 训练——在 RL 梯度更新时屏蔽掉"检索到的 token"（如向量检索返回的 ID 列表、SQL 执行结果），防止这些外部信息污染梯度。

在 LocalSearch-R1 中，DSL 执行时注入的向量候选 ID（`IN ('msg_042','msg_107',...)`）也属于检索到的 token，当前论文未讨论是否对其 masking。

**建议**：在 RL 训练中引入 retrieved token masking，报告 with/without masking 的对比。这同时是技术改进，也显示了对最新工作的深入理解。

### 6.2 Curriculum Learning（从简单到复杂逻辑）

当前所有训练样本等权采样。考虑 DSL 有明确的复杂度层次，可以引入 difficulty-aware curriculum：

- Phase 1: 简单 AND 查询（纯结构化或单路语义）
- Phase 2: 多路向量检索组合
- Phase 3: OR/NOT/嵌套逻辑

工作量约 2-3 天，如果能证明 curriculum 进一步提升 2-3 个点，成为额外的技术贡献。参考：UR2、ZeroSearch 均采用类似策略。

### 6.3 更多域的泛化验证

当前只有 email + ArxivQA 两个域，是 reviewer 6rE4 的核心质疑之一。建议新增一个域：

- **产品文档/技术手册**：有结构化字段（产品名、版本号、分类）+ 非结构化内容（描述、FAQ），与 email 域差异大
- **个人文件系统**：文件名、创建时间、文件类型（结构化）+ 文件内容（非结构化），贴合 "local search" 主题

从 2 → 3 个域，在 reviewer 眼中的可信度提升是非线性的。

### 6.4 与 SQL-R1 正面对标

最新出现的 SQL-R1（arXiv preprint, 2025）用 RL 训练 NL-to-SQL，7B 超越更大模型。这与 LocalSearch-R1 的 SQL 部分直接相关：

- 在 Related Work 中讨论 SQL-R1
- 指出 SQL-R1 只处理纯结构化查询，不支持向量检索
- 在 action space ablation 中包含 "SQL-only（类 SQL-R1）" 对比行

### 6.5 与 Text2VectorSQL 的正面对比

Text2VectorSQL 是最直接的 DSL 设计竞品（SQL + 向量检索），但用 prompting/SFT 而非 RL。这是完美的 ablation：**同样的 DSL 设计，RL vs SFT 的差距有多大？**

如果能证明 RL 显著优于 SFT 在 DSL 生成上的效果，直接回应了 "RL 是否必要" 的质疑。

### 6.6 端到端延迟的真实 Edge 部署数据

当前延迟数据在 A100 上测量。如果能补充在消费级硬件（如 MacBook M-series 或中端 GPU）上的延迟，会加强 "local search" 的定位可信度。诚实报告即使延迟变高，也比回避强——reviewer 关注的是真实性，不只是数字。

---

## 7. 综合建议

### 必须做（P0）

1. **Action space ablation** — 整个论文论证核心，缺失则主会概率 <15%
2. **补齐 hybrid baselines** — RRF, Self-querying RAG, GPT-4o few-shot DSL
3. **准确描述 DeepRetrieval 和 Text2VectorSQL** — 不要让 reviewer 认为没认真读竞品

### 强烈推荐（P1）

4. 按逻辑复杂度分层报告性能
5. Orchestration 行为分析（训练前 vs 训练后）
6. 引入 Retrieved Token Masking 并消融

### 时间紧张时的取舍

如果 9.5 周内做不完所有实验，优先级排序：
**Action space ablation > hybrid baselines > complexity breakdown > orchestration analysis > 其余**

时间预警：实验如果有 GPU 不够、baseline 复现困难等 blocker，会非常紧张。建议第一周就启动 P0 实验，写作任务后移。
