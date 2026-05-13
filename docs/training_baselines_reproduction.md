# Training Baselines 复现方案（详细调研版）

## 概述

根据论文体系，需要复现的训练类 baseline 有三类：
1. **DeepRetrieval** (B3) — RL 训练布尔查询生成
2. **Search-R1** (B2+) — RL 训练多轮搜索推理
3. **Text2VectorSQL** (B5-SFT) — SFT 训练 VectorSQL 生成

以及一个关键消融实验：
4. **E2 Action Space Ablation** — 验证完整 DSL 的必要性

---

## 1. DeepRetrieval 复现

### 论文信息
- **标题**: DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning
- **GitHub**: https://github.com/pat-jj/DeepRetrieval
- **Base Model**: Qwen2.5-3B-Instruct
- **RL 算法**: PPO (via veRL/verl 框架)
- **核心思想**: 训练 LLM 生成 Boolean Query（AND/OR/NOT），直接提交给搜索引擎

### 原始方法细节

**查询格式**:
```
("wireless" OR "bluetooth") AND "headphones" AND ("noise cancelling" OR "ANC") NOT "wired"
```

**Reward 设计**（分层）:
- Format reward: 语法正确 +1, 语法错误 -1, 生成空/重复 -4
- Retrieval reward (tiered):
  - Recall ≥ 0.6 → +4
  - Recall ≥ 0.4 → +2
  - Recall ≥ 0.2 → +1
  - Recall > 0 → +0.5
  - Recall = 0 → -1

**训练配置**:
- PPO, batch_size=256, KL coeff=0.001
- Max new tokens: 1024
- Temperature: 1.0
- 4× A100 80GB

### 适配到本项目的方案

**方案 A（推荐）: 直接在我们的 DSL 上用 PPO 替代 GRPO**

不需要改 DSL 格式，只需要将现有 GRPO 训练切换为 PPO，验证 RL 算法选择的影响。

```bash
# 修改现有 run_rl_grpo.sh 的 algorithm 参数
# verl 0.7.1 同时支持 PPO 和 GRPO
python3 -m verl.trainer.main_ppo \
  --data.train_files $DATA_DIR/product_rl_train.parquet \
  --data.val_files $DATA_DIR/product_rl_test.parquet \
  --actor_rollout_ref.model.path $SFT_MODEL \
  --critic.model.path $SFT_MODEL \
  --algorithm.kl_ctrl.kl_coeff 0.001 \
  --actor_rollout_ref.rollout.tensor_model_parallel_size 1 \
  ...
```

优点: 最公平的 ablation，仅变量为 RL 算法
缺点: PPO 需要额外的 critic model，GPU 显存需求更大

**方案 B: 复现 Boolean Query 范式**

训练模型生成布尔查询而非我们的 DSL，使用 BM25 执行。

1. 改训练数据：将 DSL 格式的 ground truth 转换为布尔查询格式
2. 改 reward function：用布尔查询在 BM25 上执行，计算 Recall
3. 改 system prompt：指导模型生成布尔查询

```python
# 布尔查询 reward (参照 DeepRetrieval)
def compute_boolean_reward(query_str, gt_ids, bm25_index):
    # Parse boolean query
    try:
        results = execute_boolean_query(query_str, bm25_index)
    except SyntaxError:
        return -1.0  # format error
    
    if not results:
        return -1.0
    
    recall = len(set(results) & set(gt_ids)) / len(gt_ids)
    if recall >= 0.6: return 4.0
    elif recall >= 0.4: return 2.0
    elif recall >= 0.2: return 1.0
    elif recall > 0: return 0.5
    else: return -1.0
```

优点: 真实复现 DeepRetrieval 的范式差异
缺点: 需要实现布尔查询执行引擎，工作量大

### 推荐
**方案 A**，因为：
- 和我们的方法只有一个变量差异（PPO vs GRPO/DAPO）
- Reviewer 最关心的是"为什么选 GRPO 不选 PPO"
- 实现成本低，verl 直接支持 PPO

---

## 2. Search-R1 复现

### 论文信息
- **标题**: Search-R1: Training LLMs to Reason and Retrieve with Reinforcement Learning
- **GitHub**: https://github.com/PeterGriffinJin/Search-R1
- **Base Model**: Qwen2.5-7B-Base (也支持 3B)
- **RL 算法**: PPO (7B) / GRPO (3B)
- **核心思想**: 多轮搜索推理，模型自主决定何时搜索、搜索什么

### 原始方法细节

**交互格式**:
```
<think>I need to find information about...</think>
<search>query text here</search>
<information>...search results injected here...</information>
<think>Based on the results...</think>
<search>refined query</search>
<information>...more results...</information>
<think>Now I can answer...</think>
<answer>final answer</answer>
```

**关键设计**:
- 最多 4 轮搜索
- 每轮搜索检索 top-3 documents
- Reward: 纯 outcome-based (Exact Match on final answer)
- 不使用 format reward 或过程 reward
- Rollout 时每个 `<search>` tag 触发实际检索

**训练配置**:
- GRPO for 3B: group_size=16, temperature=1.0
- PPO for 7B: KL coeff=0.005
- Max search rounds: 4
- Max tokens per round: 512

### 适配到本项目的方案

**核心差异**: Search-R1 是多轮 QA 范式（搜索→推理→搜索→回答），我们是单轮 DSL 生成范式。直接复现 Search-R1 需要彻底改变任务格式。

**方案（作为 B2+ baseline）**:

1. 改训练格式为多轮搜索：
```
<think>分析用户查询的需求...</think>
<search>第一轮搜索 query</search>
<information>[检索到的产品列表]</information>
<think>根据结果进一步 refine...</think>
<search>第二轮 refined query</search>
<information>[更精确的产品列表]</information>
<answer>product_id_1, product_id_2, ...</answer>
```

2. Reward: 直接用最终 answer 中的 product_id 与 gt_document_ids 计算 Hit@1/MRR

3. 搜索引擎: 使用现有的 BM25 + ColQwen (RRF)，在 rollout 时调用

**实现复杂度**: 高
- 需要修改 verl 的 rollout 流程支持多轮交互
- 需要实现 retrieval environment（在 rollout 中执行搜索）
- Search-R1 的代码有现成的 retrieval env 可参考

### 推荐
作为论文中的 **Related Work 讨论**而非实际 baseline 实验。原因：
- 范式差异太大（multi-round QA vs single-shot DSL），不是 apple-to-apple 比较
- 实现成本极高（需要改 rollout 流程）
- 如果 reviewer 要求，可以引用 Search-R1 论文中的结果，说明我们的范式在商品搜索场景更合适

---

## 3. Text2VectorSQL 复现

### 论文信息
- **标题**: Text2VectorSQL: Towards a Unified Interface for Vector Search and SQL Queries
- **arXiv**: 2506.23071
- **GitHub**: https://github.com/OpenDCAI/Text2VectorSQL
- **Base Model**: OmniSQL-7B/14B/32B (from Qwen2.5-Coder-Instruct)
- **训练方法**: 纯 SFT（无 RL）
- **开源模型**: UniVectorSQL-7B-LoRA-merged (HuggingFace: VectorSQL/UniVectorSQL-7B-LoRA-merged)

### 原始方法细节

**VectorSQL 格式**:
```sql
SELECT title, conference 
FROM papers 
WHERE conference = 'ICDE' AND publication_year > 2022 
ORDER BY abstract_embedding <-> EMBED('query optimization') 
LIMIT 5
```

**训练数据**: SynVecSQL-24K (4-stage pipeline: DB augmentation → complexity-aware synthesis → back-translation → CoT synthesis)

**SFT 超参数** (7B):
- Learning rate: 2e-5, full fine-tuning
- Batch size: 64, epochs: 2
- Context: 8192 tokens
- DeepSpeed ZeRO3, bfloat16
- 4× A100-SXM4-80GB, ~1 day

### 适配到本项目的方案

**核心结论: 我们已有的 SFT baseline 就是最公平的 Text2VectorSQL proxy。**

理由：
1. Text2VectorSQL 用的是纯 SFT，我们的 SFT 4B (Hit@1=0.680) 已经是 SFT 方法在我们数据上的最佳表现
2. 他们的 VectorSQL 语法和我们的 DSL 语法不同，无法直接迁移
3. 他们的评估用 BIRD/Spider benchmark，与我们的电商场景完全不同
4. 公平对比应在同一数据/格式下对比 SFT vs RL

**在论文中的呈现**:
- 引用 Text2VectorSQL 作为最相关的 SFT 工作
- 指出差异：(1) 我们支持图像向量检索, (2) 我们用 RL 而非 SFT, (3) 我们的 DSL 支持更复杂的逻辑组合
- 我们的 SFT baseline (Hit@1=0.680) 作为 SFT 上界，展示 RL (0.809) 的提升

---

## 4. E2 Action Space Ablation（优先级最高）

### 目标
验证完整 DSL（SQL + Text Vector + Image Vector）的必要性，通过限制 action space 进行消融。

### 五种 Action Space

| ID | Action Space | 允许的操作 | 预期影响 |
|----|-------------|-----------|---------|
| AS-1 | SQL Only | 仅 `sql` 字段，无 `retrieval_list` | 无法做语义匹配 |
| AS-2 | SQL + Text | `sql` + text retrieval | 无法利用图像 |
| AS-3 | Text Only | 仅 text retrieval，无 sql | 无法做结构化过滤 |
| AS-4 | Full (Ours) | sql + text + image retrieval | 完整能力 |
| AS-5 | SQL + Image | `sql` + image retrieval，无 text | 主力检索缺失 |

### 实现方式

对每种 action space，修改以下三处：
1. **System prompt**: 限制 DSL schema 中可用的字段/类型
2. **Reward function**: 在 DSL 执行时，忽略/清空被禁止的部分
3. **训练数据**: 过滤掉包含被禁止操作的 ground truth（可选）

**具体代码修改** (基于现有 `reward_dsl.py`):

```python
# 在 compute_score 函数中添加 action_space 过滤
def filter_dsl_by_action_space(dsl, action_space):
    if action_space == "sql_only":
        dsl.pop("retrieval_list", None)
    elif action_space == "sql_text":
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r["type"] == "text"]
    elif action_space == "text_only":
        dsl.pop("sql", None)
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r["type"] == "text"]
    elif action_space == "sql_image":
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r["type"] == "image"]
    # "full" = no filtering
    return dsl
```

### 训练配置

每种 action space 独立训练一个 GRPO 模型：
- Base: SFT 4B checkpoint
- 训练 300 steps（根据电商 GRPO 最佳 step）
- GPU: 4× (GPUs 3-6 on 102)
- 预计时间: ~4-6 小时/variant = 总共 ~24 小时

---

## 执行计划

### Phase 1: E2 Action Space Ablation（最可行，优先执行）
- 使用现有 GRPO 训练代码
- 只需修改 system prompt 和 reward function
- 在 102 服务器上执行（需要用户提供访问）

### Phase 2: DeepRetrieval 方案 A（PPO on our DSL）
- 修改 verl 训练配置从 GRPO → PPO
- 需要额外的 critic model GPU
- 在 102 服务器上执行

### Phase 3: 论文写作
- Text2VectorSQL: 作为 Related Work 引用，SFT baseline 作 proxy
- Search-R1: 作为 Related Work 讨论范式差异
- DeepRetrieval: 如果有 PPO 结果，作为 B3 baseline

### 当前阻塞
- **102 服务器无法从本地/ai4sci 访问**（内网 10.60.200.102）
- 需要用户从同网段机器操作，或提供跳板
