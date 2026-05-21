# B3 SQL-R1 复现方案

> 优先级：P1 | 预估时间：4-5 天 | 硬件：2×48GB GPU (A100/H100)

---

## 一、原论文信息

- **论文**：SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning
- **arXiv**：2504.08600（NeurIPS 2025）
- **代码**：https://github.com/IDEA-FinAI/SQL-R1
- **原始模型**：Qwen2.5-Coder-3B/7B/14B-Instruct
- **原始数据**：
  - SFT 冷启动：SynSQL-200K（从 SynSQL-2.5M 采样，4 个难度等级各 50K）
  - RL 训练：SynSQL-Complex-5K（仅复杂样本）
  - 评估：Spider-Dev, BIRD-Dev
- **RL 算法**：GRPO，10 epoch，8×80GB GPU
- **框架**：verl v0.1 (vendored copy)，vllm==0.6.3
- **训练流程**：SFT → RL 两阶段

---

## 二、核心机制

### 2.1 单轮 SQL 生成

与 Search-R1 的多轮不同，SQL-R1 是单轮生成：

```
<think>
用户想找价格低于100的Nike品牌产品...
需要对 products 表做 WHERE 过滤...
</think>
<answer>
```sql
SELECT DISTINCT product_id FROM products WHERE brand = 'Nike' AND price < 100;
```
</answer>
```

### 2.2 Reward 函数（4 维）

| 分量 | 正分 | 负分 | 逻辑 |
|------|------|------|------|
| **Format** | +1 | -1 | 检查标签合法性：恰好 1 个 `<think>`/`</think>`/`<answer>`/`</answer>`，顺序正确，`<answer>` 内有 ` ```sql ``` ` code block |
| **Execution** | +2 | -2 | SQL 在 SQLite 上执行成功（30s 超时）/失败 |
| **Result Match** | +3 | -3 | 预测 SQL 执行结果与 GT SQL 执行结果做集合比较 |
| **Length Bonus** | 0~1.5 | 0 | 仅在全部正确时：鼓励简洁 SQL 和高效推理 |

总分范围：-3（格式错+不可执行）到 ~7.5（全对+简洁奖励）

### 2.3 Result Match 详细逻辑（exec_eval.py）

1. 对预测 SQL 和 GT SQL 都执行，获取结果集
2. 比较逻辑：
   - 列顺序无关：尝试列排列找到匹配（≤3 列全排列，>3 列采样 20 行约束排列空间）
   - 行顺序无关：除非 GT SQL 含 `ORDER BY`
   - 用 bag semantics（multiset equality）
   - `keep_distinct=False`：去掉两边的 DISTINCT
3. 超时 60s

### 2.4 Format 验证详细逻辑（synsql.py）

```python
def validate_response_structure(answer_text, processed_str):
    # 检查标签数量
    assert processed_str.count('<think>') == 1
    assert processed_str.count('</think>') == 1
    assert processed_str.count('<answer>') == 1
    assert processed_str.count('</answer>') == 1
    # 检查顺序
    assert index('<think>') < index('</think>') < index('<answer>') < index('</answer>')
    # 检查 SQL code block
    assert '```sql' in answer_text and '```' appears after sql block
    # 提取 SQL
    sql = extract_between('```sql', '```', answer_text)
    return sql, True
```

### 2.5 Length Bonus 计算

```python
LIMIT_LENGTH = 2048
pos_length = len(think_text) + len(answer_text)

if pos_length <= LIMIT_LENGTH:
    sql_ratio = len(pred_sql) / len(answer_text)  # SQL 占 answer 的比例
    length_ratio = pos_length / LIMIT_LENGTH * 0.5  # 使用了多少 token budget
    length_score = length_ratio + sql_ratio
else:
    length_score = 0.5 + len(pred_sql) / len(answer_text)
```

---

## 三、公平复现方案

### 3.1 对齐策略

| 维度 | 原论文 | 我们的复现 | 公平性说明 |
|------|--------|-----------|-----------|
| Base model | Qwen2.5-Coder-7B-Instruct | **Qwen3-4B** | 和 Ours 同模型，控制变量 |
| 训练流程 | SFT (200K) → RL (5K) | **SFT → RL** | 统一两阶段流程 |
| SFT 数据 | SynSQL-200K（多表 NL2SQL）| **我们自己的电商/邮件数据** | 用我们的数据，避免额外数据优势 |
| RL 数据 | SynSQL-Complex-5K | **我们的训练集** | 和 Ours 同训练数据 |
| 数据库 | 多表 JOIN (Spider/BIRD schema) | **单表** | 我们的 schema 是单表，SQL 更简单 |
| RL 算法 | GRPO | **GRPO** | 完全一致 |
| Reward | 4 维（格式+执行+结果+长度）| **4 维（完全复用）** | 直接复用代码 |

### 3.2 核心限制（B3 vs B5 对比点）

SQL-R1 **只能做结构化字段过滤**，无法做：
- 文本语义检索（`id IN <text_N>` → 无法表达）
- 图像语义检索（`id IN <image_N>` → 无法表达）
- 多模态组合检索

这正是论文要证明的：B5 (DSL = SQL + text + image) > B3 (pure SQL)

---

## 四、Stage 1: SFT 数据构造

### 4.1 从现有 gold_dsl 提取纯 SQL

```python
def extract_pure_sql(gold_dsl):
    """从 gold_dsl 中提取纯 SQL 条件，去掉向量检索部分"""
    sql = gold_dsl["sql"]
    # 去掉 id IN <text_N> 和 id IN <image_N> 条件
    # 去掉 id NOT IN <text_N> 和 id NOT IN <image_N> 条件
    # 保留纯结构化条件：brand=, price<, category=, rating> 等
    # 如果去掉向量条件后 WHERE 为空，则退化为 SELECT * FROM table
    cleaned_sql = remove_vector_conditions(sql)
    return cleaned_sql
```

### 4.2 SFT 数据格式

```python
{
    "prompt": "<think>\n",  # 模型从 <think> 开始生成
    "completion": """分析用户查询，需要找价格低于100的Nike品牌产品。
需要对 products 表的 brand 和 price 字段进行过滤。
</think>
<answer>
```sql
SELECT DISTINCT product_id FROM products WHERE brand = 'Nike' AND price < 100;
```
</answer>"""
}
```

### 4.3 Prompt 模板

```
<|im_start|>system
You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question and database schema, then provides the final SQL query.
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL query within ```sql ``` tags.

Database Schema:
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    brand TEXT,
    category TEXT,
    price REAL,
    rating REAL,
    num_reviews INTEGER,
    in_stock INTEGER,
    release_date TEXT,
    title TEXT,
    description TEXT,
    review_summary TEXT
);

For example:
<think>
The user wants to find products from a specific brand within a price range.
I need to filter the products table by brand and price.
</think>
<answer>
```sql
SELECT DISTINCT product_id FROM products WHERE brand = 'Sony' AND price < 200;
```
</answer>
<|im_end|>
<|im_start|>user
{natural_language_query}
<|im_end|>
<|im_start|>assistant
<think>
```

邮件场景 schema：
```sql
CREATE TABLE emails (
    message_id TEXT PRIMARY KEY,
    received_date TEXT,
    is_draft INTEGER,
    is_read INTEGER,
    is_starred INTEGER,
    is_archived INTEGER,
    thread_msg_count INTEGER,
    sender_name TEXT,
    recipient_list TEXT,
    folder_labels TEXT,
    subject TEXT,
    content TEXT,
    attachment_list TEXT
);
```

### 4.4 SFT 训练配置

- Base: Qwen3-4B
- LR: 5e-5
- Epochs: 2-3
- 数据量：1000-2000 条（从训练集采样）

---

## 五、Stage 2: RL 训练

### 5.1 训练数据 Parquet 格式

```python
{
    "data_source": "synsql",  # 必须包含 "synsql"，reward 函数靠这个 dispatch
    "prompt": [{"role": "user", "content": "<上述 prompt 模板>"}],
    "ability": "logic",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "db_id": "ecommerce",  # 映射到 SQLite 文件路径
            "sql": "SELECT DISTINCT product_id FROM products WHERE brand = 'Nike' AND price < 100"
        }
    },
    "extra_info": {"split": "train", "index": 0}
}
```

### 5.2 代码修改

**必须修改的文件**：

1. **`verl/utils/reward_score/synsql.py` line 153** — 修改 db_path：
```python
# 原代码
db_path = os.path.join('data/NL2SQL/SynSQL-2.5M/databases', db_name, db_name + '.sqlite')
# 改为
db_path = os.path.join('data/databases', db_name, db_name + '.sqlite')
```

2. **`verl/trainer/main_ppo.py`** — 确保 data_source 路由正确：
```python
def _select_rm_score_fn(data_source):
    if 'synsql' in data_source:  # 原代码已有
        return synsql.compute_score
```

3. **准备 SQLite 数据库文件**：
```
data/databases/ecommerce/ecommerce.sqlite  # 含 products 表
data/databases/email/email.sqlite          # 含 emails 表
```

### 5.3 训练超参（2×48GB）

| 参数 | 原文(8×80GB) | 我们(2×48GB) |
|------|-------------|-------------|
| `train_batch_size` | 8 | **4** |
| `ppo_mini_batch_size` | 8 | **4** |
| `ppo_micro_batch_size` | 8 | **2** |
| `rollout.n` (GRPO group size) | 8 | **4** |
| `lr` | 3e-7 | **3e-7** |
| `kl_loss_coef` | 0.001 | 0.001 |
| `kl_loss_type` | low_var_kl | low_var_kl |
| `clip_ratio` | 0.2 | 0.2 |
| `entropy_coeff` | 0.001 | 0.001 |
| `temperature` | 1.1 (训练) | **1.1** |
| `max_prompt_length` | 4096 | **2048**（单表 schema 更短） |
| `max_response_length` | 2048 | **1024**（SQL 更简单） |
| `total_epochs` | 10 | **8** |
| `tensor_model_parallel_size` | 4 | **2** |
| `gpu_memory_utilization` | 0.2 | **0.4** |
| `param_offload` | True | True |
| `grad_offload` | True | True |
| `optimizer_offload` | True | True |
| `ref.param_offload` | True | True |
| `gradient_checkpointing` | True | True |
| `save_freq` | 100 | 50 |
| `test_freq` | 100 | 50 |

**关键差异说明**：
- `temperature=1.1`：原代码实际用 1.1（论文说 0.8 是推理时的），训练用 1.1 鼓励探索
- `max_prompt_length=2048`：我们单表 schema，CREATE TABLE 很短，不需要 4096
- `max_response_length=1024`：单表 SQL 不需要 2048 的 budget

### 5.4 训练脚本

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/ecommerce_sql/train.parquet \
    data.val_files=data/ecommerce_sql/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=checkpoints/sql_r1_sft \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=8 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.critic_warmup=0
```

---

## 六、Qwen3-4B 兼容性注意事项

### 6.1 Thinking mode 冲突

与 Search-R1 同样的问题。SQL-R1 显式管理 `<think>`/`<answer>` 标签，format 验证会检查恰好 1 个 `<think>` 标签。如果 Qwen3 自动进入 thinking mode，会产生额外的 `<think>` 标签，format reward 永远 -1。

**解决方案**：
- 使用 SFT 后的 checkpoint（SFT 已经教会了正确的输出格式）
- 在 generation config 中设置 `enable_thinking=False`

### 6.2 Instruct vs Base

原论文用 Instruct 模型。但我们统一用 base → SFT → RL。SFT 阶段会教会模型：
- ChatML 格式（`<|im_start|>` / `<|im_end|>`）
- `<think>`/`<answer>` 标签格式
- SQL code block 格式

### 6.3 Tokenizer

确保 SQL 关键字和特殊标签能正确 tokenize/decode：
```python
test_sql = "SELECT DISTINCT product_id FROM products WHERE brand = 'Nike' AND price < 100;"
test_tags = "<think>\n分析...\n</think>\n<answer>\n```sql\n" + test_sql + "\n```\n</answer>"
ids = tokenizer.encode(test_tags)
assert test_sql in tokenizer.decode(ids)
```

---

## 七、评估

### 7.1 推理配置

- temperature=0.8（原论文推理设置）或 greedy (temperature=0)
- 无多轮（单轮生成）

### 7.2 评估流程

1. 模型生成 SQL
2. 在 SQLite 上执行 SQL，获取 product_id 列表
3. 与 GT 做 Hit@1, Hit@3, MRR, Recall@20（和 Ours 同指标）

**注意**：SQL-R1 的检索结果是 SQL 执行返回的集合（无排序），而 Ours 返回有序排名。对于 Hit@1：
- SQL-R1：如果执行结果恰好只有 1 个且命中 GT → Hit@1=1
- SQL-R1：如果执行结果有多个，取第一个（按 product_id 排序或随机）
- 需要定义清楚如何从 SQL 结果集转换为排序列表

建议：SQL 执行返回的所有 product_id 视为 top-K 结果，不区分顺序。

### 7.3 按维度分析

- **按 Level**：L1（简单，SQL-R1 应接近 Ours）→ L3（复杂，SQL-R1 应明显弱）
- **按 Modality**：
  - structured：SQL-R1 应接近 Ours（都能精确过滤）
  - text：SQL-R1 应明显弱（无法做语义检索，只能做 LIKE 模糊匹配）
  - image：SQL-R1 完全无法处理

### 7.4 预期结果

| 维度 | SQL-R1 vs Ours |
|------|---------------|
| Overall | SQL-R1 明显弱 |
| Structured-only queries | 接近 |
| Text queries | SQL-R1 弱（无语义检索） |
| Image queries | SQL-R1 完全失败 |
| L1 (简单) | 差距小 |
| L3 (复杂 OR/NOT/嵌套) | 差距大 |

---

## 八、时间表

| 天数 | 任务 |
|------|------|
| Day 1 | 准备 SQLite 数据库 + 从 gold_dsl 提取纯 SQL GT |
| Day 1 | SFT 数据构造 + SFT 训练（复用 Search-R1 环境） |
| Day 2 | RL 训练数据转 parquet + 修改 db_path |
| Day 2-3 | RL 训练（~10h） |
| Day 4 | 评估 + 分析 |

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| 单表 SQL 太简单，模型瞬间收敛 | 减少 epoch（8→5），早停 |
| 2 卡 OOM | 降 rollout.n 到 2，降 batch 到 2 |
| Qwen3 thinking mode 冲突 | SFT 教正确格式 + 关闭 thinking |
| SQL 执行超时（复杂 query） | 已有 30s 超时机制 |
| GT SQL 去掉向量条件后 WHERE 为空 | 退化为 SELECT *，对这类 query 报告 SQL-R1 无法处理 |
| vllm 0.6.3 + Qwen3 不兼容 | 和 Search-R1 共用解决方案 |

---

## 十、与 E2 Action Space Ablation 的关系

E2 实验的 "SQL only" 行与 SQL-R1 baseline 的区别：

| 维度 | E2 "SQL only" | SQL-R1 baseline |
|------|--------------|-----------------|
| 训练方法 | 我们自己的 RL pipeline | SQL-R1 的 RL pipeline |
| Reward | 我们的 4 维 reward | SQL-R1 的 4 维 reward |
| 外部方法 | 否（内部 ablation） | 是（外部 SOTA 复现） |

两者互补但不可替代：
- E2 证明 "在我们的框架内，完整 DSL > pure SQL"
- SQL-R1 证明 "即使用外部 SOTA 的 SQL RL 方法，仍然不如我们的 DSL"
