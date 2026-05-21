# B2 Search-R1 复现方案

> 优先级：P0 | 预估时间：5-6 天 | 硬件：2×48GB GPU (A100/H100)

---

## 一、原论文信息

- **论文**：Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **arXiv**：2503.09516
- **代码**：https://github.com/PeterGriffinJin/Search-R1
- **原始模型**：Qwen2.5-3B/7B/14B（base 和 instruct 均有）
- **原始数据**：NQ + HotpotQA（QA 数据集，HuggingFace: `PeterJinGo/nq_hotpotqa_train`）
- **原始检索器**：E5-base-v2 + FAISS flat index，topk=3
- **RL 算法**：GRPO，15 epoch，8×GPU
- **框架**：verl (vendored copy)，vllm==0.6.3

---

## 二、核心机制

### 2.1 多轮 Search-Reason 循环

LLM 在推理过程中交替生成思考和搜索指令：

```
Prompt → <think>推理</think> → <search>NL query</search>
→ [检索器返回 top-3] → <information>Doc1 Doc2 Doc3</information>
→ <think>继续推理</think> → <search>refined query</search>
→ [再次检索] → <information>...</information>
→ <think>最终推理</think> → <answer>P001</answer>
```

- 最多 `max_turns` 轮搜索（原文 v0.2 用 4 轮）
- `<information>` 块做 **state masking**，不参与梯度更新
- 每轮观测截断到 `max_obs_length=500` tokens
- 检索器是独立的 FastAPI 服务器，训练时通过 HTTP 调用

### 2.2 状态机（generation.py）

```
1. 初始化 prompt（截断到 max_start_length=2048）
2. For step in range(max_turns):
   a. 生成文本（vLLM rollout）
   b. 后处理：在第一个 </search> 或 </answer> 处截断
   c. 解析 action：regex r'<(search|answer)>(.*?)</\1>'
      - search → 发送到检索器，结果包装为 <information>...</information>
      - answer → 标记完成
      - 无效 → 注入错误提示继续
   d. 更新滚动上下文（左截断到 max_prompt_length=4096）
3. 最终生成（do_search=False，强制输出 answer）
```

### 2.3 Reward 函数

使用 `qa_em.py`（v0.2 版本，和论文一致）：

**answer 提取逻辑**（关键坑）：
- 用 `re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL)` 找所有匹配
- **需要 ≥2 个 `<answer>` 标签才返回结果**（第一个是 prompt 里的示例）
- 返回最后一个匹配的内容

**评分**：
- 提取到 answer + EM 匹配 GT → reward = 1.0
- 提取到 answer + 不匹配 → reward = 0.0（format_score）
- 未提取到 → reward = 0.0

**v0.3 format 版本**（qa_em_format.py）额外增加：
- 结构验证：检查标签状态机是否合法
- 分层评分：valid_format(0.2) + retrieval_correct(+0) + correct_answer(1.0)
- 我们用 v0.2 简单版本即可

---

## 三、公平复现方案

### 3.1 对齐策略

| 维度 | 原论文 | 我们的复现 | 公平性说明 |
|------|--------|-----------|-----------|
| Base model | Qwen2.5-7B | **Qwen3-4B** | 和 Ours 同模型，控制变量 |
| 训练流程 | base → RL | **SFT → RL** | 和 Ours 统一流程 |
| 训练数据 | NQ+HotpotQA (QA) | **我们的电商/邮件** | 必须用我们的数据 |
| 检索器 | E5-base-v2 | **ColQwen (Qwen3-Embedding-8B)** | 和 Ours 同检索器 |
| RL 算法 | GRPO | **GRPO** | 完全一致 |
| 任务定义 | QA（输出答案文本）| **检索（输出 product_id/email_id）** | 适配为检索任务 |

### 3.2 论文表述

> "For fair comparison, all RL-trained methods (Search-R1, SQL-R1, Ours) follow the same two-stage pipeline: (1) SFT on domain-specific training data to learn the output format, (2) GRPO on the same training queries. The only controlled variable is the action space."

---

## 四、Stage 1: SFT 数据构造

### 4.1 SFT 轨迹格式

从现有训练数据 + 检索器自动构造多轮搜索轨迹：

```
<think>
用户在找红色的Nike运动鞋，价格低于100美元。我需要搜索相关产品。
</think>
<search>red Nike running shoes under $100</search>
<information>
Doc 1(Title: Nike Air Max 90) Brand: Nike. Category: Running Shoes. Price: $89.99. Color: Red/White...
Doc 2(Title: Nike Revolution 6) Brand: Nike. Category: Running Shoes. Price: $65.00. Color: Red...
Doc 3(Title: Adidas Ultraboost) Brand: Adidas. Category: Running Shoes. Price: $95.00. Color: Red...
</information>
<think>
找到了几个结果。Doc 1 (Nike Air Max 90) 和 Doc 2 (Nike Revolution 6) 都符合要求：品牌是Nike，价格低于$100，红色。让我确认一下最匹配的。
</think>
<answer>P001</answer>
```

### 4.2 构造流程

```python
for query in training_queries:
    # 1. 用 NL query 调用 ColQwen 检索 top-3
    results = retriever.search(query["natural_language_query"], topk=3)
    
    # 2. 构造第一轮搜索轨迹
    think_1 = llm_generate_reasoning(query)  # 或模板生成
    search_query = query["natural_language_query"]  # 初始搜索用原始 query
    info = format_results(results)
    
    # 3. 构造最终推理 + answer
    gt_id = query["gt_document_ids"][0]
    think_2 = f"Based on the search results, {gt_id} matches the requirements."
    
    # 4. 组装完整轨迹
    trajectory = f"<think>\n{think_1}\n</think>\n<search>{search_query}</search>\n<information>\n{info}\n</information>\n<think>\n{think_2}\n</think>\n<answer>{gt_id}</answer>"
```

建议构造 **1000-2000 条** SFT 数据，覆盖各 level 和 modality。

### 4.3 SFT 训练配置

- Base: Qwen3-4B
- LR: 5e-5
- Epochs: 2-3
- 用 verl 的 `fsdp_sft_trainer.py` 或标准 SFT 脚本

---

## 五、Stage 2: RL 训练

### 5.1 Prompt 模板

```
Given a user query, find the most relevant product/email.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> P001 </answer>.
Query: {natural_language_query}
```

**注意**：prompt 里必须包含 `<answer> P001 </answer>` 示例，否则 extract_solution 找不到 ≥2 个 answer 标签，reward 永远为 0。

### 5.2 训练数据 Parquet 格式

```python
{
    "data_source": "ecommerce",  # 或 "email"
    "prompt": [{"role": "user", "content": "<上述 prompt 模板>"}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["P001", "P002", "P006"]}  # gt_document_ids
    },
    "extra_info": {"split": "train", "index": 0}
}
```

### 5.3 Reward 函数适配

在 `main_ppo.py` 的 `_select_rm_score_fn()` 中注册：

```python
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'hotpotqa', 'ecommerce', 'email']:
        return qa_em.compute_score_em
    ...
```

answer 提取后与 `ground_truth["target"]` 列表做 EM 匹配，任一命中即 reward=1.0。

### 5.4 检索服务器

包装现有 ColQwen 为 FastAPI 服务：

```python
# retrieval_server.py
from fastapi import FastAPI
app = FastAPI()

@app.post("/retrieve")
def retrieve(request):
    # request.queries: List[str], request.topk: int
    results = []
    for query in request.queries:
        docs = colqwen_retriever.search(query, topk=request.topk)
        formatted = [{
            "document": {"id": d["id"], "contents": f"{d['title']}\n{d['text']}"},
            "score": d["score"]
        } for d in docs]
        results.append(formatted)
    return {"result": results}
```

Corpus JSONL 格式：
```jsonl
{"id": "P001", "contents": "Nike Air Max 90\nBrand: Nike. Category: Running Shoes. Price: $89.99. Rating: 4.5..."}
```

### 5.5 训练超参（2×48GB）

| 参数 | 原文(8 GPU) | 我们(2 GPU) |
|------|------------|------------|
| `train_batch_size` | 512 | **64** |
| `ppo_mini_batch_size` | 256 | **32** |
| `ppo_micro_batch_size` | 64 | **8** |
| `n_agent` (GRPO group size) | 5 | **3** |
| `lr` | 1e-6 | **1e-6** |
| `lr_warmup_steps_ratio` | 0.285 | 0.285 |
| `kl_loss_coef` | 0.001 | 0.001 |
| `kl_loss_type` | low_var_kl | low_var_kl |
| `clip_ratio` | 0.2 | 0.2 |
| `entropy_coeff` | 0.001 | 0.001 |
| `temperature` | 1.0 | 1.0 |
| `max_prompt_length` | 4096 | 4096 |
| `max_response_length` | 500 | 500 |
| `max_start_length` | 2048 | 2048 |
| `max_obs_length` | 500 | 500 |
| `total_epochs` | 15 | **15** |
| `total_training_steps` | 1005 | 按数据量调整 |
| `tensor_model_parallel_size` | 1 | **1** |
| `gpu_memory_utilization` | 0.6 | **0.5** |
| `param_offload` | True | True |
| `grad_offload` | True | True |
| `optimizer_offload` | True | True |
| `state_masking` | True | True |
| `gradient_checkpointing` | True | True |
| `retriever.topk` | 3 | 3 |
| `max_turns` | 4 | **3** |
| `save_freq` | 100 | 50 |
| `test_freq` | 100 | 50 |

### 5.6 训练脚本

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files=data/ecommerce_search/train.parquet \
    data.val_files=data/ecommerce_search/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=checkpoints/search_r1_sft \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3
```

---

## 六、Qwen3-4B 兼容性注意事项

### 6.1 `<think>` 标签冲突

Qwen3 有原生 thinking mode，会和 Search-R1 的 `<think>` 标签冲突。

**解决方案**：
- 使用 Qwen3-4B base 模型（非 instruct）
- 或显式关闭 thinking mode：`enable_thinking=False`
- 验证：确保模型不会自动进入 thinking mode 产生双重 `<think>` 标签

### 6.2 Chat template

- Base 模型通常没有 `chat_template`，verl 代码会 fallback 到直接使用 prompt content（`rl_dataset.py` line 131）
- 如果 Qwen3-4B base 有 chat_template，需要验证生成的格式是否正确
- `qa_em_format.py` 中的 `is_valid_sequence()` 硬编码了 `<|im_start|>assistant`——如果用 v0.2 的 `qa_em.py` 则无此问题

### 6.3 Tokenizer

```python
# 验证特殊标签能正确 tokenize
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
for tag in ["<think>", "</think>", "<search>", "</search>", "<answer>", "</answer>", "<information>", "</information>"]:
    ids = tok.encode(tag)
    assert tok.decode(ids).strip() == tag, f"Tag {tag} roundtrip failed"

# 确保 pad_token_id
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
```

### 6.4 vLLM 兼容性

- 原代码固定 vllm==0.6.3
- Qwen3-4B 可能需要更新版 vLLM
- 如果 0.6.3 不支持 Qwen3，需要升级 vLLM 并检查 verl 集成代码兼容性
- 必须设置 `VLLM_ATTENTION_BACKEND=XFORMERS`

---

## 七、评估

### 7.1 推理

设置 `+trainer.val_only=true` 使用 verl 的验证循环，或编写独立推理脚本。

推理时：
- temperature=0（greedy decoding）
- max_turns=3（和训练一致）
- 检索器必须在线

### 7.2 指标

使用我们现有的 eval 脚本，报告：
- Hit@1, Hit@3, MRR, Recall@20（和 Ours 同指标）
- 按 Level (L1/L2/L3) 分层
- 按 Modality (structured/text/image) 分类
- 额外报告：平均搜索轮次、有效格式比例

### 7.3 预期结果

- Search-R1 在纯文本语义查询上可能接近 Ours
- 在结构化字段过滤（price<100, brand='Nike'）上应明显弱于 Ours（NL query 无法精确表达）
- 在图像查询上应明显弱于 Ours（NL → text retrieval 无法做 image similarity）
- 在复杂逻辑组合（OR/NOT/嵌套）上应显著弱于 Ours

---

## 八、时间表

| 天数 | 任务 |
|------|------|
| Day 1 | 环境搭建：clone Search-R1，安装 verl/vllm，验证 Qwen3-4B 兼容性 |
| Day 1 | 搭建 ColQwen FastAPI 检索服务器 |
| Day 2 | SFT 数据构造（自动生成搜索轨迹）+ SFT 训练 |
| Day 2 | RL 训练数据转 parquet + reward 函数适配 |
| Day 3-4 | RL 训练（~15h） |
| Day 5 | 评估 + 按 Level/Modality 分析 |

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| vllm==0.6.3 不支持 Qwen3-4B | 升级 vllm 并修改 verl 集成代码；或用 Qwen2.5-3B 作 fallback |
| 2 卡 OOM | 降 n_agent 到 2，降 batch_size 到 32 |
| Qwen3 thinking mode 冲突 | 用 base 模型 + 关闭 thinking |
| 多轮生成+检索延迟高 | 检索服务器预热 + batch 检索 |
| RL 训不起来（reward 不涨） | 检查 answer 提取逻辑（≥2 个标签）；增加 SFT 数据量 |
