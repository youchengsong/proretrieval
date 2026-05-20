# DeepRetrieval Baseline (B3) — 电商场景复现

DeepRetrieval 是基于 BM25 的检索基线方法。模型通过 RL (GRPO) 学习为用户查询生成优化的 Boolean 检索 query，然后用 Pyserini/Lucene 执行 BM25 检索，以 NDCG@k 作为 reward 信号。

## 方法概述

- **检索方式**: Pyserini LuceneSearcher (BM25)
- **模型**: Qwen3-4B，从 base model 开始 RL 训练（无 SFT 阶段）
- **RL 算法**: GRPO (verl 0.7.1)
- **Reward**: NDCG@1000（BM25 检索结果与 GT product IDs 的匹配度）
- **输出格式**: `<think>...</think><answer>{"query": "..."}</answer>`，answer 中是 Boolean query

## 数据来源

### 训练数据

- **来源文件**: `~/LocalSearch/data/ecommerce/train/queries_product_train_step9.jsonl`
  - 这是 LocalSearch pipeline 第 9 步生成的电商训练集查询（共 20,000 条）
  - 其中 1,517 条无 GT（占 7.6%），在数据准备时被过滤
- **最终训练集**: 18,483 条有 GT 的查询（`data/train.parquet`）
- **数据准备脚本**: `3_prepare_data.py`
  - 将 natural_language_query + gt_document_ids 转为 verl 格式的 parquet
  - 过滤无 GT 的查询（`if not gt_ids: continue`）

### 测试数据

- **来源文件**: `~/LocalSearch/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl`
  - LocalSearch pipeline 生成的电商测试集（3,000 条查询）
  - 每条包含 query_id, natural_language_query, gt_document_ids
- **评估指标**: Hit@1, Hit@3, MRR（TOP_K=20）

### 产品库（BM25 索引）

- **来源**: `~/LocalSearch/data/ecommerce/test/products_test.json`（10,000 产品）
- 字段: product_id, brand, color, category, text（产品描述）
- 转为 Pyserini JSONL 后建立 Lucene BM25 索引

## 复现步骤

### 环境依赖

```bash
# Python 3.11, CUDA
pip install verl==0.7.1 vllm pyserini torch transformers datasets
# Java 21（Pyserini/Lucene 需要）
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

### Step 1: 转换产品数据为 Pyserini 格式

```bash
python 1_convert_products.py
```

- 输入: `products_test.json`（原始产品 JSON）
- 输出: `jsonl_docs/products.jsonl`（每行一个文档，包含 id + contents）
- contents 拼接方式: `"Brand: {brand} Color: {color} Category: {category} {text}"`

### Step 2: 构建 BM25 索引

```bash
bash 2_build_index.sh
```

- 使用 `pyserini.index.lucene` 构建 Lucene 索引
- 输出: `indexes/lucene-index-products/`
- 验证: 脚本会自动测试搜索 "running shoes" 确认索引可用

### Step 3: 准备 RL 训练数据

```bash
python 3_prepare_data.py
```

- 读取 `queries_product_train_step9.jsonl` 和 `queries_product_test_step9_gt_v2.jsonl`
- 过滤掉无 GT 的查询
- 输出: `data/train.parquet`, `data/test.parquet`, `data/val.parquet`（val 取 test 的前 300 条）
- 每条记录包含: prompt（system + user message）, reward_model（含 gt target IDs）

### Step 4: GRPO RL 训练

```bash
bash 4_train.sh
```

关键训练参数:
- **GPU**: 6× A6000 (CUDA_VISIBLE_DEVICES=1,2,3,4,5,6)
- **Base model**: Qwen3-4B（从 base 开始，不经过 SFT）
- **Batch**: train_batch_size=192, n=4 rollouts/prompt
- **Learning rate**: 1e-6
- **Temperature**: 0.6 (rollout), 0.4 (validation)
- **Max lengths**: prompt=512, response=1024
- **Epochs**: 1
- **Checkpoint**: 每 50 步保存

Reward 函数（`reward_ecommerce.py`）:
- 解析模型输出中的 Boolean query
- 用 Pyserini BM25 检索 top-1000 结果
- 计算 NDCG@1000 作为 answer_score
- Format reward: +0.1（正确格式）或 -2（错误格式）
- 总分 = format_score + answer_score（answer_score > 0 时）

### Step 5: 合并 FSDP Checkpoint

训练使用 FSDP 分片存储，需要合并为 HuggingFace 格式才能推理:

```bash
python merge_fsdp_ckpt.py <checkpoint_dir>
# 例如: python merge_fsdp_ckpt.py checkpoints_v7/global_step_100
```

- 读取 6 个 rank shard，按维度拼接 sharded tensors
- 输出: `<checkpoint_dir>/actor/huggingface/model.safetensors`

### Step 6: 评估

```bash
python 5_eval.py --model <merged_hf_model_path> --output-name b3_deepretrieval
```

- 加载合并后的 HuggingFace 模型
- 用 vLLM 生成 Boolean query（temperature=0，greedy decoding）
- 用 Pyserini 执行 BM25 检索 top-20
- 计算 Hit@1, Hit@3, MRR
- 结果保存到 `~/LocalSearch/data/ecommerce/result/<output-name>.json`

批量评估多个 checkpoint:

```bash
python 5_eval_checkpoint.py --ckpt-dir checkpoints_v7 --steps 50,100,150
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `1_convert_products.py` | 产品数据 → Pyserini JSONL |
| `2_build_index.sh` | 构建 Lucene BM25 索引 |
| `3_prepare_data.py` | 查询数据 → verl RL 训练 parquet |
| `4_train.sh` | GRPO RL 训练脚本（verl） |
| `5_eval.py` | 单模型评估（vLLM + Pyserini） |
| `5_eval_checkpoint.py` | 批量 checkpoint 评估 |
| `6_monitor_eval.sh` | 训练中自动监控评估 |
| `reward_ecommerce.py` | RL reward 函数（NDCG@k） |
| `merge_fsdp_ckpt.py` | FSDP 分片 → HuggingFace 合并 |
| `convert_ckpt.py` | 辅助 checkpoint 格式转换 |

## 注意事项

1. **无 SFT 阶段**: DeepRetrieval 直接从 base model 开始 RL，不需要 SFT 预训练
2. **GT 过滤**: 训练数据中无 GT 的查询会被过滤掉（7.6%），避免 NDCG 计算时分母为 0 导致错误惩罚
3. **vLLM 版本**: 使用 `VLLM_USE_V1=0` 强制 v0 引擎，避免多进程 executor 问题
4. **路径修改**: 脚本中的路径是服务器 102 上的绝对路径，复现时需要根据实际环境修改
5. **训练索引 vs 测试索引**: 训练时 reward 用的是 train 产品库的索引（`lucene-index-products-train`），评估用的是 test 产品库的索引（`lucene-index-products`）
