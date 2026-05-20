# DeepRetrieval Baseline (B3) — 邮件场景复现

DeepRetrieval 是基于 BM25 的检索基线方法。模型通过 RL (GRPO) 学习为用户查询生成优化的 Boolean 检索 query，然后用 Pyserini/Lucene 执行 BM25 检索，以 NDCG@k 作为 reward 信号。

## 方法概述

- **检索方式**: Pyserini LuceneSearcher (BM25)
- **模型**: Qwen3-4B，从 base model 开始 RL 训练（无 SFT 阶段）
- **RL 算法**: GRPO (verl 0.7.1)
- **Reward**: NDCG@1000（BM25 检索结果与 GT email IDs 的匹配度）
- **输出格式**: `<think>...</think><answer>{"query": "..."}</answer>`，answer 中是 Boolean query

## 结果

| 方法 | Hit@1 | Hit@3 | MRR | Exec Rate | N |
|------|-------|-------|-----|-----------|---|
| DeepRetrieval step100 | 0.745 | 0.851 | 0.805 | 99.9% | 2569 |
| DeepRetrieval step104 | 0.743 | 0.851 | 0.805 | 99.9% | 2569 |

对比我们的 DSL 方法:

| 方法 | Hit@1 | Hit@3 | MRR | 差距 (Hit@1) |
|------|-------|-------|-----|-------------|
| DAPO step843 (Ours) | 0.909 | 0.910 | 0.910 | — |
| GRPO step843 (Ours) | 0.906 | 0.908 | 0.907 | — |
| DeepRetrieval step100 | 0.745 | 0.851 | 0.805 | -16.4pp |

> DeepRetrieval 使用 Boolean query + BM25 检索，Hit@1=0.745 远低于我们的 DSL 方法(0.909)。Boolean query 表达能力有限，无法精确指定结构化字段过滤条件（如 From/To/Date），只能通过关键词匹配近似。

## 数据来源

### 训练数据

- **来源文件**: 从 RL parquet `email_rl_train_fixed.parquet` 提取（20,000 条）
  - 包含 NL 查询和 GT document IDs
  - 所有 20,000 条记录均保留（包括无 GT 的查询）
- **最终训练集**: 20,000 条（`data/train.parquet`）

### 测试数据

- **来源文件**: 从 RL parquet `email_rl_test_fixed.parquet` 提取（3,000 条）
- **有效评估**: N=2569（431 条无 gt_document_ids 的 query 不参与指标计算）
- **评估指标**: Hit@1, Hit@3, MRR（TOP_K=20）

### 邮件库（BM25 索引）

- **训练索引**: `email_train.parquet` 中 10,000 封邮件
- **测试索引**: `email_test.json` 中 5,000 封邮件
- 字段: Message_ID, Date, From, To, Subject, Message
- Contents 拼接: `"From: {From} To: {To} Subject: {Subject} {Message}"`

## 复现步骤

### 环境依赖

```bash
pip install verl==0.7.1 vllm pyserini torch transformers datasets
pip install 'numpy<2.0.0' 'transformers>=4.56.0,<5'
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

### Step 1: 转换邮件数据为 Pyserini 格式

```bash
python 1_convert_emails.py
```

- 输入: `email_train.parquet` + `email_test.json`
- 输出: `jsonl_docs/emails_train.jsonl`, `jsonl_docs/emails_test.jsonl`

### Step 2: 构建 BM25 索引

```bash
bash 2_build_index.sh
```

- 训练索引: `indexes/lucene-index-emails-train/`（10,000 docs，训练 reward 用）
- 测试索引: `indexes/lucene-index-emails-test/`（5,000 docs，评估用）

### Step 3: 准备 RL 训练数据

```bash
python 3_prepare_data.py
```

- 从 RL parquet 提取 NL 查询和 GT IDs
- 输出: `data/train.parquet`(20,000), `data/test.parquet`(3,000), `data/val.parquet`(300)

### Step 4: GRPO RL 训练

```bash
bash 4_train.sh
```

关键训练参数:
- **GPU**: 2× RTX PRO 6000 Blackwell (96GB)
- **Base model**: Qwen3-4B（从 base 开始，不经过 SFT）
- **Batch**: train_batch_size=192, n=4 rollouts/prompt
- **Learning rate**: 1e-6
- **Temperature**: 0.6 (rollout), 0.4 (validation)
- **Max lengths**: prompt=512, response=1024
- **Total steps**: 104 (1 epoch)
- **Checkpoint**: 每 50 步保存
- **训练时长**: ~2.5 小时

训练曲线: score mean 从 -0.89 (step 1) 上升到 0.628 (step 100)。

### Step 5: 合并 FSDP Checkpoint

```bash
python merge_fsdp_ckpt.py checkpoints/global_step_100
```

- 使用 base 模型 weight shapes 作为参考正确合并 FSDP 分片
- 处理 DTensor 和 bfloat16 转换（通过 numpy 中间格式避免 safetensors 存储问题）
- 输出: `checkpoints/global_step_100/actor/huggingface/model.safetensors`

### Step 6: 评估

```bash
python 5_eval.py --model checkpoints/global_step_100/actor/huggingface --output-name deepretrieval_email_step100
```

- 使用 `multiprocessing.set_start_method('spawn')` 避免 JVM/CUDA fork 冲突
- LLM 初始化在 Pyserini 之前，避免 CUDA 重初始化错误
- Temperature=0 greedy decoding，max_tokens=512

## 文件说明

| 文件 | 说明 |
|------|------|
| `1_convert_emails.py` | 邮件数据 → Pyserini JSONL |
| `2_build_index.sh` | 构建 Lucene BM25 索引（train + test） |
| `3_prepare_data.py` | 查询数据 → verl RL 训练 parquet |
| `4_train.sh` | GRPO RL 训练脚本（verl） |
| `5_eval.py` | 评估脚本（vLLM + Pyserini BM25） |
| `reward_email.py` | RL reward 函数（NDCG@1000） |
| `merge_fsdp_ckpt.py` | FSDP 分片 → HuggingFace 合并 |

## 注意事项

1. **无 SFT 阶段**: 直接从 base model 开始 RL，与电商 DeepRetrieval 一致
2. **保留所有训练数据**: 不过滤无 GT 的查询（与电商不同），保持 20,000 条
3. **FSDP 合并需参考 base model shapes**: 因为 FSDP 等分参数可能使两个 shard 形状相同（如 embedding 各占一半 vocab），简单的 shape 比较会误判为 replicated
4. **vLLM CUDA fork 问题**: 评估时需设置 `multiprocessing.set_start_method('spawn')` 并在 Pyserini 之前初始化 LLM
5. **训练服务器**: Blackwell (port 39581)，checkpoints 在 `/root/autodl-tmp/DeepRetrieval_email/checkpoints/`
