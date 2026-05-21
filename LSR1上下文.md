# LocalSearch-R1 EMNLP 2026 — 项目上下文与进展

> 最后更新: 2026-05-19 02:25 CST

## 一、项目概述

**目标**: 复现两个 RL 训练的 baseline（Search-R1 和 SQL-R1），用于 LocalSearch-R1 EMNLP 2026 论文（deadline: 2026-05-25）。

**训练流程**: SFT（教格式）→ RL/GRPO（优化正确率）

**基础模型**: Qwen/Qwen3-4B (4.02B params)

**四个训练任务**:
| 任务 | 领域 | 状态 |
|------|------|------|
| Search-R1 ecommerce | 产品搜索 | SFT ✅ → RL 启动中 |
| Search-R1 email | 邮件搜索 | SFT ✅ → RL 排队 |
| SQL-R1 ecommerce | SQL查询 | SFT 待训 |
| SQL-R1 email | SQL查询 | SFT 待训 |

---

## 二、远程服务器连接

**Blackwell 服务器 (AutoDL)**:
```
SSH:  ssh -p 34907 root@connect.westb.seetacloud.com
密码: JUmCDt0QqyKf
SCP:  scp -P 34907 本地文件 root@connect.westb.seetacloud.com:远程路径
```

**硬件**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM each), 50 CPU cores, 240GB RAM

**本地 expect 脚本（自动 SSH/SCP）**:
- SSH: `expect /tmp/sc2.exp "远程命令"`
- SCP: `expect /tmp/scp_to_blackwell.exp 本地路径 远程路径`

---

## 三、服务器目录结构

```
/root/autodl-tmp/
├── data/
│   ├── ecommerce/
│   │   └── product_rl_train_filtered.parquet    # 20000条，产品搜索 RL 数据
│   └── email/
│       ├── email_rl_train_fixed.parquet          # 20000条，邮件搜索 RL 数据
│       └── email_test.json                       # 5000封邮件语料（Enron风格）
├── search_r1/
│   ├── sft_data/
│   │   ├── ecommerce_search_sft.jsonl           # 18483条 SFT 数据
│   │   └── email_search_sft.jsonl               # 16828条 SFT 数据
│   ├── rl_data/
│   │   ├── ecommerce_search_rl_train.parquet    # RL 训练数据
│   │   └── email_search_rl_train.parquet
│   └── checkpoints/
│       ├── ecommerce_sft_v2/final/              # ✅ SFT checkpoint
│       ├── email_sft_v2/final/                  # ✅ SFT checkpoint
│       ├── search_r1_rl_ecommerce_v2/           # RL 输出目录
│       └── search_r1_rl_email_v2/
├── sql_r1/
│   ├── sft_data/
│   │   ├── ecommerce_sql_sft.jsonl             # 20000条
│   │   └── email_sql_sft.jsonl                 # 20000条
│   ├── rl_data/
│   └── checkpoints/
├── scripts/
│   ├── train_sft_v2.py                         # SFT 训练脚本（修复版）
│   ├── fix_sft_data_v2.py                      # SFT 数据生成脚本
│   ├── run_search_rl.sh                        # Search-R1 RL 启动脚本
│   ├── reward_search_r1.py                     # RL reward 函数
│   ├── reward_sql_r1.py                        # SQL reward 函数
│   ├── test_sft_search_v2.py                   # SFT 测试脚本
│   └── check_rl_pipeline.py                    # RL 预检脚本
└── logs/
    ├── sft_v3_train.log                        # ecommerce SFT 日志
    ├── email_sft_v3.log                        # email SFT 日志
    ├── search_rl_v2.log                        # RL 训练日志
    ├── reward_debug.log                        # Reward 调试日志
    └── auto_rl_pipeline.log
```

---

## 四、训练数据来源与格式

### SFT 数据（由 fix_sft_data_v2.py 生成）:
- **输入**: RL parquet 中的 query + ground truth ID
- **过程**: BM25 从语料库检索相关结果，拼装成对话格式
- **Search-R1 格式**: system + user + assistant(`<think>→<search>→<information>→<think>→<answer>`)
- **SQL-R1 格式**: system + user + assistant(`<think>→<answer>```sql...```)

### RL 数据 (parquet):
- **columns**: `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`
- **prompt**: numpy array, 包含 1 条 user message（含系统指令和查询）
- **reward_model**: JSON string `{"style": "rule", "ground_truth": {"target": ["ID1", "ID2", ...]}}`

---

## 五、踩过的坑（重要！）

### 1. Qwen3 apply_chat_template 剥离标签
**问题**: `tokenizer.apply_chat_template()` 会自动剥离 assistant 消息中的 `<search>` 和 `<information>` 标签，将其当作 tool call 处理。导致 SFT 训练数据中搜索格式被完全删除（573 tokens 缩减到 189 tokens）。
**修复**: 手动拼接 `<|im_start|>role\ncontent<|im_end|>` 格式，绕过模板解析。

### 2. enable_thinking=False 注入空 think 块
**问题**: `enable_thinking=False` + `add_generation_prompt=True` 时，模板自动注入 `<think>\n\n</think>\n\n`，推理时模型跳过 think/search。
**注意**: 此参数仅影响推理（add_generation_prompt=True），训练时（add_generation_prompt=False）无影响。

### 3. Label 未做 masking
**问题**: 原始 `labels = input_ids.copy()` 让模型在 system+user 内容上也计算 loss，稀释 assistant 学习信号。
**修复**: 找到 `<|im_start|>assistant\n` 位置，之前的 labels 全设 -100。

### 4. Reward 函数 ground_truth 格式
**RL parquet 中**: `ground_truth` 是 `{"target": ["ID1", ...]}` (dict)
**SFT 生成时用的原始 parquet**: `ground_truth` 是 `["ID1", ...]` (list)
**修复**: reward 函数同时处理 list/dict/str 三种格式。

### 5. batch_size 与 OOM
**ecommerce**: avg ~920 tokens, batch_size=32 可以跑
**email**: avg ~921 tokens, batch_size=32 OOM → 降到 16 (grad_accum=2)

### 6. TRL 与 transformers 5.8.1 不兼容
**问题**: SFTTrainer 参数名变更（max_length→max_seq_length, processing_class→tokenizer 等）
**修复**: 弃用 TRL，直接用 transformers.Trainer

---

## 六、关键训练参数

### SFT (train_sft_v2.py):
- DeepSpeed ZeRO-2 + CPU optimizer offload
- gradient_checkpointing=True
- lr=2e-5, cosine scheduler, warmup_ratio=0.05
- 手动 chat template（不用 apply_chat_template）
- Label masking: 仅 assistant 内容计算 loss

### RL (GRPO via verl):
- verl 0.7.1 + vllm 0.17.1
- algorithm: GRPO (adv_estimator=grpo)
- train_batch_size=32, n=4 (rollout samples)
- actor lr=1e-6, KL loss coef=0.001
- vllm rollout: tensor_parallel=2, gpu_memory_utilization=0.5
- FSDP param_offload + optimizer_offload
- 15 epochs, save_freq=2340

---

## 七、如何查看训练进度

```bash
# SSH 到服务器
ssh -p 34907 root@connect.westb.seetacloud.com
# 密码: JUmCDt0QqyKf

# 查看当前运行的 screen 会话
screen -ls

# 查看 RL 训练日志
tail -f /root/autodl-tmp/logs/search_rl_v2.log

# 查看 reward 统计
cat /root/autodl-tmp/logs/reward_debug.log

# 查看 GPU 状态
nvidia-smi

# 查看 SFT 训练 loss
grep 'loss' /root/autodl-tmp/logs/sft_v3_train.log | tail -5
grep 'loss' /root/autodl-tmp/logs/email_sft_v3.log | tail -5
```

---

## 八、当前状态 (2026-05-19 02:25)

- ✅ Search-R1 ecommerce SFT 完成 (loss: 3.585 → 0.053)
- ✅ Search-R1 email SFT 完成 (loss → 0.053)
- 🔄 Search-R1 ecommerce RL 刚启动 (screen: search_rl)
- ⏳ Search-R1 email RL 排队
- ⏳ SQL-R1 SFT + RL 待开始
- ⚠️ SFT 模型推理效果待验证（需确认 `<think>/<search>/<answer>` 格式正确）
