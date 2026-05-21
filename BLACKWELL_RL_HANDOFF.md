# Blackwell RL 训练完整交接文档

> 写于 2026-05-18 12:35 UTC，供新 session 接手

---

## 1. 项目背景

LocalSearch-R1 是一篇 EMNLP 2026 投稿（截止 2026-05-25），核心是用 RL 训练 LLM 做多模态检索。

当前任务：在 Blackwell 服务器上复现两个 baseline（Search-R1 和 SQL-R1）的 RL GRPO 训练。

- **Search-R1**: 搜索引擎交互式检索 baseline
- **SQL-R1**: SQL 查询式检索 baseline
- 每个 baseline 各有 ecommerce 和 email 两个场景，共 **4 个 RL 训练任务**
- 所有 4 个 SFT checkpoint 已完成，RL 是唯一剩余的训练工作

---

## 2. 服务器连接

### Blackwell 服务器（AutoDL 中国）
```
SSH: ssh -p 34907 root@connect.westb.seetacloud.com
密码: JUmCDt0QqyKf
```

### 从本地 Ubuntu 机器连接的 expect 脚本

**执行远程命令**（`/tmp/sc2.exp`）：
```expect
#!/usr/bin/expect -f
set timeout 60
set cmd [lindex $argv 0]
spawn ssh -p 34907 -o StrictHostKeyChecking=no root@connect.westb.seetacloud.com
expect {
    -nocase "password" { send "JUmCDt0QqyKf\r"; exp_continue }
    "# " { }
    "$ " { }
}
send "$cmd\r"
expect {
    "# " { }
    "$ " { }
    timeout { }
}
send "exit\r"
expect eof
```
用法: `expect /tmp/sc2.exp "远程命令"`

**SCP 上传文件**（`/tmp/scp_to_blackwell.exp`）：
```expect
#!/usr/bin/expect -f
set timeout 600
set src [lindex $argv 0]
set dst [lindex $argv 1]
spawn scp -P 34907 -o StrictHostKeyChecking=no $src root@connect.westb.seetacloud.com:$dst
expect {
    -nocase "password" { send "JUmCDt0QqyKf\r"; exp_continue }
    eof
}
foreach {pid spawnid os_error_flag value} [wait] break
exit $value
```
用法: `expect /tmp/scp_to_blackwell.exp 本地路径 远程路径`

---

## 3. 服务器硬件和软件

| 项目 | 值 |
|------|----|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB each |
| GPU 架构 | SM_120 (Compute Capability 12.0) |
| CPU/内存 | 50核 / 240GB |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| vllm | 0.8.6.dev0 (从 0.8.5 源码编译，支持 SM_120) |
| verl | 0.7.1 |
| transformers | 5.8.1 |
| Python | 3.12 |
| 系统盘 | 30GB (44%已用) |
| 数据盘 | 350GB /root/autodl-tmp (11%已用) |

---

## 4. 目录结构

```
/root/autodl-tmp/
├── scripts/
│   ├── run_rl_all_vllm.sh          # 主 RL 训练脚本（4步顺序执行）
│   ├── reward_search_r1.py          # Search-R1 reward 函数
│   ├── reward_sql_r1.py             # SQL-R1 reward 函数
│   └── patch_verl_vllm_compat.py    # vllm 兼容性补丁脚本（已执行过）
├── search_r1/
│   ├── checkpoints/
│   │   ├── search_r1_sft_ecommerce/final/   # SFT checkpoint ✅
│   │   ├── search_r1_sft_email/final/       # SFT checkpoint ✅
│   │   ├── search_r1_rl_ecommerce/          # RL output (训练中)
│   │   └── search_r1_rl_email/              # RL output (待训练)
│   └── rl_data/
│       ├── ecommerce_search_rl_train.parquet  # 20000 条
│       └── email_search_rl_train.parquet      # 20000 条
├── sql_r1/
│   ├── checkpoints/
│   │   ├── sql_r1_sft_ecommerce/final/      # SFT checkpoint ✅
│   │   ├── sql_r1_sft_email/final/          # SFT checkpoint ✅
│   │   ├── sql_r1_rl_ecommerce/             # RL output (待训练)
│   │   └── sql_r1_rl_email/                 # RL output (待训练)
│   └── rl_data/
│       ├── ecommerce_sql_rl_train.parquet     # 20000 条
│       └── email_sql_rl_train.parquet         # 20000 条
├── vllm_build/                               # vllm 源码编译目录
├── logs/
│   ├── rl_all_progress.log                   # 总进度日志
│   ├── rl_v10_master.log                     # 当前运行的完整日志
│   ├── search_r1_rl_ecommerce_vllm.log       # Step 1 详细日志
│   └── ...
```

---

## 5. 训练参数

### 训练脚本: `/root/autodl-tmp/scripts/run_rl_all_vllm.sh`

4个顺序步骤，用 `verl.trainer.main_ppo` + `algorithm.adv_estimator=grpo`：

| 步骤 | 模型 | 数据 | Epochs | Save Freq | LR | Temp | Max Resp Len |
|------|------|------|--------|-----------|-----|------|-------------|
| 1. Search-R1 ecommerce | search_r1_sft_ecommerce | 20000 | 15 | 2340 | 1e-6 | 1.0 | 512 |
| 2. Search-R1 email | search_r1_sft_email | 20000 | 15 | 2340 | 1e-6 | 1.0 | 512 |
| 3. SQL-R1 ecommerce | sql_r1_sft_ecommerce | 20000 | 8 | 1250 | 3e-7 | 1.1 | 1024 |
| 4. SQL-R1 email | sql_r1_sft_email | 20000 | 8 | 1250 | 3e-7 | 1.1 | 1024 |

共同参数:
- `rollout.name=vllm`, `tensor_model_parallel_size=2`, `gpu_memory_utilization=0.5`
- `rollout.n=4` (GRPO group size)
- `rollout.top_k=-1`, `rollout.do_sample=True`
- `train_batch_size=32`, `ppo_mini_batch_size=32`, `ppo_micro_batch_size_per_gpu=2`
- `fsdp_config.param_offload=True`, `fsdp_config.optimizer_offload=True`
- `use_kl_loss=True`, `kl_loss_coef=0.001`, `entropy_coeff=0.001`
- `data.return_raw_chat=True`

环境变量:
```bash
CUDA_VISIBLE_DEVICES=0,1
VLLM_USE_V1=1
TORCH_COMPILE_DISABLE=1
HF_ENDPOINT=https://hf-mirror.com
WANDB_MODE=disabled
```

---

## 6. Parquet 数据格式

以 `ecommerce_search_rl_train.parquet` 为例：

```python
Columns: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
Types:
  data_source    str
  prompt         object  # numpy array of dicts: [{'role': 'user', 'content': '...'}]
  ability        str
  reward_model   str     # ⚠️ JSON 字符串，不是 dict！
  extra_info     str     # ⚠️ JSON 字符串，不是 dict！
```

`reward_model` 示例值:
```json
{"style": "rule", "ground_truth": {"target": ["B07GC9M3L6", ...], "db_name": "ecommerce"}}
```

`extra_info` 示例值:
```json
{"query_id": "q_00001", "index": 0, "level": 2, "n": 3, "modalities": ["structured", "image"], "split": "train"}
```

**关键问题**: verl 0.7.1 代码到处假设 `reward_model` 和 `extra_info` 是 Python dict，但 parquet 序列化后变成了 JSON 字符串。这导致了大量 `'str' object has no attribute 'get'` 和 `string indices must be integers` 错误。

---

## 7. 已修复的所有问题（10个）

### 7.1 vllm API 不兼容问题（verl 0.7.1 期望 vllm 0.9+ 但我们用 0.8.5）

**#1: `all_special_tokens_extended` AttributeError**
- 文件: `/root/autodl-tmp/vllm_build/vllm/transformers_utils/tokenizer.py` (约第83行)
- 原因: transformers 5.8.1 删了 `tokenizer.all_special_tokens_extended`
- 修复: `getattr(tokenizer, "all_special_tokens_extended", tokenizer.all_special_tokens)`

**#2: `torch.compile` / cudagraph 崩溃 (SM_120)**
- 原因: vllm V1 的 `compilation_config: {"cudagraph_mode": "FULL_AND_PIECEWISE"}` 在 Blackwell 上崩溃
- 修复: `export TORCH_COMPILE_DISABLE=1`

**#3: AsyncLLM 缺少 5 个方法**
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- 缺失方法: `reset_mm_cache`, `wait_for_requests_to_drain`, `pause_generation`, `resume_generation`, `get_supported_tasks`
- 修复: 在文件头部插入了 monkey-patch 代码块 `_patch_asyncllm_for_085()`，给 AsyncLLM 类添加 no-op 方法
- 额外: 也直接注释了 `reset_mm_cache` 调用（约第457行），给 `get_supported_tasks` 加了 `hasattr` 保护（约第465行），给 `wait_for_requests_to_drain` 加了 `hasattr` 保护（约第673行）
- 额外: monkey-patch 还添加了 fake `output_processor` 和 `engine_core` 属性

**#4: `process_weights_after_loading` ImportError**
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/utils.py` (约第200行)
- 修复: `try/except ImportError: pass` 包裹

**#5: `top_k=0` ValueError**
- 原因: vllm 0.8.5 要求 top_k 必须是 -1（禁用）或 >= 1
- 修复: 训练脚本里改成 `top_k=-1`

**#6: `multi_modal_data={}` 空 dict 触发多模态检查**
- 文件: `vllm_async_server.py` 约第614行
- 原因: verl 总是创建 `multi_modal_data = {}`，即使没有图像/视频数据也传给 vllm，vllm 0.8.5 把非 None 当作多模态请求
- 修复: `TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data if multi_modal_data else None)`

### 7.2 数据格式问题（JSON 字符串 vs dict）

**#7: `extra_info` 在 rl_dataset.py 中是 JSON 字符串**
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/utils/dataset/rl_dataset.py` (约第359行)
- 修复: 在原有 `if ... is None` 判断后添加:
```python
elif isinstance(row_dict["extra_info"], str):
    import json as _json
    row_dict["extra_info"] = _json.loads(row_dict["extra_info"])
```

**#8: `reward_model` 在 rl_dataset.py 中是 JSON 字符串**
- 文件: 同上，约第363行
- 修复: 在 `index = row_dict.get(...)` 行前添加:
```python
if "reward_model" in row_dict and isinstance(row_dict["reward_model"], str):
    import json as _json_rm
    try:
        row_dict["reward_model"] = _json_rm.loads(row_dict["reward_model"])
    except (ValueError, KeyError):
        pass
```

**#9: `reward_model` 在 naive.py reward manager 中是 JSON 字符串**
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/naive.py` (约第43行)
- 修复:
```python
_rm = data_item.non_tensor_batch["reward_model"]
if isinstance(_rm, str):
    _rm = __import__("json").loads(_rm)
ground_truth = _rm["ground_truth"]
```
- 同时修了 `extra_info` 读取（约第44行）

**#10: `reward_model` 在 ray_trainer.py 和 batch.py 中**
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py` (第436行, 第528行)
- 文件: `/root/miniconda3/lib/python3.12/site-packages/verl/workers/reward_manager/batch.py` (第62行, 第116行)
- 修复: 内联 `json.loads` 转换

---

## 8. 当前状态（截至 2026-05-18 12:35 UTC）

### 训练正在运行！
- Screen session: `rl_v10`（PID 594521，启动于 12:07:17）
- 当前在 **Step 1/4: Search-R1 RL ecommerce**
- 已到 **step 59**，每步约 32 秒
- GPU 内存: 80.4GB allocated / 92.9GB reserved（每卡）
- CPU 内存: 218-219GB

### ⚠️ 严重问题: Reward 全为 0

```
critic/score/mean:0.0
critic/score/max:0.0
critic/rewards/mean:0.0
zeros=200/200 ones=0/200
last_reason=no_match / no_answer
```

模型生成的回复非常短（avg 47-49 tokens），reward 函数找不到有效答案。可能原因:
1. reward 函数期望特定格式的输出（如 `<answer>...</answer>` 标签），但模型生成的格式不对
2. 模型没有正确使用搜索工具，直接给出了无法匹配的答案
3. prompt 格式或 chat template 可能有问题

**这是目前最紧迫需要调查的问题**。训练在跑，但 reward=0 意味着模型什么也学不到。

### 监控命令

```bash
# 查看当前训练 step 和 reward
expect /tmp/sc2.exp "grep 'step:' /root/autodl-tmp/logs/rl_v10_master.log | tail -3"

# 查看 reward worker 输出
expect /tmp/sc2.exp "grep 'reward_search_r1' /root/autodl-tmp/logs/rl_v10_master.log | tail -5"

# 查看总进度
expect /tmp/sc2.exp "cat /root/autodl-tmp/logs/rl_all_progress.log"

# 查看 GPU 使用
expect /tmp/sc2.exp "nvidia-smi"

# 查看 screen session
expect /tmp/sc2.exp "screen -ls"
```

---

## 9. Reward 函数

### Search-R1 Reward: `/root/autodl-tmp/scripts/reward_search_r1.py`
需要检查的关键点:
- `compute_score` 函数签名和参数格式
- 期望的模型输出格式（是否需要 `<answer>` 标签、search 工具调用等）
- `ground_truth` 参数是 dict `{"target": [...], "db_name": "..."}`

### SQL-R1 Reward: `/root/autodl-tmp/scripts/reward_sql_r1.py`
- 类似结构，检查 SQL 执行和结果匹配逻辑

### 查看 reward 脚本:
```bash
expect /tmp/sc2.exp "cat /root/autodl-tmp/scripts/reward_search_r1.py"
expect /tmp/sc2.exp "cat /root/autodl-tmp/scripts/reward_sql_r1.py"
```

---

## 10. 被修改过的文件清单

服务器上以下文件被手动修改过（不是原始安装版本）：

1. `/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - monkey-patch 代码块（文件头部）
   - `reset_mm_cache` 注释
   - `get_supported_tasks` hasattr 保护
   - `wait_for_requests_to_drain` hasattr 保护
   - `multi_modal_data` None 转换
   
2. `/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/utils.py`
   - `process_weights_after_loading` try/except

3. `/root/miniconda3/lib/python3.12/site-packages/verl/utils/dataset/rl_dataset.py`
   - `extra_info` JSON 解析
   - `reward_model` JSON 解析

4. `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/naive.py`
   - `reward_model` JSON 解析
   - `extra_info` JSON 解析

5. `/root/miniconda3/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py`
   - `reward_model` JSON 解析（第436行和第528行）

6. `/root/miniconda3/lib/python3.12/site-packages/verl/workers/reward_manager/batch.py`
   - `reward_model` JSON 解析（第62行和第116行）

7. `/root/autodl-tmp/vllm_build/vllm/transformers_utils/tokenizer.py`
   - `all_special_tokens_extended` getattr fallback

---

## 11. 下一步行动

### 最紧急: 调查 Reward = 0 的问题
1. 查看 reward 函数代码，理解期望的输出格式
2. 查看模型实际生成的内容: `grep -A 5 'response_length' /root/autodl-tmp/logs/rl_v10_master.log | tail -20`
3. 手动跑一条测试看 reward 函数是否正常工作
4. 如果是 prompt/chat_template 问题，可能需要修改数据或训练参数

### 如果需要重启训练
```bash
# 停止当前训练
expect /tmp/sc2.exp "screen -X -S rl_v10 quit; ray stop --force"

# 修改脚本后重启
expect /tmp/scp_to_blackwell.exp /tmp/blackwell_scripts/run_rl_all_vllm.sh /root/autodl-tmp/scripts/run_rl_all_vllm.sh
expect /tmp/sc2.exp "screen -dmS rl_v11 bash -c 'export HYDRA_FULL_ERROR=1; bash /root/autodl-tmp/scripts/run_rl_all_vllm.sh 2>&1 | tee /root/autodl-tmp/logs/rl_v11_master.log'"
```

### 如果遇到更多 JSON 字符串问题
其他可能需要修复的文件（如果用到的话）：
- `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/dapo.py` (第61-62行)
- `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/gdpo.py` (第44-45行)
- `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/limited.py` (第408-409行)
- `/root/miniconda3/lib/python3.12/site-packages/verl/experimental/reward_loop/reward_manager/remote.py` (第84-85行)
- `/root/miniconda3/lib/python3.12/site-packages/verl/workers/reward_manager/dapo.py` (第92行)

### 本地脚本副本
训练脚本的本地副本在: `/tmp/blackwell_scripts/run_rl_all_vllm.sh`
修改后用 `expect /tmp/scp_to_blackwell.exp` 上传。

### 训练完成后
- 验证 checkpoint 目录下有保存的模型
- 每个 checkpoint 需要跑推理测试验证训练效果
- 用户的要求: "checkpoint保存了吗？按总step除以4来保存checkpoint，每出一个checkpoint，跑一下测试，验证一下训练效果"

---

## 12. 本地项目信息

- GitHub: `https://github.com/zjw49246/LocalSearch-R1`（私有）
- 本地 clone: `/home/ubuntu/LocalSearch-R1/`
- 项目详细背景: `/home/ubuntu/LocalSearch-R1/CLAUDE.md`
- 原有进度: `/home/ubuntu/LocalSearch-R1/BLACKWELL_PROGRESS.md`
