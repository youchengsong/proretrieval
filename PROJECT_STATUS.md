# LocalSearch-R1 项目进展（2026-05-14 更新）

> 接手须知：先读 CLAUDE.md 了解项目全貌和API凭证，本文件补充最新进展。

---

## 一、当前正在做的事

### DeepRetrieval RL 真复现（B3 baseline, P0）

**状态**：🟢 **训练中** — 102服务器 GPUs 2-5 (A6000×4)

**训练配置**：
- Base model: Qwen3-4B (`/data/LLM_Models/Qwen3-4B`)
- RL算法: GRPO, batch=64, n=4 rollouts, lr=1e-6
- max_response_length=1024 (Qwen3 thinking需更长输出)
- FSDP param_offload + optimizer_offload (解决OOM)
- gpu_memory_utilization=0.4 (vLLM)
- 总步数: 2880, 每步~84s, 预计~67小时 (约5月17日完成)
- Checkpoint 每100步保存
- WandB: `LocalSearch-R1/deepretrieval_ecommerce_qwen3_4b`

**服务器路径** (102)：
- 训练脚本: `~/DeepRetrieval_ecommerce/4_train.sh`
- Reward函数: `~/DeepRetrieval_ecommerce/reward_ecommerce.py`
- 数据: `~/DeepRetrieval_ecommerce/data/{train,val,test}.parquet` (train=18,483, val=300, test=2,737)
- Lucene索引: `~/DeepRetrieval_ecommerce/indexes/lucene-index-products/` (3000 docs)
- Checkpoints: `~/DeepRetrieval_ecommerce/checkpoints/`
- 日志: `~/DeepRetrieval_ecommerce/train_nohup.log`

**早期指标 (step 1-2)**：
- reward/mean: -1.34 (大部分format fail, 预期中)
- clip_ratio: 0.67 (67%触及1024 token上限, 比512时97%好很多)
- peak GPU memory: 34.7GB/48GB (安全)
- 无OOM/错误

---

## 二、实验结果汇总

### 电商（N=3000，Hit@1 主指标）

**已完成的大规模实验**：
| 类别 | 方法 | Hit@1 | MRR | N |
|------|------|-------|-----|---|
| B1 | BM25 | 0.548 | 0.630 | 2737 |
| B1 | BGE-Reranker (best B1) | 0.593 | 0.664 | 2737 |
| B5 | SFT 4B | 0.680 | 0.705 | 3000 |
| B5 | **GRPO step300 (best)** | **0.809** | **0.835** | 3000 |
| B5 | DAPO step420 | 0.808 | 0.833 | 3000 |
| B5 | GPT-5.5 | 0.667 | 0.689 | 2998 |
| B5 | Opus 4.7 | 0.642 | 0.655 | 2998 |

**已完成但样本量太小（需扩到3000）**：
| 类别 | 方法 | Hit@1 | MRR | N |
|------|------|-------|-----|---|
| B2 | HyDE | 0.511 | 0.588 | 268 |
| B2 | IRCoT | 0.563 | 0.631 | 268 |
| B2 | RankGPT | 0.743 | 0.759 | 268 |
| B3 | LangChain SQR | 0.590 | 0.650 | 268 |
| B3 | DeepRetrieval API | 0.623 | 0.682 | 268 |
| B3 | SQL-R1 API | 0.925 | 0.942 | 200 |
| B3 | Few-shot GPT-5.5 | 0.253 | 0.268 | 300 |
| B3 | Few-shot Opus 4.7 | 0.250 | 0.262 | 300 |

结果文件位置：
- 本地：`results/ecommerce/eval_*.json`, `results/ecommerce/b1_*.json`
- ai4sci：`~/LocalSearch/data/ecommerce/result/eval_*.json`

### 邮件（N=3000）

| 方法 | Hit@1 | MRR | 状态 |
|------|-------|-----|------|
| DAPO step843 (best) | 0.909 | 0.910 | ✅ |
| GRPO step843 | 0.906 | 0.907 | ✅ |
| SFT | 0.891 | 0.892 | ✅ |
| GPT-5.4 | ? | ? | ⚠️ 预测完成未eval |
| Opus 4.7 | ? | ? | ⚠️ 预测完成未eval |
| GPT-5.5 | ? | ? | ⚠️ 预测完成未eval |

邮件商业模型预测文件在 ai4sci：`~/LocalSearch/data/email/result/predictions_email_{gpt54,opus47}.json`

---

## 三、需要训练的3个Baseline

1. **DeepRetrieval** (B3, RL) — 🟢 训练中 (102, 预计5月17日完成)
2. **Search-R1** (B2, RL) — 多轮搜索推理 → 未开始，实现复杂（需改verl rollout支持多轮）
3. **Text2VectorSQL** (B5, SFT) — VectorSQL 纯SFT → 文档建议用现有SFT baseline作proxy

详见 `docs/training_baselines_reproduction.md`

---

## 四、待办优先级

1. 🟢 DeepRetrieval RL训练 — 训练中，~67h，预计5月17日完成
2. 🔴 邮件商业模型eval — 在ai4sci跑eval_email.py
3. 🔴 B2/B3扩全量3000 — 纯API调用，在ai4sci跑
4. 🔴 Qwen3-8B系列 — 需要在102训练
5. 🟡 Search-R1复现
6. 🟡 GraphRAG (B4)
7. 🟢 E2 Action Space Ablation
8. 🟢 论文写作

---

## 五、服务器连接速查

### ai4sci (4090, 通过cloudflared)
```bash
# 下载 cloudflared
curl -L -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64 && chmod +x /tmp/cloudflared

# SSH via expect
cat > /tmp/ssh_cmd.exp << 'EXPEOF'
spawn ssh -o "ProxyCommand=/tmp/cloudflared access ssh --hostname %h" -o StrictHostKeyChecking=no sunzhen@ai4sci.claude-code-manager.com {<COMMAND>}
expect "password:"
send "12345678\r"
expect eof
EXPEOF
expect /tmp/ssh_cmd.exp 2>&1
```

### 102 (A6000×7, 内网)
```bash
# 从Mac可直接连接
cat > /tmp/ssh_102.exp << 'EXPEOF'
spawn ssh -o StrictHostKeyChecking=no sunzhen@10.60.200.102 {<COMMAND>}
expect "password:"
send "sunzhen\r"
expect eof
EXPEOF
expect /tmp/ssh_102.exp 2>&1
```
- GPU 0: shaohuwang占用 (36GB vLLM)
- GPU 1: embedding_server.py (30GB, DeepRetrieval不需要)
- GPU 2-5: DeepRetrieval训练中 (~18.7GB each)
- GPU 6: 空闲

### GitHub
```bash
# 私有仓库
git clone https://nbdss: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@github.com/zjw49246/LocalSearch-R1.git
```
