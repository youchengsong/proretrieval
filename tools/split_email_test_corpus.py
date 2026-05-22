#!/usr/bin/env python3
"""
把邮件测试集语料库从 10000 压缩到 5000：
  - 保留全部 3033 个 gold document
  - 随机补充 1967 个非 gold document
  - 同步更新所有相关文件（corpus、features、descriptions、queries GT）
"""

import json
import random
import pandas as pd
from pathlib import Path

random.seed(42)

BASE = Path("/data/youchengsong/DSL-R1/data/email/test")

# ── Step 1: 收集所有 gold_ids ─────────────────────────────────────────────────
gold_ids = set()
queries = []
with open(BASE / "queries_email_test_step9.jsonl") as f:
    for line in f:
        q = json.loads(line)
        queries.append(q)
        for gid in (q.get("gold_ids") or []):
            gold_ids.add(gid)

print(f"Gold IDs (去重): {len(gold_ids)}")

# ── Step 2: 构建新的 5000 文档集合 ───────────────────────────────────────────
email_df = pd.read_parquet(BASE / "email_test.parquet")
print(f"原始语料库: {len(email_df)} 条")

all_ids = set(email_df["Message_ID"].tolist())
non_gold_ids = list(all_ids - gold_ids)
random.shuffle(non_gold_ids)

extra = 5000 - len(gold_ids)
sampled_ids = gold_ids | set(non_gold_ids[:extra])
print(f"新语料库: {len(sampled_ids)} 条 (gold={len(gold_ids)}, extra={extra})")

# ── Step 3: 过滤 email_test.parquet ──────────────────────────────────────────
new_email_df = email_df[email_df["Message_ID"].isin(sampled_ids)].reset_index(drop=True)
new_email_df.to_parquet(BASE / "email_test.parquet", index=False)
print(f"email_test.parquet → {len(new_email_df)} 条")

# ── Step 4: 过滤 subject/message features ────────────────────────────────────
sf = pd.read_parquet(BASE / "test_subject_features.parquet")
new_sf = sf[sf["message_id"].isin(sampled_ids)].reset_index(drop=True)
new_sf.to_parquet(BASE / "test_subject_features.parquet", index=False)
print(f"test_subject_features.parquet → {len(new_sf)} 条")

mf = pd.read_parquet(BASE / "test_message_features.parquet")
new_mf = mf[mf["message_id"].isin(sampled_ids)].reset_index(drop=True)
new_mf.to_parquet(BASE / "test_message_features.parquet", index=False)
print(f"test_message_features.parquet → {len(new_mf)} 条")

# ── Step 5: 过滤 email_descriptions_test.jsonl ───────────────────────────────
kept = 0
with open(BASE / "email_descriptions_test.jsonl") as fin, \
     open(BASE / "_email_descriptions_test_new.jsonl", "w") as fout:
    for line in fin:
        obj = json.loads(line)
        mid = obj.get("message_id")
        if mid in sampled_ids:
            fout.write(line)
            kept += 1
(BASE / "_email_descriptions_test_new.jsonl").rename(BASE / "email_descriptions_test.jsonl")
print(f"email_descriptions_test.jsonl → {kept} 条")

# ── Step 6: 更新 queries 的 gt_document_ids（过滤掉不在新语料库中的 doc）──────
def update_queries(input_path, output_path):
    updated = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            q = json.loads(line)
            orig_gt = q.get("gt_document_ids") or []
            new_gt = [d for d in orig_gt if d in sampled_ids]
            if len(new_gt) != len(orig_gt):
                updated += 1
            q["gt_document_ids"] = new_gt
            fout.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{output_path.name}: {updated} 条 query 的 GT 被缩减")

update_queries(BASE / "queries_email_test_step8.jsonl",
               BASE / "_queries_email_test_step8_new.jsonl")
(BASE / "_queries_email_test_step8_new.jsonl").rename(BASE / "queries_email_test_step8.jsonl")

update_queries(BASE / "queries_email_test_step9.jsonl",
               BASE / "_queries_email_test_step9_new.jsonl")
(BASE / "_queries_email_test_step9_new.jsonl").rename(BASE / "queries_email_test_step9.jsonl")

# ── Step 7: 验证 ──────────────────────────────────────────────────────────────
print("\n=== 验证 ===")
check_df = pd.read_parquet(BASE / "email_test.parquet")
print(f"email_test.parquet: {len(check_df)} 条")

remaining_gold = gold_ids & set(check_df["Message_ID"].tolist())
print(f"gold_ids 保留: {len(remaining_gold)}/{len(gold_ids)}")

empty_gt = 0
with open(BASE / "queries_email_test_step9.jsonl") as f:
    for line in f:
        q = json.loads(line)
        if not q.get("gt_document_ids"):
            empty_gt += 1
print(f"GT 变空的 query 数: {empty_gt}")
print("Done.")
