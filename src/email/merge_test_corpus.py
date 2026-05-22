#!/usr/bin/env python3
"""
等待 corpus desc 生成完成后，合并 test + corpus 所有相关文件。

合并后：
  test/email_test.parquet              (2596 + 7404 = 10000)
  test/email_descriptions_test.jsonl  (2596 + 7404 = 10000)
  test/test_subject_features.parquet  (2596 + 7404 = 10000)
  test/test_message_features.parquet  (2596 + 7404 = 10000)
"""

import json
import time
import pandas as pd
from pathlib import Path

DATA   = Path("/data/youchengsong/DSL-R1/data/email")
TEST   = DATA / "test"
TOTAL  = 7404

# ── 等待 corpus 描述完成 ────────────────────────────────────────────────────────
print("Waiting for corpus descriptions to complete...")
while True:
    s = sum(1 for _ in open(DATA / "corpus_subject_pos.jsonl"))
    m = sum(1 for _ in open(DATA / "corpus_message_pos.jsonl"))
    print(f"  subject_pos: {s}/{TOTAL}  message_pos: {m}/{TOTAL}", flush=True)
    if s >= TOTAL and m >= TOTAL:
        print("Corpus descriptions complete!")
        break
    time.sleep(30)

# ── 等待 email_descriptions_corpus.jsonl 写完 ──────────────────────────────────
import subprocess, time as t
while True:
    r = subprocess.run(["pgrep", "-f", "gen_corpus_features_and_desc"], capture_output=True)
    if r.returncode != 0:
        break
    print("Still merging desc output...", flush=True)
    t.sleep(5)

# 确认 email_descriptions_corpus.jsonl 存在
desc_corpus = DATA / "email_descriptions_corpus.jsonl"
timeout = 60
while not desc_corpus.exists() and timeout > 0:
    t.sleep(2); timeout -= 2
if not desc_corpus.exists():
    # 手动生成 merge output
    print("Manually merging desc output...")
    done_subj, done_msg = {}, {}
    with open(DATA / "corpus_subject_pos.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            done_subj[obj["message_id"]] = obj["subject_pos"]
    with open(DATA / "corpus_message_pos.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            done_msg[obj["message_id"]] = obj["message_pos"]
    with open(desc_corpus, "w") as f:
        for mid in set(done_subj) | set(done_msg):
            f.write(json.dumps({"message_id": mid,
                                "subject_pos": done_subj.get(mid),
                                "message_pos": done_msg.get(mid)},
                               ensure_ascii=False) + "\n")
    print(f"  Written {len(done_subj | done_msg.keys())} records")

# ── 1. email_test.parquet ──────────────────────────────────────────────────────
df_test   = pd.read_parquet(TEST / "email_test.parquet")
df_corpus = pd.read_parquet(DATA / "email_corpus.parquet")
merged = pd.concat([df_test, df_corpus], ignore_index=True)
merged.to_parquet(TEST / "email_test.parquet", index=False)
print(f"email_test.parquet: {len(df_test)} + {len(df_corpus)} = {len(merged)}")

# ── 2. email_descriptions_test.jsonl ─────────────────────────────────────────
lines_test = open(TEST / "email_descriptions_test.jsonl").readlines()
lines_corp = open(desc_corpus).readlines()
with open(TEST / "email_descriptions_test.jsonl", "w") as f:
    f.writelines(lines_test + lines_corp)
print(f"email_descriptions_test.jsonl: {len(lines_test)} + {len(lines_corp)} = {len(lines_test)+len(lines_corp)}")

# ── 3. test_subject_features.parquet ─────────────────────────────────────────
sf_test   = pd.read_parquet(TEST / "test_subject_features.parquet")
sf_corpus = pd.read_parquet(DATA / "corpus_subject_features.parquet")
sf_merged = pd.concat([sf_test, sf_corpus], ignore_index=True)
sf_merged.to_parquet(TEST / "test_subject_features.parquet", index=False)
print(f"test_subject_features.parquet: {len(sf_test)} + {len(sf_corpus)} = {len(sf_merged)}")

# ── 4. test_message_features.parquet ─────────────────────────────────────────
mf_test   = pd.read_parquet(TEST / "test_message_features.parquet")
mf_corpus = pd.read_parquet(DATA / "corpus_message_features.parquet")
mf_merged = pd.concat([mf_test, mf_corpus], ignore_index=True)
mf_merged.to_parquet(TEST / "test_message_features.parquet", index=False)
print(f"test_message_features.parquet: {len(mf_test)} + {len(mf_corpus)} = {len(mf_merged)}")

print("\nAll merged successfully!")
