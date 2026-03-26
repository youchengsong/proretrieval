#!/usr/bin/env python3
"""
用 Qwen3-Embedding-8B 生成邮件 Subject 和 Message 字段的特征向量。
支持分片并行：--shard 0 --total 4

输入: enron_final.parquet (Message_ID, Subject, Message)
输出:
  email_subject_features_shard{N}.parquet  →  email_subject_features.parquet
  email_message_features_shard{N}.parquet  →  email_message_features.parquet

用法:
  CUDA_VISIBLE_DEVICES=2 python gen_email_text_features.py --field subject --shard 0 --total 4
  CUDA_VISIBLE_DEVICES=4 python gen_email_text_features.py --field subject --shard 1 --total 4
  ...
  python gen_email_text_features.py --field subject --merge
  python gen_email_text_features.py --field message --merge

Message 字段截断到 2000 字符（约 500 token），避免超出模型限制。
"""

import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR   = "/data/youchengsong/DSL-R1/data"
MODEL_PATH = "/data/youchengsong/models/Qwen3-Embedding-8B"
INPUT_FILE = f"{DATA_DIR}/enron_final.parquet"

# Message 截断长度（字符数）
MESSAGE_TRUNCATE = 2000
BATCH_SIZE = 32


def get_paths(field: str, shard: int = None, total: int = None):
    if shard is not None:
        return f"{DATA_DIR}/email_{field}_features_shard{shard}.parquet"
    return f"{DATA_DIR}/email_{field}_features.parquet"


def run_shard(field: str, shard: int, total: int):
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    indices = list(range(shard, len(df), total))
    df = df.iloc[indices].reset_index(drop=True)
    print(f"[{field} shard {shard}/{total}] {len(df)} rows, GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if field == "subject":
        texts = df["Subject"].fillna("").tolist()
        max_seq = 128
    else:
        texts = df["Message"].fillna("").str[:MESSAGE_TRUNCATE].tolist()
        max_seq = 512

    model = SentenceTransformer(MODEL_PATH, device="cuda")
    model.max_seq_length = max_seq
    print(f"  max_seq_length: {max_seq}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  embedding shape: {embeddings.shape}")

    out_path = get_paths(field, shard, total)
    result = pd.DataFrame({
        "message_id":      df["Message_ID"].values,
        "text_features":   list(embeddings),
    })
    result.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")


def merge(field: str, total: int):
    parts = []
    for i in range(total):
        path = get_paths(field, i, total)
        parts.append(pd.read_parquet(path))
        print(f"  Loaded shard {i}: {len(parts[-1])} rows")

    merged = pd.concat(parts, ignore_index=True)
    original = pd.read_parquet(INPUT_FILE, columns=["Message_ID"])
    original = original.rename(columns={"Message_ID": "message_id"})
    merged = original.merge(merged, on="message_id", how="left")

    out_path = f"{DATA_DIR}/email_{field}_features.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Merged {len(merged)} rows → {out_path}")

    for i in range(total):
        os.remove(get_paths(field, i, total))
    print("Shard files removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=["subject", "message"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge(args.field, args.total)
    else:
        run_shard(args.field, args.shard, args.total)
