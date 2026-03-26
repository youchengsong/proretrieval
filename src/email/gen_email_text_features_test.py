#!/usr/bin/env python3
"""
用 Qwen3-Embedding-8B 生成测试集邮件 Subject 和 Message 字段的特征向量。

输入: email_test.parquet (Message_ID, Subject, Message)
输出:
  test_subject_features.parquet  (message_id, text_features)
  test_message_features.parquet  (message_id, text_features)

用法:
  CUDA_VISIBLE_DEVICES=3 python gen_email_text_features_test.py --field subject
  CUDA_VISIBLE_DEVICES=5 python gen_email_text_features_test.py --field message
"""

import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR   = "/data/youchengsong/DSL-R1/data/email"
MODEL_PATH = "/data/youchengsong/models/Qwen3-Embedding-8B"
INPUT_FILE = f"{DATA_DIR}/email_test.parquet"
MESSAGE_TRUNCATE = 2000
BATCH_SIZE = 32


def run(field: str):
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    print(f"[{field}] {len(df)} rows, GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if field == "subject":
        texts   = df["Subject"].fillna("").tolist()
        max_seq = 128
        out_path = f"{DATA_DIR}/test_subject_features.parquet"
    else:
        texts   = df["Message"].fillna("").str[:MESSAGE_TRUNCATE].tolist()
        max_seq = 512
        out_path = f"{DATA_DIR}/test_message_features.parquet"

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

    result = pd.DataFrame({
        "message_id":    df["Message_ID"].values,
        "text_features": list(embeddings),
    })
    result.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=["subject", "message"], required=True)
    args = parser.parse_args()
    run(args.field)
