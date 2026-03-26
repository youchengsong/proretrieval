#!/usr/bin/env python3
"""
用 Qwen3-Embedding-8B 生成电商 text 字段的特征向量。
支持分片并行：--shard 0 --total 4 表示处理第 0 份（共 4 份）

输入: products_final.parquet (product_id, text)
输出: product_text_features_shard{N}.parquet  或  product_text_features.parquet（合并）

用法:
  CUDA_VISIBLE_DEVICES=2 python gen_product_text_features.py --shard 0 --total 4
  CUDA_VISIBLE_DEVICES=4 python gen_product_text_features.py --shard 1 --total 4
  CUDA_VISIBLE_DEVICES=5 python gen_product_text_features.py --shard 2 --total 4
  CUDA_VISIBLE_DEVICES=6 python gen_product_text_features.py --shard 3 --total 4
  python gen_product_text_features.py --merge  # 合并
"""

import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR   = "/data/youchengsong/DSL-R1/data"
MODEL_PATH = "/data/youchengsong/models/Qwen3-Embedding-8B"
INPUT_FILE  = f"{DATA_DIR}/products_final.parquet"
OUTPUT_FILE = f"{DATA_DIR}/product_text_features.parquet"

BATCH_SIZE = 32


def run_shard(shard: int, total: int):
    df = pd.read_parquet(INPUT_FILE, columns=["product_id", "text"])
    # 按 shard 切片
    indices = list(range(shard, len(df), total))
    df = df.iloc[indices].reset_index(drop=True)
    print(f"[shard {shard}/{total}] {len(df)} products, GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model = SentenceTransformer(MODEL_PATH, device="cuda")
    model.max_seq_length = 512

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"[shard {shard}] embedding shape: {embeddings.shape}")

    out_path = f"{DATA_DIR}/product_text_features_shard{shard}.parquet"
    result = pd.DataFrame({
        "product_id":    df["product_id"].values,
        "text_features": list(embeddings),
    })
    result.to_parquet(out_path, index=False)
    print(f"[shard {shard}] Saved to {out_path}")


def merge(total: int):
    parts = []
    for i in range(total):
        path = f"{DATA_DIR}/product_text_features_shard{i}.parquet"
        parts.append(pd.read_parquet(path))
        print(f"Loaded shard {i}: {len(parts[-1])} rows")
    merged = pd.concat(parts, ignore_index=True)
    # 按原始顺序排列
    original = pd.read_parquet(INPUT_FILE, columns=["product_id"])
    merged = original.merge(merged, on="product_id", how="left")
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"Merged {len(merged)} rows → {OUTPUT_FILE}")
    # 清理 shard 文件
    for i in range(total):
        os.remove(f"{DATA_DIR}/product_text_features_shard{i}.parquet")
    print("Shard files removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge(args.total)
    else:
        run_shard(args.shard, args.total)
