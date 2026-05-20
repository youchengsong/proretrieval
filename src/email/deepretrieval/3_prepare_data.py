#!/usr/bin/env python3
"""Convert email queries to DeepRetrieval training parquet format.

Reads from existing RL parquet files (which contain NL queries and GT IDs)
and creates new parquet files with DeepRetrieval-specific Boolean query prompts.
Also supports reading from JSONL files if available.
"""
import json
import os
import random

import numpy as np
import pandas as pd
from datasets import Dataset

DATA_DIR = "/root/autodl-tmp/data/email"
TRAIN_RL_PARQUET = os.path.join(DATA_DIR, "email_rl_train_fixed.parquet")
TEST_RL_PARQUET = os.path.join(DATA_DIR, "test/email_rl_test_fixed.parquet")
TRAIN_JSONL = os.path.join(DATA_DIR, "train/queries_email_train_step9.jsonl")
TEST_JSONL = os.path.join(DATA_DIR, "test/queries_email_test_step9.jsonl")
OUTPUT_DIR = "/root/autodl-tmp/DeepRetrieval_email/data"

SYSTEM_PROMPT = """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.

You are an email search query optimizer. Given a user's email search query, generate an optimized Boolean search query to retrieve the most relevant emails from an email database using BM25 retrieval.

The email database contains fields: From (sender email), To (recipient emails), Subject, and Message body text. Your Boolean query will be matched against these fields.

Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{"query": "...."}
</answer>.
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."""


def extract_from_rl_parquet(filepath):
    """Extract NL queries and GT IDs from existing RL parquet files."""
    df = pd.read_parquet(filepath)
    queries = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        nl_query = ""
        for msg in prompt:
            if msg.get("role") == "user":
                nl_query = msg["content"]
                break

        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt_ids = rm.get("ground_truth", [])
        if isinstance(gt_ids, np.ndarray):
            gt_ids = gt_ids.tolist()
        if isinstance(gt_ids, dict):
            gt_ids = gt_ids.get("target", [])
            if isinstance(gt_ids, np.ndarray):
                gt_ids = gt_ids.tolist()

        extra = row.get("extra_info", {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        if isinstance(extra, dict):
            qid = extra.get("query_id", str(idx))
        else:
            qid = str(idx)

        queries.append({
            "query_id": qid,
            "natural_language_query": nl_query,
            "gt_document_ids": gt_ids,
        })
    return queries


def load_queries_jsonl(filepath):
    queries = []
    with open(filepath) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_queries(jsonl_path, parquet_path):
    """Try JSONL first (has richer data), fall back to RL parquet."""
    if os.path.exists(jsonl_path):
        print(f"Loading from JSONL: {jsonl_path}")
        return load_queries_jsonl(jsonl_path)
    elif os.path.exists(parquet_path):
        print(f"Loading from RL parquet: {parquet_path}")
        return extract_from_rl_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"Neither {jsonl_path} nor {parquet_path} found")


def make_dataset(queries, split, data_source_name="email_sparse"):
    records = []
    for idx, q in enumerate(queries):
        gt_ids = q.get("gt_document_ids") or []
        query_text = q.get("natural_language_query") or q.get("query", "")
        if not query_text:
            continue
        record = {
            "data_source": f"{data_source_name}_{split}",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ],
            "ability": "email_retrieval",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": gt_ids},
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "query_id": q.get("query_id", str(idx)),
            },
        }
        records.append(record)
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_queries = load_queries(TRAIN_JSONL, TRAIN_RL_PARQUET)
    test_queries = load_queries(TEST_JSONL, TEST_RL_PARQUET)
    print(f"Loaded {len(train_queries)} train, {len(test_queries)} test queries")

    train_records = make_dataset(train_queries, "train")
    test_records = make_dataset(test_queries, "test")
    print(f"Valid records: {len(train_records)} train, {len(test_records)} test")

    random.seed(42)
    random.shuffle(train_records)

    val_size = min(300, len(test_records))
    val_records = test_records[:val_size]

    Dataset.from_list(train_records).to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
    Dataset.from_list(test_records).to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))
    Dataset.from_list(val_records).to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"))
    print(f"Saved: train={len(train_records)}, test={len(test_records)}, val={len(val_records)}")


if __name__ == "__main__":
    main()
