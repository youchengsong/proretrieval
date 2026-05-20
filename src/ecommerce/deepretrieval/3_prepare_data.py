#!/usr/bin/env python3
"""Convert ecommerce queries to DeepRetrieval training parquet format.
Fixed: proper chat format (system + user messages, no embedded special tokens).
"""
import json
import os
import random
from datasets import Dataset

TRAIN_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/train/queries_product_train_step9.jsonl"
TEST_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
OUTPUT_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/data"

SYSTEM_PROMPT = """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.

You are a product search query optimizer. Given a customer's product search query, generate an optimized Boolean search query to retrieve the most relevant products from a product database using BM25 retrieval.

The product database contains fields: Brand, Color, Category, and product description text. Your Boolean query will be matched against these fields.

Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{"query": "...."}
</answer>.
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."""

def load_queries(filepath):
    queries = []
    with open(filepath) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def make_dataset(queries, split, data_source_name="ecommerce_product_sparse"):
    records = []
    for idx, q in enumerate(queries):
        gt_ids = q.get("gt_document_ids") or q.get("gt_product_ids") or []
        if not gt_ids:
            continue
        query_text = q.get("natural_language_query") or q.get("query", "")
        record = {
            "data_source": f"{data_source_name}_{split}",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ],
            "ability": "product_retrieval",
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
    train_queries = load_queries(TRAIN_FILE)
    test_queries = load_queries(TEST_FILE)
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
