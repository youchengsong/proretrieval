#!/usr/bin/env python3
"""Convert ecommerce queries to DeepRetrieval training parquet format.

Follows the exact format from DeepRetrieval's data_preprocess scripts:
- prompt: Qwen2.5-Instruct chat template with system + user message
- reward_model.ground_truth.target: list of ground truth product IDs
- data_source: dataset name + split (used to select reward function)
"""
import json
import os
import random
from datasets import Dataset

TRAIN_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/train/queries_product_train_step9.jsonl"
TEST_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
OUTPUT_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/data"

INSTRUCTION = """You are a product search query optimizer. Given a customer's product search query, generate an optimized Boolean search query to retrieve the most relevant products from a product database using BM25 retrieval.

The product database contains fields: Brand, Color, Category, and product description text. Your Boolean query will be matched against these fields."""

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
{instruction}
Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{{"query": "...."}}
</answer>.
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

Here's the customer query:
{query}
Assistant: Let me think step by step.
<think>
"""


def load_queries(filepath):
    queries = []
    with open(filepath) as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)
    return queries


def make_dataset(queries, split, data_source_name="ecommerce_product_sparse"):
    records = []
    for idx, q in enumerate(queries):
        gt_ids = q.get("gt_document_ids") or q.get("gt_product_ids") or []
        if not gt_ids:
            continue

        query_text = q.get("natural_language_query") or q.get("query", "")
        prompt_text = PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION,
            query=query_text,
        )

        record = {
            "data_source": f"{data_source_name}_{split}",
            "prompt": [{"role": "user", "content": prompt_text}],
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

    train_ds = Dataset.from_list(train_records)
    test_ds = Dataset.from_list(test_records)
    val_ds = Dataset.from_list(val_records)

    train_ds.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
    test_ds.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))
    val_ds.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"))

    print(f"Saved parquet files to {OUTPUT_DIR}")
    print(f"  train: {len(train_ds)} samples")
    print(f"  test:  {len(test_ds)} samples")
    print(f"  val:   {len(val_ds)} samples")

    sample = train_records[0]
    prompt_words = len(sample["prompt"][0]["content"].split())
    print(f"  Sample prompt length: {prompt_words} words")


if __name__ == "__main__":
    main()
