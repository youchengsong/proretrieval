#!/usr/bin/env python3
"""Convert ecommerce queries to DeepRetrieval training parquet format.

For verl 0.7.1 with custom_reward_function interface.
Format matches the existing RL training data schema (product_rl_test.parquet).

DeepRetrieval trains the model to generate Boolean queries for BM25 retrieval.
The model output format: <think>...</think><answer>{"query": "..."}</answer>
"""
import json
import os
import argparse
from datasets import Dataset


SYSTEM_PROMPT = """You are an expert in product search query generation. Given a natural language product search query, generate a Boolean search query using AND, OR, NOT operators and parentheses to retrieve relevant products via BM25.

The product database contains product titles, descriptions, brands, and categories as searchable text. Your Boolean query will be matched against all these fields.

Show your reasoning in <think> </think> tags. Then provide your Boolean query in JSON format within <answer> </answer> tags.

Example:
<think>
The user wants wireless noise-cancelling headphones. I should combine key terms.
</think>
<answer>
{"query": "(\"wireless\" OR \"bluetooth\") AND \"headphones\" AND (\"noise cancelling\" OR \"ANC\")"}
</answer>

Important:
- Use double quotes around phrases
- Use AND, OR, NOT operators (uppercase)
- Use parentheses to group terms logically
- Focus on product attributes, features, and descriptive terms"""


def load_queries(filepath):
    queries = []
    with open(filepath) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--test_file', required=True)
    args = parser.parse_args()

    train_queries = load_queries(args.train_file)
    test_queries = load_queries(args.test_file)
    print(f"Loaded {len(train_queries)} train, {len(test_queries)} test queries")

    def build_records(queries, split):
        records = []
        for i, q in enumerate(queries):
            gt_ids = q.get("gt_document_ids") or []
            if not gt_ids:
                continue
            nl_query = q.get("natural_language_query", "")
            records.append({
                "data_source": f"ecommerce_{split}",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": nl_query},
                ],
                "ability": "product_retrieval",
                "reward_model": json.dumps({
                    "style": "rule",
                    "ground_truth": {"target": gt_ids},
                }),
                "extra_info": json.dumps({
                    "query_id": q.get("query_id", ""),
                    "index": i,
                    "level": q.get("level"),
                    "n": q.get("n"),
                    "modalities": q.get("modalities", []),
                    "split": split,
                }),
            })
        return records

    train_records = build_records(train_queries, "train")
    test_records = build_records(test_queries, "test")
    val_records = build_records(test_queries[:300], "val")
    print(f"Valid: {len(train_records)} train, {len(test_records)} test, {len(val_records)} val")

    train_ds = Dataset.from_list(train_records).shuffle(seed=42)
    test_ds = Dataset.from_list(test_records)
    val_ds = Dataset.from_list(val_records)

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.output_dir, "test.parquet"))
    val_ds.to_parquet(os.path.join(args.output_dir, "val.parquet"))

    print(f"\nSaved to {args.output_dir}/")
    print(f"  train: {len(train_ds)}")
    print(f"  test:  {len(test_ds)}")
    print(f"  val:   {len(val_ds)}")

    print(f"\n=== Sample (train[0]) ===")
    row = train_ds[0]
    print(f"data_source: {row['data_source']}")
    print(f"prompt[0] (system): {row['prompt'][0]['content'][:100]}...")
    print(f"prompt[1] (user):   {row['prompt'][1]['content']}")
    gt = json.loads(row['reward_model'])['ground_truth']['target']
    print(f"ground_truth: {len(gt)} doc IDs, first 3: {gt[:3]}")
    print(f"extra_info: {row['extra_info']}")
