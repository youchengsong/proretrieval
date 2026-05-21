#!/usr/bin/env python3
"""Prepare email SQL-R1 RL training data in verl parquet format.

Reads queries_email_train_step9.jsonl, filters out queries without GT,
and creates parquet files for verl GRPO training.

Usage:
    python prepare_data.py
"""
import json, os
import numpy as np
import pandas as pd

DATA_DIR = "/home/sunzhen/LocalSearch/data/email"
TRAIN_QUERIES = os.path.join(DATA_DIR, "train/queries_email_train_step9.jsonl")
TEST_QUERIES = os.path.join(DATA_DIR, "test/queries_email_test_step9.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "sql_r1")

SYSTEM_PROMPT = """You are a helpful email search SQL expert assistant.
The assistant first thinks about how to write the SQL WHERE clause by analyzing the user query and database schema, then provides the final WHERE clause.
The reasoning process and WHERE clause are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL WHERE clause within ```sql ``` tags.

Database Schema:
CREATE TABLE emails (
    message_id TEXT PRIMARY KEY,
    "From" TEXT,
    "To" TEXT,
    Subject TEXT,
    Message TEXT,
    Date TEXT,
    year INTEGER,
    day INTEGER
);

Rules:
1. Use exact match (=) for "From" and "To" fields with full email addresses.
2. Use LIKE with % wildcards for Subject and Message text search (case-insensitive).
3. Use exact comparisons (=, >, <, >=, <=, BETWEEN) for year and day.
4. Use AND/OR/NOT to combine conditions as needed.
5. When the query mentions email content or topics, use Message LIKE '%keyword%' or Subject LIKE '%keyword%'.

For example:
<think>
The user is looking for emails from john.smith@enron.com about meeting schedules in 2001. Let me analyze:
1. Sender filter: "from john.smith@enron.com" → exact match on "From" field
2. Topic: "meeting schedules" → text search on Subject and Message
3. Year: "in 2001" → year = 2001
I'll combine with AND since all conditions must be satisfied.
</think>
<answer>
The query filters for sender, meeting-related content, and year 2001:
```sql
"From" = 'john.smith@enron.com' AND (Subject LIKE '%meeting%' OR Message LIKE '%meeting%') AND (Subject LIKE '%schedule%' OR Message LIKE '%schedule%') AND year = 2001
```
</answer>

Another example:
<think>
The user wants emails to sara.shackleton@enron.com or emails about gas pipeline contracts. Let me analyze:
1. Recipient: "to sara.shackleton@enron.com" → exact match on "To"
2. Topic: "gas pipeline contracts" → text search
These are OR conditions since either one should match.
</think>
<answer>
The query uses OR to combine recipient filter and topic search:
```sql
"To" = 'sara.shackleton@enron.com' OR (Subject LIKE '%gas%' AND Subject LIKE '%pipeline%') OR (Message LIKE '%gas%' AND Message LIKE '%pipeline%' AND Message LIKE '%contract%')
```
</answer>"""


def load_queries(path):
    queries = []
    with open(path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def make_parquet(queries, output_path, split_name):
    rows = []
    skipped = 0
    for q in queries:
        gt_ids = q.get("gt_document_ids") or []

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["natural_language_query"]},
        ]

        reward_model = {
            "style": "rule",
            "ground_truth": {
                "gt_email_ids": np.array(gt_ids, dtype=object),
            },
        }

        extra_info = {
            "query_id": q["query_id"],
            "level": q.get("level"),
            "n": q.get("n"),
            "modalities": q.get("modalities"),
            "split": split_name,
        }

        rows.append({
            "data_source": "email_sql",
            "prompt": prompt,
            "ability": "retrieval",
            "reward_model": reward_model,
            "extra_info": extra_info,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_empty_gt = sum(1 for q in queries if not (q.get("gt_document_ids") or []))
    print(f"  {split_name}: {len(rows)} samples saved ({n_empty_gt} with empty GT, kept)")
    return len(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training queries...")
    train_queries = load_queries(TRAIN_QUERIES)
    print(f"  Total: {len(train_queries)}")

    print("Loading test queries...")
    test_queries = load_queries(TEST_QUERIES)
    print(f"  Total: {len(test_queries)}")

    print("\nCreating training parquet...")
    n_train = make_parquet(train_queries, os.path.join(OUTPUT_DIR, "train.parquet"), "train")

    print("Creating test parquet...")
    n_test = make_parquet(test_queries, os.path.join(OUTPUT_DIR, "test.parquet"), "test")

    print("Creating val parquet (first 300 of test)...")
    val_queries = test_queries[:300]
    n_val = make_parquet(val_queries, os.path.join(OUTPUT_DIR, "val.parquet"), "val")

    print(f"\nDone: train={n_train}, test={n_test}, val={n_val}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
