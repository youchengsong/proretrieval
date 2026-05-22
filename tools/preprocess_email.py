#!/usr/bin/env python3
"""
Convert queries_email_{train/test}_step9.jsonl → verl parquet format.

Output:
    /data/youchengsong/DSL-R1/data/email/train/email_train.parquet
    /data/youchengsong/DSL-R1/data/email/test/email_test_rl.parquet
"""

import json
import os
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/data/youchengsong/DSL-R1/data/email")
SPLITS = {
    "train": (BASE / "train/queries_email_train_step9.jsonl",
              BASE / "train/email_rl_train.parquet"),
    "test":  (BASE / "test/queries_email_test_step9.jsonl",
              BASE / "test/email_rl_test.parquet"),
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert email search assistant. Given a natural language query, generate a DSL to retrieve matching emails from the Enron email corpus.

## Output format
Output ONLY a JSON object wrapped in <answer></answer>:
<answer>
{
  "sql": "WHERE <conditions>",
  "retrieval_list": [
    {"type": "text", "field": "<field>", "query": "<search text>"}
  ]
}
</answer>

## Email schema
Structured fields (use directly in SQL):
- "From"     TEXT    -- sender email address
- "To"       TEXT    -- recipient email address
- year       INTEGER -- year the email was sent

Text fields (must use <text_x> placeholder in SQL):
- Subject    -- email subject line
- Message    -- email body content

## DSL rules
- For text fields: use `message_id IN <text_x>` (positive) or `message_id NOT IN <text_x>` (negative)
- Assign <text_0>, <text_1>, ... in order of appearance in SQL
- For structured fields: use standard SQL operators (=, >, <, >=, <=, BETWEEN)
- Combine conditions with AND / OR / NOT

## Examples

Query: emails with a subject about quarterly earnings report
<answer>
{"sql": "WHERE message_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "Subject", "query": "quarterly earnings report"}]}
</answer>

Query: emails from john.smith@enron.com sent in 2001
<answer>
{"sql": "WHERE \"From\" = 'john.smith@enron.com' AND year = 2001", "retrieval_list": []}
</answer>

Query: emails sent to all.employees@enron.com about merger announcement
<answer>
{"sql": "WHERE \"To\" = 'all.employees@enron.com' AND message_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "Message", "query": "merger announcement restructuring"}]}
</answer>

Query: emails with a subject about options trading, or any email from marcus.nettelton@enron.com
<answer>
{"sql": "WHERE message_id IN <text_0> OR \"From\" = 'marcus.nettelton@enron.com'", "retrieval_list": [{"type": "text", "field": "Subject", "query": "options trading"}]}
</answer>

Query: emails from jeff.dasovich@enron.com that are not about internal meeting schedule
<answer>
{"sql": "WHERE \"From\" = 'jeff.dasovich@enron.com' AND message_id NOT IN <text_0>", "retrieval_list": [{"type": "text", "field": "Subject", "query": "internal meeting schedule"}]}
</answer>

Query: emails sent between 2000 and 2002 containing merger discussion
<answer>
{"sql": "WHERE year >= 2000 AND year <= 2002 AND message_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "Message", "query": "merger discussion"}]}
</answer>
"""


def convert_split(input_path: Path, output_path: Path, split_name: str):
    rows = []
    skipped = 0
    with open(input_path) as f:
        for idx, line in enumerate(f):
            q = json.loads(line)
            nl = q.get("natural_language_query")
            gt = q.get("gt_document_ids")
            if not nl:
                skipped += 1
                continue
            rows.append({
                "data_source": "email",
                "prompt": json.dumps([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": nl},
                ], ensure_ascii=False),
                "ability": "retrieval",
                "reward_model": json.dumps({
                    "style": "rule",
                    "ground_truth": gt,
                    "dsl_ground_truth": json.dumps(q["dsl"], ensure_ascii=False),
                }, ensure_ascii=False),
                "extra_info": json.dumps({
                    "query_id":   q["query_id"],
                    "index":      idx,
                    "level":      q.get("level"),
                    "n":          q.get("n"),
                    "modalities": q.get("modalities", []),
                    "split":      split_name,
                }, ensure_ascii=False),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"[{split_name}] {len(df)} rows → {output_path}  (skipped {skipped})")


if __name__ == "__main__":
    for split_name, (inp, out) in SPLITS.items():
        convert_split(inp, out, split_name)
    print("Done.")
