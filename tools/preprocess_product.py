#!/usr/bin/env python3
"""
Convert queries_product_{train/test}_step9.jsonl → verl parquet format.

Output:
    /data/youchengsong/DSL-R1/data/ecommerce/train/product_rl_train.parquet
    /data/youchengsong/DSL-R1/data/ecommerce/test/product_rl_test.parquet
"""

import json
import os
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/data/youchengsong/DSL-R1/data/ecommerce")
SPLITS = {
    "train": (BASE / "train/queries_product_train_step9.jsonl",
              BASE / "train/product_rl_train.parquet"),
    "test":  (BASE / "test/queries_product_test_step9.jsonl",
              BASE / "test/product_rl_test.parquet"),
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert product search assistant. Given a natural language query, generate a DSL to retrieve matching products from an e-commerce catalog.

## Output format
Output ONLY a JSON object wrapped in <answer></answer>:
<answer>
{
  "sql": "WHERE <conditions>",
  "retrieval_list": [
    {"type": "text|image", "field": "<field>", "query": "<search text>"}
  ]
}
</answer>

## Product schema
Structured fields (use directly in SQL):
- brand          TEXT    -- product brand name
- category       TEXT    -- product category
- price          REAL    -- product price in USD
- rating         REAL    -- average customer rating (0.0–5.0)
- num_reviews    INTEGER -- number of customer reviews
- in_stock       INTEGER -- availability (1=in stock, 0=out of stock)
- release_date   TEXT    -- product release date (YYYY-MM-DD)

Text fields (must use <text_x> placeholder in SQL):
- title          -- product title
- description    -- product description
- review_summary -- summarized customer reviews

Image fields (must use <image_x> placeholder in SQL):
- image_url      -- product image

## DSL rules
- For text fields: use `product_id IN <text_x>` (positive) or `product_id NOT IN <text_x>` (negative)
- For image fields: use `product_id IN <image_x>` (positive) or `product_id NOT IN <image_x>` (negative)
- Assign <text_0>, <text_1>, ... and <image_0>, <image_1>, ... independently in order of appearance
- For structured fields: use standard SQL operators (=, >, <, >=, <=, BETWEEN)
- Combine conditions with AND / OR / NOT

## Examples

Query: Nike running shoes under $100
<answer>
{"sql": "WHERE brand = 'Nike' AND price < 100 AND product_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "title", "query": "running shoes"}]}
</answer>

Query: highly rated Sony headphones with more than 500 reviews
<answer>
{"sql": "WHERE brand = 'Sony' AND rating >= 4.5 AND num_reviews > 500", "retrieval_list": []}
</answer>

Query: products that look like a vintage leather handbag
<answer>
{"sql": "WHERE product_id IN <image_0>", "retrieval_list": [{"type": "image", "field": "image_url", "query": "vintage leather handbag brown aged texture"}]}
</answer>

Query: The North Face jackets, or products priced exactly $7.14, or something with an emerald green crystal necklace appearance
<answer>
{"sql": "WHERE brand = 'The North Face' OR price = 7.14 OR product_id IN <image_0>", "retrieval_list": [{"type": "image", "field": "image_url", "query": "emerald green crystal vintage statement necklace"}]}
</answer>

Query: in-stock electronics with good reviews about battery life, priced between $50 and $200
<answer>
{"sql": "WHERE in_stock = 1 AND price >= 50 AND price <= 200 AND product_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "review_summary", "query": "battery life performance"}]}
</answer>

Query: women's shoes that are visually similar to strappy heels but are not from Gucci
<answer>
{"sql": "WHERE product_id IN <image_0> AND brand != 'Gucci'", "retrieval_list": [{"type": "image", "field": "image_url", "query": "women strappy high heels elegant"}]}
</answer>

Query: products with waterproof description and a 5-star rating released after 2022
<answer>
{"sql": "WHERE rating = 5.0 AND release_date > '2022-12-31' AND product_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "description", "query": "waterproof weather resistant"}]}
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
                "data_source": "ecommerce",
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
