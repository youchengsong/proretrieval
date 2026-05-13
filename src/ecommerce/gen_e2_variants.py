#!/usr/bin/env python3
"""Generate E2 Action Space Ablation training/test data variants.

Creates modified parquet files with restricted system prompts for each action space.
"""
import pandas as pd, json, os, copy, numpy as np

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce"
TRAIN_FILE = os.path.join(DATA_DIR, "train/product_rl_train_filtered.parquet")
TEST_FILE = os.path.join(DATA_DIR, "test/product_rl_test_filtered.parquet")
OUT_DIR = os.path.join(DATA_DIR, "e2_variants")
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPTS = {
    "sql_only": """You are an expert product search assistant. Given a natural language query, generate a DSL to retrieve matching products from an e-commerce catalog.

## Output format
Output ONLY a JSON object wrapped in <answer></answer>:
<answer>
{
  "sql": "WHERE <conditions>",
  "retrieval_list": []
}
</answer>

## Product schema
Structured fields (use in SQL WHERE clause):
- brand          TEXT    -- product brand name
- category       TEXT    -- product category
- price          REAL    -- product price in USD
- average_rating REAL    -- average customer rating (0.0-5.0)
- rating_number  INTEGER -- number of customer reviews
- color          TEXT    -- product color

## DSL rules
- Use standard SQL operators (=, >, <, >=, <=, BETWEEN, LIKE, IN)
- Combine conditions with AND / OR / NOT
- retrieval_list must always be empty []
- String matching is case-insensitive

## Examples

Query: Nike running shoes under $100
<answer>
{"sql": "WHERE brand = 'Nike' AND price < 100", "retrieval_list": []}
</answer>

Query: highly rated Sony headphones with more than 500 reviews
<answer>
{"sql": "WHERE brand = 'Sony' AND average_rating >= 4.5 AND rating_number > 500", "retrieval_list": []}
</answer>""",

    "sql_text": """You are an expert product search assistant. Given a natural language query, generate a DSL to retrieve matching products from an e-commerce catalog.

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

## Product schema
Structured fields (use directly in SQL):
- brand          TEXT    -- product brand name
- category       TEXT    -- product category
- price          REAL    -- product price in USD
- average_rating REAL    -- average customer rating (0.0-5.0)
- rating_number  INTEGER -- number of customer reviews
- color          TEXT    -- product color

Text fields (must use <text_x> placeholder in SQL):
- title          -- product title
- description    -- product description

## DSL rules
- For text fields: use `product_id IN <text_x>` (positive) or `product_id NOT IN <text_x>` (negative)
- Assign <text_0>, <text_1>, ... in order of appearance
- For structured fields: use standard SQL operators (=, >, <, >=, <=, BETWEEN, LIKE)
- Combine conditions with AND / OR / NOT

## Examples

Query: Nike running shoes under $100
<answer>
{"sql": "WHERE brand = 'Nike' AND price < 100 AND product_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "title", "query": "running shoes"}]}
</answer>

Query: highly rated Sony headphones with more than 500 reviews
<answer>
{"sql": "WHERE brand = 'Sony' AND average_rating >= 4.5 AND rating_number > 500", "retrieval_list": []}
</answer>""",

    "text_only": """You are an expert product search assistant. Given a natural language query, generate a DSL to retrieve matching products from an e-commerce catalog.

## Output format
Output ONLY a JSON object wrapped in <answer></answer>:
<answer>
{
  "sql": "WHERE product_id IN <text_0>",
  "retrieval_list": [
    {"type": "text", "field": "<field>", "query": "<search text>"}
  ]
}
</answer>

## Product schema
Text fields (use <text_x> placeholder in SQL):
- title          -- product title
- description    -- product description

## DSL rules
- Use `product_id IN <text_x>` to include text search results
- Use `product_id NOT IN <text_x>` to exclude text search results
- Assign <text_0>, <text_1>, ... in order of appearance
- Combine conditions with AND / OR / NOT
- SQL WHERE clause should ONLY contain product_id IN/NOT IN conditions

## Examples

Query: Nike running shoes under $100
<answer>
{"sql": "WHERE product_id IN <text_0>", "retrieval_list": [{"type": "text", "field": "title", "query": "Nike running shoes under $100"}]}
</answer>

Query: vintage leather handbag but not crossbody
<answer>
{"sql": "WHERE product_id IN <text_0> AND product_id NOT IN <text_1>", "retrieval_list": [{"type": "text", "field": "title", "query": "vintage leather handbag"}, {"type": "text", "field": "title", "query": "crossbody bag"}]}
</answer>""",

    "sql_image": """You are an expert product search assistant. Given a natural language query, generate a DSL to retrieve matching products from an e-commerce catalog.

## Output format
Output ONLY a JSON object wrapped in <answer></answer>:
<answer>
{
  "sql": "WHERE <conditions>",
  "retrieval_list": [
    {"type": "image", "field": "image_url", "query": "<visual description>"}
  ]
}
</answer>

## Product schema
Structured fields (use directly in SQL):
- brand          TEXT    -- product brand name
- category       TEXT    -- product category
- price          REAL    -- product price in USD
- average_rating REAL    -- average customer rating (0.0-5.0)
- rating_number  INTEGER -- number of customer reviews
- color          TEXT    -- product color

Image fields (must use <image_x> placeholder in SQL):
- image_url      -- product image

## DSL rules
- For image fields: use `product_id IN <image_x>` (positive) or `product_id NOT IN <image_x>` (negative)
- Assign <image_0>, <image_1>, ... in order of appearance
- For structured fields: use standard SQL operators (=, >, <, >=, <=, BETWEEN, LIKE)
- Combine conditions with AND / OR / NOT

## Examples

Query: Nike running shoes under $100
<answer>
{"sql": "WHERE brand = 'Nike' AND price < 100 AND product_id IN <image_0>", "retrieval_list": [{"type": "image", "field": "image_url", "query": "running shoes"}]}
</answer>

Query: highly rated Sony headphones with more than 500 reviews
<answer>
{"sql": "WHERE brand = 'Sony' AND average_rating >= 4.5 AND rating_number > 500", "retrieval_list": []}
</answer>""",
}


def make_variant(input_parquet, variant_name, system_prompt, output_path):
    df = pd.read_parquet(input_parquet)
    new_prompts = []
    for _, row in df.iterrows():
        prompt_val = row['prompt']
        if isinstance(prompt_val, str):
            msgs = json.loads(prompt_val)
        elif isinstance(prompt_val, (list, np.ndarray)):
            msgs = list(prompt_val)
            if isinstance(msgs[0], str):
                msgs = [json.loads(m) if isinstance(m, str) else m for m in msgs]
        else:
            msgs = list(prompt_val)
        msgs[0]['content'] = system_prompt
        new_prompts.append(json.dumps(msgs, ensure_ascii=False))
    df = df.copy()
    df['prompt'] = new_prompts
    df.to_parquet(output_path, index=False)
    print(f"  {variant_name}: {len(df)} samples -> {output_path}")


for variant, prompt in SYSTEM_PROMPTS.items():
    print(f"Generating {variant}...")
    make_variant(TRAIN_FILE, variant, prompt,
                 os.path.join(OUT_DIR, f"train_{variant}.parquet"))
    make_variant(TEST_FILE, variant, prompt,
                 os.path.join(OUT_DIR, f"test_{variant}.parquet"))

print("\nDone! All variants in", OUT_DIR)
