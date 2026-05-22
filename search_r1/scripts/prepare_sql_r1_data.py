"""
Prepare SQL-R1 training data from existing LocalSearch data.
1. Convert SFT data: extract pure SQL from gold_dsl, create SQL-R1 format
2. Convert RL data: create parquet with SQL ground truth
3. Create SQLite databases for reward function execution
"""

import json
import re
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = "/root/autodl-tmp/sql_r1"
DATA_DIR = "/root/autodl-tmp/data"

ECOMMERCE_SYSTEM_PROMPT = """You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question and database schema, then provides the final SQL query.
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL query within ```sql ``` tags.

Database Schema:
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    brand TEXT,
    color TEXT,
    category TEXT,
    price REAL,
    average_rating REAL,
    rating_number INTEGER
);

For example:
<think>
The user wants to find products from a specific brand within a price range.
I need to filter the products table by brand and price.
</think>
<answer>
```sql
SELECT DISTINCT product_id FROM products WHERE brand = 'Sony' AND price < 200;
```
</answer>"""

EMAIL_SYSTEM_PROMPT = """You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question and database schema, then provides the final SQL query.
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL query within ```sql ``` tags.

Database Schema:
CREATE TABLE emails (
    message_id TEXT PRIMARY KEY,
    received_date TEXT,
    is_draft INTEGER,
    is_read INTEGER,
    is_starred INTEGER,
    is_archived INTEGER,
    thread_msg_count INTEGER,
    sender_name TEXT,
    recipient_list TEXT,
    folder_labels TEXT,
    subject TEXT,
    content TEXT,
    attachment_list TEXT
);

For example:
<think>
The user wants to find emails from a specific sender that are unread.
I need to filter the emails table by sender_name and is_read.
</think>
<answer>
```sql
SELECT DISTINCT message_id FROM emails WHERE sender_name = 'John' AND is_read = 0;
```
</answer>"""


def remove_vector_conditions(sql_where):
    """Remove vector retrieval placeholders from SQL WHERE clause."""
    if not sql_where:
        return sql_where

    cleaned = sql_where
    cleaned = re.sub(r"\s*(?:AND|OR)\s+(?:product_id|message_id)\s+(?:NOT\s+)?IN\s+<(?:text|image)_\d+>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:product_id|message_id)\s+(?:NOT\s+)?IN\s+<(?:text|image)_\d+>\s*(?:AND|OR)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:product_id|message_id)\s+(?:NOT\s+)?IN\s+<(?:text|image)_\d+>", "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"WHERE\s*(AND|OR)\s+", "WHERE ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(AND|OR)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(AND|OR)\s+(AND|OR)", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    if re.match(r"^\s*WHERE\s*$", cleaned, re.IGNORECASE) or not cleaned:
        return ""

    return cleaned


def extract_sql_from_dsl(dsl_str, id_field="product_id", table_name="products"):
    """Extract pure SQL from a DSL JSON string."""
    try:
        if isinstance(dsl_str, str):
            dsl = json.loads(dsl_str)
        else:
            dsl = dsl_str
    except (json.JSONDecodeError, TypeError):
        return None

    sql_body = dsl.get("sql", "")
    if not sql_body:
        return None

    cleaned = remove_vector_conditions(sql_body)
    if not cleaned:
        return f"SELECT DISTINCT {id_field} FROM {table_name}"

    return f"SELECT DISTINCT {id_field} FROM {table_name} {cleaned}"


def create_sft_data_sql_r1(sft_jsonl_path, output_path, scenario="ecommerce"):
    """Convert existing SFT data to SQL-R1 format."""
    system_prompt = ECOMMERCE_SYSTEM_PROMPT if scenario == "ecommerce" else EMAIL_SYSTEM_PROMPT
    id_field = "product_id" if scenario == "ecommerce" else "message_id"
    table_name = "products" if scenario == "ecommerce" else "emails"

    records = []
    with open(sft_jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])

            user_msg = None
            assistant_msg = None
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]

            if not user_msg or not assistant_msg:
                continue

            m = re.search(r"<answer>(.*?)</answer>", assistant_msg, re.DOTALL)
            if not m:
                continue

            try:
                dsl = json.loads(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                continue

            sql = extract_sql_from_dsl(dsl, id_field, table_name)
            if not sql:
                continue

            think_text = f"The user is looking for specific items. Let me analyze the query and write a SQL query to filter the {table_name} table based on the structured conditions mentioned."

            completion = f"""<think>
{think_text}
</think>
<answer>
```sql
{sql};
```
</answer>"""

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": completion}
                ]
            }
            records.append(record)

    np.random.seed(42)
    if len(records) > 2000:
        indices = np.random.choice(len(records), 2000, replace=False)
        records = [records[i] for i in sorted(indices)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SQL-R1 SFT] {scenario}: {len(records)} samples -> {output_path}")
    return records


def create_rl_data_sql_r1(rl_parquet_path, output_path, scenario="ecommerce"):
    """Convert existing RL parquet to SQL-R1 format."""
    system_prompt = ECOMMERCE_SYSTEM_PROMPT if scenario == "ecommerce" else EMAIL_SYSTEM_PROMPT
    id_field = "product_id" if scenario == "ecommerce" else "message_id"
    table_name = "products" if scenario == "ecommerce" else "emails"

    df = pd.read_parquet(rl_parquet_path)
    print(f"[SQL-R1 RL] {scenario}: loaded {len(df)} rows from {rl_parquet_path}")

    records = []
    for idx, row in df.iterrows():
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, np.ndarray):
            prompt_msgs = prompt_msgs.tolist()

        user_query = None
        for msg in prompt_msgs:
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        if not user_query:
            continue

        reward_model = row.get("reward_model", "{}")
        if isinstance(reward_model, str):
            try:
                rm = json.loads(reward_model)
            except:
                continue
        else:
            rm = reward_model

        gt = rm.get("ground_truth", [])
        if gt is None:
            gt = []
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except:
                gt = []
        if not isinstance(gt, list):
            gt = list(gt) if gt else []

        dsl_gt = rm.get("dsl_ground_truth", "")
        if isinstance(dsl_gt, str) and dsl_gt:
            try:
                dsl_obj = json.loads(dsl_gt)
                gt_sql = extract_sql_from_dsl(dsl_obj, id_field, table_name)
            except:
                gt_sql = f"SELECT DISTINCT {id_field} FROM {table_name}"
        else:
            gt_sql = f"SELECT DISTINCT {id_field} FROM {table_name}"

        extra_info = row.get("extra_info", "{}")
        if isinstance(extra_info, str):
            try:
                extra = json.loads(extra_info)
            except:
                extra = {}
        elif isinstance(extra_info, dict):
            extra = extra_info
        else:
            extra = {}

        new_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        record = {
            "data_source": f"sql_r1_{scenario}",
            "prompt": new_prompt,
            "ability": "sql",
            "reward_model": json.dumps({
                "style": "rule",
                "ground_truth": {
                    "gt_ids": gt if isinstance(gt, list) else list(gt),
                    "gt_sql": gt_sql,
                    "db_name": scenario,
                    "id_field": id_field,
                    "table_name": table_name
                }
            }),
            "extra_info": json.dumps(extra)
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[SQL-R1 RL] {scenario}: {len(records)} samples -> {output_path}")


def create_sqlite_db(json_path, db_path, scenario="ecommerce"):
    """Create SQLite database from product/email JSON."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if scenario == "ecommerce":
        cur.execute("""
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY,
                brand TEXT, color TEXT, category TEXT,
                price REAL, average_rating REAL, rating_number INTEGER
            )
        """)
        for p in data:
            cur.execute("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?)",
                (p.get("product_id"), p.get("brand"), p.get("color"),
                 p.get("category"), p.get("price"),
                 p.get("average_rating"), p.get("rating_number")))
    else:
        cur.execute("""
            CREATE TABLE emails (
                message_id TEXT PRIMARY KEY,
                received_date TEXT, is_draft INTEGER, is_read INTEGER,
                is_starred INTEGER, is_archived INTEGER, thread_msg_count INTEGER,
                sender_name TEXT, recipient_list TEXT, folder_labels TEXT,
                subject TEXT, content TEXT, attachment_list TEXT
            )
        """)
        for e in data:
            cur.execute("INSERT OR IGNORE INTO emails VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (e.get("message_id"), e.get("received_date"), e.get("is_draft"),
                 e.get("is_read"), e.get("is_starred"), e.get("is_archived"),
                 e.get("thread_msg_count"), e.get("sender_name"),
                 e.get("recipient_list"), e.get("folder_labels"),
                 e.get("subject"), e.get("content"), e.get("attachment_list")))

    conn.commit()
    count = cur.execute(f"SELECT COUNT(*) FROM {'products' if scenario == 'ecommerce' else 'emails'}").fetchone()[0]
    conn.close()
    print(f"[SQLite] {scenario}: {count} rows -> {db_path}")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    # Create SQLite databases
    create_sqlite_db(
        f"{DATA_DIR}/ecommerce/products_test.json",
        f"{BASE_DIR}/databases/ecommerce.sqlite",
        "ecommerce"
    )
    create_sqlite_db(
        f"{DATA_DIR}/email/email_test.json",
        f"{BASE_DIR}/databases/email.sqlite",
        "email"
    )

    # Create SQL-R1 SFT data
    create_sft_data_sql_r1(
        f"{DATA_DIR}/ecommerce/sft_train.jsonl",
        f"{BASE_DIR}/sft_data/ecommerce_sql_sft.jsonl",
        "ecommerce"
    )
    create_sft_data_sql_r1(
        f"{DATA_DIR}/email/sft_train.jsonl",
        f"{BASE_DIR}/sft_data/email_sql_sft.jsonl",
        "email"
    )

    # Create SQL-R1 RL data
    create_rl_data_sql_r1(
        f"{DATA_DIR}/ecommerce/product_rl_train_filtered.parquet",
        f"{BASE_DIR}/rl_data/ecommerce_sql_rl_train.parquet",
        "ecommerce"
    )
    create_rl_data_sql_r1(
        f"{DATA_DIR}/email/email_rl_train_fixed.parquet",
        f"{BASE_DIR}/rl_data/email_sql_rl_train.parquet",
        "email"
    )

    print("\n[DONE] All SQL-R1 data prepared.")
