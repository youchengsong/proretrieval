"""
Prepare Search-R1 training data.
Single-turn version: model generates <think>reasoning</think><search>NL query</search>
Reward function internally performs BM25 retrieval and checks GT match.
"""

import json
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = "/root/autodl-tmp/search_r1"
DATA_DIR = "/root/autodl-tmp/data"

ECOMMERCE_SEARCH_PROMPT = """Given a user query, find the most relevant product by searching the product catalog.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> P001 </answer>.
Query: """

EMAIL_SEARCH_PROMPT = """Given a user query, find the most relevant email by searching the email archive.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> MSG001 </answer>.
Query: """


def create_sft_data_search_r1(sft_jsonl_path, output_path, scenario="ecommerce"):
    """Create SFT data in Search-R1 multi-turn format.

    Format: <think>reasoning</think><search>NL query</search>
    <information>Doc 1 ...</information>
    <think>analysis</think><answer>ID</answer>
    """
    id_field = "product_id" if scenario == "ecommerce" else "message_id"
    prompt_template = ECOMMERCE_SEARCH_PROMPT if scenario == "ecommerce" else EMAIL_SEARCH_PROMPT

    records = []
    with open(sft_jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])

            user_msg = None
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break

            assistant_msg = None
            for msg in messages:
                if msg["role"] == "assistant":
                    assistant_msg = msg["content"]
                    break

            if not user_msg or not assistant_msg:
                continue

            m = re.search(r"<answer>(.*?)</answer>", assistant_msg, re.DOTALL)
            if not m:
                continue

            try:
                dsl = json.loads(m.group(1).strip())
            except:
                continue

            gt_retrieval_queries = []
            for r in dsl.get("retrieval_list", []):
                if r.get("query"):
                    gt_retrieval_queries.append(r["query"])

            search_query = gt_retrieval_queries[0] if gt_retrieval_queries else user_msg

            completion = f"""<think>
The user is searching for specific items. Let me formulate a search query to find relevant results.
</think>
<search>{search_query}</search>
<information>
[Search results would appear here during training]
</information>
<think>
Based on the search results, I can identify the most relevant item.
</think>
<answer> {id_field}_placeholder </answer>"""

            record = {
                "messages": [
                    {"role": "user", "content": prompt_template + user_msg},
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

    print(f"[Search-R1 SFT] {scenario}: {len(records)} samples -> {output_path}")


def create_rl_data_search_r1(rl_parquet_path, output_path, scenario="ecommerce"):
    """Convert existing RL parquet to Search-R1 format."""
    prompt_template = ECOMMERCE_SEARCH_PROMPT if scenario == "ecommerce" else EMAIL_SEARCH_PROMPT

    df = pd.read_parquet(rl_parquet_path)
    print(f"[Search-R1 RL] {scenario}: loaded {len(df)} rows")

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
            {"role": "user", "content": prompt_template + user_query}
        ]

        record = {
            "data_source": f"search_r1_{scenario}",
            "prompt": new_prompt,
            "ability": "search",
            "reward_model": json.dumps({
                "style": "rule",
                "ground_truth": {
                    "target": gt if isinstance(gt, list) else list(gt),
                    "db_name": scenario
                }
            }),
            "extra_info": json.dumps(extra)
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[Search-R1 RL] {scenario}: {len(records)} samples -> {output_path}")


def build_bm25_corpus(json_path, corpus_path, scenario="ecommerce"):
    """Build BM25 corpus from product/email data."""
    with open(json_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    corpus = []
    for item in data:
        if scenario == "ecommerce":
            doc_id = item.get("product_id", "")
            text_parts = []
            for field in ["brand", "category", "color", "text"]:
                v = item.get(field, "")
                if v:
                    text_parts.append(str(v))
            text = " ".join(text_parts)
            title = item.get("brand", "") + " " + item.get("category", "")
        else:
            doc_id = item.get("message_id", "")
            text_parts = []
            for field in ["sender_name", "subject", "content", "folder_labels"]:
                v = item.get(field, "")
                if v:
                    text_parts.append(str(v))
            text = " ".join(text_parts)
            title = item.get("subject", "")

        corpus.append({
            "id": doc_id,
            "title": title.strip(),
            "text": text.strip()
        })

    with open(corpus_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[BM25 Corpus] {scenario}: {len(corpus)} docs -> {corpus_path}")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    # Build BM25 corpus for reward function
    build_bm25_corpus(
        f"{DATA_DIR}/ecommerce/products_test.json",
        f"{BASE_DIR}/corpus/ecommerce_corpus.jsonl",
        "ecommerce"
    )
    build_bm25_corpus(
        f"{DATA_DIR}/email/email_test.json",
        f"{BASE_DIR}/corpus/email_corpus.jsonl",
        "email"
    )

    # Create Search-R1 SFT data
    create_sft_data_search_r1(
        f"{DATA_DIR}/ecommerce/sft_train.jsonl",
        f"{BASE_DIR}/sft_data/ecommerce_search_sft.jsonl",
        "ecommerce"
    )
    create_sft_data_search_r1(
        f"{DATA_DIR}/email/sft_train.jsonl",
        f"{BASE_DIR}/sft_data/email_search_sft.jsonl",
        "email"
    )

    # Create Search-R1 RL data
    create_rl_data_search_r1(
        f"{DATA_DIR}/ecommerce/product_rl_train_filtered.parquet",
        f"{BASE_DIR}/rl_data/ecommerce_search_rl_train.parquet",
        "ecommerce"
    )
    create_rl_data_search_r1(
        f"{DATA_DIR}/email/email_rl_train_fixed.parquet",
        f"{BASE_DIR}/rl_data/email_search_rl_train.parquet",
        "email"
    )

    print("\n[DONE] All Search-R1 data prepared.")
