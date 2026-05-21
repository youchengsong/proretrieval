"""
从本地文件合并 SFT 数据：
  product_rl_train.parquet         → prompt (system + user)
  queries_product_train_step9.jsonl → dsl (gold answer)
输出: train.jsonl
"""
import json
import pandas as pd

BASE_DIR   = "/home/sunzhen/sft_localSearch"
PARQUET    = f"{BASE_DIR}/ecommerce/train/product_rl_train.parquet"
STEP9      = f"{BASE_DIR}/ecommerce/train/queries_product_train_step9.jsonl"
OUT_PATH   = f"{BASE_DIR}/train.jsonl"

print("Reading parquet...")
df = pd.read_parquet(PARQUET)
id2prompt = {}
for _, row in df.iterrows():
    extra  = json.loads(row["extra_info"])
    qid    = extra["query_id"]
    prompt = json.loads(row["prompt"])
    id2prompt[qid] = prompt
print(f"  {len(id2prompt)} prompts loaded.")

print("Reading step9 DSL...")
id2dsl = {}
with open(STEP9) as f:
    for line in f:
        d = json.loads(line)
        id2dsl[d["query_id"]] = d["dsl"]
print(f"  {len(id2dsl)} DSLs loaded.")

print("Merging and writing...")
count = 0
with open(OUT_PATH, "w") as fout:
    for qid, prompt_msgs in id2prompt.items():
        if qid not in id2dsl:
            continue
        dsl      = id2dsl[qid]
        response = "<answer>\n" + json.dumps(dsl, ensure_ascii=False) + "\n</answer>"
        messages = prompt_msgs + [{"role": "assistant", "content": response}]
        fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        count += 1

print(f"Done. {count} samples written to {OUT_PATH}")
