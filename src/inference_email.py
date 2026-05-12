"""
邮件测试集推理：用邮件 SFT 模型生成 DSL
输出 JSON 格式，包含原 query
"""
import json
import os
import pandas as pd
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/s3-migration/sft_localSearch_email/output/final"
TEST_PARQUET = "/home/sunzhen/LocalSearch/data/email/test/email_rl_test.parquet"
TEST_STEP9 = "/home/sunzhen/LocalSearch/data/email/test/queries_email_test_step9.jsonl"
OUTPUT_DIR = "/home/sunzhen/LocalSearch/data/email/result"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions_step9.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读原 query
print("Loading queries...")
id2query = {}
with open(TEST_STEP9) as f:
    for line in f:
        d = json.loads(line)
        id2query[d["query_id"]] = d["natural_language_query"]

# 读 prompt
print("Loading test data...")
df = pd.read_parquet(TEST_PARQUET)
query_ids = []
prompts = []
for _, row in df.iterrows():
    extra = json.loads(row["extra_info"])
    query_ids.append(extra["query_id"])
    messages = json.loads(row["prompt"])
    prompts.append(messages)
print(f"Total test samples: {len(prompts)}")

# 加载模型
print("Loading model...")
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.9,
)
tokenizer = llm.get_tokenizer()

# 格式化
print("Formatting prompts...")
formatted = []
for msgs in prompts:
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    formatted.append(text)

# 推理
print("Running inference...")
sampling_params = SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"])
outputs = llm.generate(formatted, sampling_params)

# 解析并保存
print("Parsing and saving...")
results = []
count_ok = 0
count_fail = 0
for qid, output in zip(query_ids, outputs):
    text = output.outputs[0].text.strip()
    dsl = None
    if "<answer>" in text:
        try:
            json_str = text.split("<answer>")[1].strip()
            dsl = json.loads(json_str)
            count_ok += 1
        except (json.JSONDecodeError, IndexError):
            count_fail += 1
    else:
        try:
            dsl = json.loads(text)
            count_ok += 1
        except json.JSONDecodeError:
            count_fail += 1
    results.append({
        "query_id": qid,
        "natural_language_query": id2query.get(qid, ""),
        "dsl": dsl,
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done. {count_ok} OK, {count_fail} failed. Saved to {OUTPUT_FILE}")
