"""
电商测试集推理：用 SFT 模型生成 DSL

用法:
  python inference_ecommerce.py                          # 默认输出 predictions.jsonl
  python inference_ecommerce.py -o predictions_step9     # 输出 predictions_step9.jsonl
  python inference_ecommerce.py --model /path/to/model   # 指定模型路径
"""
import argparse
import json
import os
import pandas as pd
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/data/s3-migration/sft_localSearch/output/final"
TEST_PARQUET = "/home/sunzhen/LocalSearch/data/ecommerce/test/product_rl_test.parquet"
OUTPUT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"

parser = argparse.ArgumentParser(description="电商测试集 DSL 推理")
parser.add_argument("-o", "--output_name", default="predictions",
                    help="输出文件名（不含扩展名），默认 predictions")
parser.add_argument("--model", default=DEFAULT_MODEL, help="模型路径")
args = parser.parse_args()

OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{args.output_name}.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    model=args.model,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.9,
)

tokenizer = llm.get_tokenizer()

# 格式化 prompt（用 chat template）
print("Formatting prompts...")
formatted = []
for msgs in prompts:
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted.append(text)

# 推理
print("Running inference...")
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
    stop=["</answer>"],
)

outputs = llm.generate(formatted, sampling_params)

# 解析结果并保存
print("Parsing and saving...")
count_ok = 0
count_fail = 0
with open(OUTPUT_FILE, "w") as fout:
    for qid, output in zip(query_ids, outputs):
        text = output.outputs[0].text.strip()

        # 提取 <answer>...</answer> 之间的 JSON
        dsl = None
        if "<answer>" in text:
            try:
                json_str = text.split("<answer>")[1].strip()
                dsl = json.loads(json_str)
                count_ok += 1
            except (json.JSONDecodeError, IndexError):
                dsl = None
                count_fail += 1
        else:
            # 可能直接输出了 JSON
            try:
                dsl = json.loads(text)
                count_ok += 1
            except json.JSONDecodeError:
                dsl = None
                count_fail += 1

        fout.write(json.dumps({
            "query_id": qid,
            "dsl": dsl,
        }, ensure_ascii=False) + "\n")

print(f"Done. {count_ok} OK, {count_fail} failed to parse.")
print(f"Saved to {OUTPUT_FILE}")
