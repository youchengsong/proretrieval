"""Inference with 8B model on Blackwell."""
import json, os, sys
import pandas as pd
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/rl_8b/ecommerce/merged_step150"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/rl_8b/ecommerce/predictions.jsonl"
TEST_DATA = "/root/autodl-tmp/data/ecommerce/test/product_rl_test.parquet"

print(f"Model: {MODEL}")
print(f"Output: {OUTPUT}")

df = pd.read_parquet(TEST_DATA)
query_ids, prompts = [], []
for _, row in df.iterrows():
    extra = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
    query_ids.append(extra["query_id"])
    p = row["prompt"]
    prompts.append(p.tolist() if hasattr(p, 'tolist') else (json.loads(p) if isinstance(p, str) else p))
print(f"{len(prompts)} samples")

llm = LLM(model=MODEL, trust_remote_code=True, max_model_len=2048,
          gpu_memory_utilization=0.85, tensor_parallel_size=2, dtype="bfloat16")
tokenizer = llm.get_tokenizer()
formatted = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts]
outputs = llm.generate(formatted, SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"]))

ok, fail = 0, 0
with open(OUTPUT, "w") as f:
    for qid, out in zip(query_ids, outputs):
        text = out.outputs[0].text.strip()
        dsl = None
        if "<answer>" in text:
            try:
                dsl = json.loads(text.split("<answer>")[1].strip())
                ok += 1
            except (json.JSONDecodeError, IndexError):
                fail += 1
        else:
            try:
                dsl = json.loads(text)
                ok += 1
            except json.JSONDecodeError:
                fail += 1
        f.write(json.dumps({"query_id": qid, "dsl": dsl}, ensure_ascii=False) + "\n")
print(f"Done! {ok} OK, {fail} failed. Saved to {OUTPUT}")
