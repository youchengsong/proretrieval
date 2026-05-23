#!/usr/bin/env python3
"""E2 Action Space Ablation - Inference on test set.

Usage:
    python inference_e2.py --model_path /path/to/model \
        --test_data /path/to/test.parquet \
        --output /path/to/pred.json \
        --action_space sql_only
"""
import argparse, json, os, re
import pandas as pd
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action_space", default="full",
                        choices=["sql_only", "sql_text", "text_only", "sql_image", "full"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading test data from {args.test_data}...")
    df = pd.read_parquet(args.test_data)
    print(f"Total samples: {len(df)}")

    query_ids = []
    prompts_raw = []
    gt_ids_list = []

    for _, row in df.iterrows():
        prompt_val = row["prompt"]
        if isinstance(prompt_val, str):
            msgs = json.loads(prompt_val)
        else:
            msgs = list(prompt_val)

        prompts_raw.append(msgs)

        qid = None
        gt_ids = []
        if "extra_info" in row and pd.notna(row.get("extra_info")):
            extra = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
            qid = extra.get("query_id")
        if "reward_model" in row and pd.notna(row.get("reward_model")):
            rm = json.loads(row["reward_model"]) if isinstance(row["reward_model"], str) else row["reward_model"]
            if qid is None:
                qid = rm.get("query_id")
            gt_ids = rm.get("ground_truth", [])

        query_ids.append(qid or str(_))
        gt_ids_list.append(gt_ids)

    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()

    formatted = []
    for msgs in prompts_raw:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        formatted.append(text)

    print(f"Running inference ({len(formatted)} queries)...")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</answer>"],
    )
    outputs = llm.generate(formatted, sampling_params)

    results = []
    ok = fail = 0
    for qid, gt_ids, output in zip(query_ids, gt_ids_list, outputs):
        text = output.outputs[0].text.strip()
        if not text.endswith("</answer>"):
            text += "</answer>"

        dsl = None
        m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
        if m:
            try:
                dsl = json.loads(m.group(1).strip())
                ok += 1
            except json.JSONDecodeError:
                fail += 1
        else:
            try:
                dsl = json.loads(text)
                ok += 1
            except json.JSONDecodeError:
                fail += 1

        results.append({
            "query_id": qid,
            "dsl": dsl,
            "raw_output": text,
            "gt_document_ids": gt_ids,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. {ok} parsed, {fail} failed. Saved to {args.output}")


if __name__ == "__main__":
    main()
