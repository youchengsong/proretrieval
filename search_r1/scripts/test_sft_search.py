"""Quick SFT quality check for Search-R1 models."""
import json, sys, torch, argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parquet_path", type=str, required=True, help="RL parquet to sample prompts from")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading prompts from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    samples = df.sample(n=min(args.n_samples, len(df)), random_state=42)

    format_ok = 0
    format_fail = 0

    for i, (_, row) in enumerate(samples.iterrows()):
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        rm = json.loads(row["reward_model"])
        gt = rm["ground_truth"] if isinstance(rm["ground_truth"], list) else rm["ground_truth"]["target"]

        messages = prompt if isinstance(prompt, list) else [prompt]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        has_think = "<think>" in response and "</think>" in response
        has_search = "<search>" in response and "</search>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        ok = has_think and (has_search or has_answer)

        if ok:
            format_ok += 1
        else:
            format_fail += 1

        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{args.n_samples}")
        print(f"Query: {messages[-1]['content'][:150]}...")
        print(f"Ground truth: {gt[:3]}...")
        print(f"Format check: think={has_think} search={has_search} answer={has_answer} => {'PASS' if ok else 'FAIL'}")
        print(f"Response (first 500 chars):\n{response[:500]}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {format_ok}/{format_ok+format_fail} passed format check")
    if format_ok >= (format_ok + format_fail) * 0.6:
        print("SFT quality: GOOD - ready for RL")
    else:
        print("SFT quality: POOR - may need investigation")

if __name__ == "__main__":
    main()
