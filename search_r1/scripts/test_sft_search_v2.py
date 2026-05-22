"""Quick SFT quality check for Search-R1 models - with system prompt."""
import json, sys, torch, argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SEARCH_SYSTEM = """You are a helpful search assistant. Given a user query, find the most relevant product by searching the product catalog.
You must conduct reasoning inside <think> and </think> first every time you get new information, then search using <search> and </search>, and finally answer with <answer> and </answer>."""

EMAIL_SYSTEM = """You are a helpful search assistant. Given a user query, find the most relevant email by searching the email archive.
You must conduct reasoning inside <think> and </think> first every time you get new information, then search using <search> and </search>, and finally answer with <answer> and </answer>."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="ecommerce", choices=["ecommerce", "email"])
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--with_system", action="store_true", help="Prepend system message")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print(f"Loading prompts from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    samples = df.sample(n=min(args.n_samples, len(df)), random_state=42)

    system_msg = SEARCH_SYSTEM if args.domain == "ecommerce" else EMAIL_SYSTEM
    format_ok = 0
    total = 0

    for i, (_, row) in enumerate(samples.iterrows()):
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        messages = prompt if isinstance(prompt, list) else [prompt]

        rm = json.loads(row["reward_model"])
        gt = rm["ground_truth"] if isinstance(rm["ground_truth"], list) else rm["ground_truth"]["target"]

        if args.with_system:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": system_msg}] + messages

        # Build template manually to avoid Qwen3 stripping tags
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        has_think = "<think>" in response and "</think>" in response
        has_search = "<search>" in response and "</search>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        ok = has_think and (has_search or has_answer)
        if ok:
            format_ok += 1
        total += 1

        print(f"\n{'='*60}")
        print(f"Sample {i+1} | think={has_think} search={has_search} answer={has_answer} => {'PASS' if ok else 'FAIL'}")
        print(f"Query: {messages[-1]['content'][:120]}")
        print(f"GT: {gt[:3]}")
        print(f"Response:\n{response[:600]}")

    print(f"\n{'='*60}")
    print(f"RESULT: {format_ok}/{total} passed (with_system={args.with_system})")

if __name__ == "__main__":
    main()
