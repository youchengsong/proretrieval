#!/usr/bin/env python3
"""Evaluate a DeepRetrieval checkpoint on ecommerce test queries.

Uses GPU 0 (separate from training GPUs 1-6) with low memory footprint.
Loads checkpoint, generates boolean queries via vLLM, runs BM25, computes metrics.
"""
import argparse
import json
import os
import re
import sys
import time
import glob
import multiprocessing

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

INDEX_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/indexes/lucene-index-products"
TEST_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"

INSTRUCTION = """You are a product search query optimizer. Given a customer's product search query, generate an optimized Boolean search query to retrieve the most relevant products from a product database using BM25 retrieval.

The product database contains fields: Brand, Color, Category, and product description text. Your Boolean query will be matched against these fields."""

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
{instruction}
Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{{"query": "...."}}
</answer>.
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

Here's the customer query:
{query}
Assistant: Let me think step by step.
<think>
"""

TOP_K = 20


def find_model_path(checkpoint_dir):
    """Find the actual model directory within a verl checkpoint."""
    candidates = [
        os.path.join(checkpoint_dir, "actor"),
        os.path.join(checkpoint_dir, "actor", "huggingface"),
        checkpoint_dir,
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    for root, dirs, files in os.walk(checkpoint_dir):
        if "config.json" in files:
            return root
    return checkpoint_dir


def extract_query(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[-1].strip())
            return data.get("query", "")
        except json.JSONDecodeError:
            pass
    return None


def run_eval_vllm(model_path, queries, gpu_mem=0.22, max_model_len=1536):
    """Run inference with vLLM on GPU 0."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    prompts = [
        PROMPT_TEMPLATE.format(instruction=INSTRUCTION, query=q["natural_language_query"])
        for q in queries
    ]

    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        results.append(output.outputs[0].text)
    return results


def run_eval_transformers(model_path, queries, batch_size=1):
    """Fallback: run inference with transformers (slower but more memory-efficient)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model.eval()

    prompts = [
        PROMPT_TEMPLATE.format(instruction=INSTRUCTION, query=q["natural_language_query"])
        for q in queries
    ]

    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        for j, seq in enumerate(out):
            text = tokenizer.decode(seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            results.append(text)
        if (i // batch_size) % 100 == 0:
            print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to verl checkpoint dir (e.g., checkpoints/.../global_step_50)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test queries")
    parser.add_argument("--gpu-mem", type=float, default=0.22, help="vLLM gpu_memory_utilization")
    parser.add_argument("--use-transformers", action="store_true", help="Use transformers instead of vLLM")
    parser.add_argument("--output-dir", type=str, default="/home/sunzhen/DeepRetrieval_ecommerce/eval_results")
    args = parser.parse_args()

    model_path = find_model_path(args.checkpoint)
    print(f"Model path: {model_path}")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"ERROR: config.json not found in {model_path}")
        sys.exit(1)

    step_match = re.search(r"global_step_(\d+)", args.checkpoint)
    step_name = f"step_{step_match.group(1)}" if step_match else os.path.basename(args.checkpoint)

    queries = []
    with open(TEST_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} test queries")

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(INDEX_DIR)
    print(f"Loaded index: {searcher.num_docs} documents")

    t0 = time.time()
    try:
        if args.use_transformers:
            raise RuntimeError("forced transformers mode")
        print("Running inference with vLLM...")
        generated_texts = run_eval_vllm(model_path, queries, args.gpu_mem)
    except Exception as e:
        print(f"vLLM failed ({e}), falling back to transformers...")
        generated_texts = run_eval_transformers(model_path, queries)
    gen_time = time.time() - t0
    print(f"Generation took {gen_time:.1f}s")

    predictions = {}
    format_ok = 0
    for q, text in zip(queries, generated_texts):
        qid = q["query_id"]
        bool_query = extract_query(text)
        if bool_query:
            format_ok += 1
            try:
                hits = searcher.search(bool_query, k=TOP_K)
                ranked = [hit.docid for hit in hits]
            except Exception:
                ranked = []
        else:
            ranked = []
        predictions[qid] = ranked

    hit1 = hit3 = mrr = n = 0
    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:TOP_K]
        n += 1
        if ranked and ranked[0] in gt:
            hit1 += 1
        if any(r in gt for r in ranked[:3]):
            hit3 += 1
        for j, r in enumerate(ranked):
            if r in gt:
                mrr += 1.0 / (j + 1)
                break

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "step": step_name,
        "checkpoint": args.checkpoint,
        "n_queries": n,
        "hit1": round(hit1 / n, 4) if n else 0,
        "hit3": round(hit3 / n, 4) if n else 0,
        "mrr": round(mrr / n, 4) if n else 0,
        "format_ok": format_ok,
        "format_rate": round(format_ok / len(queries), 4),
        "gen_time_s": round(gen_time, 1),
    }

    result_file = os.path.join(args.output_dir, f"{step_name}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"DeepRetrieval Eval: {step_name} (N={n})")
    print(f"  Hit@1:  {result['hit1']:.4f}")
    print(f"  Hit@3:  {result['hit3']:.4f}")
    print(f"  MRR:    {result['mrr']:.4f}")
    print(f"  Format: {format_ok}/{len(queries)} ({result['format_rate']:.1%})")
    print(f"  Time:   {gen_time:.1f}s")
    print(f"{'='*50}")

    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "a") as f:
        f.write(f"{step_name}\tHit@1={result['hit1']:.4f}\tHit@3={result['hit3']:.4f}\tMRR={result['mrr']:.4f}\tFmt={result['format_rate']:.1%}\n")
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
