#!/usr/bin/env python3
"""Evaluate DeepRetrieval model on email test queries.

Loads a trained checkpoint (or base model), generates boolean queries,
runs Pyserini BM25 retrieval, and computes Hit@1/3, MRR metrics.

Init LLM before Pyserini to avoid CUDA fork issues with JVM.
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import re

from vllm import LLM, SamplingParams

INDEX_DIR = "/root/autodl-tmp/DeepRetrieval_email/indexes/lucene-index-emails-test"
TEST_JSONL = "/root/autodl-tmp/data/email/test/queries_email_test_step9.jsonl"
TEST_PARQUET = "/root/autodl-tmp/data/email/test/email_rl_test_fixed.parquet"
OUTPUT_DIR = "/root/autodl-tmp/data/email/result"

INSTRUCTION = """You are an email search query optimizer. Given a user's email search query, generate an optimized Boolean search query to retrieve the most relevant emails from an email database using BM25 retrieval.

The email database contains fields: From (sender email), To (recipient emails), Subject, and Message body text. Your Boolean query will be matched against these fields."""

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
{instruction}
Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{{"query": "...."}}
</answer>.
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

Here's the email search query:
{query}
Assistant: Let me think step by step.
<think>
"""

TOP_K = 20


def extract_query(text):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, text, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[-1].strip())
            return data.get("query", "")
        except json.JSONDecodeError:
            pass
    return None


def load_test_queries():
    """Load test queries from JSONL or RL parquet."""
    if os.path.exists(TEST_JSONL):
        queries = []
        with open(TEST_JSONL) as f:
            for line in f:
                queries.append(json.loads(line))
        return queries

    import numpy as np
    import pandas as pd
    df = pd.read_parquet(TEST_PARQUET)
    queries = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        nl_query = ""
        for msg in prompt:
            if msg.get("role") == "user":
                nl_query = msg["content"]
                break
        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt_ids = rm.get("ground_truth", [])
        if isinstance(gt_ids, np.ndarray):
            gt_ids = gt_ids.tolist()
        if isinstance(gt_ids, dict):
            gt_ids = gt_ids.get("target", [])
            if isinstance(gt_ids, np.ndarray):
                gt_ids = gt_ids.tolist()
        queries.append({
            "query_id": f"qe_{idx:05d}",
            "natural_language_query": nl_query,
            "gt_document_ids": gt_ids,
        })
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="deepretrieval_email")
    parser.add_argument("--gpu-memory", type=float, default=0.8)
    args = parser.parse_args()

    queries = load_test_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} test queries")

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    prompts = []
    for q in queries:
        prompt = PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION,
            query=q["natural_language_query"],
        )
        prompts.append(prompt)

    print("Generating boolean queries...")
    outputs = llm.generate(prompts, sampling_params)

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(INDEX_DIR)
    print(f"Loaded index: {searcher.num_docs} documents")

    predictions = {}
    for q, output in zip(queries, outputs):
        qid = q["query_id"]
        generated = output.outputs[0].text
        bool_query = extract_query(generated)
        if bool_query:
            try:
                hits = searcher.search(bool_query, k=TOP_K)
                ranked = [hit.docid for hit in hits]
            except Exception:
                ranked = []
        else:
            ranked = []
        predictions[qid] = ranked

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{args.output_name}.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_file}")

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

    print(f"\nDeepRetrieval Email Results (N={n}):")
    print(f"  Hit@1: {hit1/n:.4f}")
    print(f"  Hit@3: {hit3/n:.4f}")
    print(f"  MRR:   {mrr/n:.4f}")
    nonempty = sum(1 for v in predictions.values() if v)
    print(f"  Non-empty: {nonempty}/{len(predictions)}")


if __name__ == "__main__":
    main()
