#!/usr/bin/env python3
"""Evaluate DeepRetrieval model on ecommerce test queries.

Loads a trained checkpoint (or base model), generates boolean queries,
runs Pyserini BM25 retrieval, and computes Hit@1/3, MRR metrics.
"""
import argparse
import json
import os
import re

from pyserini.search.lucene import LuceneSearcher
from vllm import LLM, SamplingParams

INDEX_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/indexes/lucene-index-products"
TEST_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
OUTPUT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="b3_deepretrieval_faithful")
    parser.add_argument("--gpu-memory", type=float, default=0.8)
    args = parser.parse_args()

    queries = []
    with open(TEST_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} test queries")

    searcher = LuceneSearcher(INDEX_DIR)
    print(f"Loaded index: {searcher.num_docs} documents")

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

    predictions = {}
    for q, output in zip(queries, outputs):
        qid = q["query_id"]
        generated = output.outputs[0].text
        bool_query = extract_query(generated)
        if bool_query:
            hits = searcher.search(bool_query, k=TOP_K)
            ranked = [hit.docid for hit in hits]
        else:
            ranked = []
        predictions[qid] = ranked

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

    print(f"\nDeepRetrieval Faithful Results (N={n}):")
    print(f"  Hit@1: {hit1/n:.4f}")
    print(f"  Hit@3: {hit3/n:.4f}")
    print(f"  MRR:   {mrr/n:.4f}")
    nonempty = sum(1 for v in predictions.values() if v)
    print(f"  Non-empty: {nonempty}/{len(predictions)}")


if __name__ == "__main__":
    main()
