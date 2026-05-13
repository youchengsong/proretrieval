#!/usr/bin/env python3
"""B2 Baseline: IRCoT — Interleaved Retrieval Chain-of-Thought for product search.

Adapted from Trivedi et al. 2023. Multi-round: retrieve → reason → refine → retrieve again.
Round 1: Encode original query, retrieve top-50.
Round 2: GPT-5.4 analyzes query + Round 1 results, generates refined search description → retrieve top-50.
Merge both rounds via RRF → final top-20.
"""
import argparse, json, os, time, sys
import numpy as np
import pandas as pd
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"
GT_FILE = os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_test.json")
TEXT_FEAT_FILE = os.path.join(DATA_DIR, "product_text_features_test.parquet")
QWEN_PATH = os.environ.get("QWEN_PATH", "/home/sunzhen/models/Qwen3-Embedding-8B")
TOP_K = 20
RETRIEVE_K = 50
RRF_K = 60

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

REFINE_PROMPT = """You are an ecommerce product search expert. A customer searched for: "{query}"

Here are the top results from the initial retrieval:
{products}

Based on the search query and these initial results, think step by step:
1. What specific product attributes is the customer looking for?
2. Which aspects of the query are NOT well-covered by these results?
3. What additional search terms or descriptions would help find better matches?

Now write a refined, detailed product description that captures ALL aspects the customer is looking for, especially the ones missing from initial results. Focus on the specific requirements (brand, color, size, features, category, price range) mentioned in the query.

Refined search description:"""


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def rrf_merge(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    pid_list = [p["product_id"] for p in products]
    pmap = {}
    for p in products:
        desc = f"Brand: {p.get('brand','N/A')} | Color: {p.get('color','N/A')} | Category: {p.get('category','N/A')} | Price: {p.get('price','N/A')} | Rating: {p.get('average_rating','N/A')} | {(p.get('text','') or '')[:150]}"
        pmap[p["product_id"]] = desc
    print(f"Loaded {len(products)} products")

    print("Loading product text features...")
    t_df = pd.read_parquet(TEXT_FEAT_FILE)
    t_mat = np.stack(t_df["text_features"].values).astype(np.float32)
    t_mat = t_mat / np.linalg.norm(t_mat, axis=1, keepdims=True)

    device = f"cuda:{args.gpu}"
    print(f"Loading Qwen3-Embedding on {device} (fp16)...")
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(QWEN_PATH, device=device, model_kwargs={"torch_dtype": torch.float16})

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    log_jsonl = os.path.join(RESULT_DIR, "ircot_log.jsonl")
    existing = {}
    if os.path.exists(log_jsonl):
        with open(log_jsonl) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d["final_ranked"]
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [q for q in queries if q["query_id"] not in existing]
    print(f"  Todo: {len(todo)} queries")

    for i, q in enumerate(todo):
        qid = q["query_id"]
        query_text = q["natural_language_query"]

        # Round 1: encode original query → retrieve
        q_vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)
        scores_r1 = (q_vec @ t_mat.T)[0]
        r1_idx = np.argsort(-scores_r1)[:RETRIEVE_K]
        r1_pids = [pid_list[idx] for idx in r1_idx]

        # Build product summary for GPT
        prod_text = ""
        for j, pid in enumerate(r1_pids[:10]):
            prod_text += f"[{j+1}] {pmap.get(pid, 'N/A')}\n"

        prompt = REFINE_PROMPT.format(query=query_text, products=prod_text)

        refined_desc = query_text
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_completion_tokens=300,
                )
                refined_desc = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        # Round 2: encode refined description → retrieve
        r_vec = model.encode([refined_desc], normalize_embeddings=True).astype(np.float32)
        scores_r2 = (r_vec @ t_mat.T)[0]
        r2_idx = np.argsort(-scores_r2)[:RETRIEVE_K]
        r2_pids = [pid_list[idx] for idx in r2_idx]

        # Merge Round 1 + Round 2 via RRF
        final_ranked = rrf_merge([r1_pids, r2_pids], k=RRF_K)[:TOP_K]

        existing[qid] = final_ranked
        with open(log_jsonl, "a") as f:
            f.write(json.dumps({"query_id": qid, "final_ranked": final_ranked,
                                "refined_desc": refined_desc[:200]}, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] done")

    # Save predictions
    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]

    output_file = os.path.join(RESULT_DIR, "b2_ircot.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_file}")

    # Evaluate
    overall = {"hit1": 0, "hit3": 0, "mrr": 0, "n": 0}
    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:TOP_K]
        h1 = 1.0 if ranked and ranked[0] in gt else 0.0
        h3 = 1.0 if any(r in gt for r in ranked[:3]) else 0.0
        rr = 0.0
        for j, r in enumerate(ranked):
            if r in gt:
                rr = 1.0 / (j + 1)
                break
        overall["hit1"] += h1; overall["hit3"] += h3; overall["mrr"] += rr; overall["n"] += 1

    n = overall["n"]
    print(f"\nIRCoT Results (N={n}):")
    print(f"  Hit@1: {overall['hit1']/n:.4f}")
    print(f"  Hit@3: {overall['hit3']/n:.4f}")
    print(f"  MRR:   {overall['mrr']/n:.4f}")

    eval_file = os.path.join(RESULT_DIR, "eval_b2_ircot.json")
    with open(eval_file, "w") as f:
        json.dump({"hit1": overall["hit1"]/n, "hit3": overall["hit3"]/n, "mrr": overall["mrr"]/n, "count": n}, f, indent=2)
    print(f"Saved eval to {eval_file}")


if __name__ == "__main__":
    main()
