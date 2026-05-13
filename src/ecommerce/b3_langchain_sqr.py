#!/usr/bin/env python3
"""B3 Baseline: LangChain-style Self-Query Retrieval (SQR).

GPT-5.4 extracts structured filters (brand, color, category, price) from the query,
applies them to filter products, then ranks by text embedding similarity.
Represents the 'constrained structured query' paradigm without RL training.
"""
import argparse, json, os, re, time
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

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

SQR_PROMPT = """You are a product search filter generator. Given a customer's search query, extract structured filters and a text search query.

Available filter fields:
- brand (string, exact match)
- color (string, exact match)
- category (string, one of: {categories})
- price_min (number, minimum price)
- price_max (number, maximum price)
- min_rating (number, minimum average rating, 1-5)

Output a JSON object with:
- "filters": dict of field->value for applicable filters (omit fields not mentioned in query)
- "text_query": a refined text search query capturing the semantic intent

Only extract filters when the query CLEARLY specifies them. Do not guess.

Customer query: {query}

Output JSON only, no other text:"""


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def parse_sqr_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```json?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"filters": {}, "text_query": ""}


def apply_filters(products, filters):
    filtered = products
    if "brand" in filters and filters["brand"]:
        brand = filters["brand"].lower()
        filtered = [p for p in filtered if brand in (p.get("brand", "") or "").lower()]
    if "color" in filters and filters["color"]:
        color = filters["color"].lower()
        filtered = [p for p in filtered if color in (p.get("color", "") or "").lower()]
    if "category" in filters and filters["category"]:
        cat = filters["category"].lower()
        filtered = [p for p in filtered if cat in (p.get("category", "") or "").lower()]
    if "price_min" in filters:
        try:
            pmin = float(filters["price_min"])
            filtered = [p for p in filtered if (p.get("price") or 0) >= pmin]
        except (ValueError, TypeError):
            pass
    if "price_max" in filters:
        try:
            pmax = float(filters["price_max"])
            filtered = [p for p in filtered if (p.get("price") or 99999) <= pmax]
        except (ValueError, TypeError):
            pass
    if "min_rating" in filters:
        try:
            mrat = float(filters["min_rating"])
            filtered = [p for p in filtered if (p.get("average_rating") or 0) >= mrat]
        except (ValueError, TypeError):
            pass
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    pid_list = [p["product_id"] for p in products]
    pid_to_idx = {p["product_id"]: i for i, p in enumerate(products)}
    print(f"Loaded {len(products)} products")

    categories = sorted(set(p.get("category", "") for p in products if p.get("category")))
    cat_str = ", ".join(categories[:30])

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

    log_jsonl = os.path.join(RESULT_DIR, "sqr_log.jsonl")
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

        prompt = SQR_PROMPT.format(query=query_text, categories=cat_str)

        sqr_result = {"filters": {}, "text_query": query_text}
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()
                sqr_result = parse_sqr_response(raw)
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        filters = sqr_result.get("filters", {})
        text_query = sqr_result.get("text_query", query_text) or query_text

        filtered = apply_filters(products, filters)

        # If filters are too aggressive (< 5 results), fall back to all products
        if len(filtered) < 5:
            filtered = products
            filters = {}

        filtered_pids = [p["product_id"] for p in filtered]
        filtered_indices = [pid_to_idx[pid] for pid in filtered_pids]
        filtered_mat = t_mat[filtered_indices]

        q_vec = model.encode([text_query], normalize_embeddings=True).astype(np.float32)
        scores = (q_vec @ filtered_mat.T)[0]
        top_local = np.argsort(-scores)[:TOP_K]
        final_ranked = [filtered_pids[idx] for idx in top_local]

        existing[qid] = final_ranked
        with open(log_jsonl, "a") as f:
            f.write(json.dumps({
                "query_id": qid, "final_ranked": final_ranked,
                "filters": filters, "text_query": text_query[:200],
                "filtered_count": len(filtered)
            }, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] done")

    # Save predictions
    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]

    output_file = os.path.join(RESULT_DIR, "b3_langchain_sqr.json")
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
    print(f"\nSQR Results (N={n}):")
    print(f"  Hit@1: {overall['hit1']/n:.4f}")
    print(f"  Hit@3: {overall['hit3']/n:.4f}")
    print(f"  MRR:   {overall['mrr']/n:.4f}")

    eval_file = os.path.join(RESULT_DIR, "eval_b3_langchain_sqr.json")
    with open(eval_file, "w") as f:
        json.dump({"hit1": overall["hit1"]/n, "hit3": overall["hit3"]/n, "mrr": overall["mrr"]/n, "count": n}, f, indent=2)
    print(f"Saved eval to {eval_file}")


if __name__ == "__main__":
    main()
