#!/usr/bin/env python3
"""B4 Baseline: LightRAG — Graph-based Knowledge Retrieval.

Uses LightRAG (EMNLP 2025) for graph-augmented retrieval on product data.
Pipeline: product text → entity/relation extraction → KG construction →
          hybrid query (KG traversal + vector similarity) → ranked product list.

This is the proper graph-based retrieval baseline, complementary to our DSL approach
(graph-edge structure vs field-constraint structure).
"""
import argparse, asyncio, json, math, os, re, sys, time
import numpy as np
from collections import defaultdict

DATA_DIR = os.environ.get("DATA_DIR", "/home/sunzhen/LocalSearch/data/ecommerce/test")
RESULT_DIR = os.environ.get("RESULT_DIR", "/home/sunzhen/LocalSearch/data/ecommerce/result")
GT_FILE = os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_test.json")

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

TOP_K = 20
WORKING_DIR = os.path.join(RESULT_DIR, "lightrag_workdir")


def format_product_text(p):
    """Format product as a rich text document for LightRAG ingestion."""
    parts = [f"Product ID: {p['product_id']}"]
    if p.get("brand"):
        parts.append(f"Brand: {p['brand']}")
    if p.get("color"):
        parts.append(f"Color: {p['color']}")
    if p.get("category"):
        parts.append(f"Category: {p['category']}")
    if p.get("price"):
        parts.append(f"Price: ${p['price']}")
    if p.get("average_rating"):
        parts.append(f"Rating: {p['average_rating']}")
    if p.get("rating_number"):
        parts.append(f"Number of ratings: {p['rating_number']}")
    if p.get("text"):
        parts.append(f"Description: {p['text'][:800]}")
    return "\n".join(parts)


def load_queries(gt_file):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


async def create_lightrag(working_dir, max_async=4):
    """Create and initialize LightRAG instance with Azure OpenAI + local embeddings."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from openai import AsyncAzureOpenAI
    from sentence_transformers import SentenceTransformer

    llm_client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
    )

    print(f"Loading embedding model {EMBED_MODEL}...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    async def azure_llm_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ):
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        resp = await llm_client.chat.completions.create(
            model=AZURE_DEPLOY,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_completion_tokens=kwargs.get("max_tokens", 4000),
        )
        return resp.choices[0].message.content

    async def local_embed_func(texts: list[str]) -> np.ndarray:
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=azure_llm_func,
        llm_model_name=AZURE_DEPLOY,
        llm_model_max_async=max_async,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=512,
            func=local_embed_func,
        ),
        embedding_func_max_async=1,
        chunk_token_size=600,
        chunk_overlap_token_size=50,
        entity_extract_max_gleaning=0,
        top_k=TOP_K * 2,
        chunk_top_k=TOP_K * 2,
    )

    await rag.initialize_storages()
    return rag


def extract_product_ids_from_context(context_text, product_ids_set):
    """Extract product IDs from LightRAG context output, preserving order."""
    if not context_text:
        return []
    found = []
    seen = set()
    for pid in product_ids_set:
        if pid in context_text and pid not in seen:
            pos = context_text.index(pid)
            found.append((pos, pid))
            seen.add(pid)
    found.sort(key=lambda x: x[0])
    return [pid for _, pid in found]


def extract_product_ids_from_raw_data(raw_data, product_ids_set):
    """Extract product IDs from raw_data structured output."""
    if not raw_data:
        return []
    found = []
    seen = set()

    data = raw_data.get("data", {})

    for chunk in data.get("chunks", []):
        content = chunk.get("content", "")
        source_id = chunk.get("source_id", "")
        file_path = chunk.get("file_path", "")
        for candidate in [source_id, file_path]:
            if candidate in product_ids_set and candidate not in seen:
                found.append(candidate)
                seen.add(candidate)
        for pid in product_ids_set:
            if pid in content and pid not in seen:
                found.append(pid)
                seen.add(pid)

    for ref in data.get("references", []):
        source_id = ref.get("source_id", "")
        file_path = ref.get("file_path", "")
        for candidate in [source_id, file_path]:
            if candidate in product_ids_set and candidate not in seen:
                found.append(candidate)
                seen.add(candidate)

    for entity in data.get("entities", []):
        desc = entity.get("description", "")
        source = entity.get("source_id", "")
        if source:
            for pid in product_ids_set:
                if pid in source and pid not in seen:
                    found.append(pid)
                    seen.add(pid)
        for pid in product_ids_set:
            if pid in desc and pid not in seen:
                found.append(pid)
                seen.add(pid)

    return found


async def index_products(rag, products, batch_size=50):
    """Index all products into LightRAG."""
    texts = [format_product_text(p) for p in products]
    ids = [p["product_id"] for p in products]

    total = len(texts)
    t0 = time.time()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        try:
            await rag.ainsert(batch_texts, ids=batch_ids)
        except Exception as e:
            print(f"  Batch {start}-{end} failed: {e}")
            for i in range(len(batch_texts)):
                try:
                    await rag.ainsert([batch_texts[i]], ids=[batch_ids[i]])
                except Exception as e2:
                    print(f"    Product {batch_ids[i]} failed: {e2}")
        elapsed = time.time() - t0
        done = end
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  Indexed [{done}/{total}] {rate:.1f}/s")


async def query_products(rag, queries, product_ids_set, mode="hybrid"):
    """Query LightRAG for each query, extract ranked product IDs."""
    from lightrag import QueryParam

    predictions = {}
    total = len(queries)
    t0 = time.time()

    for i, q in enumerate(queries):
        qid = q["query_id"]
        nl = q["natural_language_query"]

        try:
            result = await rag.aquery_llm(
                nl,
                param=QueryParam(
                    mode=mode,
                    only_need_context=True,
                    top_k=TOP_K * 2,
                    chunk_top_k=TOP_K * 2,
                ),
            )

            ranked = []
            raw_data = result if isinstance(result, dict) else {}

            ranked = extract_product_ids_from_raw_data(raw_data, product_ids_set)

            if len(ranked) < TOP_K:
                context_text = ""
                if isinstance(result, dict):
                    llm_resp = result.get("llm_response", {})
                    context_text = llm_resp.get("content", "") or ""
                elif isinstance(result, str):
                    context_text = result
                extra = extract_product_ids_from_context(context_text, product_ids_set)
                for pid in extra:
                    if pid not in set(ranked):
                        ranked.append(pid)

            predictions[qid] = ranked[:TOP_K]

        except Exception as e:
            print(f"  Query {qid} failed: {e}")
            predictions[qid] = []

        if (i + 1) % 100 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Query [{i+1}/{total}] {rate:.1f}/s")

    return predictions


def evaluate(predictions, queries, top_k=20):
    """Evaluate predictions against ground truth."""
    ks = [1, 3, 10, 20]
    hit1 = hit3 = mrr = n = 0
    ndcg = {k: 0.0 for k in ks}

    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or q.get("gold_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:top_k]
        n += 1

        if ranked and ranked[0] in gt:
            hit1 += 1
        if any(r in gt for r in ranked[:3]):
            hit3 += 1
        for j, r in enumerate(ranked):
            if r in gt:
                mrr += 1.0 / (j + 1)
                break

        for k in ks:
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
            if ideal > 0:
                dcg = sum(
                    1.0 / math.log2(i + 2)
                    for i, doc in enumerate(ranked[:k])
                    if doc in gt
                )
                ndcg[k] += dcg / ideal

    if n == 0:
        print("No valid queries!")
        return {}

    result = {
        "n": n,
        "hit1": hit1 / n,
        "hit3": hit3 / n,
        "mrr": mrr / n,
    }
    for k in ks:
        result[f"ndcg@{k}"] = ndcg[k] / n

    print(f"\nLightRAG Results (N={n}):")
    print(f"  Hit@1:    {result['hit1']:.4f}")
    print(f"  Hit@3:    {result['hit3']:.4f}")
    print(f"  MRR:      {result['mrr']:.4f}")
    for k in ks:
        print(f"  NDCG@{k:<3}  {result[f'ndcg@{k}']:.4f}")

    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix"])
    parser.add_argument("--max-async", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing if already done")
    parser.add_argument("--query-only", action="store_true", help="Only run queries")
    args = parser.parse_args()

    queries = load_queries(GT_FILE)
    if args.limit:
        queries = queries[:args.limit]
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    product_ids_set = set(p["product_id"] for p in products)
    print(f"Loaded {len(products)} products")

    os.makedirs(WORKING_DIR, exist_ok=True)

    print("Initializing LightRAG...")
    rag = await create_lightrag(WORKING_DIR, max_async=args.max_async)

    if not args.skip_index and not args.query_only:
        t0 = time.time()
        print(f"Indexing {len(products)} products...")
        await index_products(rag, products)
        print(f"Indexing done in {time.time() - t0:.0f}s")

    print(f"\nQuerying with mode={args.mode}...")
    predictions = await query_products(rag, queries, product_ids_set, mode=args.mode)

    output_file = os.path.join(RESULT_DIR, "b4_lightrag.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_file}")

    result = evaluate(predictions, queries)

    eval_file = os.path.join(RESULT_DIR, "eval_b4_lightrag.json")
    with open(eval_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved eval to {eval_file}")


if __name__ == "__main__":
    asyncio.run(main())
