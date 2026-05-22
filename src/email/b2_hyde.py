#!/usr/bin/env python3
"""B2 Baseline: HyDE (Hypothetical Document Embeddings) for email retrieval."""
import argparse, json, os, time
import numpy as np
import pandas as pd
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/email/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/email/result"
GT_FILE = os.path.join(DATA_DIR, "queries_email_test_step9.jsonl")
EMAILS_FILE = os.path.join(DATA_DIR, "email_test.json")
MSG_FEAT_FILE = os.path.join(DATA_DIR, "test_message_features.parquet")
QWEN_PATH = os.environ.get("QWEN_PATH", "/home/sunzhen/models/Qwen3-Embedding-8B")
TOP_K = 20

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

HYDE_PROMPT = """You are an email archive assistant. Given a user's search query about emails, write a hypothetical email that would perfectly match this query. Include realistic From/To addresses, Subject line, Date, and email body content. Be specific and realistic.

User query: {query}

Hypothetical email:"""


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def generate_hypothetical_docs(queries, output_jsonl):
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    existing = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d["hyp_doc"]
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [q for q in queries if q["query_id"] not in existing]
    print(f"  Todo: {len(todo)} queries")

    for i, q in enumerate(todo):
        prompt = HYDE_PROMPT.format(query=q["natural_language_query"])
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_completion_tokens=300,
                )
                hyp_doc = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {q['query_id']}: {e}")
                time.sleep(2 ** retry)
                hyp_doc = ""

        with open(output_jsonl, "a") as f:
            f.write(json.dumps({"query_id": q["query_id"], "hyp_doc": hyp_doc}, ensure_ascii=False) + "\n")
        existing[q["query_id"]] = hyp_doc

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] generated")

    return existing


def encode_and_retrieve(queries, hyp_docs, device="cuda:0"):
    import torch
    from sentence_transformers import SentenceTransformer

    print("Loading precomputed message features...")
    mf = pd.read_parquet(MSG_FEAT_FILE)
    eid_list = mf["message_id"].values.tolist()
    m_mat = np.stack(mf["text_features"].values).astype(np.float32)
    m_mat = m_mat / np.linalg.norm(m_mat, axis=1, keepdims=True)

    print(f"Loading Qwen3-Embedding on {device} (fp16)...")
    model = SentenceTransformer(QWEN_PATH, device=device, model_kwargs={"torch_dtype": torch.float16})

    doc_texts = []
    qid_order = []
    for q in queries:
        qid = q["query_id"]
        doc = hyp_docs.get(qid, "")
        if not doc:
            doc = q["natural_language_query"]
        doc_texts.append(doc)
        qid_order.append(qid)

    print(f"Encoding {len(doc_texts)} hypothetical docs...")
    doc_vecs = model.encode(doc_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    doc_vecs = doc_vecs.astype(np.float32)

    print("Computing similarities...")
    scores = doc_vecs @ m_mat.T

    predictions = {}
    for i, qid in enumerate(qid_order):
        top_idx = np.argsort(-scores[i])[:TOP_K]
        predictions[qid] = [eid_list[idx] for idx in top_idx]

    return predictions


def evaluate(predictions, queries):
    overall = {"hit1": 0, "hit3": 0, "mrr": 0, "n": 0}
    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or q.get("gold_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:TOP_K]
        h1 = 1.0 if ranked and ranked[0] in gt else 0.0
        h3 = 1.0 if any(r in gt for r in ranked[:3]) else 0.0
        rr = 0.0
        for i, r in enumerate(ranked):
            if r in gt:
                rr = 1.0 / (i + 1)
                break
        overall["hit1"] += h1
        overall["hit3"] += h3
        overall["mrr"] += rr
        overall["n"] += 1
    n = overall["n"]
    if n == 0:
        return {}
    return {"hit1": overall["hit1"]/n, "hit3": overall["hit3"]/n, "mrr": overall["mrr"]/n, "count": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gen-only", action="store_true")
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    hyp_jsonl = os.path.join(RESULT_DIR, "hyde_hypothetical_docs.jsonl")
    print("Step 1: Generating hypothetical emails via GPT-5.4...")
    hyp_docs = generate_hypothetical_docs(queries, hyp_jsonl)
    print(f"  Total hypothetical docs: {len(hyp_docs)}")

    if args.gen_only:
        print("gen-only mode, stopping here.")
        return

    print("Step 2: Encoding & retrieving...")
    predictions = encode_and_retrieve(queries, hyp_docs, device=f"cuda:{args.gpu}")

    output_file = os.path.join(RESULT_DIR, "b2_hyde.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_file}")

    print("Step 3: Evaluating...")
    results = evaluate(predictions, queries)
    print(f"\nHyDE Results (N={results['count']}):")
    print(f"  Hit@1: {results['hit1']:.4f}")
    print(f"  Hit@3: {results['hit3']:.4f}")
    print(f"  MRR:   {results['mrr']:.4f}")

    eval_file = os.path.join(RESULT_DIR, "eval_b2_hyde.json")
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved eval to {eval_file}")


if __name__ == "__main__":
    main()
