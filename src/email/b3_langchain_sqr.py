#!/usr/bin/env python3
"""B3 Baseline: LangChain-style Self-Query Retrieval (SQR) for email.

GPT-5.4 extracts structured filters (from, to, date, subject keywords) from the query,
applies them to filter emails, then ranks by text embedding similarity.
"""
import argparse, json, os, re, time
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

SQR_PROMPT = """You are an email search filter generator. Given a user's search query about emails, extract structured filters and a text search query.

Available filter fields:
- from_addr (string, substring match on sender address)
- to_addr (string, substring match on recipient address)
- subject_keyword (string, keyword that must appear in subject line)
- year (integer, e.g. 2000, 2001)
- date_keyword (string, keyword in the date field, e.g. "Jan", "October", "Mon")

Output a JSON object with:
- "filters": dict of field->value for applicable filters (omit fields not mentioned in query)
- "text_query": a refined text search query capturing the semantic intent

Only extract filters when the query CLEARLY specifies them. Do not guess.

User query: {query}

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


def apply_filters(emails, filters):
    filtered = emails
    if "from_addr" in filters and filters["from_addr"]:
        val = filters["from_addr"].lower()
        filtered = [e for e in filtered if val in (e.get("From", "") or "").lower()]
    if "to_addr" in filters and filters["to_addr"]:
        val = filters["to_addr"].lower()
        filtered = [e for e in filtered if val in (e.get("To", "") or "").lower()]
    if "subject_keyword" in filters and filters["subject_keyword"]:
        val = filters["subject_keyword"].lower()
        filtered = [e for e in filtered if val in (e.get("Subject", "") or "").lower()]
    if "year" in filters and filters["year"]:
        try:
            yr = str(int(filters["year"]))
            filtered = [e for e in filtered if yr in (e.get("Date", "") or "")]
        except (ValueError, TypeError):
            pass
    if "date_keyword" in filters and filters["date_keyword"]:
        val = filters["date_keyword"].lower()
        filtered = [e for e in filtered if val in (e.get("Date", "") or "").lower()]
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    emails = json.load(open(EMAILS_FILE))
    eid_list = [e["Message_ID"] for e in emails]
    eid_to_idx = {e["Message_ID"]: i for i, e in enumerate(emails)}
    print(f"Loaded {len(emails)} emails")

    print("Loading precomputed message features...")
    mf = pd.read_parquet(MSG_FEAT_FILE)
    feat_ids = mf["message_id"].values.tolist()
    m_mat = np.stack(mf["text_features"].values).astype(np.float32)
    m_mat = m_mat / np.linalg.norm(m_mat, axis=1, keepdims=True)
    feat_id_to_idx = {mid: i for i, mid in enumerate(feat_ids)}

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

        prompt = SQR_PROMPT.format(query=query_text)

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

        filtered = apply_filters(emails, filters)

        if len(filtered) < 5:
            filtered = emails
            filters = {}

        filtered_eids = [e["Message_ID"] for e in filtered]
        filtered_feat_indices = [feat_id_to_idx[eid] for eid in filtered_eids if eid in feat_id_to_idx]
        filtered_eids_valid = [eid for eid in filtered_eids if eid in feat_id_to_idx]

        if not filtered_feat_indices:
            filtered_feat_indices = list(range(len(feat_ids)))
            filtered_eids_valid = feat_ids

        filtered_mat = m_mat[filtered_feat_indices]

        q_vec = model.encode([text_query], normalize_embeddings=True).astype(np.float32)
        scores = (q_vec @ filtered_mat.T)[0]
        top_local = np.argsort(-scores)[:TOP_K]
        final_ranked = [filtered_eids_valid[idx] for idx in top_local]

        existing[qid] = final_ranked
        with open(log_jsonl, "a") as f:
            f.write(json.dumps({
                "query_id": qid, "final_ranked": final_ranked,
                "filters": filters, "text_query": text_query[:200],
                "filtered_count": len(filtered)
            }, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] done")

    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]

    output_file = os.path.join(RESULT_DIR, "b3_langchain_sqr.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_file}")

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
