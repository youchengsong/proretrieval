#!/usr/bin/env python3
"""
B1 Pure Retrieval Baselines for Email (all-in-one)
Methods: BM25, ColQwen (dense text), BGE-M3, RRF, BGE-reranker
No image modality for email.
"""
import argparse
import json
import os
import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR = "/home/sunzhen/LocalSearch/data/email/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/email/result"
EMAILS_FILE = os.path.join(DATA_DIR, "email_test.json")
GT_FILE = os.path.join(DATA_DIR, "queries_email_test_step9.jsonl")
MSG_FEAT_FILE = os.path.join(DATA_DIR, "test_message_features.parquet")
SUBJ_FEAT_FILE = os.path.join(DATA_DIR, "test_subject_features.parquet")
QWEN_PATH = "/home/sunzhen/models/Qwen3-Embedding-8B"
TOP_K = 20


def load_data():
    print("Loading data...")
    with open(EMAILS_FILE) as f:
        emails = json.load(f)
    eid_list = [e["Message_ID"] for e in emails]
    eid_to_idx = {eid: i for i, eid in enumerate(eid_list)}
    texts = []
    for e in emails:
        parts = []
        if e.get("Subject"):
            parts.append(f"Subject: {e['Subject']}")
        if e.get("From"):
            parts.append(f"From: {e['From']}")
        if e.get("To"):
            parts.append(f"To: {e['To']}")
        if e.get("Date"):
            parts.append(f"Date: {e['Date']}")
        if e.get("Message"):
            parts.append(e["Message"][:1000])
        texts.append(" ".join(parts))

    queries = []
    with open(GT_FILE) as f:
        for line in f:
            queries.append(json.loads(line))

    print(f"  Emails: {len(emails)}, Queries: {len(queries)}")
    return emails, eid_list, eid_to_idx, texts, queries


def evaluate(predictions, queries, method_name):
    hit1 = hit3 = mrr = recall = 0
    by_level = defaultdict(lambda: {"hit1": 0, "hit3": 0, "mrr": 0, "recall": 0, "n": 0})
    by_mod = defaultdict(lambda: {"hit1": 0, "hit3": 0, "mrr": 0, "recall": 0, "n": 0})
    n = 0

    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or q.get("gold_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:TOP_K]
        n += 1

        h1 = 1.0 if ranked and ranked[0] in gt else 0.0
        h3 = 1.0 if any(r in gt for r in ranked[:3]) else 0.0
        rr = 0.0
        for i, r in enumerate(ranked):
            if r in gt:
                rr = 1.0 / (i + 1)
                break
        rec = len(set(ranked) & gt) / len(gt) if gt else 0.0

        hit1 += h1; hit3 += h3; mrr += rr; recall += rec

        level = q.get("level", 0)
        by_level[level]["hit1"] += h1
        by_level[level]["hit3"] += h3
        by_level[level]["mrr"] += rr
        by_level[level]["recall"] += rec
        by_level[level]["n"] += 1

        for mod in q.get("modalities", []):
            by_mod[mod]["hit1"] += h1
            by_mod[mod]["hit3"] += h3
            by_mod[mod]["mrr"] += rr
            by_mod[mod]["recall"] += rec
            by_mod[mod]["n"] += 1

    if n == 0:
        print(f"  {method_name}: No valid queries!")
        return {}

    result = {
        "method": method_name, "n": n,
        "hit1": hit1/n, "hit3": hit3/n, "mrr": mrr/n, "recall": recall/n,
        "by_level": {str(k): {kk: vv/v["n"] if kk != "n" else vv for kk, vv in v.items()} for k, v in sorted(by_level.items())},
        "by_modality": {k: {kk: vv/v["n"] if kk != "n" else vv for kk, vv in v.items()} for k, v in sorted(by_mod.items())},
    }

    print(f"\n{'='*50}")
    print(f"{method_name} (N={n})")
    print(f"  Hit@1:   {hit1/n:.4f}")
    print(f"  Hit@3:   {hit3/n:.4f}")
    print(f"  MRR:     {mrr/n:.4f}")
    print(f"  Recall:  {recall/n:.4f}")
    print(f"\n  By Level:")
    for lv in sorted(by_level):
        d = by_level[lv]
        nn = d["n"]
        print(f"    L{lv}: n={nn}  Hit@1={d['hit1']/nn:.4f}  Hit@3={d['hit3']/nn:.4f}  MRR={d['mrr']/nn:.4f}")
    if by_mod:
        print(f"\n  By Modality:")
        for mod in sorted(by_mod):
            d = by_mod[mod]
            nn = d["n"]
            print(f"    {mod}: n={nn}  Hit@1={d['hit1']/nn:.4f}  Hit@3={d['hit3']/nn:.4f}")

    return result


def save_predictions(predictions, method_name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"b1_{method_name}.json")
    with open(path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"  Saved to {path}")


def save_eval(result, method_name):
    if not result:
        return
    path = os.path.join(RESULT_DIR, f"eval_b1_{method_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Eval saved to {path}")


# ── Method 1: BM25 ───────────────────────────────────────────
def run_bm25(texts, eid_list, queries):
    from rank_bm25 import BM25Okapi
    print("\n[BM25] Building index...")
    tokenized = [re.findall(r'\w+', t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)

    predictions = {}
    for i, q in enumerate(queries):
        qtokens = re.findall(r'\w+', q["natural_language_query"].lower())
        scores = bm25.get_scores(qtokens)
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        predictions[q["query_id"]] = [eid_list[j] for j in top_idx]
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(queries)}]")

    return predictions


# ── Method 2: ColQwen (Dense Text - message features) ────────
def run_colqwen(eid_list, queries):
    import torch
    from sentence_transformers import SentenceTransformer

    print("\n[ColQwen] Loading precomputed message features...")
    mf = pd.read_parquet(MSG_FEAT_FILE)
    m_ids = mf["message_id"].values.tolist()
    m_mat = np.stack(mf["text_features"].values).astype(np.float32)

    print("[ColQwen] Loading Qwen3-Embedding-8B (fp16)...")
    model = SentenceTransformer(QWEN_PATH, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512

    nl_queries = [q["natural_language_query"] for q in queries]
    print(f"[ColQwen] Encoding {len(nl_queries)} queries...")
    q_embs = model.encode(nl_queries, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

    del model
    torch.cuda.empty_cache()

    print("[ColQwen] Computing similarities...")
    scores = q_embs @ m_mat.T
    predictions = {}
    for i, q in enumerate(queries):
        top_idx = np.argsort(scores[i])[::-1][:TOP_K]
        predictions[q["query_id"]] = [m_ids[j] for j in top_idx]

    return predictions


# ── Method 3: BGE-M3 ─────────────────────────────────────────
def run_bgem3(texts, eid_list, queries):
    from FlagEmbedding import BGEM3FlagModel

    print("\n[BGE-M3] Loading model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("[BGE-M3] Encoding emails...")
    doc_embs = model.encode(texts, batch_size=32, max_length=512)["dense_vecs"]

    nl_queries = [q["natural_language_query"] for q in queries]
    print(f"[BGE-M3] Encoding {len(nl_queries)} queries...")
    q_embs = model.encode(nl_queries, batch_size=32, max_length=512)["dense_vecs"]

    del model
    import torch; torch.cuda.empty_cache()

    print("[BGE-M3] Computing similarities...")
    scores = q_embs @ doc_embs.T
    predictions = {}
    for i, q in enumerate(queries):
        top_idx = np.argsort(scores[i])[::-1][:TOP_K]
        predictions[q["query_id"]] = [eid_list[j] for j in top_idx]

    return predictions


# ── Method 4: RRF (BM25 + ColQwen) ───────────────────────────
def run_rrf(bm25_preds, colqwen_preds, queries, k=60):
    print(f"\n[RRF] Fusing BM25 + ColQwen (k={k})...")
    predictions = {}
    for q in queries:
        qid = q["query_id"]
        bm25_ranked = bm25_preds.get(qid, [])
        cq_ranked = colqwen_preds.get(qid, [])

        rrf_scores = defaultdict(float)
        for rank, eid in enumerate(bm25_ranked):
            rrf_scores[eid] += 1.0 / (k + rank + 1)
        for rank, eid in enumerate(cq_ranked):
            rrf_scores[eid] += 1.0 / (k + rank + 1)

        sorted_eids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K]
        predictions[qid] = sorted_eids

    return predictions


# ── Method 5: BGE-Reranker ───────────────────────────────────
def run_reranker(first_stage_preds, texts, eid_list, eid_to_idx, queries, top_n=100):
    from FlagEmbedding import FlagReranker

    print(f"\n[BGE-Reranker] Loading model...")
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    predictions = {}
    total = len(queries)
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        nl = q["natural_language_query"]
        candidates = first_stage_preds.get(qid, [])[:top_n]
        if not candidates:
            predictions[qid] = []
            continue

        pairs = [[nl, texts[eid_to_idx[eid]]] for eid in candidates if eid in eid_to_idx]
        valid_eids = [eid for eid in candidates if eid in eid_to_idx]

        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(valid_eids, scores), key=lambda x: x[1], reverse=True)
        predictions[qid] = [eid for eid, _ in ranked[:TOP_K]]

        if (qi + 1) % 500 == 0 or qi + 1 == total:
            print(f"  [{qi+1}/{total}]")

    del reranker
    import torch; torch.cuda.empty_cache()
    return predictions


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all",
                        help="Comma-separated: bm25,colqwen,bgem3,rrf,reranker or 'all'")
    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods != "all" else \
        ["bm25", "colqwen", "bgem3", "rrf", "reranker"]

    emails, eid_list, eid_to_idx, texts, queries = load_data()
    results = {}

    if "bm25" in methods:
        t0 = time.time()
        results["bm25"] = run_bm25(texts, eid_list, queries)
        save_predictions(results["bm25"], "bm25")
        r = evaluate(results["bm25"], queries, "BM25")
        save_eval(r, "bm25")
        print(f"  Time: {time.time()-t0:.0f}s")

    if "colqwen" in methods:
        t0 = time.time()
        results["colqwen"] = run_colqwen(eid_list, queries)
        save_predictions(results["colqwen"], "colqwen")
        r = evaluate(results["colqwen"], queries, "ColQwen (Dense Text)")
        save_eval(r, "colqwen")
        print(f"  Time: {time.time()-t0:.0f}s")

    if "rrf" in methods:
        bm25_p = results.get("bm25")
        cq_p = results.get("colqwen")
        if not bm25_p:
            path = os.path.join(RESULT_DIR, "b1_bm25.json")
            if os.path.exists(path):
                bm25_p = json.load(open(path))
        if not cq_p:
            path = os.path.join(RESULT_DIR, "b1_colqwen.json")
            if os.path.exists(path):
                cq_p = json.load(open(path))
        if bm25_p and cq_p:
            results["rrf"] = run_rrf(bm25_p, cq_p, queries)
            save_predictions(results["rrf"], "rrf")
            r = evaluate(results["rrf"], queries, "RRF (BM25+ColQwen)")
            save_eval(r, "rrf")
        else:
            print("[RRF] Skipped: need BM25 and ColQwen results first")

    if "bgem3" in methods:
        t0 = time.time()
        results["bgem3"] = run_bgem3(texts, eid_list, queries)
        save_predictions(results["bgem3"], "bgem3")
        r = evaluate(results["bgem3"], queries, "BGE-M3")
        save_eval(r, "bgem3")
        print(f"  Time: {time.time()-t0:.0f}s")

    if "reranker" in methods:
        first_stage = results.get("rrf")
        if not first_stage:
            path = os.path.join(RESULT_DIR, "b1_rrf.json")
            if os.path.exists(path):
                first_stage = json.load(open(path))
        if first_stage:
            t0 = time.time()
            results["reranker"] = run_reranker(first_stage, texts, eid_list, eid_to_idx, queries)
            save_predictions(results["reranker"], "reranker")
            r = evaluate(results["reranker"], queries, "BGE-Reranker (on RRF)")
            save_eval(r, "reranker")
            print(f"  Time: {time.time()-t0:.0f}s")
        else:
            print("[Reranker] Skipped: need RRF results first")

    print("\n" + "="*50)
    print("All done!")


if __name__ == "__main__":
    main()
