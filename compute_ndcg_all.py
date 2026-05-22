#!/usr/bin/env python3
"""Compute NDCG@1/3/10/20 for all result files (both ranked-list and DSL-eval formats)."""
import json, math, os, sys
from collections import defaultdict

def load_gt(gt_file):
    """Load ground truth: {qid: set(gt_ids)}"""
    gt = {}
    with open(gt_file) as f:
        for line in f:
            d = json.loads(line)
            qid = d["query_id"]
            ids = d.get("gt_document_ids") or d.get("gold_ids") or []
            if ids:
                gt[qid] = set(ids)
    return gt

def dcg(ranked, gt, k):
    score = 0.0
    for i, doc in enumerate(ranked[:k]):
        if doc in gt:
            score += 1.0 / math.log2(i + 2)
    return score

def idcg(gt_size, k):
    score = 0.0
    for i in range(min(gt_size, k)):
        score += 1.0 / math.log2(i + 2)
    return score

def compute_ndcg(ranked, gt, k):
    ideal = idcg(len(gt), k)
    if ideal == 0:
        return 0.0
    return dcg(ranked, gt, k) / ideal

def eval_ranked_file(filepath, gt, ks=[1, 3, 10, 20]):
    """Evaluate a file in {qid: [ranked_id_list]} format."""
    data = json.load(open(filepath))
    metrics = {f"NDCG@{k}": 0.0 for k in ks}
    n = 0
    by_level = defaultdict(lambda: {f"NDCG@{k}": 0.0 for k in ks} | {"n": 0})

    for qid, ranked in data.items():
        if qid not in gt:
            continue
        n += 1
        for k in ks:
            metrics[f"NDCG@{k}"] += compute_ndcg(ranked, gt[qid], k)

    if n == 0:
        return None
    return {k: v / n for k, v in metrics.items()} | {"n": n}

def eval_dsl_eval_file(filepath, gt, ks=[1, 3, 10, 20]):
    """Evaluate a DSL eval file that has per_query results with hit info.
    For DSL methods, results are unordered sets. We compute NDCG assuming
    hits appear at best possible positions (optimistic NDCG)."""
    data = json.load(open(filepath))
    if "per_query" not in data:
        return None

    per_query = data["per_query"]
    metrics = {f"NDCG@{k}": 0.0 for k in ks}
    n = 0

    for pq in per_query:
        qid = pq["query_id"]
        if qid not in gt:
            continue
        gt_set = gt[qid]
        result_size = pq.get("result_size", 0) or 0
        gt_size = pq.get("gt_size", len(gt_set)) or len(gt_set)
        recall = pq.get("recall", 0) or 0

        n_hits = round(recall * gt_size) if gt_size > 0 else 0
        n += 1

        for k in ks:
            ideal = idcg(gt_size, k)
            if ideal == 0:
                continue
            effective_results = min(result_size, k)
            effective_hits = min(n_hits, effective_results)
            d = 0.0
            for i in range(effective_hits):
                d += 1.0 / math.log2(i + 2)
            metrics[f"NDCG@{k}"] += d / ideal

    if n == 0:
        return None
    return {k: v / n for k, v in metrics.items()} | {"n": n}

def main():
    ecom_gt_file = "data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
    email_gt_file = "data/email/test/queries_email_test_step9.jsonl"

    ecom_gt = load_gt(ecom_gt_file)
    email_gt = load_gt(email_gt_file)

    print(f"Loaded GT: ecommerce={len(ecom_gt)}, email={len(email_gt)}")

    # === Ecommerce ranked-list files ===
    ecom_ranked = {
        "B1 BM25": "data/ecommerce/result/b1_bm25.json",
        "B1 ColQwen": "data/ecommerce/result/b1_colqwen.json",
        "B1 RRF": "data/ecommerce/result/b1_rrf.json",
        "B1 BGE-M3": "data/ecommerce/result/b1_bgem3.json",
        "B1 BGE-Reranker": "data/ecommerce/result/b1_reranker.json",
        "B1 Dense Image": "data/ecommerce/result/b1_dense_image.json",
        "B2 HyDE": "data/ecommerce/result/b2_hyde.json",
        "B2 IRCoT": "data/ecommerce/result/b2_ircot.json",
        "B2 RankGPT": "data/ecommerce/result/b2_rankgpt.json",
        "B3 DeepRetrieval": "data/ecommerce/result/b3_deepretrieval_full.json",
        "B3 SQR": "data/ecommerce/result/b3_langchain_sqr.json",
        "B3 SQL-R1 zero-shot": "data/ecommerce/result/b3_sql_r1.json",
    }

    # === Email ranked-list files ===
    email_ranked = {
        "B1 BM25": "data/email/result/b1_bm25.json",
        "B1 ColQwen": "data/email/result/b1_colqwen.json",
        "B1 RRF": "data/email/result/b1_rrf.json",
        "B1 BGE-M3": "data/email/result/b1_bgem3.json",
        "B1 BGE-Reranker": "data/email/result/b1_reranker.json",
    }

    # === Ecommerce DSL eval files (per_query with recall) ===
    ecom_dsl = {
        "4B Base": "data/ecommerce/result/eval_base_gt_v2.json",
        "4B SFT": "data/ecommerce/result/eval_predictions_sft_4b_gt_v2.json",
        "4B GRPO step200": "data/ecommerce/result/eval_rl_step200_gt_v2.json",
        "4B GRPO step300": "data/ecommerce/result/eval_rl_step300_gt_v2.json",
        "4B GRPO step550": "data/ecommerce/result/eval_rl_step550_gt_v2.json",
        "4B DAPO step420": "data/ecommerce/result/eval_dapo_step420_gt_v2.json",
        "8B Base": "data/ecommerce/result/eval_predictions_base8b.json",
        "8B SFT": "data/ecommerce/result/eval_predictions_sft8b.json",
        "8B GRPO step150": "data/ecommerce/result/eval_predictions_grpo8b_step150.json",
        "GPT-5.4": "data/ecommerce/result/eval_gpt54_azure_gt_v2_nocase.json",
        "GPT-5.4 thinking": "data/ecommerce/result/eval_gpt54_thinking_gt_v2_nocase.json",
        "GPT-5.5": "data/ecommerce/result/eval_predictions_gpt55_full.json",
        "Opus 4.6": "data/ecommerce/result/eval_claude_opus_gt_v2_nocase.json",
        "Opus 4.7": "data/ecommerce/result/eval_predictions_opus47.json",
        "SFT 1.7B": "data/ecommerce/result/eval_sft_1.7b.json",
        "4B GRPO step1200": "data/ecommerce/result/eval_rl_step1200_gt_v2.json",
        "Gemini 3.1 Pro": "data/ecommerce/result/eval_gemini_31pro_v2_gt_v2_nocase.json",
        "B5 Few-shot GPT-5.5": "data/ecommerce/result/eval_predictions_fewshot_gpt55.json",
        "B5 Few-shot Opus 4.7": "data/ecommerce/result/eval_predictions_fewshot_opus47.json",
    }

    # === Email DSL eval files ===
    email_dsl = {
        "4B Base": "data/email/result/eval_predictions_base_email_step9_gt.json",
        "4B SFT": "data/email/result/eval_predictions_sft_email_step9_gt.json",
        "4B GRPO step843": "data/email/result/eval_predictions_grpo_step843_step9_gt.json",
        "4B DAPO step843": "data/email/result/eval_predictions_dapo_step843_step9_gt.json",
        "GPT-5.4": "data/email/result/eval_predictions_email_gpt54_list_step9_gt.json",
        "GPT-5.5": "data/email/result/eval_predictions_email_gpt55_list_step9_gt.json",
        "Opus 4.7": "data/email/result/eval_predictions_email_opus47_list_step9_gt.json",
    }

    all_results = {}

    # Ecommerce ranked
    print("\n" + "=" * 70)
    print("ECOMMERCE — Ranked-list Baselines")
    print("=" * 70)
    print(f"{'Method':<25} {'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'N':>6}")
    print("-" * 70)
    for name, path in ecom_ranked.items():
        if not os.path.exists(path):
            print(f"  {name}: FILE NOT FOUND ({path})")
            continue
        r = eval_ranked_file(path, ecom_gt)
        if r:
            print(f"{name:<25} {r['NDCG@1']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['NDCG@20']:>8.4f} {r['n']:>6}")
            all_results[f"ecom_ranked_{name}"] = r

    # Ecommerce DSL
    print("\n" + "=" * 70)
    print("ECOMMERCE — DSL Methods (optimistic NDCG: hits at best positions)")
    print("=" * 70)
    print(f"{'Method':<25} {'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'N':>6}")
    print("-" * 70)
    for name, path in ecom_dsl.items():
        if not os.path.exists(path):
            print(f"  {name}: FILE NOT FOUND")
            continue
        r = eval_dsl_eval_file(path, ecom_gt)
        if r:
            print(f"{name:<25} {r['NDCG@1']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['NDCG@20']:>8.4f} {r['n']:>6}")
            all_results[f"ecom_dsl_{name}"] = r
        else:
            print(f"  {name}: NO DATA")

    # Email ranked
    print("\n" + "=" * 70)
    print("EMAIL — Ranked-list Baselines")
    print("=" * 70)
    print(f"{'Method':<25} {'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'N':>6}")
    print("-" * 70)
    for name, path in email_ranked.items():
        if not os.path.exists(path):
            print(f"  {name}: FILE NOT FOUND ({path})")
            continue
        r = eval_ranked_file(path, email_gt)
        if r:
            print(f"{name:<25} {r['NDCG@1']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['NDCG@20']:>8.4f} {r['n']:>6}")
            all_results[f"email_ranked_{name}"] = r

    # Email DSL
    print("\n" + "=" * 70)
    print("EMAIL — DSL Methods (optimistic NDCG: hits at best positions)")
    print("=" * 70)
    print(f"{'Method':<25} {'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'N':>6}")
    print("-" * 70)
    for name, path in email_dsl.items():
        if not os.path.exists(path):
            print(f"  {name}: FILE NOT FOUND")
            continue
        r = eval_dsl_eval_file(path, email_gt)
        if r:
            print(f"{name:<25} {r['NDCG@1']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['NDCG@20']:>8.4f} {r['n']:>6}")
            all_results[f"email_dsl_{name}"] = r
        else:
            print(f"  {name}: NO DATA")

    with open("data/ndcg_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to data/ndcg_all_results.json")


if __name__ == "__main__":
    main()
