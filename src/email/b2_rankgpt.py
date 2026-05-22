#!/usr/bin/env python3
"""B2 Baseline: RankGPT — LLM listwise reranking on top of RRF results for email."""
import argparse, json, os, re, time
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/email/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/email/result"
GT_FILE = os.path.join(DATA_DIR, "queries_email_test_step9.jsonl")
EMAILS_FILE = os.path.join(DATA_DIR, "email_test.json")
RRF_FILE = os.path.join(RESULT_DIR, "b1_rrf.json")
TOP_K = 20

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

RANKGPT_PROMPT = """I will provide you with {num} emails, each indicated by a numerical identifier [1] to [{num}]. Rank the emails based on their relevance to the search query: "{query}"

{passages}

Search query: "{query}"
Rank the {num} emails above based on their relevance to the search query. The emails should be listed in descending order using identifiers. The most relevant email should be listed first. The output format should be a list of identifiers, e.g., [3] > [1] > [2]. Only respond with the ranking, no other text."""


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def build_email_map(emails_file):
    emails = json.load(open(emails_file))
    emap = {}
    for e in emails:
        desc = f"From: {e.get('From','N/A')} | To: {e.get('To','N/A')} | Subject: {e.get('Subject','N/A')} | Date: {e.get('Date','N/A')} | {(e.get('Message','') or '')[:200]}"
        emap[e["Message_ID"]] = desc
    return emap


def parse_ranking(text, num):
    ids = re.findall(r'\[(\d+)\]', text)
    ranked = []
    seen = set()
    for id_str in ids:
        idx = int(id_str)
        if 1 <= idx <= num and idx not in seen:
            ranked.append(idx)
            seen.add(idx)
    for i in range(1, num + 1):
        if i not in seen:
            ranked.append(i)
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    qmap = {q["query_id"]: q for q in queries}
    print(f"Loaded {len(queries)} queries")

    emap = build_email_map(EMAILS_FILE)
    print(f"Loaded {len(emap)} emails")

    rrf = json.load(open(RRF_FILE))
    print(f"Loaded RRF results for {len(rrf)} queries")

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    output_jsonl = os.path.join(RESULT_DIR, "rankgpt_rerank_log.jsonl")
    existing = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d["reranked"]
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [q for q in queries if q["query_id"] not in existing and q["query_id"] in rrf]
    print(f"  Todo: {len(todo)} queries")

    for i, q in enumerate(todo):
        qid = q["query_id"]
        candidates = rrf[qid][:TOP_K]
        num = len(candidates)

        passages = ""
        for j, eid in enumerate(candidates):
            desc = emap.get(eid, "N/A")
            passages += f"[{j+1}] {desc}\n"

        prompt = RANKGPT_PROMPT.format(
            num=num, query=q["natural_language_query"], passages=passages
        )

        reranked_eids = candidates
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=200,
                )
                rank_text = resp.choices[0].message.content.strip()
                ranked_indices = parse_ranking(rank_text, num)
                reranked_eids = [candidates[idx - 1] for idx in ranked_indices]
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        existing[qid] = reranked_eids
        with open(output_jsonl, "a") as f:
            f.write(json.dumps({"query_id": qid, "reranked": reranked_eids}, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] reranked")

    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]
        elif qid in rrf:
            predictions[qid] = rrf[qid][:TOP_K]

    output_file = os.path.join(RESULT_DIR, "b2_rankgpt.json")
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
    print(f"\nRankGPT Results (N={n}):")
    print(f"  Hit@1: {overall['hit1']/n:.4f}")
    print(f"  Hit@3: {overall['hit3']/n:.4f}")
    print(f"  MRR:   {overall['mrr']/n:.4f}")

    eval_file = os.path.join(RESULT_DIR, "eval_b2_rankgpt.json")
    with open(eval_file, "w") as f:
        json.dump({"hit1": overall["hit1"]/n, "hit3": overall["hit3"]/n, "mrr": overall["mrr"]/n, "count": n}, f, indent=2)


if __name__ == "__main__":
    main()
