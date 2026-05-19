#!/usr/bin/env python3
"""B3 Baseline: DeepRetrieval-style boolean query + BM25 retrieval.

Prompts GPT-5.4 to generate structured boolean keyword queries from natural
language, then executes BM25 retrieval. Represents the 'boolean query generation'
paradigm (as in DeepRetrieval) adapted to product retrieval.

Unlike our DSL approach (SQL + vector), this uses only keyword-based retrieval
with LLM-generated query reformulation.
"""
import argparse, json, math, os, re, time
from collections import Counter
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"
GT_FILE = os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_test.json")

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

BOOL_PROMPT = """You are a search query optimizer. Given a customer's product search query, generate an optimized boolean keyword query for BM25 retrieval.

Rules:
1. Use AND to combine mandatory terms.
2. Use OR within parentheses for synonyms/alternatives.
3. Include brand names, colors, materials, product types, and key features.
4. Expand abbreviations and include common synonyms.
5. Do NOT include price or rating — BM25 cannot filter numerics.
6. Output ONLY the boolean query, nothing else.

Examples:
Query: "Nike black running shoes for marathon"
Output: Nike AND black AND (running OR marathon) AND (shoes OR sneakers)

Query: "wireless noise cancelling headphones under $200"
Output: wireless AND (noise cancelling OR noise-canceling OR ANC) AND (headphones OR earphones)

Query: "red leather laptop bag for 15 inch"
Output: (red OR crimson) AND (leather) AND (laptop AND bag) AND (15 inch OR 15")

Query: "highly rated stainless steel water bottle"
Output: (stainless steel) AND (water bottle OR thermos OR flask)

Customer query: {query}
Output:"""

TOP_K = 20


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def tokenize(text):
    if not text:
        return []
    return re.findall(r'\w+', text.lower())


class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_lens = []
        self.tf = []
        self.df = Counter()
        for doc in corpus:
            tokens = tokenize(doc)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.tf.append(tf)
            for t in set(tokens):
                self.df[t] += 1
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1

    def score(self, query_tokens, doc_idx):
        s = 0.0
        dl = self.doc_lens[doc_idx]
        for t in query_tokens:
            if t not in self.tf[doc_idx]:
                continue
            tf_val = self.tf[doc_idx][t]
            df_val = self.df.get(t, 0)
            idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
            tf_norm = (tf_val * (self.k1 + 1)) / (tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            s += idf * tf_norm
        return s

    def search(self, query_tokens, top_k=20):
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


def parse_boolean_query(bool_query):
    """Parse boolean query into a flat list of keywords for BM25."""
    cleaned = bool_query.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r'\bAND\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bOR\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bNOT\b', ' ', cleaned, flags=re.IGNORECASE)
    return tokenize(cleaned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    pid_list = [p["product_id"] for p in products]
    print(f"Loaded {len(products)} products")

    corpus = []
    for p in products:
        parts = [
            p.get("brand", "") or "",
            p.get("color", "") or "",
            p.get("category", "") or "",
            p.get("text", "") or "",
        ]
        corpus.append(" ".join(parts))

    print("Building BM25 index...")
    bm25 = SimpleBM25(corpus)
    print("  BM25 index built")

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    log_file = os.path.join(RESULT_DIR, "b3_deepretrieval_log.jsonl")
    existing = {}
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [q for q in queries if q["query_id"] not in existing]
    print(f"  Todo: {len(todo)} queries")

    for i, q in enumerate(todo):
        qid = q["query_id"]
        query_text = q["natural_language_query"]

        prompt = BOOL_PROMPT.format(query=query_text)
        bool_query = query_text
        raw_output = ""

        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=200,
                )
                raw_output = resp.choices[0].message.content.strip()
                bool_query = raw_output
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        query_tokens = parse_boolean_query(bool_query)
        if not query_tokens:
            query_tokens = tokenize(query_text)

        results = bm25.search(query_tokens, top_k=TOP_K)
        ranked = [pid_list[idx] for idx, score in results if score > 0]

        existing[qid] = {
            "query_id": qid,
            "ranked": ranked,
            "bool_query": bool_query[:500],
            "result_count": len(ranked),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] done")

    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]["ranked"]

    output_file = os.path.join(RESULT_DIR, "b3_deepretrieval.json")
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

    print(f"\nDeepRetrieval Results (N={n}):")
    print(f"  Hit@1: {hit1/n:.4f}")
    print(f"  Hit@3: {hit3/n:.4f}")
    print(f"  MRR:   {mrr/n:.4f}")


if __name__ == "__main__":
    main()
