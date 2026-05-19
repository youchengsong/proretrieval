#!/usr/bin/env python3
"""B3 Baseline: SQL-R1 style pure SQL retrieval.

Prompts GPT-5.4 to generate a pure SQL WHERE clause from natural language queries.
Executes against SQLite product table. No vector retrieval — tests whether
structured filtering alone suffices for product retrieval.

Represents the 'SQL-only' paradigm (as in SQL-R1, DAIL-SQL, etc.) adapted to
our product retrieval task.
"""
import argparse, json, os, re, sqlite3, time
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"
GT_FILE = os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_test.json")

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

SQL_PROMPT = """You are a product search SQL generator. Given a customer's search query, generate a SQL WHERE clause to find matching products.

Table schema:
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    brand TEXT,
    color TEXT,
    category TEXT,
    price REAL,
    average_rating REAL,
    rating_number INTEGER,
    text TEXT
);

Rules:
1. Output ONLY the WHERE clause (without "WHERE" keyword), nothing else.
2. Use LIKE with % wildcards for text/brand/color/category matching (case-insensitive).
3. Use exact comparisons (=, >, <, >=, <=, BETWEEN) for price and average_rating.
4. Use AND/OR to combine conditions as needed.
5. If the query mentions a product description/feature, use text LIKE '%keyword%'.
6. If no structured filters apply, output: 1=1

Examples:
Query: "Nike black running shoes under $100"
Output: brand LIKE '%Nike%' AND color LIKE '%black%' AND text LIKE '%running%' AND text LIKE '%shoes%' AND price < 100

Query: "highly rated wireless headphones"
Output: text LIKE '%wireless%' AND text LIKE '%headphones%' AND average_rating >= 4.0

Query: "red laptop bag"
Output: color LIKE '%red%' AND text LIKE '%laptop%' AND text LIKE '%bag%'

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


def build_db(products):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE products (
        product_id TEXT PRIMARY KEY, brand TEXT, color TEXT, category TEXT,
        price REAL, average_rating REAL, rating_number INTEGER, text TEXT
    )""")
    for p in products:
        conn.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
            (p.get("product_id"), p.get("brand"), p.get("color"),
             p.get("category"), p.get("price"), p.get("average_rating"),
             p.get("rating_number"), p.get("text", "")),
        )
    conn.commit()
    return conn


def execute_sql(conn, where_clause):
    where_clause = where_clause.strip().rstrip(";")
    if not where_clause or where_clause == "1=1":
        return []
    sql = f"SELECT product_id FROM products WHERE {where_clause} LIMIT {TOP_K}"
    try:
        rows = conn.execute(sql).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    print(f"Loaded {len(products)} products")

    conn = build_db(products)

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    log_file = os.path.join(RESULT_DIR, "b3_sql_r1_log.jsonl")
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

        prompt = SQL_PROMPT.format(query=query_text)
        where_clause = "1=1"
        raw_output = ""

        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=300,
                )
                raw_output = resp.choices[0].message.content.strip()
                raw_output = re.sub(r"^```sql\s*", "", raw_output)
                raw_output = re.sub(r"\s*```$", "", raw_output)
                where_clause = raw_output
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        ranked = execute_sql(conn, where_clause)

        existing[qid] = {
            "query_id": qid,
            "ranked": ranked,
            "where_clause": where_clause[:500],
            "result_count": len(ranked),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            ok = sum(1 for v in existing.values() if v["result_count"] > 0)
            print(f"  [{i+1}/{len(todo)}] done, {ok}/{len(existing)} had results")

    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]["ranked"]

    output_file = os.path.join(RESULT_DIR, "b3_sql_r1.json")
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

    print(f"\nSQL-R1 Results (N={n}):")
    print(f"  Hit@1: {hit1/n:.4f}")
    print(f"  Hit@3: {hit3/n:.4f}")
    print(f"  MRR:   {mrr/n:.4f}")
    nonempty = sum(1 for v in existing.values() if v["result_count"] > 0)
    print(f"  Exec Rate: {nonempty}/{len(existing)} ({nonempty/len(existing)*100:.1f}%)")


if __name__ == "__main__":
    main()
