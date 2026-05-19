#!/usr/bin/env python3
"""B4 Baseline: GraphRAG-style knowledge graph retrieval.

Builds a lightweight knowledge graph from product data using LLM entity/relation
extraction, then retrieves products by matching query entities to the graph.

This is a simplified version of Microsoft's GraphRAG adapted for product retrieval.
Instead of the full GraphRAG pipeline (which is designed for long-form text corpora),
we build a product-centric KG and use entity matching + graph traversal for retrieval.

Paradigm: Knowledge-Graph Retrieval — complementary to our DSL approach
(graph-edge structure vs field-constraint structure).
"""
import argparse, json, math, os, re, time
from collections import Counter, defaultdict
from openai import AzureOpenAI

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"
GT_FILE = os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_test.json")

AZURE_ENDPOINT = "https://algorithm.cognitiveservices.azure.com/"
AZURE_KEY = "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX"
AZURE_DEPLOY = "gpt-5.4"

EXTRACT_PROMPT = """Extract key entities and relationships from this product description.

Product ID: {pid}
Brand: {brand}
Color: {color}
Category: {category}
Price: ${price}
Rating: {rating}
Description: {text}

Output a JSON object with:
- "entities": list of {{type, name}} where type is one of: product_type, feature, material, use_case, target_audience
- "relationships": list of {{source, relation, target}} connecting this product to concepts

Be specific and extract at least 3-5 entities. Output JSON only:"""

QUERY_EXTRACT_PROMPT = """Extract search entities from this customer query.

Query: {query}

Output a JSON object with:
- "entities": list of {{type, name}} where type is one of: brand, color, product_type, feature, material, use_case, target_audience, price_range
- "constraints": list of specific requirements (e.g., "price under 50", "rating above 4")

Output JSON only:"""

TOP_K = 20


def load_queries(gt_file, limit=None):
    queries = []
    with open(gt_file) as f:
        for line in f:
            queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def parse_json(text):
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
    return None


class ProductKG:
    """Lightweight product knowledge graph."""

    def __init__(self):
        self.entity_to_products = defaultdict(set)
        self.product_entities = defaultdict(set)
        self.product_data = {}

    def add_product(self, pid, product, entities):
        self.product_data[pid] = product

        brand = (product.get("brand") or "").lower().strip()
        if brand:
            key = f"brand:{brand}"
            self.entity_to_products[key].add(pid)
            self.product_entities[pid].add(key)

        color = (product.get("color") or "").lower().strip()
        if color:
            key = f"color:{color}"
            self.entity_to_products[key].add(pid)
            self.product_entities[pid].add(key)

        category = (product.get("category") or "").lower().strip()
        if category:
            key = f"category:{category}"
            self.entity_to_products[key].add(pid)
            self.product_entities[pid].add(key)

        if entities:
            for ent in entities:
                etype = (ent.get("type") or "entity").lower()
                ename = (ent.get("name") or "").lower().strip()
                if ename and len(ename) > 1:
                    key = f"{etype}:{ename}"
                    self.entity_to_products[key].add(pid)
                    self.product_entities[pid].add(key)

    def search(self, query_entities, constraints=None, top_k=20):
        """Score products by entity overlap with query."""
        scores = Counter()

        for ent in query_entities:
            etype = (ent.get("type") or "entity").lower()
            ename = (ent.get("name") or "").lower().strip()
            if not ename:
                continue

            exact_key = f"{etype}:{ename}"
            if exact_key in self.entity_to_products:
                for pid in self.entity_to_products[exact_key]:
                    scores[pid] += 3.0

            for key, pids in self.entity_to_products.items():
                if ename in key or key.split(":", 1)[-1] in ename:
                    for pid in pids:
                        scores[pid] += 1.0

        if constraints:
            for c in constraints:
                c_lower = c.lower()
                price_match = re.search(r'(?:under|below|less than|<)\s*\$?(\d+)', c_lower)
                if price_match:
                    max_price = float(price_match.group(1))
                    for pid in list(scores.keys()):
                        p = self.product_data.get(pid, {})
                        if (p.get("price") or 0) > max_price:
                            scores[pid] *= 0.1

                price_match2 = re.search(r'(?:above|over|more than|>)\s*\$?(\d+)', c_lower)
                if price_match2:
                    min_price = float(price_match2.group(1))
                    for pid in list(scores.keys()):
                        p = self.product_data.get(pid, {})
                        if (p.get("price") or 99999) < min_price:
                            scores[pid] *= 0.1

                rating_match = re.search(r'rating\s*(?:above|over|>=?)\s*(\d+\.?\d*)', c_lower)
                if rating_match:
                    min_rating = float(rating_match.group(1))
                    for pid in list(scores.keys()):
                        p = self.product_data.get(pid, {})
                        if (p.get("average_rating") or 0) < min_rating:
                            scores[pid] *= 0.5

        ranked = scores.most_common(top_k)
        return [pid for pid, score in ranked if score > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index-limit", type=int, default=None,
                        help="Limit products indexed (for cost control)")
    args = parser.parse_args()

    queries = load_queries(GT_FILE, limit=args.limit)
    print(f"Loaded {len(queries)} queries")

    products = json.load(open(PRODUCTS_FILE))
    print(f"Loaded {len(products)} products")

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version="2024-12-01-preview",
    )

    # Phase 1: Build KG (extract entities from products)
    kg = ProductKG()
    kg_cache_file = os.path.join(RESULT_DIR, "b4_graphrag_kg_cache.jsonl")
    kg_cache = {}
    if os.path.exists(kg_cache_file):
        with open(kg_cache_file) as f:
            for line in f:
                d = json.loads(line)
                kg_cache[d["product_id"]] = d.get("entities", [])
        print(f"  Loaded {len(kg_cache)} cached entity extractions")

    products_to_index = products
    if args.index_limit:
        products_to_index = products[:args.index_limit]

    todo_products = [p for p in products_to_index if p["product_id"] not in kg_cache]
    print(f"  Need to extract entities for {len(todo_products)} products")

    for i, p in enumerate(todo_products):
        pid = p["product_id"]
        prompt = EXTRACT_PROMPT.format(
            pid=pid,
            brand=p.get("brand", ""),
            color=p.get("color", ""),
            category=p.get("category", ""),
            price=p.get("price", ""),
            rating=p.get("average_rating", ""),
            text=(p.get("text", "") or "")[:500],
        )

        entities = []
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=500,
                )
                raw = resp.choices[0].message.content.strip()
                parsed = parse_json(raw)
                if parsed:
                    entities = parsed.get("entities", [])
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(2 ** retry)
                else:
                    print(f"  Failed entity extraction for {pid}: {e}")

        kg_cache[pid] = entities
        with open(kg_cache_file, "a") as f:
            f.write(json.dumps({"product_id": pid, "entities": entities}, ensure_ascii=False) + "\n")

        if (i + 1) % 100 == 0:
            print(f"  Entity extraction: [{i+1}/{len(todo_products)}]")

    for p in products:
        pid = p["product_id"]
        entities = kg_cache.get(pid, [])
        kg.add_product(pid, p, entities)

    print(f"  KG built: {len(kg.entity_to_products)} entity types, {len(kg.product_data)} products")

    # Phase 2: Query
    log_file = os.path.join(RESULT_DIR, "b4_graphrag_log.jsonl")
    existing = {}
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"  Found {len(existing)} existing query results, resuming...")

    todo_queries = [q for q in queries if q["query_id"] not in existing]
    print(f"  Todo: {len(todo_queries)} queries")

    for i, q in enumerate(todo_queries):
        qid = q["query_id"]
        query_text = q["natural_language_query"]

        prompt = QUERY_EXTRACT_PROMPT.format(query=query_text)
        query_entities = []
        constraints = []

        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()
                parsed = parse_json(raw)
                if parsed:
                    query_entities = parsed.get("entities", [])
                    constraints = parsed.get("constraints", [])
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(2 ** retry)
                else:
                    print(f"  Failed query extraction for {qid}: {e}")

        ranked = kg.search(query_entities, constraints, top_k=TOP_K)

        existing[qid] = {
            "query_id": qid,
            "ranked": ranked,
            "query_entities": query_entities[:10],
            "result_count": len(ranked),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  Query: [{i+1}/{len(todo_queries)}] done")

    predictions = {}
    for q in queries:
        qid = q["query_id"]
        if qid in existing:
            predictions[qid] = existing[qid]["ranked"]

    output_file = os.path.join(RESULT_DIR, "b4_graphrag.json")
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

    print(f"\nGraphRAG Results (N={n}):")
    print(f"  Hit@1: {hit1/n:.4f}")
    print(f"  Hit@3: {hit3/n:.4f}")
    print(f"  MRR:   {mrr/n:.4f}")


if __name__ == "__main__":
    main()
