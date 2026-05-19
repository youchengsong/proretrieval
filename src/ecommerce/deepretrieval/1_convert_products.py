#!/usr/bin/env python3
"""Convert product data to Pyserini-compatible JSONL format for BM25 indexing."""
import json
import os
import sys

INPUT_FILE = "/home/sunzhen/LocalSearch/data/ecommerce/test/products_test.json"
OUTPUT_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/jsonl_docs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "products.jsonl")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE) as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for p in products:
            parts = []
            if p.get("brand"):
                parts.append(f"Brand: {p['brand']}")
            if p.get("color"):
                parts.append(f"Color: {p['color']}")
            if p.get("category"):
                parts.append(f"Category: {p['category']}")
            if p.get("text"):
                parts.append(p["text"])
            contents = " ".join(parts)

            doc = {"id": p["product_id"], "contents": contents}
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {len(products)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
