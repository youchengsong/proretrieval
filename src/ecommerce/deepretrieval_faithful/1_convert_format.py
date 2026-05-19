#!/usr/bin/env python3
"""Convert ecommerce product data to Pyserini-compatible JSONL format.

Mirrors the pattern from DeepRetrieval's src/Lucene/msmarco_beir/1_convert_format.py.
Converts product JSON to JSONL with 'id' and 'contents' fields for Lucene indexing.
"""
import json
import os
import argparse


def convert_products(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "docs.jsonl")

    with open(input_file) as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products from {input_file}")

    with open(output_file, "w", encoding="utf-8") as out:
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

    print(f"Wrote {len(products)} documents to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/sunzhen/LocalSearch/data/ecommerce/test/products_test.json")
    parser.add_argument("--output_dir", default="code/data/local_index_search/ecommerce/jsonl_docs")
    args = parser.parse_args()

    # Also check for train products and merge if available
    train_input = args.input.replace("test/products_test", "train/products_train")
    if os.path.exists(train_input) and train_input != args.input:
        print(f"Found train products: {train_input}")
        with open(args.input) as f:
            test_products = json.load(f)
        with open(train_input) as f:
            train_products = json.load(f)

        seen_ids = set()
        merged = []
        for p in train_products + test_products:
            if p["product_id"] not in seen_ids:
                seen_ids.add(p["product_id"])
                merged.append(p)

        merged_file = os.path.join(args.output_dir, "merged_products.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(merged_file, "w") as f:
            json.dump(merged, f)
        print(f"Merged {len(merged)} unique products (train: {len(train_products)}, test: {len(test_products)})")
        convert_products(merged_file, args.output_dir)
    else:
        convert_products(args.input, args.output_dir)
