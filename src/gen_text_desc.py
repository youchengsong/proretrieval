#!/usr/bin/env python3
"""
Generate text_pos and text_neg for all 13,334 products.

Phase 1: API (deepseek-v3-2-251201) → text_pos for each product
Phase 2: API (deepseek-v3-2-251201) → text_neg (given text_pos, generate a different product description)

Output: /data/youchengsong/DSL-R1/data/text_descriptions.jsonl

Usage:
    python gen_text_descriptions.py --api_key tum_xxx
    python gen_text_descriptions.py --api_key tum_xxx --test 20   # test on 20 products
    python gen_text_descriptions.py --api_key tum_xxx --phase 1   # text_pos only
    python gen_text_descriptions.py --api_key tum_xxx --phase 2   # text_neg only
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("/data/youchengsong/DSL-R1/data")
INPUT_FILE = DATA_DIR / "products_final.parquet"

TEXT_POS_FILE = DATA_DIR / "text_pos.jsonl"
OUTPUT_FILE   = DATA_DIR / "text_descriptions.jsonl"

API_BASE    = "https://token.claude-code-manager.com/v1"
MODEL       = "deepseek-v3-2-251201"
CONCURRENCY = 20

TEXT_POS_SYSTEM = "You are a product search assistant. Extract concise semantic search queries from product descriptions."
TEXT_POS_PROMPT = """Product description:
{text}

Extract 3-5 key semantic phrases (as a short search query, space-separated) that describe this product's core features, use case, and materials.
Output ONLY the query, no explanation, no punctuation."""

TEXT_NEG_SYSTEM = "You are a product search assistant."
TEXT_NEG_PROMPT = """Product category: {category}
Product features: {text_pos}

Describe a DIFFERENT product in the same category (5-8 words) with clearly different features, materials, or use case.
Output ONLY the description, no explanation."""


# ── Load data ─────────────────────────────────────────────────────────────────
def load_products():
    df = pd.read_parquet(INPUT_FILE, columns=["product_id", "text", "category"])
    print(f"Loaded {len(df)} products, {df['category'].nunique()} categories")
    return df


# ── Phase 1: Generate text_pos ────────────────────────────────────────────────
async def gen_text_pos_one(client, sem, product_id, text):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": TEXT_POS_SYSTEM},
                        {"role": "user",   "content": TEXT_POS_PROMPT.format(text=text[:2000])},
                    ],
                    max_tokens=60,
                    temperature=0.3,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                return product_id, msg.content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {product_id}: {e}")
                    return product_id, None
                await asyncio.sleep(2 ** attempt)


async def phase1_gen_text_pos(api_key, records, concurrency):
    done = {}
    if TEXT_POS_FILE.exists():
        with open(TEXT_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["product_id"]] = obj["text_pos"]
    print(f"Phase 1: {len(done)} already done")

    remaining = [r for r in records if r["product_id"] not in done]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("Phase 1: all done, skipping")
        return done

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)
    tasks  = [gen_text_pos_one(client, sem, r["product_id"], r["text"]) for r in remaining]

    fout = open(TEXT_POS_FILE, "a", encoding="utf-8")
    with tqdm(total=len(tasks), desc="text_pos") as pbar:
        for coro in asyncio.as_completed(tasks):
            pid, result = await coro
            if result:
                done[pid] = result
                fout.write(json.dumps({"product_id": pid, "text_pos": result}, ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(1)
    fout.close()

    print(f"Phase 1 done: {len(done)} text_pos generated")
    return done


# ── Phase 2: Generate text_neg via API ───────────────────────────────────────
async def gen_text_neg_one(client, sem, product_id, category, text_pos):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": TEXT_NEG_SYSTEM},
                        {"role": "user",   "content": TEXT_NEG_PROMPT.format(
                            category=category, text_pos=text_pos)},
                    ],
                    max_tokens=60,
                    temperature=0.7,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                return product_id, msg.content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {product_id}: {e}")
                    return product_id, None
                await asyncio.sleep(2 ** attempt)


async def phase2_gen_text_neg(api_key, text_pos_dict, id_to_cat, concurrency):
    done_neg = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("text_neg"):
                    done_neg[obj["product_id"]] = obj["text_neg"]
    print(f"Phase 2: {len(done_neg)} text_neg already done")

    remaining = [(pid, id_to_cat.get(pid, ""), tp)
                 for pid, tp in text_pos_dict.items() if pid not in done_neg]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("Phase 2: all done, skipping")
        return done_neg

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)
    tasks  = [gen_text_neg_one(client, sem, pid, cat, tp) for pid, cat, tp in remaining]

    fout = open(OUTPUT_FILE, "a", encoding="utf-8")
    with tqdm(total=len(tasks), desc="text_neg") as pbar:
        for coro in asyncio.as_completed(tasks):
            pid, result = await coro
            if result:
                done_neg[pid] = result
                fout.write(json.dumps({
                    "product_id": pid,
                    "text_pos":   text_pos_dict[pid],
                    "text_neg":   result,
                }, ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(1)
    fout.close()

    print(f"Phase 2 done: {len(done_neg)} text_neg generated")
    return done_neg


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",     default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--phase",       type=int, default=0,
                        help="0=all, 1=text_pos only, 2=text_neg only")
    parser.add_argument("--test",        type=int, default=0,
                        help="Only process N products (for testing)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set --api_key or env var API_KEY")

    df = load_products()
    if args.test > 0:
        df = df.iloc[:args.test]
        print(f"Test mode: processing {len(df)} products")

    records   = df[["product_id", "text"]].to_dict("records")
    id_to_cat = dict(zip(df["product_id"], df["category"]))

    if args.phase in (0, 1):
        text_pos_dict = await phase1_gen_text_pos(args.api_key, records, args.concurrency)
    else:
        text_pos_dict = {}
        pids = set(df["product_id"])
        if TEXT_POS_FILE.exists():
            with open(TEXT_POS_FILE) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["product_id"] in pids:
                        text_pos_dict[obj["product_id"]] = obj["text_pos"]
        print(f"Loaded {len(text_pos_dict)} text_pos from file")

    if args.phase in (0, 2):
        await phase2_gen_text_neg(args.api_key, text_pos_dict, id_to_cat, args.concurrency)

    print(f"\nAll done. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
