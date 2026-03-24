#!/usr/bin/env python3
"""
Generate image_pos and image_neg for all products with downloaded images.

Phase 1: Gemini vision API → image_pos (describe image as a search query)
Phase 2: Gemini text API  → image_neg (given image_pos, generate a different product description)

Images are read from: /data/youchengsong/DSL-R1/image/{product_id}.jpg
Output:               /data/youchengsong/DSL-R1/data/image_descriptions.jsonl

Usage:
    python gen_image_descriptions.py --api_key tum_xxx
    python gen_image_descriptions.py --api_key tum_xxx --test 20   # test on 20 products
    python gen_image_descriptions.py --api_key tum_xxx --phase 1   # image_pos only
    python gen_image_descriptions.py --api_key tum_xxx --phase 2   # image_neg only
"""

import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/data/youchengsong/DSL-R1/data")
IMAGE_DIR = Path("/data/youchengsong/DSL-R1/image")

INPUT_FILE   = DATA_DIR / "products_final.parquet"
IMG_POS_FILE = DATA_DIR / "image_pos.jsonl"
OUTPUT_FILE  = DATA_DIR / "image_descriptions.jsonl"

API_BASE = "https://token.claude-code-manager.com/v1"
MODEL    = "gemini-2.5-flash"

CONCURRENCY    = 80
IMAGE_MAX_SIZE = 768  # 1 Gemini tile = 768×768
MIN_WORDS      = 5    # retry if output has fewer words

IMAGE_POS_PROMPT = (
    "You are an e-commerce search expert. Look at this product image carefully and write a search query that would find this exact product.\n\n"
    "Requirements:\n"
    "- Must be 6 to 12 words\n"
    "- Include: product type + color/material + key visual feature\n"
    "- Examples: 'red stainless steel insulated water bottle with lid', 'black wireless over-ear noise canceling headphones', 'white ceramic coffee mug with handle'\n"
    "- Output ONLY the query, no punctuation, no explanation."
)

IMAGE_NEG_PROMPT = (
    "Category: {category}\n"
    "This product looks like: \"{image_pos}\"\n\n"
    "Write a search query (6-10 words) for a DIFFERENT product in the same category "
    "with clearly different appearance, color, material, or style.\n"
    "- Must NOT describe the same product\n"
    "- Examples of contrast: different color, different material, different form factor\n"
    "- Output ONLY the query, no punctuation, no explanation."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_products():
    df = pd.read_parquet(INPUT_FILE, columns=["product_id", "category"])
    return dict(zip(df["product_id"], df["category"]))


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > IMAGE_MAX_SIZE:
        scale = IMAGE_MAX_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Phase 1: Generate image_pos ───────────────────────────────────────────────
async def gen_image_pos_one(client, sem, product_id: str, image_path: Path):
    async with sem:
        b64 = encode_image(image_path)
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": IMAGE_POS_PROMPT},
                        ],
                    }],
                    max_tokens=512,
                    temperature=0.3,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                result = msg.content.strip()
                if len(result.split()) < MIN_WORDS:
                    raise ValueError(f"output too short: '{result}'")
                return product_id, result
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {product_id}: {e}")
                    return product_id, None
                await asyncio.sleep(2 ** attempt)


async def phase1_gen_image_pos(api_key, todo_pids, concurrency):
    done = {}
    if IMG_POS_FILE.exists():
        with open(IMG_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["product_id"]] = obj["image_pos"]
    print(f"Phase 1: {len(done)} already done")

    remaining = [pid for pid in todo_pids if pid not in done]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("Phase 1: all done, skipping")
        return done

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)
    tasks  = [gen_image_pos_one(client, sem, pid, IMAGE_DIR / f"{pid}.jpg") for pid in remaining]

    fout = open(IMG_POS_FILE, "a", encoding="utf-8")
    with tqdm(total=len(tasks), desc="image_pos") as pbar:
        for coro in asyncio.as_completed(tasks):
            pid, result = await coro
            if result:
                done[pid] = result
                fout.write(json.dumps({"product_id": pid, "image_pos": result}, ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(1)
    fout.close()

    print(f"Phase 1 done: {len(done)} image_pos generated")
    return done


# ── Phase 2: Generate image_neg via API ───────────────────────────────────────
async def gen_image_neg_one(client, sem, product_id, category, image_pos):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content":
                        IMAGE_NEG_PROMPT.format(category=category, image_pos=image_pos)}],
                    max_tokens=80,
                    temperature=0.7,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                result = msg.content.strip()
                if len(result.split()) < MIN_WORDS:
                    raise ValueError(f"output too short: '{result}'")
                return product_id, result
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {product_id}: {e}")
                    return product_id, None
                await asyncio.sleep(2 ** attempt)


async def phase2_gen_image_neg(api_key, image_pos_dict, id_to_cat, concurrency):
    done_neg = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("image_neg"):
                    done_neg[obj["product_id"]] = obj["image_neg"]
    print(f"Phase 2: {len(done_neg)} image_neg already done")

    remaining = [(pid, id_to_cat.get(pid, ""), ip)
                 for pid, ip in image_pos_dict.items() if pid not in done_neg]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("Phase 2: all done, skipping")
        return done_neg

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)
    tasks  = [gen_image_neg_one(client, sem, pid, cat, ip) for pid, cat, ip in remaining]

    fout = open(OUTPUT_FILE, "a", encoding="utf-8")
    with tqdm(total=len(tasks), desc="image_neg") as pbar:
        for coro in asyncio.as_completed(tasks):
            pid, result = await coro
            if result:
                done_neg[pid] = result
                fout.write(json.dumps({
                    "product_id": pid,
                    "image_pos":  image_pos_dict[pid],
                    "image_neg":  result,
                }, ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(1)
    fout.close()

    print(f"Phase 2 done: {len(done_neg)} image_neg generated")
    return done_neg


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",     default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--phase",       type=int, default=0,
                        help="0=all, 1=image_pos only, 2=image_neg only")
    parser.add_argument("--test",        type=int, default=0,
                        help="Only process N products (for testing)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set --api_key or env var API_KEY")

    id_to_cat = load_products()

    # Determine which products to process
    available = sorted({p.stem for p in IMAGE_DIR.glob("*.jpg")} & set(id_to_cat.keys()))
    if args.test > 0:
        available = available[:args.test]
        print(f"Test mode: processing {len(available)} products")

    if args.phase in (0, 1):
        image_pos_dict = await phase1_gen_image_pos(args.api_key, available, args.concurrency)
    else:
        image_pos_dict = {}
        if IMG_POS_FILE.exists():
            with open(IMG_POS_FILE) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["product_id"] in set(available):
                        image_pos_dict[obj["product_id"]] = obj["image_pos"]
        print(f"Loaded {len(image_pos_dict)} image_pos from file")

    if args.phase in (0, 2):
        await phase2_gen_image_neg(args.api_key, image_pos_dict, id_to_cat, args.concurrency)

    print(f"\nAll done. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
