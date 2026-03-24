#!/usr/bin/env python3
"""
Download product images from image_url in products_final.parquet.

Images are saved as: /data/youchengsong/DSL-R1/image/{product_id}.jpg
Skips already-downloaded images (safe to rerun).

Usage:
    python download_images.py
    python download_images.py --workers 32 --timeout 10
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR  = Path("/data/youchengsong/DSL-R1/data")
IMAGE_DIR = Path("/data/youchengsong/DSL-R1/image")
INPUT_FILE = DATA_DIR / "products_final.parquet"

FAILED_LOG = DATA_DIR / "image_download_failed.txt"

WORKERS = 16
TIMEOUT = 15


def download_one(product_id: str, url: str, out_dir: Path, timeout: int):
    dest = out_dir / f"{product_id}.jpg"
    if dest.exists():
        return product_id, "skip"
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return product_id, "ok"
    except Exception as e:
        return product_id, f"fail: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE, columns=["product_id", "image_url"])
    df = df.dropna(subset=["image_url"])
    print(f"Total products with image_url: {len(df)}")

    already = sum(1 for _ in IMAGE_DIR.glob("*.jpg"))
    print(f"Already downloaded: {already}, remaining: {len(df) - already}")

    records = df[["product_id", "image_url"]].to_dict("records")

    failed = []
    ok = skip = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, r["product_id"], r["image_url"], IMAGE_DIR, args.timeout): r["product_id"]
            for r in records
        }
        with tqdm(total=len(futures), desc="downloading") as pbar:
            for fut in as_completed(futures):
                pid, status = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    failed.append((pid, status))
                pbar.update(1)

    print(f"\nDone: {ok} downloaded, {skip} skipped, {len(failed)} failed")

    if failed:
        with open(FAILED_LOG, "w") as f:
            for pid, reason in failed:
                f.write(f"{pid}\t{reason}\n")
        print(f"Failed list saved to {FAILED_LOG}")


if __name__ == "__main__":
    main()
