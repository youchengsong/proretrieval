#!/usr/bin/env python3
"""Run Claude Opus 4.7 inference on 1000 ecommerce samples."""
import asyncio
import json
import re
import time
import sys
from openai import AsyncOpenAI

BASE_URL = "http://47.94.22.126/v1"
API_KEY = "tum_ywUTn-A_TYQ9Q2xkLCxpicD6c6VVC_Y5gHL0VLMbEyk"
MODEL = "anthropic/claude-opus-4.7"
DATA_FILE = "/tmp/ecom_1000_samples.json"
OUTPUT_JSONL = "/tmp/opus47_ecom_1000.jsonl"
OUTPUT_JSON = "/tmp/opus47_ecom_1000.json"
CONCURRENCY = 10
MAX_RETRIES = 3
RETRY_DELAY = 5


def fix_system_prompt(sp):
    sp = sp.replace(
        "- rating         REAL    -- average customer rating (0.0–5.0)",
        "- average_rating REAL    -- average customer rating (0.0–5.0)"
    ).replace(
        "- num_reviews    INTEGER -- number of customer reviews",
        "- rating_number  INTEGER -- number of customer reviews"
    ).replace(
        "- in_stock       INTEGER -- availability (1=in stock, 0=out of stock)\n",
        ""
    ).replace(
        "- release_date   TEXT    -- product release date (YYYY-MM-DD)\n",
        ""
    ).replace(
        "rating >= 4.5 AND num_reviews > 500",
        "average_rating >= 4.5 AND rating_number > 500"
    ).replace(
        "rating = 5.0 AND release_date > '2022-12-31'",
        "average_rating = 5.0"
    ).replace(
        "in_stock = 1 AND price >= 50 AND price <= 200",
        "price >= 50 AND price <= 200"
    ).replace(
        "rating = 4.5",
        "average_rating = 4.5"
    ).replace(
        "products with waterproof description and a 5-star rating released after 2022",
        "products with waterproof description and a 5-star rating"
    ).replace(
        "in-stock electronics with good reviews about battery life, priced between $50 and $200",
        "electronics with good reviews about battery life, priced between $50 and $200"
    )
    return sp


def extract_dsl(text):
    if not text:
        return None
    if "<answer>" in text:
        try:
            json_str = text.split("<answer>")[1].split("</answer>")[0].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def call_one(client, system_prompt, sample, semaphore):
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample["user_query"]},
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
                text = response.choices[0].message.content or ""
                text = text.strip()
                dsl = extract_dsl(text)
                usage = response.usage
                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "error": None if dsl else "parse_failed",
                    "attempts": attempt + 1,
                }
            except Exception as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return {
                    "query_id": sample["query_id"],
                    "dsl": None,
                    "raw_output": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": f"api_error: {str(e)[:200]}",
                    "attempts": attempt + 1,
                }


async def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    system_prompt = fix_system_prompt(data["system_prompt"])
    samples = data["samples"]
    print(f"Running {len(samples)} samples with {MODEL}, concurrency={CONCURRENCY}")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Resume from existing results
    existing = {}
    try:
        with open(OUTPUT_JSONL) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"Found {len(existing)} existing results, resuming...")
    except FileNotFoundError:
        pass

    todo = [s for s in samples if s["query_id"] not in existing]
    print(f"Remaining: {len(todo)} queries")

    if not todo:
        print("All done.")
    else:
        start_time = time.time()
        completed = 0
        ok_count = 0
        fail_count = 0
        total_in = 0
        total_out = 0
        total = len(todo)
        lock = asyncio.Lock()

        async def process(sample):
            nonlocal completed, ok_count, fail_count, total_in, total_out
            result = await call_one(client, system_prompt, sample, semaphore)
            async with lock:
                with open(OUTPUT_JSONL, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                total_in += result["input_tokens"]
                total_out += result["output_tokens"]
                if result["dsl"] is not None:
                    ok_count += 1
                else:
                    fail_count += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    qps = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{total}] OK={ok_count} Fail={fail_count} "
                          f"QPS={qps:.1f} Time={elapsed:.0f}s "
                          f"Tokens(in={total_in}, out={total_out})")

        tasks = [process(s) for s in todo]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        print(f"\nDone in {elapsed:.0f}s. OK={ok_count}, Fail={fail_count}")
        print(f"Total tokens: input={total_in}, output={total_out}")

    # Generate eval-format JSON
    all_results = {}
    with open(OUTPUT_JSONL) as f:
        for line in f:
            d = json.loads(line)
            all_results[d["query_id"]] = d

    predictions = []
    failed = 0
    for s in samples:
        qid = s["query_id"]
        if qid in all_results and all_results[qid].get("dsl") is not None:
            predictions.append({"query_id": qid, "dsl": all_results[qid]["dsl"]})
        else:
            failed += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)

    print(f"\nFinal: {len(predictions)} valid, {failed} failed → {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
