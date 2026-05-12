#!/usr/bin/env python3
"""Run GPT-5.5 inference on 82 missing ecommerce samples."""
import asyncio
import json
import time
from openai import AsyncAzureOpenAI

AZURE_ENDPOINT = "https://zhouj-mp1gbpux-eastus2.cognitiveservices.azure.com/"
API_KEY = "0PGxfkUTBoEmHcj3mheyc5Ep6qQuibrATMi7aHMFJqyyMkSdZDMPJQQJ99CEACHYHv6XJ3w3AAAAACOGhsiT"
DEPLOYMENT = "gpt-5.5"
API_VERSION = "2024-12-01-preview"
DATA_FILE = "/tmp/gpt55_missing_82.json"
OUTPUT_JSONL = "/tmp/gpt55_missing_82.jsonl"
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
                    model=DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample["user_query"]},
                    ],
                    max_completion_tokens=4096,
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

    client = AsyncAzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )
    semaphore = asyncio.Semaphore(CONCURRENCY)

    existing = {}
    try:
        with open(OUTPUT_JSONL) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"Found {len(existing)} existing, resuming...")
    except FileNotFoundError:
        pass

    todo = [s for s in samples if s["query_id"] not in existing]
    print(f"GPT-5.5: Running {len(todo)} remaining samples...")

    if not todo:
        print("All done.")
        return

    start_time = time.time()
    completed = ok_count = fail_count = total_in = total_out = 0
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
            if completed % 20 == 0 or completed == total:
                elapsed = time.time() - start_time
                print(f"  [{completed}/{total}] OK={ok_count} Fail={fail_count} Time={elapsed:.0f}s")

    await asyncio.gather(*[process(s) for s in todo])
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s. OK={ok_count}, Fail={fail_count}")


if __name__ == "__main__":
    asyncio.run(main())
