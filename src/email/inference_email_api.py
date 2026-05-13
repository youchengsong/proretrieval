#!/usr/bin/env python3
"""Email scenario commercial API inference (GPT-5.5 / GPT-5.4 / Opus 4.7)."""
import argparse, asyncio, json, os, time
import pandas as pd

TEST_PARQUET = "/home/sunzhen/LocalSearch/data/email/test/email_rl_test_fixed.parquet"
OUTPUT_DIR = "/home/sunzhen/LocalSearch/data/email/result"
MAX_RETRIES = 3
RETRY_DELAY = 5

API_CONFIGS = {
    "gpt55": {
        "endpoint": "https://zhouj-mp1gbpux-eastus2.cognitiveservices.azure.com/",
        "key": "0PGxfkUTBoEmHcj3mheyc5Ep6qQuibrATMi7aHMFJqyyMkSdZDMPJQQJ99CEACHYHv6XJ3w3AAAAACOGhsiT",
        "model": "gpt-5.5",
        "api_version": "2024-12-01-preview",
        "provider": "azure",
    },
    "gpt54": {
        "endpoint": "https://algorithm.cognitiveservices.azure.com/",
        "key": "AM6qC7ShNQFK4lCcwzakQ8hT9MkBiXQWtaGsbHa9YNiJJpXcdIKHJQQJ99CDACYeBjFXJ3w3AAAAACOGOFOX",
        "model": "gpt-5.4",
        "api_version": "2024-12-01-preview",
        "provider": "azure",
    },
    "opus47": {
        "base_url": "http://47.94.22.126/v1",
        "key": "tum_ywUTn-A_TYQ9Q2xkLCxpicD6c6VVC_Y5gHL0VLMbEyk",
        "model": "anthropic/claude-opus-4.7",
        "provider": "openai",
    },
}


def load_samples(limit=None):
    df = pd.read_parquet(TEST_PARQUET)
    samples = []
    for _, row in df.iterrows():
        extra = json.loads(row["extra_info"])
        msgs = list(row["prompt"])
        msgs = [json.loads(m) if isinstance(m, str) else m for m in msgs]
        system_prompt = msgs[0]["content"] if msgs else ""
        user_query = msgs[1]["content"] if len(msgs) > 1 else ""
        samples.append({
            "query_id": extra["query_id"],
            "system_prompt": system_prompt,
            "user_query": user_query,
        })
    if limit:
        samples = samples[:limit]
    return samples


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


async def call_api(client, model, sample, semaphore, is_azure=False):
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sample["system_prompt"]},
                        {"role": "user", "content": sample["user_query"]},
                    ],
                    "temperature": 0.3,
                    "max_completion_tokens": 4096,
                }
                response = await client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content.strip() if response.choices else ""
                dsl = extract_dsl(text)
                usage = response.usage
                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "error": None if dsl else "parse_failed",
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
                }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, choices=list(API_CONFIGS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    cfg = API_CONFIGS[args.api]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"predictions_email_{args.api}.json")
    log_file = os.path.join(OUTPUT_DIR, f"predictions_email_{args.api}.jsonl")

    samples = load_samples(limit=args.limit)
    print(f"Loaded {len(samples)} samples for {args.api}")

    existing = {}
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [s for s in samples if s["query_id"] not in existing]
    print(f"  Todo: {len(todo)} samples")

    if not todo:
        print("All done!")
    else:
        if cfg["provider"] == "azure":
            from openai import AsyncAzureOpenAI
            client = AsyncAzureOpenAI(
                azure_endpoint=cfg["endpoint"],
                api_key=cfg["key"],
                api_version=cfg["api_version"],
            )
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=cfg.get("base_url"),
                api_key=cfg["key"],
            )

        semaphore = asyncio.Semaphore(args.concurrency)
        is_azure = cfg["provider"] == "azure"

        batch_size = 20
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            tasks = [call_api(client, cfg["model"], s, semaphore, is_azure) for s in batch]
            results = await asyncio.gather(*tasks)

            with open(log_file, "a") as f:
                for r in results:
                    existing[r["query_id"]] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            done = len(existing)
            total = len(samples)
            errors = sum(1 for r in results if r.get("error"))
            print(f"  [{done}/{total}] batch done, {errors} errors in batch")

    predictions = {}
    for qid, r in existing.items():
        if r.get("dsl"):
            predictions[qid] = r["dsl"]
        elif r.get("raw_output"):
            predictions[qid] = r["raw_output"]

    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(predictions)} predictions to {output_file}")

    ok = sum(1 for r in existing.values() if r.get("dsl"))
    fail = sum(1 for r in existing.values() if r.get("error"))
    total_in = sum(r.get("input_tokens", 0) for r in existing.values())
    total_out = sum(r.get("output_tokens", 0) for r in existing.values())
    print(f"Stats: {ok} ok, {fail} failed, {total_in} input tokens, {total_out} output tokens")


if __name__ == "__main__":
    asyncio.run(main())
