#!/usr/bin/env python3
"""B5 Few-shot DSL generation with commercial APIs (GPT-5.5, Opus 4.7, Gemini).

Uses the same system prompt + few-shot examples as the Qwen3-4B few-shot ICL experiment,
but calls commercial model APIs instead.
"""
import argparse, json, os, time, sys
import pandas as pd

DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"
TEST_PARQUET = os.path.join(DATA_DIR, "product_rl_test.parquet")

API_CONFIGS = {
    "gpt55": {
        "type": "azure",
        "endpoint": "https://zhouj-mp1gbpux-eastus2.cognitiveservices.azure.com/",
        "key": "0PGxfkUTBoEmHcj3mheyc5Ep6qQuibrATMi7aHMFJqyyMkSdZDMPJQQJ99CEACHYHv6XJ3w3AAAAACOGhsiT",
        "deploy": "gpt-5.5",
        "api_version": "2024-12-01-preview",
    },
    "opus47": {
        "type": "openai_compat",
        "base_url": "http://47.94.22.126/v1",
        "key": "tum_ywUTn-A_TYQ9Q2xkLCxpicD6c6VVC_Y5gHL0VLMbEyk",
        "model": "anthropic/claude-opus-4.7",
    },
    "gemini": {
        "type": "openai_compat",
        "base_url": "https://token.claude-code-manager.com/v1",
        "key": "tum_VWYxGAiOLi0t0FJCjvDpAjibhNMui7BVJRWLucx83Q0",
        "model": "google/gemini-2.5-pro",
    },
}


def make_client(config):
    if config["type"] == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=config["endpoint"],
            api_key=config["key"],
            api_version=config["api_version"],
        ), config["deploy"]
    else:
        from openai import OpenAI
        return OpenAI(
            base_url=config["base_url"],
            api_key=config["key"],
        ), config["model"]


def parse_dsl(text):
    if "<answer>" in text:
        try:
            json_str = text.split("<answer>")[1]
            if "</answer>" in json_str:
                json_str = json_str.split("</answer>")[0]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, choices=list(API_CONFIGS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = API_CONFIGS[args.api]
    client, model_name = make_client(config)
    print(f"Using API: {args.api}, model: {model_name}")

    df = pd.read_parquet(TEST_PARQUET)
    samples = []
    for _, row in df.iterrows():
        extra = json.loads(row["extra_info"])
        msgs = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"].tolist()
        samples.append({"query_id": extra["query_id"], "messages": msgs})

    if args.limit:
        samples = samples[:args.limit]
    print(f"Total samples: {len(samples)}")

    output_jsonl = os.path.join(RESULT_DIR, f"predictions_fewshot_{args.api}.jsonl")
    existing = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                d = json.loads(line)
                existing.add(d["query_id"])
        print(f"  Found {len(existing)} existing, resuming...")

    todo = [s for s in samples if s["query_id"] not in existing]
    print(f"  Todo: {len(todo)}")

    ok = fail = 0
    for i, s in enumerate(todo):
        qid = s["query_id"]
        messages = s["messages"]

        dsl = None
        for retry in range(3):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                }
                if config["type"] == "azure":
                    kwargs["max_completion_tokens"] = 512
                else:
                    kwargs["max_tokens"] = 512
                    kwargs["temperature"] = 0

                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content.strip()
                dsl = parse_dsl(text)
                break
            except Exception as e:
                print(f"  Retry {retry+1} for {qid}: {e}")
                time.sleep(2 ** retry)

        if dsl:
            ok += 1
        else:
            fail += 1

        with open(output_jsonl, "a") as f:
            f.write(json.dumps({"query_id": qid, "dsl": dsl}, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail}")

    print(f"\nDone. ok={ok}, fail={fail}")

    # Convert to eval format
    all_results = {}
    with open(output_jsonl) as f:
        for line in f:
            d = json.loads(line)
            all_results[d["query_id"]] = d

    predictions = []
    for s in samples:
        qid = s["query_id"]
        if qid in all_results:
            predictions.append({"query_id": qid, "dsl": all_results[qid].get("dsl")})

    output_json = os.path.join(RESULT_DIR, f"predictions_fewshot_{args.api}.json")
    with open(output_json, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved {len(predictions)} predictions to {output_json}")


if __name__ == "__main__":
    main()
