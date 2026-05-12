#!/usr/bin/env python3
"""
闭源 API baseline 推理脚本（GPT / Claude / Gemini / DeepSeek 统一版）

支持三种 provider:
  - openai:    GPT-5.4, o4-mini 等（官方 API）
  - anthropic: Claude Sonnet 4.6, Opus 4.6 等（官方 API）
  - proxy:     Gemini, DeepSeek 等（OpenAI 兼容代理）

用法:
  # GPT-5.4（无 thinking，默认）
  python inference_closedapi.py --provider openai --model gpt-5.4 --api_key sk-xxx -o predictions_gpt54

  # GPT-5.4（开 thinking）
  python inference_closedapi.py --provider openai --model gpt-5.4 --api_key sk-xxx --reasoning_effort medium -o predictions_gpt54_thinking

  # o4-mini
  python inference_closedapi.py --provider openai --model o4-mini --api_key sk-xxx -o predictions_o4mini

  # Claude Sonnet 4.6
  python inference_closedapi.py --provider anthropic --model claude-sonnet-4-6-20250514 --api_key sk-ant-xxx -o predictions_claude_sonnet

  # Gemini（通过代理）
  python inference_closedapi.py --provider proxy --model gemini-3.1-pro-preview --api_key tum_xxx --base_url https://token.claude-code-manager.com/v1 -o predictions_gemini

  # 验证模式（只打印修正后的 prompt，不调 API）
  python inference_closedapi.py --verify_prompt

  # dry run（跑 N 条验证）
  python inference_closedapi.py --provider openai --model gpt-5.4 --api_key sk-xxx --dry_run 5 -o test
"""
import argparse
import asyncio
import json
import os
import sys
import time
import pandas as pd

# ── 配置 ──────────────────────────────────────────────────────
TEST_PARQUET = "/home/sunzhen/LocalSearch/data/ecommerce/test/product_rl_test.parquet"
OUTPUT_DIR = "/home/sunzhen/LocalSearch/data/ecommerce/result"

MAX_RETRIES = 3
RETRY_DELAY = 5


# ── Prompt 修正 ──────────────────────────────────────────────
def fix_system_prompt(system_prompt: str) -> str:
    """
    修正 prompt 中的 schema 字段名和示例，使其与实际数据库一致。

    数据库实际字段: product_id, brand, color, category, price, average_rating, rating_number
    Prompt 中错误的字段:
      - rating         → average_rating
      - num_reviews    → rating_number
      - in_stock       → 数据库中不存在，需移除
      - release_date   → 数据库中不存在，需移除
    """
    # 1. 修正 schema 描述
    system_prompt = system_prompt.replace(
        "- rating         REAL    -- average customer rating (0.0\u20135.0)",
        "- average_rating REAL    -- average customer rating (0.0\u20135.0)"
    ).replace(
        "- num_reviews    INTEGER -- number of customer reviews",
        "- rating_number  INTEGER -- number of customer reviews"
    ).replace(
        "- in_stock       INTEGER -- availability (1=in stock, 0=out of stock)\n",
        ""
    ).replace(
        "- release_date   TEXT    -- product release date (YYYY-MM-DD)\n",
        ""
    )

    # 2. 修正 examples 中的 SQL 字段名
    system_prompt = system_prompt.replace(
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
    )

    # 3. 修正 examples 的自然语言描述
    system_prompt = system_prompt.replace(
        "products with waterproof description and a 5-star rating released after 2022",
        "products with waterproof description and a 5-star rating"
    ).replace(
        "in-stock electronics with good reviews about battery life, priced between $50 and $200",
        "electronics with good reviews about battery life, priced between $50 and $200"
    )

    return system_prompt


# ── DSL 提取 ─────────────────────────────────────────────────
def extract_dsl(text: str):
    """从模型输出提取 <answer>...</answer> 中的 DSL JSON"""
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


# ── 数据加载 ─────────────────────────────────────────────────
def load_samples():
    """加载测试数据并修正 prompt"""
    df = pd.read_parquet(TEST_PARQUET)
    samples = []
    for _, row in df.iterrows():
        extra = json.loads(row["extra_info"])
        prompt = list(row["prompt"])
        system_prompt = ""
        user_query = ""
        for m in prompt:
            if isinstance(m, str):
                m = json.loads(m)
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                user_query = m["content"]

        system_prompt = fix_system_prompt(system_prompt)

        samples.append({
            "query_id": extra["query_id"],
            "system_prompt": system_prompt,
            "user_query": user_query,
        })
    return samples


# ── Prompt 验证 ──────────────────────────────────────────────
def verify_prompt():
    """打印修正后的 prompt 并做完整性检查"""
    samples = load_samples()
    sp = samples[0]["system_prompt"]

    print("=" * 60)
    print("修正后的 System Prompt:")
    print("=" * 60)
    print(sp)
    print()

    # 检查是否还有错误字段名
    errors = []
    for bad_field in ["num_reviews", "in_stock", "release_date"]:
        if bad_field in sp:
            errors.append(f"仍包含错误字段: {bad_field}")

    # 检查 SQL 和 schema 区域是否还有独立的 "rating" 字段名
    # 排除自然语言描述中的 "rating"（如 "5-star rating", "customer rating"）
    import re
    cleaned = sp.replace("average_rating", "XXX").replace("rating_number", "YYY")
    for m in re.finditer(r'\brating\b', cleaned):
        context = cleaned[max(0, m.start()-20):m.end()+20]
        # 跳过自然语言中的 rating（出现在注释或 query 描述中）
        if any(phrase in context for phrase in ["customer rating", "star rating", "5-star"]):
            continue
        errors.append(f"仍有 SQL 字段 'rating' 出现: ...{context}...")

    # 检查正确字段名是否存在
    for good_field in ["average_rating", "rating_number"]:
        if good_field not in sp:
            errors.append(f"缺少正确字段: {good_field}")

    # 检查不应存在的字段
    for removed in ["in_stock", "release_date"]:
        if removed in sp:
            errors.append(f"未删除的字段: {removed}")

    if errors:
        print("!!! PROMPT 检查失败 !!!")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("PROMPT 检查通过:")
        print("  - average_rating / rating_number 已替换")
        print("  - in_stock / release_date 已移除")
        print("  - examples 已修正")
        print(f"  - 总共 {len(samples)} 条测试数据")
        return True


# ── API 调用（Azure OpenAI Chat Completions，支持 thinking） ──
async def call_azure_openai(client, model, sample, semaphore, reasoning_effort=None):
    """调用 Azure OpenAI Chat Completions API（支持 reasoning_effort）"""
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sample["system_prompt"]},
                        {"role": "user", "content": sample["user_query"]},
                    ],
                    "max_completion_tokens": 4096,
                }
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort

                response = await client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                text = text.strip()
                dsl = extract_dsl(text)

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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


# ── API 调用（OpenAI Chat Completions / Proxy） ──────────────
async def call_openai_chat(client, model, sample, semaphore):
    """调用 OpenAI Chat Completions API 或 OpenAI 兼容代理（无 thinking）"""
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
                }
                # o 系列模型不支持 temperature
                if model.startswith("o"):
                    del kwargs["temperature"]

                response = await client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content.strip() if response.choices else ""
                dsl = extract_dsl(text)

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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


# ── API 调用（OpenAI Responses API，带 thinking） ────────────
async def call_openai_responses(client_sync, model, sample, semaphore, reasoning_effort="medium"):
    """调用 OpenAI Responses API（带 reasoning/thinking）
    注意：Responses API 目前没有原生 async 版本，用 asyncio.to_thread 包装同步调用"""
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                def _call():
                    return client_sync.responses.create(
                        model=model,
                        instructions=sample["system_prompt"],
                        input=[
                            {"role": "user", "content": sample["user_query"]},
                        ],
                        reasoning={"effort": reasoning_effort, "summary": "concise"},
                    )

                response = await asyncio.to_thread(_call)

                # 从 response 中提取文本（跳过 reasoning item）
                text = ""
                for item in response.output:
                    if getattr(item, "type", None) == "message" and hasattr(item, "content") and item.content:
                        for part in item.content:
                            if hasattr(part, "text"):
                                text += part.text
                text = text.strip()
                dsl = extract_dsl(text)

                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0

                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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


# ── API 调用（Anthropic / Vertex AI） ────────────────────────
async def call_anthropic(client_sync, model, sample, semaphore):
    """调用 Anthropic Claude API（通过 Vertex AI 同步客户端 + asyncio.to_thread）"""
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                def _call():
                    return client_sync.messages.create(
                        model=model,
                        max_tokens=16000,
                        thinking={"type": "adaptive"},
                        system=sample["system_prompt"],
                        messages=[
                            {"role": "user", "content": sample["user_query"]},
                        ],
                    )

                response = await asyncio.to_thread(_call)
                text = ""
                for block in response.content:
                    if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                        text += block.text
                text = text.strip()
                dsl = extract_dsl(text)

                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0

                return {
                    "query_id": sample["query_id"],
                    "dsl": dsl,
                    "raw_output": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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


# ── 主推理逻辑 ───────────────────────────────────────────────
async def run_inference(args):
    # 初始化 client
    use_responses_api = (args.provider == "openai" and args.reasoning_effort is not None)
    client = None
    client_sync = None

    if args.provider == "vertex":
        # Claude via Google Vertex AI
        if args.gcp_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gcp_credentials
        from anthropic import AnthropicVertex
        project = args.gcp_project or "claude-project-489006"
        client_sync = AnthropicVertex(project_id=project, region=args.gcp_region)
        print(f"Using Vertex AI: project={project}, region={args.gcp_region}")
    elif args.provider == "anthropic":
        from anthropic import Anthropic
        client_sync = Anthropic(api_key=args.api_key)
    elif args.provider == "azure":
        # Azure OpenAI 用 Chat Completions API（支持 reasoning_effort）
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(
            api_key=args.api_key,
            api_version=args.azure_api_version,
            azure_endpoint=args.azure_endpoint,
        )
        # Azure 用 deployment name 作为 model
        args.model = args.azure_deployment
        print(f"Using Azure OpenAI: endpoint={args.azure_endpoint}, deployment={args.azure_deployment}")
        if args.reasoning_effort:
            print(f"Reasoning effort: {args.reasoning_effort}")
    else:
        from openai import AsyncOpenAI, OpenAI
        kwargs = {"api_key": args.api_key}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        client = AsyncOpenAI(**kwargs)
        if use_responses_api:
            client_sync = OpenAI(**kwargs)
            print(f"Using Responses API with reasoning effort: {args.reasoning_effort}")

    # 加载数据
    print("Loading test data...")
    samples = load_samples()
    print(f"Total test samples: {len(samples)}")

    # 断点续跑
    output_file = os.path.join(OUTPUT_DIR, f"{args.output_name}.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                existing[d["query_id"]] = d
        print(f"Found {len(existing)} existing results, skipping...")

    todo = [s for s in samples if s["query_id"] not in existing]
    if args.dry_run > 0:
        todo = todo[:args.dry_run]
    print(f"Remaining: {len(todo)} queries")

    if not todo:
        print("All done, skipping inference.")
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        start_time = time.time()

        completed = 0
        total = len(todo)
        ok_count = 0
        fail_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        async def process_and_save(sample, fout, lock):
            nonlocal completed, ok_count, fail_count, total_input_tokens, total_output_tokens

            if args.provider in ("anthropic", "vertex"):
                result = await call_anthropic(client_sync, args.model, sample, semaphore)
            elif args.provider == "azure":
                result = await call_azure_openai(client, args.model, sample, semaphore, args.reasoning_effort)
            elif use_responses_api:
                result = await call_openai_responses(client_sync, args.model, sample, semaphore, args.reasoning_effort)
            else:
                result = await call_openai_chat(client, args.model, sample, semaphore)

            async with lock:
                fout.write(json.dumps({
                    "query_id": result["query_id"],
                    "dsl": result["dsl"],
                    "raw_output": result["raw_output"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "error": result["error"],
                    "attempts": result["attempts"],
                }, ensure_ascii=False) + "\n")
                fout.flush()

                completed += 1
                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                if result["dsl"] is not None:
                    ok_count += 1
                else:
                    fail_count += 1

                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    qps = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{total}] OK={ok_count} Fail={fail_count} "
                          f"QPS={qps:.1f} Elapsed={elapsed:.0f}s "
                          f"Tokens(in={total_input_tokens}, out={total_output_tokens})")

        lock = asyncio.Lock()
        with open(output_file, "a") as fout:
            tasks = [process_and_save(s, fout, lock) for s in todo]
            await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        print(f"\nInference done in {elapsed:.0f}s")
        print(f"OK={ok_count}, Fail={fail_count}")
        print(f"Total tokens: input={total_input_tokens}, output={total_output_tokens}")

    # 生成 eval 用的 JSON（只保留成功的）
    all_results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                all_results[d["query_id"]] = d

    predictions = []
    failed_ids = []
    for s in samples:
        qid = s["query_id"]
        if qid in all_results and all_results[qid].get("dsl") is not None:
            predictions.append({
                "query_id": qid,
                "dsl": all_results[qid]["dsl"],
            })
        else:
            failed_ids.append(qid)

    json_file = os.path.join(OUTPUT_DIR, f"{args.output_name}.json")
    with open(json_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False)

    print(f"\nFinal: {len(predictions)} valid predictions, {len(failed_ids)} excluded")
    print(f"Saved to {json_file}")

    total_in = sum(r.get("input_tokens", 0) for r in all_results.values())
    total_out = sum(r.get("output_tokens", 0) for r in all_results.values())
    print(f"Total token usage: input={total_in}, output={total_out}")


# ── 入口 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="闭源 API baseline 推理（GPT / Claude / Gemini / DeepSeek 统一版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=["openai", "azure", "anthropic", "vertex", "proxy"], default="openai",
                        help="API provider: openai, azure (Azure OpenAI), anthropic, vertex (Claude via GCP), proxy")
    parser.add_argument("--model", default="gpt-5.4", help="模型名")
    parser.add_argument("--api_key", required=False, help="API key")
    parser.add_argument("--base_url", default=None, help="自定义 API base URL（proxy 模式必填）")
    parser.add_argument("--gcp_credentials", default=None, help="GCP service account JSON 文件路径（vertex 模式）")
    parser.add_argument("--gcp_project", default=None, help="GCP project ID（vertex 模式）")
    parser.add_argument("--gcp_region", default="us-east5", help="GCP region（vertex 模式，默认 us-east5）")
    parser.add_argument("--azure_endpoint", default="https://algorithm.cognitiveservices.azure.com/",
                        help="Azure OpenAI endpoint (azure 模式)")
    parser.add_argument("--azure_deployment", default="gpt-5.4",
                        help="Azure deployment name (azure 模式)")
    parser.add_argument("--azure_api_version", default="2024-12-01-preview",
                        help="Azure API version (azure 模式)")
    parser.add_argument("-o", "--output_name", default="predictions", help="输出文件名（不含扩展名）")
    parser.add_argument("--concurrency", type=int, default=50, help="并发数")
    parser.add_argument("--dry_run", type=int, default=0, help="只跑前 N 条验证（0=全跑）")
    parser.add_argument("--reasoning_effort", default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="GPT-5.4 reasoning effort（默认不设=none）")
    parser.add_argument("--verify_prompt", action="store_true",
                        help="只验证修正后的 prompt，不调 API")

    args = parser.parse_args()

    # 验证模式
    if args.verify_prompt:
        ok = verify_prompt()
        sys.exit(0 if ok else 1)

    # 检查 API key（vertex 模式用 GCP credentials 不需要 api_key）
    if args.provider == "vertex":
        if not args.gcp_credentials:
            print("ERROR: vertex 模式需要 --gcp_credentials 指定 service account JSON")
            sys.exit(1)
    elif not args.api_key:
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "proxy": "PROXY_API_KEY",
        }
        env_key = env_keys.get(args.provider)
        args.api_key = os.environ.get(env_key) if env_key else None
        if not args.api_key:
            print(f"ERROR: 请设置 --api_key 或对应的环境变量")
            sys.exit(1)

    if args.provider == "proxy" and not args.base_url:
        print("ERROR: proxy 模式需要设置 --base_url")
        sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    if args.dry_run:
        print(f"Dry run: {args.dry_run} samples")
    print()

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
