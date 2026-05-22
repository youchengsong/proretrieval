#!/usr/bin/env python3
"""
补全 email_descriptions_test.jsonl 中缺失的 205 条描述。
"""

import asyncio
import json
import os
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm

BASE      = Path("/data/youchengsong/DSL-R1/data/email/test")
CORPUS    = BASE / "email_test.parquet"
DESC_FILE = BASE / "email_descriptions_test.jsonl"

API_BASE    = "https://token.claude-code-manager.com/v1"
MODEL       = "deepseek-v3-2-251201"
CONCURRENCY = 20
MAX_MSG     = 1000

SUBJECT_SYS    = "You are an email search assistant."
SUBJECT_PROMPT = """Email subject: "{subject}"

Write a short search query (3-8 words) that a user might type to find an email about this topic.
Use general topic keywords only, do NOT include specific person names or internal identifiers.
Output ONLY the query, no explanation, no punctuation."""

MESSAGE_SYS    = "You are an email search assistant."
MESSAGE_PROMPT = """Email content:
{message}

Write a short search query (8-15 words) that captures the main topic and key information of this email.
Use general topic keywords only, do NOT include specific person names, company-internal codes, or proper nouns.
Output ONLY the query, no explanation, no punctuation."""


async def call_api(client, sem, mid, system, prompt):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": prompt}],
                    max_tokens=256,
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("empty")
                return mid, content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {mid}: {e}")
                    return mid, None
                await asyncio.sleep(2 ** attempt)


async def main(api_key):
    # 找缺失的 ID
    corpus = pd.read_parquet(CORPUS)
    corpus_ids = set(corpus["Message_ID"].tolist())

    desc_ids = set()
    with open(DESC_FILE) as f:
        for line in f:
            desc_ids.add(json.loads(line)["message_id"])

    missing_ids = corpus_ids - desc_ids
    print(f"缺失描述的文档数: {len(missing_ids)}")
    if not missing_ids:
        print("无缺失，退出。")
        return

    # 获取这些文档的内容
    missing_df = corpus[corpus["Message_ID"].isin(missing_ids)]

    records = [
        {"message_id": row["Message_ID"],
         "subject":    str(row["Subject"]) if pd.notna(row["Subject"]) else "",
         "message":    str(row["Message"])[:MAX_MSG] if pd.notna(row["Message"]) else ""}
        for _, row in missing_df.iterrows()
    ]

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(CONCURRENCY)

    # 并发生成 subject_pos 和 message_pos
    subj_tasks = [
        (r["message_id"], call_api(client, sem, r["message_id"],
                                   SUBJECT_SYS,
                                   SUBJECT_PROMPT.format(subject=r["subject"])))
        for r in records
    ]
    msg_tasks = [
        (r["message_id"], call_api(client, sem, r["message_id"],
                                   MESSAGE_SYS,
                                   MESSAGE_PROMPT.format(message=r["message"])))
        for r in records
    ]

    done_subj, done_msg = {}, {}

    all_coros = (
        [("subj", coro) for _, coro in subj_tasks] +
        [("msg",  coro) for _, coro in msg_tasks]
    )

    async def tagged(kind, coro):
        result = await coro
        return kind, result

    tasks = [tagged(kind, coro) for kind, coro in all_coros]

    with tqdm(total=len(tasks), desc="generating") as pbar:
        for fut in asyncio.as_completed(tasks):
            kind, (mid, result) = await fut
            if result:
                if kind == "subj":
                    done_subj[mid] = result
                else:
                    done_msg[mid] = result
            pbar.update(1)

    # 追加到描述文件
    appended = 0
    with open(DESC_FILE, "a", encoding="utf-8") as f:
        for mid in missing_ids:
            obj = {
                "message_id":  mid,
                "subject_pos": done_subj.get(mid),
                "message_pos": done_msg.get(mid),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            appended += 1

    print(f"追加 {appended} 条 → {DESC_FILE}")

    # 验证
    total = 0
    with open(DESC_FILE) as f:
        for _ in f:
            total += 1
    print(f"描述文件总条数: {total} / {len(corpus_ids)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Set --api_key or env var API_KEY")
    asyncio.run(main(args.api_key))
