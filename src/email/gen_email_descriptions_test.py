#!/usr/bin/env python3
"""
Generate subject_pos and message_pos for test set emails (no neg).

Input:  /data/youchengsong/DSL-R1/data/email/email_test.parquet
Output: /data/youchengsong/DSL-R1/data/email/email_descriptions_test.jsonl

Intermediate:
  /data/youchengsong/DSL-R1/data/email/subject_pos_test.jsonl
  /data/youchengsong/DSL-R1/data/email/message_pos_test.jsonl

Usage:
    python gen_email_descriptions_test.py --api_key tum_xxx
    python gen_email_descriptions_test.py --api_key tum_xxx --test 20
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
DATA_DIR         = Path("/data/youchengsong/DSL-R1/data/email")
INPUT_FILE       = DATA_DIR / "email_test.parquet"
SUBJECT_POS_FILE = DATA_DIR / "subject_pos_test.jsonl"
MESSAGE_POS_FILE = DATA_DIR / "message_pos_test.jsonl"
OUTPUT_FILE      = DATA_DIR / "email_descriptions_test.jsonl"

API_BASE          = "https://token.claude-code-manager.com/v1"
MODEL             = "deepseek-v3-2-251201"
CONCURRENCY       = 20
MESSAGE_MAX_CHARS = 1000

# ── Prompts ───────────────────────────────────────────────────────────────────
SUBJECT_POS_SYSTEM = "You are an email search assistant."
SUBJECT_POS_PROMPT = """Email subject: "{subject}"

Write a short search query (3-8 words) that a user might type to find an email about this topic.
Use general topic keywords only, do NOT include specific person names or internal identifiers.
Output ONLY the query, no explanation, no punctuation."""

MESSAGE_POS_SYSTEM = "You are an email search assistant."
MESSAGE_POS_PROMPT = """Email content:
{message}

Write a short search query (8-15 words) that captures the main topic and key information of this email.
Use general topic keywords only, do NOT include specific person names, company-internal codes, or proper nouns.
Output ONLY the query, no explanation, no punctuation."""


# ── Load data ─────────────────────────────────────────────────────────────────
def load_emails():
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    print(f"Loaded {len(df)} test emails")
    return df


# ── API call helper ───────────────────────────────────────────────────────────
async def call_api(client, sem, msg_id, system, prompt, label):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.3,
                )
                content = resp.choices[0].message.content if resp.choices else None
                if not content or not content.strip():
                    raise ValueError("empty response")
                return msg_id, content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED [{label}] {msg_id}: {e}")
                    return msg_id, None
                await asyncio.sleep(2 ** attempt)


async def tagged(coro, kind):
    result = await coro
    return kind, result


# ── Generate subject_pos + message_pos ────────────────────────────────────────
async def gen_pos(api_key, records, concurrency):
    done_subj = {}
    if SUBJECT_POS_FILE.exists():
        with open(SUBJECT_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done_subj[obj["message_id"]] = obj["subject_pos"]

    done_msg = {}
    if MESSAGE_POS_FILE.exists():
        with open(MESSAGE_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done_msg[obj["message_id"]] = obj["message_pos"]

    remaining_subj = [r for r in records if r["message_id"] not in done_subj]
    remaining_msg  = [r for r in records if r["message_id"] not in done_msg]
    print(f"subject_pos remaining={len(remaining_subj)}, message_pos remaining={len(remaining_msg)}")

    if not remaining_subj and not remaining_msg:
        print("All done, skipping API calls")
        return done_subj, done_msg

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    tasks = (
        [tagged(call_api(client, sem, r["message_id"],
                         SUBJECT_POS_SYSTEM,
                         SUBJECT_POS_PROMPT.format(subject=r["subject"]),
                         "subject_pos"), "subj")
         for r in remaining_subj]
        +
        [tagged(call_api(client, sem, r["message_id"],
                         MESSAGE_POS_SYSTEM,
                         MESSAGE_POS_PROMPT.format(message=r["message"][:MESSAGE_MAX_CHARS]),
                         "message_pos"), "msg")
         for r in remaining_msg]
    )

    fout_subj = open(SUBJECT_POS_FILE, "a", encoding="utf-8")
    fout_msg  = open(MESSAGE_POS_FILE, "a", encoding="utf-8")

    with tqdm(total=len(tasks), desc="generating pos") as pbar:
        for fut in asyncio.as_completed(tasks):
            kind, (mid, result) = await fut
            if result:
                if kind == "subj":
                    done_subj[mid] = result
                    fout_subj.write(json.dumps({"message_id": mid, "subject_pos": result}, ensure_ascii=False) + "\n")
                    fout_subj.flush()
                else:
                    done_msg[mid] = result
                    fout_msg.write(json.dumps({"message_id": mid, "message_pos": result}, ensure_ascii=False) + "\n")
                    fout_msg.flush()
            pbar.update(1)

    fout_subj.close()
    fout_msg.close()
    print(f"Done: subject_pos={len(done_subj)}, message_pos={len(done_msg)}")
    return done_subj, done_msg


# ── Merge into output ─────────────────────────────────────────────────────────
def merge_output(done_subj, done_msg):
    all_mids = set(done_subj) | set(done_msg)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for mid in all_mids:
            obj = {
                "message_id":  mid,
                "subject_pos": done_subj.get(mid),
                "message_pos": done_msg.get(mid),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_mids)} records to {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",     default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--test",        type=int, default=0,
                        help="Only process N emails (for testing)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set --api_key or env var API_KEY")

    df = load_emails()
    if args.test > 0:
        df = df.iloc[:args.test]
        print(f"Test mode: processing {len(df)} emails")

    records = [
        {
            "message_id": row["Message_ID"],
            "subject":    row["Subject"],
            "message":    row["Message"],
        }
        for _, row in df.iterrows()
    ]

    done_subj, done_msg = await gen_pos(args.api_key, records, args.concurrency)
    merge_output(done_subj, done_msg)
    print(f"\nAll done. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
