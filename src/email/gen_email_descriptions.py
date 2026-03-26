#!/usr/bin/env python3
"""
Generate subject_pos, subject_neg, message_pos, message_neg for 10,000 emails.

Phase 1: API (deepseek-v3-2-251201) → subject_pos + message_pos (concurrent)
Phase 2: API (deepseek-v3-2-251201) → subject_neg + message_neg (from pos)

Input:  /data/youchengsong/DSL-R1/data/enron_final.parquet
Output: /data/youchengsong/DSL-R1/data/email_descriptions.jsonl

Intermediate:
  /data/youchengsong/DSL-R1/data/subject_pos.jsonl
  /data/youchengsong/DSL-R1/data/message_pos.jsonl

Notes:
  - Message body truncated to 1000 chars before sending to API
  - Concurrency: 20 parallel API calls
  - Checkpoint/resume supported via intermediate files
  - Retry up to 3 times on API failure

Usage:
    python gen_email_descriptions.py --api_key tum_xxx
    python gen_email_descriptions.py --api_key tum_xxx --test 20
    python gen_email_descriptions.py --api_key tum_xxx --phase 1
    python gen_email_descriptions.py --api_key tum_xxx --phase 2
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
DATA_DIR         = Path("/data/youchengsong/DSL-R1/data")
INPUT_FILE       = DATA_DIR / "enron_final.parquet"
SUBJECT_POS_FILE = DATA_DIR / "subject_pos.jsonl"
MESSAGE_POS_FILE = DATA_DIR / "message_pos.jsonl"
OUTPUT_FILE      = DATA_DIR / "email_descriptions.jsonl"

API_BASE          = "https://token.claude-code-manager.com/v1"
MODEL             = "deepseek-v3-2-251201"
CONCURRENCY       = 20
MESSAGE_MAX_CHARS = 1000  # truncate message body to this length before API call

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

SUBJECT_NEG_SYSTEM = "You are an email search assistant."
SUBJECT_NEG_PROMPT = """Subject search query: "{subject_pos}"

Write a search query (3-8 words) for an email about a COMPLETELY DIFFERENT topic and domain.
The negative query must NOT share any theme, domain, or concept with the positive query.

Examples:
- pos: "quarterly budget review finance" → neg: "employee onboarding training schedule"
- pos: "power plant maintenance outage" → neg: "client dinner reservation confirmation"
- pos: "contract negotiation legal terms" → neg: "vacation leave request approval"
- pos: "lost item found office desk" → neg: "software deployment rollback plan"
- pos: "holiday weekend emergency contact" → neg: "annual performance review process"

Output ONLY the query, no explanation, no punctuation."""

MESSAGE_NEG_SYSTEM = "You are an email search assistant."
MESSAGE_NEG_PROMPT = """Message search query: "{message_pos}"

Write a search query (8-15 words) for an email with COMPLETELY DIFFERENT content, topic, and domain.
The negative query must NOT share any theme, domain, or concept with the positive query.

Examples:
- pos: "natural gas pipeline capacity trading deal western region" → neg: "employee performance review feedback mid-year evaluation form"
- pos: "meeting agenda board directors quarterly earnings report" → neg: "software system upgrade deployment schedule IT infrastructure"
- pos: "found lost item office desk contact information" → neg: "quarterly financial forecast revenue projection budget planning"
- pos: "vacation coverage request meeting notes out of office" → neg: "energy trading contract terms pipeline capacity agreement"
- pos: "client invoice payment overdue reminder accounting department" → neg: "team building event outdoor activity planning logistics"

Output ONLY the query, no explanation, no punctuation."""


# ── Load data ─────────────────────────────────────────────────────────────────
def load_emails():
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    print(f"Loaded {len(df)} emails")
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


# wrap coroutine with a tag so we can distinguish results from mixed task list
async def tagged(coro, kind):
    result = await coro
    return kind, result


# ── Phase 1: subject_pos + message_pos ────────────────────────────────────────
async def phase1(api_key, records, concurrency):
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
    print(f"Phase 1: subject_pos remaining={len(remaining_subj)}, message_pos remaining={len(remaining_msg)}")

    if not remaining_subj and not remaining_msg:
        print("Phase 1: all done, skipping")
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

    with tqdm(total=len(tasks), desc="phase1") as pbar:
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
    print(f"Phase 1 done: subject_pos={len(done_subj)}, message_pos={len(done_msg)}")
    return done_subj, done_msg


# ── Phase 2: subject_neg + message_neg ────────────────────────────────────────
async def phase2(api_key, done_subj, done_msg, concurrency):
    done_out = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done_out[obj["message_id"]] = obj

    remaining_subj_neg = [(mid, sp) for mid, sp in done_subj.items()
                          if not done_out.get(mid, {}).get("subject_neg")]
    remaining_msg_neg  = [(mid, mp) for mid, mp in done_msg.items()
                          if not done_out.get(mid, {}).get("message_neg")]
    print(f"Phase 2: subject_neg remaining={len(remaining_subj_neg)}, message_neg remaining={len(remaining_msg_neg)}")

    if not remaining_subj_neg and not remaining_msg_neg:
        print("Phase 2: all done, skipping")
        return

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    tasks = (
        [tagged(call_api(client, sem, mid,
                         SUBJECT_NEG_SYSTEM,
                         SUBJECT_NEG_PROMPT.format(subject_pos=sp),
                         "subject_neg"), "subj_neg")
         for mid, sp in remaining_subj_neg]
        +
        [tagged(call_api(client, sem, mid,
                         MESSAGE_NEG_SYSTEM,
                         MESSAGE_NEG_PROMPT.format(message_pos=mp),
                         "message_neg"), "msg_neg")
         for mid, mp in remaining_msg_neg]
    )

    neg_subj = {mid: done_out[mid]["subject_neg"] for mid in done_out if done_out[mid].get("subject_neg")}
    neg_msg  = {mid: done_out[mid]["message_neg"]  for mid in done_out if done_out[mid].get("message_neg")}

    with tqdm(total=len(tasks), desc="phase2") as pbar:
        for fut in asyncio.as_completed(tasks):
            kind, (mid, result) = await fut
            if result:
                if kind == "subj_neg":
                    neg_subj[mid] = result
                else:
                    neg_msg[mid] = result
            pbar.update(1)

    # Write merged output
    all_mids = set(done_subj) | set(done_msg)
    fout = open(OUTPUT_FILE, "w", encoding="utf-8")
    written = 0
    for mid in all_mids:
        obj = {
            "message_id":  mid,
            "subject_pos": done_subj.get(mid),
            "subject_neg": neg_subj.get(mid),
            "message_pos": done_msg.get(mid),
            "message_neg": neg_msg.get(mid),
        }
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written += 1
    fout.close()
    print(f"Phase 2 done: {written} records written to {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",     default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--phase",       type=int, default=0,
                        help="0=all, 1=pos only, 2=neg only")
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

    done_subj, done_msg = {}, {}

    if args.phase in (0, 1):
        done_subj, done_msg = await phase1(args.api_key, records, args.concurrency)
    else:
        mids = {r["message_id"] for r in records}
        if SUBJECT_POS_FILE.exists():
            with open(SUBJECT_POS_FILE) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["message_id"] in mids:
                        done_subj[obj["message_id"]] = obj["subject_pos"]
        if MESSAGE_POS_FILE.exists():
            with open(MESSAGE_POS_FILE) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["message_id"] in mids:
                        done_msg[obj["message_id"]] = obj["message_pos"]
        print(f"Loaded subject_pos={len(done_subj)}, message_pos={len(done_msg)}")

    if args.phase in (0, 2):
        await phase2(args.api_key, done_subj, done_msg, args.concurrency)

    print(f"\nAll done. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
