#!/usr/bin/env python3
"""
为 email_corpus.parquet 生成：
  - corpus_subject_features.parquet  (Qwen3 embedding)
  - corpus_message_features.parquet  (Qwen3 embedding)
  - email_descriptions_corpus.jsonl  (subject_pos + message_pos only)

用法:
  CUDA_VISIBLE_DEVICES=4 python gen_corpus_features_and_desc.py --task subject_emb
  CUDA_VISIBLE_DEVICES=6 python gen_corpus_features_and_desc.py --task message_emb
  python gen_corpus_features_and_desc.py --task desc --api_key tum_xxx
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

DATA_DIR   = Path("/data/youchengsong/DSL-R1/data/email")
INPUT_FILE = DATA_DIR / "email_corpus.parquet"
MODEL_PATH = "/data/youchengsong/models/Qwen3-Embedding-8B"
API_BASE   = "https://token.claude-code-manager.com/v1"
MODEL      = "deepseek-v3-2-251201"
CONCURRENCY       = 20
MESSAGE_TRUNCATE  = 2000
BATCH_SIZE        = 32

SUBJECT_POS_FILE = DATA_DIR / "corpus_subject_pos.jsonl"
MESSAGE_POS_FILE = DATA_DIR / "corpus_message_pos.jsonl"
OUTPUT_DESC_FILE = DATA_DIR / "email_descriptions_corpus.jsonl"

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


# ── Embedding ─────────────────────────────────────────────────────────────────
def run_embedding(field: str):
    from sentence_transformers import SentenceTransformer
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    print(f"[{field} emb] {len(df)} rows, GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if field == "subject":
        texts    = df["Subject"].fillna("").tolist()
        max_seq  = 128
        out_path = DATA_DIR / "corpus_subject_features.parquet"
    else:
        texts    = df["Message"].fillna("").str[:MESSAGE_TRUNCATE].tolist()
        max_seq  = 512
        out_path = DATA_DIR / "corpus_message_features.parquet"

    model = SentenceTransformer(MODEL_PATH, device="cuda")
    model.max_seq_length = max_seq

    embeddings = model.encode(texts, batch_size=BATCH_SIZE,
                               show_progress_bar=True, normalize_embeddings=True)
    print(f"  shape: {embeddings.shape}")

    pd.DataFrame({
        "message_id":    df["Message_ID"].values,
        "text_features": list(embeddings),
    }).to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


# ── API descriptions (pos only) ───────────────────────────────────────────────
async def call_api(client, sem, msg_id, system, prompt, label):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                               {"role": "user",   "content": prompt}],
                    max_tokens=256, temperature=0.3,
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
    return kind, await coro


async def run_desc(api_key: str, concurrency: int):
    df = pd.read_parquet(INPUT_FILE, columns=["Message_ID", "Subject", "Message"])
    print(f"[desc] {len(df)} rows")

    done_subj, done_msg = {}, {}
    if SUBJECT_POS_FILE.exists():
        with open(SUBJECT_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done_subj[obj["message_id"]] = obj["subject_pos"]
    if MESSAGE_POS_FILE.exists():
        with open(MESSAGE_POS_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done_msg[obj["message_id"]] = obj["message_pos"]

    records = [{"message_id": r["Message_ID"], "subject": r["Subject"],
                "message": r["Message"]} for _, r in df.iterrows()]

    remaining_subj = [r for r in records if r["message_id"] not in done_subj]
    remaining_msg  = [r for r in records if r["message_id"] not in done_msg]
    print(f"  subject_pos remaining={len(remaining_subj)}, message_pos remaining={len(remaining_msg)}")

    if remaining_subj or remaining_msg:
        client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
        sem    = asyncio.Semaphore(concurrency)
        tasks  = (
            [tagged(call_api(client, sem, r["message_id"], SUBJECT_POS_SYSTEM,
                             SUBJECT_POS_PROMPT.format(subject=r["subject"]), "subj"), "subj")
             for r in remaining_subj]
            +
            [tagged(call_api(client, sem, r["message_id"], MESSAGE_POS_SYSTEM,
                             MESSAGE_POS_PROMPT.format(message=r["message"][:MESSAGE_TRUNCATE]), "msg"), "msg")
             for r in remaining_msg]
        )
        fout_s = open(SUBJECT_POS_FILE, "a", encoding="utf-8")
        fout_m = open(MESSAGE_POS_FILE, "a", encoding="utf-8")
        with tqdm(total=len(tasks), desc="desc") as pbar:
            for fut in asyncio.as_completed(tasks):
                kind, (mid, result) = await fut
                if result:
                    if kind == "subj":
                        done_subj[mid] = result
                        fout_s.write(json.dumps({"message_id": mid, "subject_pos": result}, ensure_ascii=False) + "\n")
                        fout_s.flush()
                    else:
                        done_msg[mid] = result
                        fout_m.write(json.dumps({"message_id": mid, "message_pos": result}, ensure_ascii=False) + "\n")
                        fout_m.flush()
                pbar.update(1)
        fout_s.close(); fout_m.close()

    all_mids = set(done_subj) | set(done_msg)
    with open(OUTPUT_DESC_FILE, "w", encoding="utf-8") as f:
        for mid in all_mids:
            f.write(json.dumps({"message_id": mid, "subject_pos": done_subj.get(mid),
                                "message_pos": done_msg.get(mid)}, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_mids)} records to {OUTPUT_DESC_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["subject_emb", "message_emb", "desc"], required=True)
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    if args.task == "subject_emb":
        run_embedding("subject")
    elif args.task == "message_emb":
        run_embedding("message")
    elif args.task == "desc":
        if not args.api_key:
            raise ValueError("--api_key required")
        asyncio.run(run_desc(args.api_key, args.concurrency))
