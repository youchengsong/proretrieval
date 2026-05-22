#!/usr/bin/env python3
"""
Sample 404 extra records from enron_test not in test/train,
generate descriptions (subject_pos + message_pos) and embeddings,
then merge into test directory to reach 10000 total.

Usage:
    python gen_extra404.py --api_key KEY --task desc
    python gen_extra404.py --task subject_emb   # CUDA_VISIBLE_DEVICES=X
    python gen_extra404.py --task message_emb   # CUDA_VISIBLE_DEVICES=X
    python gen_extra404.py --task merge
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd

SEED = 42
DATA_DIR   = Path("/data/youchengsong/DSL-R1/data")
EMAIL_DIR  = DATA_DIR / "email"
TEST_DIR   = EMAIL_DIR / "test"
TRAIN_DIR  = EMAIL_DIR / "train"

EXTRA_PARQUET  = EMAIL_DIR / "extra404.parquet"
EXTRA_SUBJ_POS = EMAIL_DIR / "extra404_subject_pos.jsonl"
EXTRA_MSG_POS  = EMAIL_DIR / "extra404_message_pos.jsonl"
EXTRA_DESC     = EMAIL_DIR / "extra404_descriptions.jsonl"
EXTRA_SUBJ_EMB = EMAIL_DIR / "extra404_subject_features.parquet"
EXTRA_MSG_EMB  = EMAIL_DIR / "extra404_message_features.parquet"

API_BASE    = "https://token.claude-code-manager.com/v1"
MODEL       = "deepseek-v3-2-251201"
CONCURRENCY = 80


def sample_extra404():
    enron = pd.read_parquet(DATA_DIR / "enron_test.parquet")
    test  = pd.read_parquet(TEST_DIR / "email_test.parquet")
    train = pd.read_parquet(TRAIN_DIR / "email_train.parquet")
    used_ids = set(test["Message_ID"].tolist()) | set(train["Message_ID"].tolist())
    remaining = enron[~enron["Message_ID"].isin(used_ids)]
    sampled = remaining.sample(n=404, random_state=SEED)
    sampled.to_parquet(EXTRA_PARQUET, index=False)
    print(f"Saved {len(sampled)} records to {EXTRA_PARQUET}")
    return sampled


SUBJECT_POS_SYSTEM = "You are an email search assistant."
SUBJECT_POS_PROMPT = """Email subject line:
{subject}

Write a concise semantic search query (5-8 words) that captures the key topic of this email subject.
Output ONLY the query, no explanation."""

MESSAGE_POS_SYSTEM = "You are an email search assistant."
MESSAGE_POS_PROMPT = """Email content (first 1000 chars):
{message}

Write a concise semantic search query (8-12 words) that captures the main topic and intent of this email.
Output ONLY the query, no explanation."""


async def gen_one(client, sem, mid, prompt_system, prompt_user, retries=3):
    import asyncio
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user",   "content": prompt_user},
                    ],
                    max_tokens=60,
                    temperature=0.3,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                return mid, msg.content.strip()
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  FAILED {mid}: {e}")
                    return mid, None
                await asyncio.sleep(2 ** attempt)


async def gen_desc(api_key, df):
    from openai import AsyncOpenAI
    from tqdm import tqdm

    done_subj = {}
    if EXTRA_SUBJ_POS.exists():
        with open(EXTRA_SUBJ_POS) as f:
            for line in f:
                obj = json.loads(line)
                done_subj[obj["Message_ID"]] = obj["subject_pos"]

    done_msg = {}
    if EXTRA_MSG_POS.exists():
        with open(EXTRA_MSG_POS) as f:
            for line in f:
                obj = json.loads(line)
                done_msg[obj["Message_ID"]] = obj["message_pos"]

    print(f"subject_pos done: {len(done_subj)}, message_pos done: {len(done_msg)}")

    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(CONCURRENCY)

    subj_remaining = [(row["Message_ID"], row["Subject"]) for _, row in df.iterrows()
                      if row["Message_ID"] not in done_subj]
    if subj_remaining:
        print(f"Generating {len(subj_remaining)} subject_pos...")
        tasks = [gen_one(client, sem, mid,
                         SUBJECT_POS_SYSTEM,
                         SUBJECT_POS_PROMPT.format(subject=str(subj)[:500]))
                 for mid, subj in subj_remaining]
        with open(EXTRA_SUBJ_POS, "a", encoding="utf-8") as fout:
            with tqdm(total=len(tasks), desc="subject_pos") as pbar:
                for coro in asyncio.as_completed(tasks):
                    mid, result = await coro
                    if result:
                        done_subj[mid] = result
                        fout.write(json.dumps({"Message_ID": mid, "subject_pos": result}, ensure_ascii=False) + "\n")
                        fout.flush()
                    pbar.update(1)

    msg_remaining = [(row["Message_ID"], row["Message"]) for _, row in df.iterrows()
                     if row["Message_ID"] not in done_msg]
    if msg_remaining:
        print(f"Generating {len(msg_remaining)} message_pos...")
        tasks = [gen_one(client, sem, mid,
                         MESSAGE_POS_SYSTEM,
                         MESSAGE_POS_PROMPT.format(message=str(msg)[:1000]))
                 for mid, msg in msg_remaining]
        with open(EXTRA_MSG_POS, "a", encoding="utf-8") as fout:
            with tqdm(total=len(tasks), desc="message_pos") as pbar:
                for coro in asyncio.as_completed(tasks):
                    mid, result = await coro
                    if result:
                        done_msg[mid] = result
                        fout.write(json.dumps({"Message_ID": mid, "message_pos": result}, ensure_ascii=False) + "\n")
                        fout.flush()
                    pbar.update(1)

    print("Merging descriptions...")
    with open(EXTRA_DESC, "w", encoding="utf-8") as fout:
        for mid in df["Message_ID"]:
            sp = done_subj.get(mid)
            mp = done_msg.get(mid)
            fout.write(json.dumps({
                "Message_ID": mid,
                "subject_pos": sp,
                "message_pos": mp,
            }, ensure_ascii=False) + "\n")
    print(f"Saved {EXTRA_DESC}")


def gen_embeddings(df, field):
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel

    MODEL_NAME = "/data/youchengsong/models/Qwen3-Embedding-8B"
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model on {device}")

    if field == "subject":
        texts = df["Subject"].fillna("").tolist()
        out_file = EXTRA_SUBJ_EMB
    else:
        texts = df["Message"].fillna("").tolist()
        out_file = EXTRA_MSG_EMB

    BATCH = 32
    all_embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"  {i+len(batch)}/{len(texts)}")

    embs = np.concatenate(all_embs, axis=0)
    result_df = pd.DataFrame({"Message_ID": df["Message_ID"].tolist(), "embedding": list(embs)})
    result_df.to_parquet(out_file, index=False)
    print(f"Saved {out_file} ({len(result_df)} rows, dim={embs.shape[1]})")


def merge_into_test():
    extra_df   = pd.read_parquet(EXTRA_PARQUET)
    extra_subj = pd.read_parquet(EXTRA_SUBJ_EMB)
    extra_msg  = pd.read_parquet(EXTRA_MSG_EMB)

    for src, dest_path in [
        (extra_df,   TEST_DIR / "email_test.parquet"),
        (extra_subj, TEST_DIR / "test_subject_features.parquet"),
        (extra_msg,  TEST_DIR / "test_message_features.parquet"),
    ]:
        dest = pd.read_parquet(dest_path)
        merged = pd.concat([dest, src], ignore_index=True)
        merged.to_parquet(dest_path, index=False)
        print(f"{dest_path.name}: {len(dest)} + {len(src)} = {len(merged)}")

    desc_dest = TEST_DIR / "email_descriptions_test.jsonl"
    with open(EXTRA_DESC) as fin, open(desc_dest, "a", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    with open(desc_dest) as f:
        n = sum(1 for _ in f)
    print(f"email_descriptions_test.jsonl: {n} total")
    print("Merge done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sample", "desc", "subject_emb", "message_emb", "merge"], required=True)
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    args = parser.parse_args()

    if args.task == "sample":
        sample_extra404()
    elif args.task == "desc":
        if not EXTRA_PARQUET.exists():
            sample_extra404()
        df = pd.read_parquet(EXTRA_PARQUET)
        if not args.api_key:
            raise ValueError("--api_key required")
        asyncio.run(gen_desc(args.api_key, df))
    elif args.task in ("subject_emb", "message_emb"):
        if not EXTRA_PARQUET.exists():
            sample_extra404()
        df = pd.read_parquet(EXTRA_PARQUET)
        field = "subject" if args.task == "subject_emb" else "message"
        gen_embeddings(df, field)
    elif args.task == "merge":
        merge_into_test()


if __name__ == "__main__":
    main()
