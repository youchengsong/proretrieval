#!/usr/bin/env python3
"""
Check cosine similarity between text_pos and text_neg for all products.
Reports:
  - Count and ratio of pairs with similarity < 0.4
  - Saves product_ids with similarity >= 0.4 to a file

Usage:
    python check_text_neg_similarity.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/data/youchengsong/DSL-R1/data")
INPUT_FILE  = DATA_DIR / "text_descriptions.jsonl"
OUTPUT_FILE = DATA_DIR / "text_neg_high_sim_ids.txt"  # ids with sim >= 0.4

EMB_MODEL      = "/data/youchengsong/models/Qwen3-Embedding-8B"
BATCH_SIZE     = 64
DEVICE         = torch.device("cuda:2")
THRESHOLD      = 0.4


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)


def embed_texts(texts, tokenizer, model):
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="embedding"):
        batch = texts[i: i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def main():
    # Load data
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("text_pos") and obj.get("text_neg"):
                records.append(obj)
    print(f"Loaded {len(records)} records")

    pids     = [r["product_id"] for r in records]
    pos_texts = [r["text_pos"]  for r in records]
    neg_texts = [r["text_neg"]  for r in records]

    # Load model
    print(f"Loading {EMB_MODEL} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    model     = AutoModel.from_pretrained(EMB_MODEL, trust_remote_code=True,
                                          torch_dtype=torch.float16).to(DEVICE).eval()

    # Embed
    print("Embedding text_pos...")
    pos_emb = embed_texts(pos_texts, tokenizer, model)
    print("Embedding text_neg...")
    neg_emb = embed_texts(neg_texts, tokenizer, model)

    # Cosine similarity (vectors already normalized)
    sims = (pos_emb * neg_emb).sum(axis=1)

    # Stats
    below = (sims < THRESHOLD).sum()
    total = len(sims)
    print(f"\n=== Results ===")
    print(f"Total pairs:         {total}")
    print(f"Similarity < {THRESHOLD}:   {below} ({100 * below / total:.1f}%)")
    print(f"Similarity >= {THRESHOLD}:  {total - below} ({100 * (total - below) / total:.1f}%)")
    print(f"Mean similarity:     {sims.mean():.4f}")
    print(f"Median similarity:   {np.median(sims):.4f}")
    print(f"Min / Max:           {sims.min():.4f} / {sims.max():.4f}")

    # Save high-similarity ids
    high_sim_ids = [pids[i] for i in range(total) if sims[i] >= THRESHOLD]
    with open(OUTPUT_FILE, "w") as f:
        for pid in high_sim_ids:
            f.write(pid + "\n")
    print(f"\nHigh-similarity IDs saved to {OUTPUT_FILE} ({len(high_sim_ids)} entries)")


if __name__ == "__main__":
    main()
