#!/usr/bin/env python3
"""
三阶段重新计算邮件测试集 GT：
  Step 1: 用 leaves/groups 中的结构化条件计算超集（去掉语义占位符）
  Step 2: 在超集内做 embedding 粗排 → Gemini 精排（并发50），得到语义匹配 IDs
  Step 3: 将语义 IDs 填回 DSL 占位符，执行完整条件，得到最终 GT

邮件只有 text 语义（Subject / Message），无 image。

用法:
  python recompute_email_test_gt.py --api-key KEY --validate 50
  python recompute_email_test_gt.py --api-key KEY
  python recompute_email_test_gt.py --api-key KEY --concurrency 30
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

# ── 路径 ───────────────────────────────────────────────────────────────────────
BASE         = Path("/data/youchengsong/DSL-R1/data/email/test")
QUERIES_PATH = BASE / "queries_email_test_step9.jsonl"
CORPUS_PATH  = BASE / "email_test.parquet"
SUBJ_FEAT    = BASE / "test_subject_features.parquet"
MSG_FEAT     = BASE / "test_message_features.parquet"
OUTPUT_PATH  = BASE / "queries_email_test_step9_gt_v2.jsonl"
VALIDATE_OUT = BASE / "gt_validate_compare_email.jsonl"
TMP_RESULTS  = BASE / "_tmp_gemini_email_results.jsonl"

QWEN_PATH    = "/data/youchengsong/models/Qwen3-Embedding-8B"
DEVICE       = "cuda:6"

API_BASE     = "https://token.claude-code-manager.com/v1"
MODEL        = "gemini-2.5-flash"

TOP_K        = 200
TEXT_BATCH   = 75
CONCURRENCY  = 50


# ── Gemini 异步调用 ────────────────────────────────────────────────────────────
async def gemini_judge_text(client, sem, retrieval_query, docs, field, retries=3):
    doc_strs = "\n\n".join(f"[{i}] {d.get(field, '')}" for i, d in enumerate(docs))
    prompt = (
        f"Search query: {retrieval_query}\n\n"
        f"Below are {len(docs)} emails (showing their '{field}' field). Return a JSON array of 0-based indices "
        f"of emails that a user searching for the above query would consider relevant results. "
        f"The email should match the core intent of the query — same topic, subject matter, or key content. "
        f"Minor differences in wording are acceptable. Example: [0, 3, 7]\n\n"
        f"{doc_strs}"
    )
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256, temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                m = re.search(r'\[[\d,\s]*\]', text)
                return json.loads(m.group()) if m else []
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  text error: {e}")
                    return []


# ── Embedding ──────────────────────────────────────────────────────────────────
def load_features():
    print("Loading corpus and features ...")
    corpus_df = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)
    corpus_df["year"] = pd.to_datetime(corpus_df["Date"], errors="coerce", utc=True).dt.year

    sf = pd.read_parquet(SUBJ_FEAT)
    mf = pd.read_parquet(MSG_FEAT)

    subj_ids = sf["message_id"].values
    subj_mat = np.stack(sf["text_features"].values).astype(np.float32)
    subj_mat = subj_mat / (np.linalg.norm(subj_mat, axis=1, keepdims=True) + 1e-9)

    msg_ids  = mf["message_id"].values
    msg_mat  = np.stack(mf["text_features"].values).astype(np.float32)
    msg_mat  = msg_mat / (np.linalg.norm(msg_mat, axis=1, keepdims=True) + 1e-9)

    return corpus_df, subj_ids, subj_mat, msg_ids, msg_mat


def embed_queries(texts):
    print(f"  Embedding {len(texts)} queries ...")
    model = SentenceTransformer(QWEN_PATH, device=DEVICE)
    model.max_seq_length = 128
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    del model; torch.cuda.empty_cache()
    return vecs


# ── 结构化叶子 mask ────────────────────────────────────────────────────────────
def struct_leaf_mask(leaf, corpus_df):
    field = leaf["field"]
    value = leaf["value"]
    n     = len(corpus_df)

    if field == "Date":
        col = corpus_df["year"]
    elif field in ("From", "To"):
        col = corpus_df[field]
    else:
        return pd.Series([False] * n, index=corpus_df.index)

    lo, hi = value.get("lo"), value.get("hi")
    op, val = value.get("op", "="), value.get("val")
    if lo is not None and hi is not None:
        return (col >= lo) & (col <= hi)
    elif op == "=":  return col == val
    elif op == "!=": return col != val
    elif op == ">":  return col >  val
    elif op == ">=": return col >= val
    elif op == "<":  return col <  val
    elif op == "<=": return col <= val
    else:            return pd.Series([True] * n, index=corpus_df.index)


# ── Step 1: 计算结构化超集 ─────────────────────────────────────────────────────
def compute_superset_indices(query, corpus_df):
    leaves   = {lf["leaf_id"]: lf for lf in query.get("leaves", [])}
    groups   = query.get("groups", [])
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")

    if not groups:
        return None

    if len(groups) == 1:
        if inner_op == "AND":
            struct_masks = []
            for lid in groups[0]:
                if leaves[lid]["modality"] != "text":
                    struct_masks.append(struct_leaf_mask(leaves[lid], corpus_df))
            if struct_masks:
                result = struct_masks[0]
                for m in struct_masks[1:]:
                    result = result & m
                return np.where(result.values)[0]
        return None

    if outer_op == "OR":
        return None

    constraints = []
    for g in groups:
        has_semantic = any(leaves[lid]["modality"] == "text" for lid in g)
        struct_lids  = [lid for lid in g if leaves[lid]["modality"] != "text"]

        if not has_semantic and struct_lids:
            masks = [struct_leaf_mask(leaves[lid], corpus_df) for lid in struct_lids]
            gmask = masks[0]
            for m in masks[1:]:
                gmask = (gmask & m) if inner_op == "AND" else (gmask | m)
            constraints.append(gmask)
        elif has_semantic and inner_op == "AND" and struct_lids:
            masks = [struct_leaf_mask(leaves[lid], corpus_df) for lid in struct_lids]
            gmask = masks[0]
            for m in masks[1:]:
                gmask = gmask & m
            constraints.append(gmask)

    if not constraints:
        return None

    result = constraints[0]
    for m in constraints[1:]:
        result = result & m
    return np.where(result.values)[0]


# ── Step 3: 填回语义 IDs，执行完整 DSL ─────────────────────────────────────────
def compute_gt_from_leaves(query, corpus_df, semantic_matches):
    leaves   = {lf["leaf_id"]: lf for lf in query.get("leaves", [])}
    groups   = query.get("groups", [])
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")

    if not leaves or not groups:
        return []

    text_leaf_ids = sorted(lid for lid, lf in leaves.items() if lf["modality"] == "text")
    leaf_semantic = {}
    for i, lid in enumerate(text_leaf_ids):
        leaf_semantic[lid] = semantic_matches.get(f"text_{i}", set())

    def leaf_mask(lid):
        lf = leaves[lid]
        is_pos = lf["is_pos"]
        if lf["modality"] == "text":
            mids = leaf_semantic.get(lid, set())
            mask = corpus_df["Message_ID"].isin(mids)
            return mask if is_pos else ~mask
        return struct_leaf_mask(lf, corpus_df)

    def group_mask(group):
        masks  = [leaf_mask(lid) for lid in group]
        result = masks[0]
        for m in masks[1:]:
            result = (result & m) if inner_op == "AND" else (result | m)
        return result

    gms    = [group_mask(g) for g in groups]
    result = gms[0]
    for m in gms[1:]:
        result = (result | m) if outer_op == "OR" else (result & m)

    return corpus_df["Message_ID"][result].tolist()


# ── 预计算每条 query 的候选集（同步） ──────────────────────────────────────────
def prepare_candidates(queries, corpus_df, subj_ids, subj_mat, msg_ids, msg_mat, subj_vecs, msg_vecs):
    prepared = []
    for q in queries:
        rl = q.get("dsl", {}).get("retrieval_list", [])
        if not rl:
            prepared.append((q, None, 0))
            continue

        superset_idx = compute_superset_indices(q, corpus_df)
        if superset_idx is not None:
            superset_size = len(superset_idx)
        else:
            superset_idx  = np.arange(len(corpus_df))
            superset_size = len(corpus_df)

        sup_mids = set(corpus_df.iloc[superset_idx]["Message_ID"])

        candidate_mids = set()
        for r in rl:
            rq = r["query"]
            if r["field"] == "Subject" and rq in subj_vecs:
                sup_s_idx = np.array([i for i, mid in enumerate(subj_ids) if mid in sup_mids])
                if len(sup_s_idx) > 0:
                    v = subj_vecs[rq]
                    sub_mat = subj_mat[sup_s_idx]
                    sims = sub_mat @ v
                    top_k = min(TOP_K, len(sup_s_idx))
                    local = np.argsort(sims)[::-1][:top_k]
                    candidate_mids.update(subj_ids[sup_s_idx[local]].tolist())
            elif r["field"] != "Subject" and rq in msg_vecs:
                sup_m_idx = np.array([i for i, mid in enumerate(msg_ids) if mid in sup_mids])
                if len(sup_m_idx) > 0:
                    v = msg_vecs[rq]
                    sub_mat = msg_mat[sup_m_idx]
                    sims = sub_mat @ v
                    top_k = min(TOP_K, len(sup_m_idx))
                    local = np.argsort(sims)[::-1][:top_k]
                    candidate_mids.update(msg_ids[sup_m_idx[local]].tolist())

        candidates = corpus_df[corpus_df["Message_ID"].isin(candidate_mids)].to_dict("records")
        prepared.append((q, candidates, superset_size))

    return prepared


# ── 单条 query 的 Gemini 精排（异步） ─────────────────────────────────────────
async def process_one_query(client, sem, q, candidates, superset_size, corpus_df, validate_n, validate_f):
    qid    = q["query_id"]
    rl     = q.get("dsl", {}).get("retrieval_list", [])
    old_gt = set(q.get("gt_document_ids") or [])

    # 纯结构化
    if candidates is None:
        new_gt = compute_gt_from_leaves(q, corpus_df, {})
        print(f"  {qid}: pure struct → new_gt={len(new_gt)}", flush=True)
        return {**q, "gt_document_ids": new_gt or None, "gt_v2_method": "struct_only"}

    semantic_matches = {}
    text_ph_idx = 0

    for r in rl:
        rq = r["query"]
        r_field = r["field"]  # Subject 或 Message

        text_tasks = []
        text_batches = []
        for start in range(0, len(candidates), TEXT_BATCH):
            batch = candidates[start:start + TEXT_BATCH]
            text_batches.append(batch)
            text_tasks.append(gemini_judge_text(client, sem, rq, batch, r_field))
        text_results = await asyncio.gather(*text_tasks)

        matched_mids = set()
        for batch, rel_idxs in zip(text_batches, text_results):
            for idx in rel_idxs:
                if 0 <= idx < len(batch):
                    matched_mids.add(batch[idx]["Message_ID"])
        semantic_matches[f"text_{text_ph_idx}"] = matched_mids
        text_ph_idx += 1

    new_gt = compute_gt_from_leaves(q, corpus_df, semantic_matches)
    sem_summary = {k: len(v) for k, v in semantic_matches.items()}
    print(f"  {qid}: superset={superset_size} candidates={len(candidates)} "
          f"semantic={sem_summary} new_gt={len(new_gt)} old_gt={len(old_gt)} "
          f"overlap={len(set(new_gt)&old_gt)}", flush=True)

    if validate_n and validate_f:
        validate_f.write(json.dumps({
            "query_id":         qid,
            "sql":              q.get("dsl", {}).get("sql", ""),
            "superset_size":    superset_size,
            "candidates_size":  len(candidates),
            "old_gt_size":      len(old_gt),
            "new_gt_size":      len(new_gt),
            "overlap":          len(set(new_gt) & old_gt),
            "new_only":         len(set(new_gt) - old_gt),
            "old_only":         len(old_gt - set(new_gt)),
            "semantic_matches": {k: sorted(v) for k, v in semantic_matches.items()},
            "new_gt":           new_gt,
        }, ensure_ascii=False) + "\n")
        validate_f.flush()

    result = {**q, "gt_document_ids": new_gt or None, "gt_v2_method": "superset+gemini+struct"}

    # 断点续跑
    with open(TMP_RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": qid, "gt_document_ids": result["gt_document_ids"],
                            "semantic_matches": {k: sorted(v) for k, v in semantic_matches.items()},
                            }, ensure_ascii=False) + "\n")

    return result


# ── 断点续跑 ──────────────────────────────────────────────────────────────────
def load_checkpoint():
    done = {}
    if TMP_RESULTS.exists():
        with open(TMP_RESULTS) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["query_id"]] = obj
    return done


# ── 主流程（异步） ────────────────────────────────────────────────────────────
async def process_queries(queries, corpus_df, subj_ids, subj_mat, msg_ids, msg_mat, api_key, concurrency, validate_n):
    target = queries[:validate_n] if validate_n else queries

    # 收集所有需要 embed 的 query
    uq_subj = list({r["query"] for q in target for r in q["dsl"]["retrieval_list"] if r["field"] == "Subject"})
    uq_msg  = list({r["query"] for q in target for r in q["dsl"]["retrieval_list"] if r["field"] != "Subject"})

    print(f"  Embedding {len(uq_subj)} subject + {len(uq_msg)} message queries ...")
    se = embed_queries(uq_subj) if uq_subj else np.empty((0, subj_mat.shape[1]))
    me = embed_queries(uq_msg)  if uq_msg  else np.empty((0, msg_mat.shape[1]))

    subj_vecs = {q: se[i] for i, q in enumerate(uq_subj)}
    msg_vecs  = {q: me[i] for i, q in enumerate(uq_msg)}

    # 断点续跑
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} queries already done")

    # 预计算候选集
    print("Preparing candidates ...")
    prepared = prepare_candidates(target, corpus_df, subj_ids, subj_mat, msg_ids, msg_mat, subj_vecs, msg_vecs)

    # 并发调 Gemini
    client = AsyncOpenAI(base_url=API_BASE, api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    validate_f = None
    if validate_n:
        validate_f = open(VALIDATE_OUT, "a")

    results = []
    tasks = []
    for q, cands, sup_size in prepared:
        qid = q["query_id"]
        if qid in checkpoint:
            q_result = {**q, "gt_document_ids": checkpoint[qid]["gt_document_ids"],
                        "gt_v2_method": "superset+gemini+struct"}
            results.append(q_result)
        else:
            tasks.append((q, cands, sup_size))

    print(f"Skipped {len(results)} from checkpoint, running {len(tasks)} ...")

    running_tasks = [
        process_one_query(client, sem, q, cands, sup_size, corpus_df, validate_n, validate_f)
        for q, cands, sup_size in tasks
    ]

    done_count = 0
    for coro in asyncio.as_completed(running_tasks):
        result = await coro
        results.append(result)
        done_count += 1
        if done_count % 100 == 0:
            print(f"  [{done_count}/{len(running_tasks)}] ...", flush=True)

    if validate_f:
        validate_f.close()

    qid_order = {q["query_id"]: i for i, q in enumerate(target)}
    results.sort(key=lambda r: qid_order[r["query_id"]])

    return results


# ── 入口 ───────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",     required=True)
    parser.add_argument("--validate",    type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    queries = [json.loads(l) for l in open(QUERIES_PATH)]
    print(f"Loaded {len(queries)} queries")

    corpus_df, subj_ids, subj_mat, msg_ids, msg_mat = load_features()

    results = await process_queries(
        queries, corpus_df, subj_ids, subj_mat, msg_ids, msg_mat,
        args.api_key, args.concurrency, args.validate,
    )

    if args.validate:
        comparisons = [json.loads(l) for l in open(VALIDATE_OUT)]
        sem_queries = [c for c in comparisons if c.get("semantic_matches")]
        print(f"\n{'='*55}")
        print(f"Validated {len(comparisons)} queries  ({len(sem_queries)} with semantic)")
        if sem_queries:
            print(f"Avg superset:  {np.mean([c['superset_size']  for c in sem_queries]):.0f}")
            print(f"Avg candidates:{np.mean([c['candidates_size']for c in sem_queries]):.0f}")
            print(f"Avg old GT:    {np.mean([c['old_gt_size']    for c in sem_queries]):.1f}")
            print(f"Avg new GT:    {np.mean([c['new_gt_size']    for c in sem_queries]):.1f}")
            print(f"Avg overlap:   {np.mean([c['overlap']        for c in sem_queries]):.1f}")
            print(f"Avg new_only:  {np.mean([c['new_only']       for c in sem_queries]):.1f}")
            print(f"Avg old_only:  {np.mean([c['old_only']       for c in sem_queries]):.1f}")
    else:
        with open(OUTPUT_PATH, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDone. → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
