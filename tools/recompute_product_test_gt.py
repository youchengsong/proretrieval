#!/usr/bin/env python3
"""
三阶段重新计算电商测试集 GT：
  Step 1: 用 leaves/groups 中的结构化条件计算超集（去掉语义占位符）
  Step 2: 在超集内做 embedding 粗排 → Gemini 精排（并发50），得到语义匹配 IDs
  Step 3: 将语义 IDs 填回 DSL 占位符，执行完整条件，得到最终 GT

用法:
  python recompute_product_test_gt.py --api-key KEY --validate 50
  python recompute_product_test_gt.py --api-key KEY
  python recompute_product_test_gt.py --api-key KEY --concurrency 30
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
from transformers import AutoModel, AutoProcessor

# ── 路径 ───────────────────────────────────────────────────────────────────────
BASE         = Path("/data/youchengsong/DSL-R1/data/ecommerce/test")
QUERIES_PATH = BASE / "queries_product_test_step9.jsonl"
CORPUS_PATH  = BASE / "products_test.parquet"
TEXT_FEAT    = BASE / "product_text_features_test.parquet"
IMAGE_FEAT   = BASE / "product_image_features_test.parquet"
OUTPUT_PATH  = BASE / "queries_product_test_step9_gt_v2.jsonl"
VALIDATE_OUT = BASE / "gt_validate_compare.jsonl"
TMP_RESULTS  = BASE / "_tmp_gemini_results.jsonl"

IMAGE_DIR    = Path("/data/youchengsong/DSL-R1/image")
QWEN_PATH    = "/data/youchengsong/models/Qwen3-Embedding-8B"
QWEN_VL_PATH = "/data/youchengsong/models/Qwen3-VL-Embedding-8B"
DEVICE       = "cuda:3"

API_BASE     = "https://token.claude-code-manager.com/v1"
MODEL        = "gemini-2.5-flash"

TOP_K        = 200
TEXT_BATCH   = 75
IMAGE_BATCH  = 10
CONCURRENCY  = 50


# ── Gemini 异步调用 ────────────────────────────────────────────────────────────
async def gemini_judge_text(client, sem, retrieval_query, docs, field, retries=3):
    doc_strs = "\n\n".join(f"[{i}] {d.get(field, '')}" for i, d in enumerate(docs))
    prompt = (
        f"Search query: {retrieval_query}\n\n"
        f"Below are {len(docs)} product descriptions. Return a JSON array of 0-based indices "
        f"of products that a user searching for the above query would consider relevant results. "
        f"The product should match the core intent of the query — same product type, function, or key attributes. "
        f"Minor differences in brand, color, or size are acceptable. Example: [0, 3, 7]\n\n"
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


async def gemini_judge_image(client, sem, retrieval_query, docs, retries=3):
    import base64
    content = [{"type": "text", "text": (
        f"Search query: {retrieval_query}\n\nBelow are {len(docs)} product images. "
        f"Return a JSON array of 0-based indices of images showing products that match "
        f"the search query. The product should be the same type, color or serve the same function "
        f"as described in the query. Example: [0, 3, 7]\n"
    )}]
    for i, d in enumerate(docs):
        content.append({"type": "text", "text": f"[{i}]"})
        img_path = IMAGE_DIR / f"{d['product_id']}.jpg"
        if img_path.exists():
            from PIL import Image
            import io
            img = Image.open(img_path).convert("RGB").resize((768, 768))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            content.append({"type": "text", "text": "(no image)"})

    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=256, temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                m = re.search(r'\[[\d,\s]*\]', text)
                return json.loads(m.group()) if m else []
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  image error: {e}")
                    return []


# ── Embedding ──────────────────────────────────────────────────────────────────
def load_features():
    print("Loading corpus and features ...")
    corpus_df = pd.read_parquet(CORPUS_PATH)
    text_df   = pd.read_parquet(TEXT_FEAT)
    image_df  = pd.read_parquet(IMAGE_FEAT)
    pid_order = corpus_df["product_id"].tolist()
    text_mat  = np.stack(text_df.set_index("product_id").loc[pid_order]["text_features"].values)
    image_mat = np.stack(image_df.set_index("product_id").loc[pid_order]["image_features"].values)
    text_mat  = text_mat  / (np.linalg.norm(text_mat,  axis=1, keepdims=True) + 1e-9)
    image_mat = image_mat / (np.linalg.norm(image_mat, axis=1, keepdims=True) + 1e-9)
    return corpus_df, text_mat, image_mat


def embed_text_queries(texts):
    print(f"  Embedding {len(texts)} text queries ...")
    model = SentenceTransformer(QWEN_PATH, device=DEVICE)
    model.max_seq_length = 128
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    del model; torch.cuda.empty_cache()
    return vecs


def embed_image_queries(texts):
    print(f"  Embedding {len(texts)} image queries (VL) ...")
    model     = AutoModel.from_pretrained(QWEN_VL_PATH, dtype=torch.float16, device_map=DEVICE).eval()
    processor = AutoProcessor.from_pretrained(QWEN_VL_PATH)
    vecs = []
    with torch.no_grad():
        for t in texts:
            messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}]]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_tensors="pt", return_dict=True,
            ).to(DEVICE)
            out = model(**inputs, output_hidden_states=True)
            v = out.hidden_states[-1][:, -1, :].float().cpu().numpy()[0]
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
    del model, processor; torch.cuda.empty_cache()
    return np.stack(vecs)


# ── 结构化叶子 mask ────────────────────────────────────────────────────────────
def struct_leaf_mask(leaf, corpus_df):
    """不使用 is_pos（结构化字段 op 已经是最终操作符）。"""
    field = leaf["field"]
    value = leaf["value"]
    n     = len(corpus_df)

    if field not in corpus_df.columns:
        return pd.Series([False] * n, index=corpus_df.index)

    col = corpus_df[field]
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
                if leaves[lid]["modality"] not in ("text", "image"):
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
        has_semantic = any(leaves[lid]["modality"] in ("text", "image") for lid in g)
        struct_lids  = [lid for lid in g if leaves[lid]["modality"] not in ("text", "image")]

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

    text_leaf_ids  = sorted(lid for lid, lf in leaves.items() if lf["modality"] == "text")
    image_leaf_ids = sorted(lid for lid, lf in leaves.items() if lf["modality"] == "image")
    leaf_semantic  = {}
    for i, lid in enumerate(text_leaf_ids):
        leaf_semantic[lid] = semantic_matches.get(f"text_{i}", set())
    for i, lid in enumerate(image_leaf_ids):
        leaf_semantic[lid] = semantic_matches.get(f"image_{i}", set())

    def leaf_mask(lid):
        lf = leaves[lid]
        is_pos = lf["is_pos"]
        if lf["modality"] in ("text", "image"):
            pids = leaf_semantic.get(lid, set())
            mask = corpus_df["product_id"].isin(pids)
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

    return corpus_df["product_id"][result].tolist()


# ── 预计算每条 query 的候选集（同步） ──────────────────────────────────────────
def prepare_candidates(queries, corpus_df, text_mat, image_mat, text_vecs, image_vecs):
    """返回列表，每个元素是 (query, candidates, superset_size) 或 (query, None, 0) 表示纯结构化。"""
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

        candidate_idx = set()
        qid = q["query_id"]
        if qid in text_vecs:
            for v in text_vecs[qid]:
                sub_mat = text_mat[superset_idx]
                sims    = sub_mat @ v
                top_k   = min(TOP_K, len(superset_idx))
                local   = np.argsort(sims)[::-1][:top_k]
                candidate_idx.update(superset_idx[local].tolist())
        if qid in image_vecs:
            for v in image_vecs[qid]:
                sub_mat = image_mat[superset_idx]
                sims    = sub_mat @ v
                top_k   = min(TOP_K, len(superset_idx))
                local   = np.argsort(sims)[::-1][:top_k]
                candidate_idx.update(superset_idx[local].tolist())

        candidates = corpus_df.iloc[sorted(candidate_idx)].to_dict("records")
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
    text_ph_idx, image_ph_idx = 0, 0

    # 每个 retrieval 单独用自己的 query 做对比
    for r in rl:
        rq = r["query"]  # retrieval_list 中的 query

        if r["type"] == "text":
            r_field = r["field"]   # retrieval_list 中指定的字段
            text_tasks = []
            text_batches = []
            for start in range(0, len(candidates), TEXT_BATCH):
                batch = candidates[start:start + TEXT_BATCH]
                text_batches.append(batch)
                text_tasks.append(gemini_judge_text(client, sem, rq, batch, r_field))
            text_results = await asyncio.gather(*text_tasks)

            matched_pids = set()
            for batch, rel_idxs in zip(text_batches, text_results):
                for idx in rel_idxs:
                    if 0 <= idx < len(batch):
                        matched_pids.add(batch[idx]["product_id"])
            semantic_matches[f"text_{text_ph_idx}"] = matched_pids
            text_ph_idx += 1

        elif r["type"] == "image":
            image_tasks = []
            image_batches = []
            for start in range(0, len(candidates), IMAGE_BATCH):
                batch = candidates[start:start + IMAGE_BATCH]
                image_batches.append(batch)
                image_tasks.append(gemini_judge_image(client, sem, rq, batch))
            image_results = await asyncio.gather(*image_tasks)

            matched_pids = set()
            for batch, rel_idxs in zip(image_batches, image_results):
                for idx in rel_idxs:
                    if 0 <= idx < len(batch):
                        matched_pids.add(batch[idx]["product_id"])
            semantic_matches[f"image_{image_ph_idx}"] = matched_pids
            image_ph_idx += 1

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

    # 断点续跑：每条完成后追加到临时文件
    with open(TMP_RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": qid, "gt_document_ids": result["gt_document_ids"],
                            "semantic_matches": {k: sorted(v) for k, v in semantic_matches.items()},
                            }, ensure_ascii=False) + "\n")

    return result


# ── 断点续跑：加载已完成的结果 ─────────────────────────────────────────────────
def load_checkpoint():
    """返回已完成的 {query_id: {"gt_document_ids": ..., "semantic_matches": ...}}"""
    done = {}
    if TMP_RESULTS.exists():
        with open(TMP_RESULTS) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["query_id"]] = obj
    return done


# ── 主流程（异步） ────────────────────────────────────────────────────────────
async def process_queries(queries, corpus_df, text_mat, image_mat, api_key, concurrency, validate_n):
    target = queries[:validate_n] if validate_n else queries

    # 批量收集 embed 任务
    text_q_map, image_q_map = {}, {}
    for q in target:
        for r in q.get("dsl", {}).get("retrieval_list", []):
            if r["type"] == "text":
                text_q_map.setdefault(q["query_id"], []).append(r["query"])
            elif r["type"] == "image":
                image_q_map.setdefault(q["query_id"], []).append(r["query"])

    all_text  = [(qid, t) for qid, ts in text_q_map.items()  for t in ts]
    all_image = [(qid, t) for qid, ts in image_q_map.items() for t in ts]

    text_vecs, image_vecs = {}, {}
    if all_text:
        vecs = embed_text_queries([t for _, t in all_text])
        for (qid, _), v in zip(all_text, vecs):
            text_vecs.setdefault(qid, []).append(v)
    if all_image:
        vecs = embed_image_queries([t for _, t in all_image])
        for (qid, _), v in zip(all_image, vecs):
            image_vecs.setdefault(qid, []).append(v)

    # 断点续跑：加载已完成的结果
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} queries already done")

    # 预计算候选集（同步）
    print("Preparing candidates ...")
    prepared = prepare_candidates(target, corpus_df, text_mat, image_mat, text_vecs, image_vecs)

    # 并发调 Gemini（跳过已完成的）
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
            # 用 checkpoint 中保存的结果
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

    # 按 query_id 排序恢复原始顺序
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

    corpus_df, text_mat, image_mat = load_features()

    results = await process_queries(
        queries, corpus_df, text_mat, image_mat,
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
