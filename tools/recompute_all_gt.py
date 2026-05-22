#!/usr/bin/env python3
"""
重新计算 4 个 step9 文件的 GT（修正顺序：先超集 → 超集内 embedding top-k → 填回全量执行）。

直接覆盖原文件中的 gt_document_ids 字段。

用法:
  # 指定 GPU 和数据集
  python recompute_all_gt.py --gpu 2 --dataset all
  python recompute_all_gt.py --gpu 2 --dataset product_train
  python recompute_all_gt.py --gpu 2 --dataset email_test

可选 dataset: product_train, product_test, email_train, email_test, all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ── 配置 ───────────────────────────────────────────────────────────────────────
BASE = Path("/data/youchengsong/DSL-R1/data")
QWEN_PATH = "/data/youchengsong/models/Qwen3-Embedding-8B"

DATASETS = {
    "product_train": {
        "queries":    BASE / "ecommerce/train/queries_product_train_step9.jsonl",
        "corpus":     BASE / "ecommerce/train/products_train.parquet",
        "text_feat":  BASE / "ecommerce/train/product_text_features_train.parquet",
        "image_feat": BASE / "ecommerce/train/product_image_features_train.parquet",
        "id_col":     "product_id",
        "type":       "product",
    },
    "product_test": {
        "queries":    BASE / "ecommerce/test/queries_product_test_step9.jsonl",
        "corpus":     BASE / "ecommerce/test/products_test.parquet",
        "text_feat":  BASE / "ecommerce/test/product_text_features_test.parquet",
        "image_feat": BASE / "ecommerce/test/product_image_features_test.parquet",
        "id_col":     "product_id",
        "type":       "product",
    },
    "email_train": {
        "queries":    BASE / "email/train/queries_email_train_step9.jsonl",
        "corpus":     BASE / "email/train/email_train.parquet",
        "subj_feat":  BASE / "email/train/train_subject_features.parquet",
        "msg_feat":   BASE / "email/train/train_message_features.parquet",
        "id_col":     "Message_ID",
        "type":       "email",
    },
    "email_test": {
        "queries":    BASE / "email/test/queries_email_test_step9.jsonl",
        "corpus":     BASE / "email/test/email_test.parquet",
        "subj_feat":  BASE / "email/test/test_subject_features.parquet",
        "msg_feat":   BASE / "email/test/test_message_features.parquet",
        "id_col":     "Message_ID",
        "type":       "email",
    },
}

TOP_K = 20
VECTOR_MODALITIES = {"text", "image"}


# ── 结构化 mask ────────────────────────────────────────────────────────────────
def struct_mask_product(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field not in df.columns:
        return None
    col = df[field]
    if "lo" in value:
        return (col >= value["lo"]) & (col <= value["hi"])
    op, val = value["op"], value["val"]
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val
    return None


def struct_mask_email(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field == "Date":
        col = df["year"]
    elif field in ("From", "To"):
        col = df[field]
    else:
        return None
    if "lo" in value:
        return (col >= value["lo"]) & (col <= value["hi"])
    op, val = value["op"], value["val"]
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val
    return None


# ── 超集计算 ──────────────────────────────────────────────────────────────────
def compute_superset(query, df, struct_mask_fn):
    """返回超集 df（结构化条件过滤后），无法缩减则返回原 df。"""
    leaves   = {l["leaf_id"]: l for l in query["leaves"]}
    groups   = query["groups"]
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")
    k        = len(groups)

    def gmasks(group):
        return [m for lid in group if leaves[lid]["modality"] not in VECTOR_MODALITIES
                for m in [struct_mask_fn(leaves[lid], df)] if m is not None]

    if k == 1:
        ms = gmasks(groups[0])
        if inner_op == "OR" and any(leaves[lid]["modality"] in VECTOR_MODALITIES for lid in groups[0]):
            return df  # OR 含语义，无法缩减
        if not ms:
            return df
        r = ms[0]
        for m in ms[1:]:
            r = r & m if inner_op == "AND" else r | m
        return df[r]

    if outer_op == "OR":
        # OR 之间：每个 group 都必须有结构化约束，否则无法缩减
        gms = []
        for g in groups:
            ms = gmasks(g)
            if not ms:
                return df
            gm = ms[0]
            for m in ms[1:]:
                gm = gm & m
            gms.append(gm)
        r = gms[0]
        for m in gms[1:]:
            r = r | m
        return df[r]
    else:
        # AND 之间：纯结构化 group 取交集
        pure = []
        for g in groups:
            if any(leaves[lid]["modality"] in VECTOR_MODALITIES for lid in g):
                continue
            ms = [struct_mask_fn(leaves[lid], df) for lid in g]
            ms = [m for m in ms if m is not None]
            if not ms:
                continue
            gm = ms[0]
            for m in ms[1:]:
                gm = gm | m
            pure.append(gm)
        if not pure:
            return df
        r = pure[0]
        for m in pure[1:]:
            r = r & m
        return df[r]


# ── 完整 DSL 执行 ─────────────────────────────────────────────────────────────
def eval_dsl(query, df, text_sets, image_sets, id_col, struct_mask_fn):
    """
    text_sets:  list of set(ids)，按 leaf_id 顺序对应每个 text 占位符。
    image_sets: list of set(ids)，按 leaf_id 顺序对应每个 image 占位符。
    在 df 上执行完整 DSL，返回满足条件的 id 列表。
    """
    leaves   = {l["leaf_id"]: l for l in query["leaves"]}
    groups   = query["groups"]
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")
    k        = len(groups)

    # 建立 leaf_id → set 索引（与原 pipeline 一致）
    ti = ii = 0
    ph = {}
    for lid in sorted(leaves):
        mod = leaves[lid]["modality"]
        if mod == "text":
            ph[lid] = ("text", ti); ti += 1
        elif mod == "image":
            ph[lid] = ("image", ii); ii += 1

    def lmask(lid):
        leaf = leaves[lid]
        if leaf["modality"] == "text":
            s = df[id_col].isin(text_sets[ph[lid][1]])
            return s if leaf["polarity"] == "pos" else ~s
        elif leaf["modality"] == "image":
            s = df[id_col].isin(image_sets[ph[lid][1]])
            return s if leaf["polarity"] == "pos" else ~s
        m = struct_mask_fn(leaf, df)
        return m if m is not None else pd.Series([True] * len(df), index=df.index)

    def combine(masks, op):
        r = masks[0]
        for m in masks[1:]:
            r = r & m if op == "AND" else r | m
        return r

    if k == 1:
        result = combine([lmask(l) for l in groups[0]], inner_op)
    elif outer_op == "OR":
        result = combine([combine([lmask(l) for l in g], "AND") for g in groups], "OR")
    else:
        result = combine([combine([lmask(l) for l in g], "OR") for g in groups], "AND")

    return df[result][id_col].tolist()


# ── 处理电商数据集 ─────────────────────────────────────────────────────────────
def process_product(cfg, device, only_none=False):
    print(f"Loading {cfg['queries']} ...")
    queries = [json.loads(l) for l in open(cfg["queries"])]
    df      = pd.read_parquet(cfg["corpus"]).reset_index(drop=True)
    tf      = pd.read_parquet(cfg["text_feat"])
    imf     = pd.read_parquet(cfg["image_feat"])

    t_ids = tf["product_id"].values
    t_mat = np.stack(tf["text_features"].values).astype(np.float32)
    i_ids = imf["product_id"].values
    i_mat = np.stack(imf["image_features"].values).astype(np.float32)

    # 收集所有需要 embed 的 query
    uq_t = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["type"] == "text"})
    uq_i = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["type"] == "image"})

    print(f"  Embedding {len(uq_t)} text + {len(uq_i)} image queries ...")

    # text embedding
    qwen = SentenceTransformer(QWEN_PATH, device=device)
    qwen.max_seq_length = 128
    te = qwen.encode(uq_t, batch_size=32, normalize_embeddings=True, show_progress_bar=True) \
         if uq_t else np.empty((0, t_mat.shape[1]))
    del qwen; torch.cuda.empty_cache()

    # image embedding (VL)
    ie = np.empty((0, i_mat.shape[1]))
    if uq_i:
        from transformers import AutoModel, AutoProcessor
        import torch.nn.functional as F
        QWEN_VL_PATH = "/data/youchengsong/models/Qwen3-VL-Embedding-8B"
        vl_model = AutoModel.from_pretrained(QWEN_VL_PATH, dtype=torch.float16, device_map=device).eval()
        vl_processor = AutoProcessor.from_pretrained(QWEN_VL_PATH)
        ie_list = []
        for idx in range(0, len(uq_i), 32):
            batch_texts = uq_i[idx:idx+32]
            messages_batch = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in batch_texts]
            texts = [vl_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
            inputs = vl_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = vl_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                embs = hidden[torch.arange(len(batch_texts), device=torch.device(device)), seq_lens]
                embs = F.normalize(embs, p=2, dim=-1).float().cpu().numpy()
            ie_list.append(embs)
        ie = np.vstack(ie_list)
        del vl_model, vl_processor; torch.cuda.empty_cache()

    tc = {q: te[i] for i, q in enumerate(uq_t)}
    ic = {q: ie[i] for i, q in enumerate(uq_i)}

    def topk(vec, mat, k):
        v = vec / (np.linalg.norm(vec) + 1e-9)
        s = mat @ v
        idx = np.argpartition(s, -k)[-k:]
        return idx[np.argsort(s[idx])[::-1]]

    stats = {"ok": 0, "empty": 0, "skipped": 0}
    for qi, q in enumerate(queries):
        if qi % 1000 == 0:
            print(f"  [{qi}/{len(queries)}] ...", flush=True)

        if only_none and q.get("gt_document_ids") is not None:
            stats["skipped"] += 1
            continue

        ret = q["dsl"]["retrieval_list"]

        # Step 1: 超集
        sup_df = compute_superset(q, df, struct_mask_product)
        sup_pids = set(sup_df["product_id"])

        # Step 2: 超集内 embedding top-k
        sup_t_idx = np.array([i for i, pid in enumerate(t_ids) if pid in sup_pids])
        sup_i_idx = np.array([i for i, pid in enumerate(i_ids) if pid in sup_pids])

        ts = []
        for r in ret:
            if r["type"] == "text" and len(sup_t_idx) > 0:
                k_ = min(TOP_K, len(sup_t_idx))
                ts.append(set(t_ids[sup_t_idx[topk(tc[r["query"]], t_mat[sup_t_idx], k_)]]))
            elif r["type"] == "text":
                ts.append(set())
        is_ = []
        for r in ret:
            if r["type"] == "image" and len(sup_i_idx) > 0:
                k_ = min(TOP_K, len(sup_i_idx))
                is_.append(set(i_ids[sup_i_idx[topk(ic[r["query"]], i_mat[sup_i_idx], k_)]]))
            elif r["type"] == "image":
                is_.append(set())

        # Step 3: 在完整语料上执行
        gt = eval_dsl(q, df, ts, is_, "product_id", struct_mask_product)

        if len(gt) == 0:
            q["gt_document_ids"] = None; stats["empty"] += 1
        else:
            q["gt_document_ids"] = gt; stats["ok"] += 1

    print(f"  ok={stats['ok']} empty={stats['empty']} skipped={stats['skipped']}")
    return queries


# ── 处理邮件数据集 ─────────────────────────────────────────────────────────────
def process_email(cfg, device, only_none=False):
    print(f"Loading {cfg['queries']} ...")
    queries = [json.loads(l) for l in open(cfg["queries"])]
    df      = pd.read_parquet(cfg["corpus"]).reset_index(drop=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.year

    sf = pd.read_parquet(cfg["subj_feat"])
    mf = pd.read_parquet(cfg["msg_feat"])
    subj_ids = sf["message_id"].values
    subj_mat = np.stack(sf["text_features"].values).astype(np.float32)
    msg_ids  = mf["message_id"].values
    msg_mat  = np.stack(mf["text_features"].values).astype(np.float32)



    # 收集 query
    uq_subj = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["field"] == "Subject"})
    uq_msg  = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["field"] != "Subject"})

    print(f"  Embedding {len(uq_subj)} subject + {len(uq_msg)} message queries ...")
    qwen = SentenceTransformer(QWEN_PATH, device=device)
    qwen.max_seq_length = 128
    se = qwen.encode(uq_subj, batch_size=32, normalize_embeddings=True, show_progress_bar=True) \
         if uq_subj else np.empty((0, subj_mat.shape[1]))
    me = qwen.encode(uq_msg, batch_size=32, normalize_embeddings=True, show_progress_bar=True) \
         if uq_msg else np.empty((0, msg_mat.shape[1]))
    del qwen; torch.cuda.empty_cache()

    sc = {q: se[i] for i, q in enumerate(uq_subj)}
    mc = {q: me[i] for i, q in enumerate(uq_msg)}

    def topk(vec, mat, k):
        v = vec / (np.linalg.norm(vec) + 1e-9)
        s = mat @ v
        idx = np.argpartition(s, -k)[-k:]
        return idx[np.argsort(s[idx])[::-1]]

    stats = {"ok": 0, "empty": 0, "skipped": 0}
    for qi, q in enumerate(queries):
        if qi % 1000 == 0:
            print(f"  [{qi}/{len(queries)}] ...", flush=True)

        if only_none and q.get("gt_document_ids") is not None:
            stats["skipped"] += 1
            continue

        # Step 1: 超集
        sup_df = compute_superset(q, df, struct_mask_email)
        sup_mids = set(sup_df["Message_ID"])

        sup_subj_idx = np.array([i for i, mid in enumerate(subj_ids) if mid in sup_mids])
        sup_msg_idx  = np.array([i for i, mid in enumerate(msg_ids)  if mid in sup_mids])

        # Step 2: 超集内 embedding top-k（邮件只有 text，无 image）
        ts = []
        for r in q["dsl"]["retrieval_list"]:
            if r["field"] == "Subject" and len(sup_subj_idx) > 0:
                k_ = min(TOP_K, len(sup_subj_idx))
                ts.append(set(subj_ids[sup_subj_idx[topk(sc[r["query"]], subj_mat[sup_subj_idx], k_)]]))
            elif r["field"] != "Subject" and len(sup_msg_idx) > 0:
                k_ = min(TOP_K, len(sup_msg_idx))
                ts.append(set(msg_ids[sup_msg_idx[topk(mc[r["query"]], msg_mat[sup_msg_idx], k_)]]))
            else:
                ts.append(set())

        # Step 3: 在完整语料上执行（邮件无 image_sets）
        gt = eval_dsl(q, df, ts, [], "Message_ID", struct_mask_email)

        if len(gt) == 0:
            q["gt_document_ids"] = None; stats["empty"] += 1
        else:
            q["gt_document_ids"] = gt; stats["ok"] += 1

    print(f"  ok={stats['ok']} empty={stats['empty']} skipped={stats['skipped']}")
    return queries


# ── 写回文件 ──────────────────────────────────────────────────────────────────
def save_queries(queries, path):
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Saved → {path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--dataset", default="all",
                        choices=["product_train", "product_test",
                                 "email_train", "email_test", "all"])
    parser.add_argument("--only-none", action="store_true",
                        help="只处理 gt_document_ids 为 None 的 query")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        cfg = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        if cfg["type"] == "product":
            queries = process_product(cfg, device, only_none=args.only_none)
        else:
            queries = process_email(cfg, device, only_none=args.only_none)

        save_queries(queries, cfg["queries"])

    print("\nAll done.")
