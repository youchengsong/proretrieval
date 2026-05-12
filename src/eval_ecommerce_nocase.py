#!/usr/bin/env python3
"""
电商测试集评估脚本
流程: 模型 DSL → 向量检索 → 填占位符 → SQLite 执行 → 对比 GT → 算指标

用法:
    CUDA_VISIBLE_DEVICES=3 python eval_ecommerce.py [--pred predictions_sft_4b.json] [--gt queries_product_test_step9_gt_v2.jsonl]
"""

import argparse, json, os, re, sqlite3, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict

# ── 配置 ──────────────────────────────────────────────────────
DATA_DIR       = "/home/sunzhen/LocalSearch/data/ecommerce/test"
RESULT_DIR     = "/home/sunzhen/LocalSearch/data/ecommerce/result"
PRODUCTS_FILE  = os.path.join(DATA_DIR, "products_test.json")
TEXT_FEAT_FILE = os.path.join(DATA_DIR, "product_text_features_test.parquet")
IMG_FEAT_FILE  = os.path.join(DATA_DIR, "product_image_features_test.parquet")
QWEN_PATH      = "/home/sunzhen/models/Qwen3-Embedding-8B"
QWEN_VL_PATH   = "/home/sunzhen/models/Qwen3-VL-Embedding-8B"
TOP_K          = 20


# ── 1. 加载数据 ──────────────────────────────────────────────
def load_data(pred_file, gt_file):
    print("Loading data...")
    with open(pred_file) as f:
        predictions = json.load(f)

    id2gt = {}
    id2meta = {}
    with open(gt_file) as f:
        for line in f:
            d = json.loads(line)
            id2gt[d["query_id"]] = d.get("gt_document_ids")
            id2meta[d["query_id"]] = {
                "level": d.get("level"),
                "n": d.get("n"),
                "modalities": d.get("modalities"),
            }

    with open(PRODUCTS_FILE) as f:
        products = json.load(f)

    print(f"  Predictions: {len(predictions)}, GT: {len(id2gt)}, Products: {len(products)}")
    return predictions, id2gt, id2meta, products


# ── 2. 创建 SQLite ───────────────────────────────────────────
def create_db(products):
    print("Creating SQLite database...")
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            brand TEXT COLLATE NOCASE,
            color TEXT COLLATE NOCASE,
            category TEXT COLLATE NOCASE,
            price REAL,
            average_rating REAL,
            rating_number INTEGER
        )
    """)
    for p in products:
        cur.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
            (p["product_id"], p.get("brand"), p.get("color"),
             p.get("category"), p.get("price"),
             p.get("average_rating"), p.get("rating_number")),
        )
    conn.commit()
    print(f"  {len(products)} products loaded into SQLite")
    return conn


# ── 3. 加载特征矩阵 ──────────────────────────────────────────
def load_features():
    print("Loading feature matrices...")
    tf = pd.read_parquet(TEXT_FEAT_FILE)
    t_ids = tf["product_id"].values
    t_mat = np.stack(tf["text_features"].values).astype(np.float32)
    t_mat = t_mat / (np.linalg.norm(t_mat, axis=1, keepdims=True) + 1e-9)

    imf = pd.read_parquet(IMG_FEAT_FILE)
    i_ids = imf["product_id"].values
    i_mat = np.stack(imf["image_features"].values).astype(np.float32)
    i_mat = i_mat / (np.linalg.norm(i_mat, axis=1, keepdims=True) + 1e-9)

    print(f"  Text: {t_mat.shape}, Image: {i_mat.shape}")
    return t_ids, t_mat, i_ids, i_mat


# ── 4. 编码 query 向量（带磁盘缓存） ─────────────────────────
EMBED_CACHE_DIR = os.path.join(RESULT_DIR, "embed_cache")


def _cache_path(pred_file):
    """根据 predictions 文件名生成缓存路径。"""
    import hashlib
    name = os.path.splitext(os.path.basename(pred_file))[0]
    return os.path.join(EMBED_CACHE_DIR, f"{name}_embeddings.npz")


def encode_queries(predictions, t_mat_dim, i_mat_dim, pred_file=None):
    cache_file = _cache_path(pred_file) if pred_file else None

    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        text_vec_map = dict(data["text_vec_map"].item())
        image_vec_map = dict(data["image_vec_map"].item())
        print(f"  Loaded: {len(text_vec_map)} text, {len(image_vec_map)} image")
        return text_vec_map, image_vec_map

    text_queries, image_queries = set(), set()
    for pred in predictions:
        dsl = pred.get("dsl")
        if not dsl or not dsl.get("retrieval_list"):
            continue
        for r in dsl["retrieval_list"]:
            q = r.get("query", "")
            if not q:
                continue
            if r["type"] == "text":
                text_queries.add(q)
            elif r["type"] == "image":
                image_queries.add(q)
    text_queries = list(text_queries)
    image_queries = list(image_queries)
    print(f"Encoding queries: {len(text_queries)} text, {len(image_queries)} image")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Text
    text_vec_map = {}
    if text_queries:
        from sentence_transformers import SentenceTransformer
        qwen = SentenceTransformer(QWEN_PATH, device=str(device), model_kwargs={"torch_dtype": torch.float16})
        qwen.max_seq_length = 128
        vecs = qwen.encode(text_queries, batch_size=32,
                           normalize_embeddings=True, show_progress_bar=True)
        for i, q in enumerate(text_queries):
            text_vec_map[q] = vecs[i]
        del qwen
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # Image
    image_vec_map = {}
    if image_queries:
        from transformers import AutoModel, AutoProcessor
        vl_model = AutoModel.from_pretrained(
            QWEN_VL_PATH, dtype=torch.float16, device_map=str(device)
        ).eval()
        vl_proc = AutoProcessor.from_pretrained(QWEN_VL_PATH)

        for i in range(0, len(image_queries), 32):
            batch = image_queries[i:i+32]
            msgs_batch = [
                [{"role": "user", "content": [{"type": "text", "text": t}]}]
                for t in batch
            ]
            texts = [
                vl_proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in msgs_batch
            ]
            inputs = vl_proc(text=texts, return_tensors="pt",
                             padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = vl_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                embs = hidden[torch.arange(len(batch), device=device), seq_lens]
                embs = F.normalize(embs, p=2, dim=-1).float().cpu().numpy()
            for j, q in enumerate(batch):
                image_vec_map[q] = embs[j]
        del vl_model, vl_proc
        torch.cuda.empty_cache()

    print(f"  Encoded: {len(text_vec_map)} text, {len(image_vec_map)} image")

    if cache_file:
        os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
        np.savez(cache_file,
                 text_vec_map=text_vec_map,
                 image_vec_map=image_vec_map)
        print(f"  Saved embeddings to {cache_file}")

    return text_vec_map, image_vec_map


# ── 5. 向量 top-K ────────────────────────────────────────────
def topk(query_vec, feat_mat, feat_ids, k):
    scores = feat_mat @ query_vec
    k = min(k, len(scores))
    if k == 0:
        return [], []
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [feat_ids[i] for i in idx], [float(scores[i]) for i in idx]


# ── 6. 执行单条 DSL ──────────────────────────────────────────
def execute_dsl(dsl, conn, t_ids, t_mat, i_ids, i_mat,
                text_vec_map, image_vec_map):
    """
    返回 (result_ids_ranked, error_str_or_None)
    """
    if not dsl:
        return [], "null_dsl"

    sql_body = dsl.get("sql", "").strip()
    retrieval_list = dsl.get("retrieval_list", [])

    if not sql_body:
        return [], "empty_sql"

    # 向量检索，得到各占位符的 ID 集合 + 分数
    text_idx = 0
    image_idx = 0
    id_scores = {}          # product_id → max cosine score
    placeholder_ids = {}    # "<text_0>" → set of ids

    for r in retrieval_list:
        query_text = r.get("query", "")
        if r["type"] == "text":
            ph = f"<text_{text_idx}>"
            text_idx += 1
            qvec = text_vec_map.get(query_text)
            if qvec is None:
                placeholder_ids[ph] = set()
                continue
            ids, scores = topk(qvec, t_mat, t_ids, TOP_K)
            placeholder_ids[ph] = set(ids)
            for pid, sc in zip(ids, scores):
                id_scores[pid] = max(id_scores.get(pid, -1), sc)
        elif r["type"] == "image":
            ph = f"<image_{image_idx}>"
            image_idx += 1
            qvec = image_vec_map.get(query_text)
            if qvec is None:
                placeholder_ids[ph] = set()
                continue
            ids, scores = topk(qvec, i_mat, i_ids, TOP_K)
            placeholder_ids[ph] = set(ids)
            for pid, sc in zip(ids, scores):
                id_scores[pid] = max(id_scores.get(pid, -1), sc)

    # 替换占位符
    exec_sql = sql_body
    for ph, ids in placeholder_ids.items():
        if ids:
            id_list = ", ".join(f"'{pid}'" for pid in ids)
            exec_sql = exec_sql.replace(ph, f"({id_list})")
        else:
            # 空集：用一个不存在的 ID，确保 IN 子句不匹配
            exec_sql = exec_sql.replace(ph, "('__EMPTY__')")

    # 补全 SQL 并执行
    if not exec_sql.upper().startswith("SELECT"):
        full_sql = f"SELECT DISTINCT product_id FROM products {exec_sql}"
    else:
        full_sql = exec_sql

    try:
        cur = conn.cursor()
        cur.execute(full_sql)
        result_ids = [row[0] for row in cur.fetchall()]
    except Exception as e:
        return [], f"sql_error: {str(e)[:100]}"

    # 按向量分数排序（有向量检索时），否则保持原序
    if id_scores:
        result_ids.sort(key=lambda pid: id_scores.get(pid, -2), reverse=True)

    return result_ids, None


# ── 7. 计算指标 ──────────────────────────────────────────────
def compute_metrics(predictions, id2gt, id2meta, conn,
                    t_ids, t_mat, i_ids, i_mat,
                    text_vec_map, image_vec_map):
    print("Executing DSLs and computing metrics...")
    per_query = []

    for i, pred in enumerate(predictions):
        qid = pred["query_id"]
        gt_ids = set(id2gt.get(qid) or [])
        meta = id2meta.get(qid, {})

        result_ids, error = execute_dsl(
            pred.get("dsl"), conn, t_ids, t_mat, i_ids, i_mat,
            text_vec_map, image_vec_map,
        )

        result_set = set(result_ids)

        # Hit@1, Hit@3
        hit1 = int(len(result_ids) > 0 and result_ids[0] in gt_ids)
        hit3 = int(any(pid in gt_ids for pid in result_ids[:3]))

        # MRR
        mrr = 0.0
        for rank, pid in enumerate(result_ids, 1):
            if pid in gt_ids:
                mrr = 1.0 / rank
                break

        # Recall, Precision
        recall = len(result_set & gt_ids) / len(gt_ids) if gt_ids else 0.0
        precision = len(result_set & gt_ids) / len(result_set) if result_set else 0.0

        per_query.append({
            "query_id": qid,
            "level": meta.get("level"),
            "n": meta.get("n"),
            "modalities": meta.get("modalities"),
            "error": error,
            "result_size": len(result_ids),
            "gt_size": len(gt_ids),
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "mrr": mrr,
            "recall": recall,
            "precision": precision,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(predictions)}")

    return per_query


def aggregate(items):
    if not items:
        return {}
    n = len(items)
    return {
        "count": n,
        "exec_success_rate": sum(1 for r in items if r["error"] is None) / n,
        "hit_at_1": sum(r["hit_at_1"] for r in items) / n,
        "hit_at_3": sum(r["hit_at_3"] for r in items) / n,
        "mrr": sum(r["mrr"] for r in items) / n,
        "recall": sum(r["recall"] for r in items) / n,
        "precision": sum(r["precision"] for r in items) / n,
        "avg_result_size": sum(r["result_size"] for r in items) / n,
    }


# ── 主入口 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=os.path.join(RESULT_DIR, "predictions_sft_4b.json"))
    parser.add_argument("--gt", default=os.path.join(DATA_DIR, "queries_product_test_step9_gt_v2.jsonl"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        pred_name = os.path.splitext(os.path.basename(args.pred))[0]
        args.output = os.path.join(RESULT_DIR, f"eval_{pred_name}.json")

    predictions, id2gt, id2meta, products = load_data(args.pred, args.gt)
    conn = create_db(products)
    t_ids, t_mat, i_ids, i_mat = load_features()
    text_vec_map, image_vec_map = encode_queries(predictions, t_mat.shape[1], i_mat.shape[1], pred_file=args.pred)

    per_query = compute_metrics(
        predictions, id2gt, id2meta, conn,
        t_ids, t_mat, i_ids, i_mat,
        text_vec_map, image_vec_map,
    )

    # 汇总
    overall = aggregate(per_query)
    overall["total"] = len(per_query)
    overall["format_success"] = sum(1 for r in per_query if r["error"] != "null_dsl")
    overall["exec_success"] = sum(1 for r in per_query if r["error"] is None)

    by_level = {}
    for lvl in [1, 2, 3]:
        items = [r for r in per_query if r["level"] == lvl]
        if items:
            by_level[str(lvl)] = aggregate(items)

    by_modality = {}
    for mod in ["structured", "text", "image"]:
        items = [r for r in per_query if mod in (r.get("modalities") or [])]
        if items:
            by_modality[mod] = aggregate(items)

    errors = defaultdict(int)
    for r in per_query:
        if r["error"]:
            err_key = r["error"][:80]
            errors[err_key] += 1

    output = {
        "overall": overall,
        "by_level": by_level,
        "by_modality": by_modality,
        "errors": dict(errors),
        "per_query": per_query,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印结果
    print(f"\n{'='*50}")
    print(f"Total: {overall['total']}")
    print(f"Format OK: {overall['format_success']} | Exec OK: {overall['exec_success']}")
    print(f"Hit@1: {overall['hit_at_1']:.4f}")
    print(f"Hit@3: {overall['hit_at_3']:.4f}")
    print(f"MRR:   {overall['mrr']:.4f}")
    print(f"Recall:{overall['recall']:.4f}")
    print(f"Avg result size: {overall['avg_result_size']:.1f}")
    print(f"\nBy Level:")
    for lvl in ["1", "2", "3"]:
        m = by_level.get(lvl)
        if m:
            print(f"  L{lvl}: n={m['count']}  Hit@1={m['hit_at_1']:.4f}  "
                  f"Hit@3={m['hit_at_3']:.4f}  MRR={m['mrr']:.4f}  "
                  f"Recall={m['recall']:.4f}")
    print(f"\nBy Modality:")
    for mod in ["structured", "text", "image"]:
        m = by_modality.get(mod)
        if m:
            print(f"  {mod}: n={m['count']}  Hit@1={m['hit_at_1']:.4f}  "
                  f"Hit@3={m['hit_at_3']:.4f}  Recall={m['recall']:.4f}")
    if errors:
        print(f"\nErrors:")
        for err, cnt in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4d} × {err}")
    print(f"\nSaved to {args.output}")
    conn.close()


if __name__ == "__main__":
    main()
