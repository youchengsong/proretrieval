#!/usr/bin/env python3
"""
邮件测试集评估 — 对齐电商评估流程
流程: 模型 DSL → 向量检索（Subject/Message 两套 features） → 填占位符 → SQLite 执行 → 对比 GT → 算指标

用法:
    CUDA_VISIBLE_DEVICES=1 python eval_email.py \
        --pred  predictions_sft_email.json \
        --gt    queries_email_test_step9.jsonl
"""

import argparse, json, os, re, sqlite3, sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

BASE_DIR          = "/root/autodl-tmp/LocalSearch/data/email"
DATA_DIR          = os.path.join(BASE_DIR, "test")
RESULT_DIR        = os.path.join(BASE_DIR, "result")
EMAIL_PARQUET     = os.path.join(DATA_DIR, "email_test.parquet")
SUBJECT_FEAT_FILE = os.path.join(DATA_DIR, "test_subject_features.parquet")
MESSAGE_FEAT_FILE = os.path.join(DATA_DIR, "test_message_features.parquet")
QWEN_PATH         = "/root/autodl-tmp/models/Qwen3-Embedding-8B"
TOP_K             = 20

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DAY_RE  = re.compile(r"\b(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{4}\b")


def parse_date(date_str):
    """从 'Thu, 26 Oct 2000 07:03:00 -0700 (PDT)' 之类提取 year / day"""
    if not date_str:
        return None, None
    y_m = YEAR_RE.search(date_str)
    d_m = DAY_RE.search(date_str)
    return (int(y_m.group(0)) if y_m else None,
            int(d_m.group(1)) if d_m else None)


def load_data(pred_file, gt_file):
    print("Loading data...")
    with open(pred_file) as f:
        predictions = json.load(f)

    id2gt, id2meta = {}, {}
    with open(gt_file) as f:
        for line in f:
            d = json.loads(line)
            id2gt[d["query_id"]] = d.get("gt_document_ids") or d.get("gold_ids") or []
            id2meta[d["query_id"]] = {
                "level": d.get("level"),
                "n": d.get("n"),
                "modalities": d.get("modalities"),
            }

    print(f"  Predictions: {len(predictions)}, GT: {len(id2gt)}")
    return predictions, id2gt, id2meta


def create_db():
    print("Creating SQLite database from email_test.parquet ...")
    df = pd.read_parquet(EMAIL_PARQUET)
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE emails (
            message_id TEXT PRIMARY KEY,
            "From" TEXT,
            "To" TEXT,
            Subject TEXT,
            Message TEXT,
            Date TEXT,
            year INTEGER,
            day INTEGER
        )
    ''')
    for _, row in df.iterrows():
        y, d = parse_date(row.get("Date") or "")
        cur.execute(
            'INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?)',
            (row["Message_ID"], row.get("From"), row.get("To"),
             row.get("Subject"), row.get("Message"), row.get("Date"),
             y, d),
        )
    conn.commit()
    print(f"  {len(df)} emails loaded into SQLite")
    return conn


def load_features():
    print("Loading feature matrices...")
    sf = pd.read_parquet(SUBJECT_FEAT_FILE)
    s_ids = sf["message_id"].values
    s_mat = np.stack(sf["text_features"].values).astype(np.float32)
    s_mat = s_mat / (np.linalg.norm(s_mat, axis=1, keepdims=True) + 1e-9)

    mf = pd.read_parquet(MESSAGE_FEAT_FILE)
    m_ids = mf["message_id"].values
    m_mat = np.stack(mf["text_features"].values).astype(np.float32)
    m_mat = m_mat / (np.linalg.norm(m_mat, axis=1, keepdims=True) + 1e-9)

    print(f"  Subject: {s_mat.shape}, Message: {m_mat.shape}")
    return s_ids, s_mat, m_ids, m_mat


EMBED_CACHE_DIR = os.path.join(RESULT_DIR, "embed_cache")


def _cache_path(pred_file):
    name = os.path.splitext(os.path.basename(pred_file))[0]
    return os.path.join(EMBED_CACHE_DIR, f"{name}_embeddings.npz")


def encode_queries(predictions, pred_file=None):
    cache_file = _cache_path(pred_file) if pred_file else None
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        vec_map = dict(data["vec_map"].item())
        print(f"  Loaded: {len(vec_map)} queries")
        return vec_map

    queries = set()
    for pred in predictions:
        dsl = pred.get("dsl")
        if not dsl or not dsl.get("retrieval_list"):
            continue
        for r in dsl["retrieval_list"]:
            q = (r.get("query") or "").strip()
            if q and r.get("type") == "text":
                queries.add(q)
    queries = list(queries)
    print(f"Encoding {len(queries)} unique text queries ...")
    if not queries:
        return {}

    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen = SentenceTransformer(QWEN_PATH, device=device)
    qwen.max_seq_length = 128
    vecs = qwen.encode(queries, batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
    vec_map = {q: vecs[i] for i, q in enumerate(queries)}
    del qwen
    torch.cuda.empty_cache()

    if cache_file:
        os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
        np.savez(cache_file, vec_map=vec_map)
        print(f"  Saved embeddings to {cache_file}")
    return vec_map


def topk(query_vec, feat_mat, feat_ids, k):
    scores = feat_mat @ query_vec
    k = min(k, len(scores))
    if k == 0:
        return [], []
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [feat_ids[i] for i in idx], [float(scores[i]) for i in idx]


def execute_dsl(dsl, conn, s_ids, s_mat, m_ids, m_mat, vec_map):
    """执行单条 DSL, 返回 (result_ids_ranked, error)"""
    if not dsl:
        return [], "null_dsl"
    sql_body = (dsl.get("sql") or "").strip()
    if not sql_body:
        return [], "empty_sql"

    retrieval_list = dsl.get("retrieval_list", []) or []
    text_idx = 0
    id_scores = {}
    placeholder_ids = {}

    for r in retrieval_list:
        if r.get("type") != "text":
            continue
        ph = f"<text_{text_idx}>"
        text_idx += 1
        query_text = (r.get("query") or "").strip()
        qvec = vec_map.get(query_text)
        if qvec is None:
            placeholder_ids[ph] = set()
            continue
        field = (r.get("field") or "").strip()
        if field.lower() == "subject":
            feat_mat, feat_ids = s_mat, s_ids
        else:
            feat_mat, feat_ids = m_mat, m_ids
        ids, scores = topk(qvec, feat_mat, feat_ids, TOP_K)
        placeholder_ids[ph] = set(ids)
        for mid, sc in zip(ids, scores):
            id_scores[mid] = max(id_scores.get(mid, -1), sc)

    exec_sql = sql_body
    for ph, ids in placeholder_ids.items():
        if ids:
            id_list = ", ".join("'" + pid.replace("'", "''") + "'" for pid in ids)
            exec_sql = exec_sql.replace(ph, f"({id_list})")
        else:
            exec_sql = exec_sql.replace(ph, "('__EMPTY__')")

    if not exec_sql.upper().lstrip().startswith("SELECT"):
        full_sql = f"SELECT DISTINCT message_id FROM emails {exec_sql}"
    else:
        full_sql = exec_sql

    try:
        cur = conn.cursor()
        cur.execute(full_sql)
        result_ids = [row[0] for row in cur.fetchall()]
    except Exception as e:
        return [], f"sql_error: {str(e)[:100]}"

    if id_scores:
        result_ids.sort(key=lambda mid: id_scores.get(mid, -2), reverse=True)
    return result_ids, None


def compute_metrics(predictions, id2gt, id2meta, conn,
                    s_ids, s_mat, m_ids, m_mat, vec_map):
    print("Executing DSLs and computing metrics...")
    per_query = []
    for i, pred in enumerate(predictions):
        qid = pred["query_id"]
        gt_ids = set(id2gt.get(qid) or [])
        meta = id2meta.get(qid, {})

        result_ids, error = execute_dsl(
            pred.get("dsl"), conn, s_ids, s_mat, m_ids, m_mat, vec_map,
        )
        result_set = set(result_ids)

        hit1 = int(len(result_ids) > 0 and result_ids[0] in gt_ids)
        hit3 = int(any(pid in gt_ids for pid in result_ids[:3]))
        mrr = 0.0
        for rank, pid in enumerate(result_ids, 1):
            if pid in gt_ids:
                mrr = 1.0 / rank
                break
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=os.path.join(RESULT_DIR, "predictions_sft_email.json"))
    parser.add_argument("--gt",   default=os.path.join(DATA_DIR, "queries_email_test_step9.jsonl"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        pred_name = os.path.splitext(os.path.basename(args.pred))[0]
        gt_tag    = "step8" if "step8" in os.path.basename(args.gt) else "step9"
        args.output = os.path.join(RESULT_DIR, f"eval_{pred_name}_{gt_tag}_gt.json")

    predictions, id2gt, id2meta = load_data(args.pred, args.gt)
    conn = create_db()
    s_ids, s_mat, m_ids, m_mat = load_features()
    vec_map = encode_queries(predictions, pred_file=args.pred)

    per_query = compute_metrics(
        predictions, id2gt, id2meta, conn,
        s_ids, s_mat, m_ids, m_mat, vec_map,
    )

    overall = aggregate(per_query)
    overall["total"] = len(per_query)
    overall["format_success"] = sum(1 for r in per_query if r["error"] != "null_dsl")
    overall["exec_success"]   = sum(1 for r in per_query if r["error"] is None)

    by_level = {}
    for lvl in sorted({r["level"] for r in per_query if r["level"] is not None}):
        items = [r for r in per_query if r["level"] == lvl]
        by_level[str(lvl)] = aggregate(items)

    by_modality = {}
    for mod in ["structured", "text", "image"]:
        items = [r for r in per_query if mod in (r.get("modalities") or [])]
        if items:
            by_modality[mod] = aggregate(items)

    errors = defaultdict(int)
    for r in per_query:
        if r["error"]:
            errors[r["error"][:80]] += 1

    output = {
        "overall": overall,
        "by_level": by_level,
        "by_modality": by_modality,
        "errors": dict(errors),
        "per_query": per_query,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"GT file: {args.gt}")
    print(f"Total: {overall['total']}")
    print(f"Format OK: {overall['format_success']} | Exec OK: {overall['exec_success']}")
    print(f"Hit@1:  {overall['hit_at_1']:.4f}")
    print(f"Hit@3:  {overall['hit_at_3']:.4f}")
    print(f"MRR:    {overall['mrr']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"Avg result size: {overall['avg_result_size']:.1f}")
    print(f"\nBy Level:")
    for lvl in sorted(by_level.keys()):
        m = by_level[lvl]
        print(f"  L{lvl}: n={m['count']}  Hit@1={m['hit_at_1']:.4f}  "
              f"Hit@3={m['hit_at_3']:.4f}  MRR={m['mrr']:.4f}  Recall={m['recall']:.4f}")
    if errors:
        print(f"\nErrors:")
        for err, cnt in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4d} × {err}")
    print(f"\nSaved to {args.output}")
    conn.close()


if __name__ == "__main__":
    main()
