#!/usr/bin/env python3
"""
从旧的 gt_validate_compare.jsonl 重新计算正确 GT。

旧文件只有统计数据（size/overlap），新文件会有 semantic_matches + new_gt。
本脚本处理两种情况：
  - 新格式（有 semantic_matches）：直接重跑 compute_gt_from_leaves，无需再调 API
  - 旧格式（只有 size 统计）：只能计算结构化部分作为下界，无法恢复 Gemini 结果

用法:
  python fix_gt_from_validate.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE         = Path("/data/youchengsong/DSL-R1/data/ecommerce/test")
VALIDATE_OUT = BASE / "gt_validate_compare.jsonl"
QUERIES_PATH = BASE / "queries_product_test_step9.jsonl"
CORPUS_PATH  = BASE / "products_test.parquet"
OUTPUT_PATH  = BASE / "gt_validate_fixed.jsonl"


def compute_gt_from_leaves(query, corpus_df, semantic_matches):
    leaves    = {lf["leaf_id"]: lf for lf in query.get("leaves", [])}
    groups    = query.get("groups", [])
    inner_op  = query.get("inner_op") or "AND"
    outer_op  = query.get("outer_op")

    if not leaves or not groups:
        return []

    text_leaf_ids  = sorted(lid for lid, lf in leaves.items() if lf["modality"] == "text")
    image_leaf_ids = sorted(lid for lid, lf in leaves.items() if lf["modality"] == "image")
    leaf_semantic  = {}
    for i, lid in enumerate(text_leaf_ids):
        leaf_semantic[lid] = set(semantic_matches.get(f"text_{i}", []))
    for i, lid in enumerate(image_leaf_ids):
        leaf_semantic[lid] = set(semantic_matches.get(f"image_{i}", []))

    def leaf_mask(lid):
        lf       = leaves[lid]
        modality = lf["modality"]
        is_pos   = lf["is_pos"]
        field    = lf["field"]
        value    = lf["value"]
        n        = len(corpus_df)

        if modality in ("text", "image"):
            pids = leaf_semantic.get(lid, set())
            mask = corpus_df["product_id"].isin(pids)
            return mask if is_pos else ~mask

        # 结构化字段：直接用 op，不用 is_pos
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

    def group_mask(group):
        masks = [leaf_mask(lid) for lid in group]
        if not masks:
            return pd.Series([False] * len(corpus_df), index=corpus_df.index)
        result = masks[0]
        for m in masks[1:]:
            result = (result & m) if inner_op == "AND" else (result | m)
        return result

    group_masks = [group_mask(g) for g in groups]
    result = group_masks[0]
    for m in group_masks[1:]:
        result = (result | m) if outer_op == "OR" else (result & m)

    return corpus_df["product_id"][result].tolist()


if __name__ == "__main__":
    corpus_df = pd.read_parquet(CORPUS_PATH)
    queries   = {q["query_id"]: q for q in (json.loads(l) for l in open(QUERIES_PATH))}
    records   = [json.loads(l) for l in open(VALIDATE_OUT)]

    new_format = [r for r in records if "semantic_matches" in r]
    old_format = [r for r in records if "semantic_matches" not in r]

    print(f"总记录: {len(records)}  新格式(有semantic_matches): {len(new_format)}  旧格式: {len(old_format)}")

    results = []

    # ── 新格式：有 semantic_matches，直接重算完整 GT ──────────────────────────
    for rec in new_format:
        qid     = rec["query_id"]
        q       = queries[qid]
        sem     = {k: set(v) for k, v in rec["semantic_matches"].items()}
        new_gt  = compute_gt_from_leaves(q, corpus_df, sem)
        old_gt  = set(q.get("gt_document_ids") or [])
        results.append({
            "query_id":    qid,
            "nl":          rec["nl"],
            "sql":         rec.get("sql", q["dsl"]["sql"]),
            "old_gt_size": len(old_gt),
            "fixed_gt_size": len(new_gt),
            "overlap":     len(set(new_gt) & old_gt),
            "format":      "new",
        })

    # ── 旧格式：只能算结构化 GT（下界），语义部分 ID 已丢失 ──────────────────
    for rec in old_format:
        qid    = rec["query_id"]
        q      = queries[qid]
        old_gt = set(q.get("gt_document_ids") or [])
        # 用空 semantic_matches，只计算结构化叶子
        struct_gt = compute_gt_from_leaves(q, corpus_df, {})
        results.append({
            "query_id":       qid,
            "nl":             rec["nl"],
            "sql":            q["dsl"]["sql"],
            "old_gt_size":    len(old_gt),
            "struct_gt_size": len(struct_gt),   # 结构化部分下界
            "gemini_size":    rec["new_gt_size"],  # Gemini 语义结果大小（ID 已丢失）
            "note":           "Gemini IDs lost, only struct GT recovered",
            "format":         "old",
        })

    # 写输出
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印汇总
    if new_format:
        print(f"\n=== 新格式修复 ({len(new_format)}条) ===")
        print(f"Avg old GT:   {np.mean([r['old_gt_size']   for r in results if r['format']=='new']):.1f}")
        print(f"Avg fixed GT: {np.mean([r['fixed_gt_size'] for r in results if r['format']=='new']):.1f}")
        print(f"Avg overlap:  {np.mean([r['overlap']       for r in results if r['format']=='new']):.1f}")

    if old_format:
        print(f"\n=== 旧格式（仅结构化GT下界，{len(old_format)}条）===")
        for r in [x for x in results if x["format"] == "old"]:
            print(f"  {r['query_id']}: struct_gt={r['struct_gt_size']}, "
                  f"gemini_semantic={r['gemini_size']}, old_gt={r['old_gt_size']}")
        print(f"\n→ 这{len(old_format)}条需用新代码重跑以获取完整 GT（含Gemini语义部分）")

    print(f"\n结果写入 {OUTPUT_PATH}")
