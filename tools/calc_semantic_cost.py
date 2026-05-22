#!/usr/bin/env python3
"""
计算测试集评估时的实际语义比对次数。

逻辑：
  1. 对每条 query，用结构化字段过滤语料库，得到超集大小 S
     - superset_sql 不为 None → 执行结构化过滤，S = 过滤后行数
     - superset_sql 为 None   → S = 全量语料库大小
  2. 该 query 的语义比对次数 = S × num_text + S × num_image
  3. 汇总所有 query
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_CORPUS  = Path("/data/youchengsong/DSL-R1/data/email/test/email_test.parquet")
EMAIL_QUERIES = Path("/data/youchengsong/DSL-R1/data/email/test/queries_email_test_step9.jsonl")

EMAIL_STRUCTURED = {"From", "To", "Date"}   # Date → year column
EMAIL_VECTOR     = {"text"}


def email_struct_mask(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field not in EMAIL_STRUCTURED:
        return None
    if field == "Date":
        col = df["year"]
    else:
        col = df[field]
    op, val = value["op"], value["val"]
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val
    return None


def email_superset_size(query, df):
    """返回该 query 在 email 语料库上的超集大小。"""
    if query.get("superset_sql") is None:
        return len(df)

    k        = query["k"]
    groups   = query["groups"]
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")
    leaves   = {l["leaf_id"]: l for l in query["leaves"]}

    def group_mask(group):
        masks = [email_struct_mask(leaves[lid], df)
                 for lid in group
                 if leaves[lid]["modality"] not in EMAIL_VECTOR]
        return [m for m in masks if m is not None]

    if k == 1:
        ms = group_mask(groups[0])
        if not ms:
            return len(df)
        r = ms[0]
        for m in ms[1:]:
            r = r & m if inner_op == "AND" else r | m
        return int(r.sum())

    elif outer_op == "OR":
        gms = []
        for g in groups:
            ms = group_mask(g)
            if not ms:
                return len(df)          # 某组无结构化 → 退化为全量
            gm = ms[0]
            for m in ms[1:]:
                gm = gm & m
            gms.append(gm)
        r = gms[0]
        for m in gms[1:]:
            r = r | m
        return int(r.sum())

    else:  # outer AND
        pure_masks = []
        for g in groups:
            if any(leaves[lid]["modality"] in EMAIL_VECTOR for lid in g):
                continue
            ms = [email_struct_mask(leaves[lid], df) for lid in g]
            ms = [m for m in ms if m is not None]
            if not ms:
                continue
            gm = ms[0]
            for m in ms[1:]:
                gm = gm | m
            pure_masks.append(gm)
        if not pure_masks:
            return len(df)
        r = pure_masks[0]
        for m in pure_masks[1:]:
            r = r & m
        return int(r.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Ecommerce
# ─────────────────────────────────────────────────────────────────────────────
PROD_CORPUS  = Path("/data/youchengsong/DSL-R1/data/ecommerce/test/products_test.parquet")
PROD_QUERIES = Path("/data/youchengsong/DSL-R1/data/ecommerce/test/queries_product_test_step9.jsonl")

PROD_STRUCTURED = {"brand", "category", "price", "average_rating", "rating_number"}
PROD_VECTOR     = {"text", "image"}


def prod_struct_mask(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field not in PROD_STRUCTURED:
        return None
    col = df[field]
    lo, hi = value.get("lo"), value.get("hi")
    if lo is not None and hi is not None:
        return (col >= lo) & (col <= hi)
    op, val = value.get("op"), value.get("val")
    if op is None:
        return None
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val
    return None


def prod_superset_size(query, df):
    if query.get("superset_sql") is None:
        return len(df)

    k        = query["k"]
    groups   = query["groups"]
    inner_op = query.get("inner_op") or "AND"
    outer_op = query.get("outer_op")
    leaves   = {l["leaf_id"]: l for l in query["leaves"]}

    def group_mask(group):
        masks = [prod_struct_mask(leaves[lid], df)
                 for lid in group
                 if leaves[lid]["modality"] not in PROD_VECTOR]
        return [m for m in masks if m is not None]

    if k == 1:
        ms = group_mask(groups[0])
        if not ms:
            return len(df)
        r = ms[0]
        for m in ms[1:]:
            r = r & m if inner_op == "AND" else r | m
        return int(r.sum())

    elif outer_op == "OR":
        gms = []
        for g in groups:
            ms = group_mask(g)
            if not ms:
                return len(df)
            gm = ms[0]
            for m in ms[1:]:
                gm = gm & m
            gms.append(gm)
        r = gms[0]
        for m in gms[1:]:
            r = r | m
        return int(r.sum())

    else:  # outer AND
        pure_masks = []
        for g in groups:
            if any(leaves[lid]["modality"] in PROD_VECTOR for lid in g):
                continue
            ms = [prod_struct_mask(leaves[lid], df) for lid in g]
            ms = [m for m in ms if m is not None]
            if not ms:
                continue
            gm = ms[0]
            for m in ms[1:]:
                gm = gm | m
            pure_masks.append(gm)
        if not pure_masks:
            return len(df)
        r = pure_masks[0]
        for m in pure_masks[1:]:
            r = r & m
        return int(r.sum())


# ─────────────────────────────────────────────────────────────────────────────
# 统计函数
# ─────────────────────────────────────────────────────────────────────────────
def analyze(name, corpus_path, queries_path, superset_fn):
    df = pd.read_parquet(corpus_path)

    # 补充 year 列（email 用）
    if "Date" in df.columns:
        df["year"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.year

    corpus_size = len(df)
    print(f"\n{'='*60}")
    print(f"{name}  corpus={corpus_size}")
    print(f"{'='*60}")

    total_text_cmp  = 0
    total_image_cmp = 0
    superset_sizes  = []
    pure_struct_count = 0   # 不需要任何语义检索的 query 数

    with open(queries_path) as f:
        queries = [json.loads(l) for l in f]

    for q in queries:
        rl = q["dsl"].get("retrieval_list", [])
        n_text  = sum(1 for r in rl if r["type"] == "text")
        n_image = sum(1 for r in rl if r["type"] == "image")

        if n_text == 0 and n_image == 0:
            pure_struct_count += 1
            superset_sizes.append(0)
            continue

        s = superset_fn(q, df)
        superset_sizes.append(s)
        total_text_cmp  += s * n_text
        total_image_cmp += s * n_image

    sizes = [s for s in superset_sizes if s > 0]
    print(f"总 query 数:          {len(queries)}")
    print(f"纯结构化(无语义):     {pure_struct_count}")
    print(f"需语义检索 query 数:  {len(queries) - pure_struct_count}")
    print(f"超集大小统计(有语义检索的):")
    print(f"  min={min(sizes)}  max={max(sizes)}  avg={np.mean(sizes):.1f}  median={np.median(sizes):.1f}")
    print(f"总文本语义比对次数:   {total_text_cmp:,}")
    print(f"总图像语义比对次数:   {total_image_cmp:,}")
    print(f"总语义比对次数:       {total_text_cmp + total_image_cmp:,}")
    print(f"(对比全量比对: text={len(queries)*corpus_size:,}, 超集节省={1 - total_text_cmp/(max(1,len(queries)*corpus_size)):.1%})")


if __name__ == "__main__":
    analyze("Email Test",     EMAIL_CORPUS, EMAIL_QUERIES, email_superset_size)
    analyze("Ecommerce Test", PROD_CORPUS,  PROD_QUERIES,  prod_superset_size)
