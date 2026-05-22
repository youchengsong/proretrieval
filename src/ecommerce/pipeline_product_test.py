#!/usr/bin/env python3
"""
电商数据集 Query 构造 Pipeline（Steps 5~8 全合并）

直接运行：
    python pipeline_product.py

内部流程：
    1. Steps 5-7 在内存中完成（生成骨架 → 填值 → 编译 DSL + superset_sql）
    2. 自动拉起 4 个子进程，每个独占一张 GPU，并行完成 Step 8
    3. 等待所有子进程完成后自动合并，输出最终文件

最终输出：
    /data/youchengsong/DSL-R1/data/ecommerce/queries_product_step8.jsonl
"""

import argparse, json, os, random, subprocess, sys
from collections import defaultdict
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/data/youchengsong/DSL-R1/data/ecommerce")
PRODUCTS_FILE  = DATA_DIR / "test/products_test.parquet"
TEXT_DESC_FILE = DATA_DIR / "test/text_descriptions_test.jsonl"
IMG_DESC_FILE  = DATA_DIR / "test/image_descriptions_test.jsonl"
TEXT_FEAT_FILE = DATA_DIR / "test/product_text_features_test.parquet"
IMG_FEAT_FILE  = DATA_DIR / "test/product_image_features_test.parquet"
QWEN_PATH      = "/data/youchengsong/models/Qwen3-Embedding-8B"
QWEN_VL_PATH   = "/data/youchengsong/models/Qwen3-VL-Embedding-8B"
PYTHON         = "/data/youchengsong/conda3/envs/DSL-R1/bin/python"

OUTPUT_FILE    = DATA_DIR / "test/queries_product_test_step8.jsonl"
TMP_STEP7_FILE = DATA_DIR / "test/_tmp_step7_test.jsonl"

GPUS          = [2, 4, 5, 6]
TOTAL_QUERIES = 3_000
SEED          = 42

# ── Step 5 参数 ───────────────────────────────────────────────────────────────
N_VALUES = [1, 2, 3, 4, 5]
N_PROBS  = [0.40, 0.20, 0.20, 0.15, 0.05]
CUT_PROB = {2: 0.30, 3: 0.40}
POS_PROB = {1: 1.0, 2: 0.90, 3: 0.80}   # 10/0, 9/1, 8/2

FIELDS = [
    {"name": "brand",          "modality": "structured", "ftype": "categorical", "weight": 2},
    {"name": "category",       "modality": "structured", "ftype": "categorical", "weight": 2},
    {"name": "price",          "modality": "structured", "ftype": "numerical",   "weight": 2},
    {"name": "average_rating", "modality": "structured", "ftype": "numerical",   "weight": 2},
    {"name": "rating_number",  "modality": "structured", "ftype": "numerical",   "weight": 1},
    {"name": "text",           "modality": "text",        "ftype": None,          "weight": 3},
    {"name": "image_url",      "modality": "image",       "ftype": None,          "weight": 2},
]
FIELD_NAMES   = [f["name"]   for f in FIELDS]
FIELD_WEIGHTS = [f["weight"] for f in FIELDS]
FIELD_META    = {f["name"]: f for f in FIELDS}

NUMERICAL_FIELDS   = {"price", "average_rating", "rating_number"}
CATEGORICAL_FIELDS = {"brand", "category"}
ROUND_DIGITS = {"price": 2, "average_rating": 1, "rating_number": 0}
CLAMP_RANGE  = {"price": (0.01, None), "average_rating": (1.0, 5.0), "rating_number": (1, None)}
VECTOR_MODALITIES  = {"text", "image"}
STRUCTURED_FIELDS  = {"brand", "category", "price", "average_rating", "rating_number"}
TOP_K       = 20
CORPUS_SIZE = 3000
MAX_GT      = CORPUS_SIZE // 2


# ── Step 5 函数 ───────────────────────────────────────────────────────────────
def get_level(n): return 1 if n == 1 else (2 if n <= 3 else 3)

def sample_groups(n, level):
    if n == 1: return [[0]], None, None
    p = CUT_PROB.get(level, 0.0)
    cuts = [i for i in range(1, n) if random.random() < p]
    groups, prev = [], 0
    for c in cuts:
        groups.append(list(range(prev, c))); prev = c
    groups.append(list(range(prev, n)))
    inner_op, outer_op = ("AND", "OR") if random.random() < 0.5 else ("OR", "AND")
    if len(groups) == 1: outer_op = None
    return groups, inner_op, outer_op

def sample_golds(groups, outer_op, all_ids, cat_index, id_to_cat):
    k = len(groups)
    if outer_op == "OR" and k > 1:
        first = random.choice(all_ids)
        pool  = [p for p in cat_index[id_to_cat[first]] if p != first]
        golds = [first]
        for _ in range(k - 1):
            if pool:
                g = random.choice(pool); pool = [p for p in pool if p != g]; golds.append(g)
            else:
                golds.append(random.choice([p for p in all_ids if p not in golds]))
        return golds
    return [random.choice(all_ids)]

def sample_fields(groups, outer_op):
    allow_cross = (outer_op == "OR")
    used_g = set(); leaf_fields = {}
    for group in groups:
        used_l = set()
        for idx in group:
            avail = FIELD_NAMES[:] if allow_cross else [f for f in FIELD_NAMES if f not in used_g]
            avail = [f for f in avail if f not in used_l]
            w     = [FIELD_WEIGHTS[FIELD_NAMES.index(f)] for f in avail]
            if not avail: avail = FIELD_NAMES[:]; w = FIELD_WEIGHTS[:]
            field = random.choices(avail, weights=w, k=1)[0]
            leaf_fields[idx] = field; used_l.add(field)
            if not allow_cross: used_g.add(field)
    return leaf_fields

def build_query(qid, all_ids, cat_index, id_to_cat):
    n = random.choices(N_VALUES, weights=N_PROBS, k=1)[0]
    level = get_level(n)
    groups, inner_op, outer_op = sample_groups(n, level)
    k = len(groups)
    gold_ids = sample_golds(groups, outer_op, all_ids, cat_index, id_to_cat)
    g2g = {i: gold_ids[i] for i in range(k)} if (outer_op == "OR" and k > 1) \
          else {i: gold_ids[0] for i in range(k)}
    lf = sample_fields(groups, outer_op)
    leaves = []
    for gi, group in enumerate(groups):
        for li in group:
            field = lf[li]
            pol = "pos" if random.random() < POS_PROB[level] else "neg"
            leaves.append({"leaf_id": li, "group": gi, "gold_id": g2g[gi],
                           "field": field, "modality": FIELD_META[field]["modality"],
                           "ftype": FIELD_META[field]["ftype"],
                           "polarity": pol, "is_pos": pol == "pos",
                           "value": None})
    return {"query_id": qid, "n": n, "level": level, "k": k,
            "inner_op": inner_op, "outer_op": outer_op,
            "groups": groups, "gold_ids": gold_ids, "leaves": leaves,
            "modalities": list({FIELD_META[l["field"]]["modality"] for l in leaves}),
            "dsl": None, "superset_sql": None,
            "natural_language_query": None, "gt_document_ids": None}


# ── Step 6 函数 ───────────────────────────────────────────────────────────────
def fill_categorical(gold_val, polarity, pool):
    if polarity == "pos": return {"op": "=", "val": gold_val}
    cands = [v for v in pool if v != gold_val]
    return {"op": "=", "val": random.choice(cands) if cands else gold_val}

def fill_numerical(gold_val, field, polarity):
    prec = int(ROUND_DIGITS[field])
    lo_c, hi_c = CLAMP_RANGE[field]
    def clamp(v):
        if lo_c is not None: v = max(v, lo_c)
        if hi_c is not None: v = min(v, hi_c)
        return v
    def fmt(v): return round(v, prec) if prec else int(v)
    if field == "average_rating":
        if polarity == "pos":
            return {"op": "=", "val": fmt(gold_val)}
        else:
            delta = random.choice([-0.1, 0.1])
            return {"op": "=", "val": fmt(clamp(gold_val + delta))}
    if polarity == "pos":
        if random.random() < 0.2: return {"op": "=", "val": fmt(gold_val)}
        lo = clamp(gold_val * 0.95)
        hi = clamp(gold_val * 1.05)
        if lo >= hi: hi = clamp(lo * 1.01 + 0.01)
        return {"lo": fmt(lo), "hi": fmt(hi)}
    else:
        if random.random() < 0.5:
            lo = clamp(gold_val * 1.01)
            hi = clamp(gold_val * 1.05)
        else:
            lo = clamp(gold_val * 0.95)
            hi = clamp(gold_val * 0.99)
        if lo >= hi: hi = clamp(lo * 1.01 + 0.01)
        return {"lo": fmt(lo), "hi": fmt(hi)}


# ── Step 7 函数 ───────────────────────────────────────────────────────────────
def _range_sql(field, value):
    if "lo" in value:
        return f"({field} >= {value['lo']} AND {field} <= {value['hi']})"
    v = f"'{value['val']}'" if isinstance(value["val"], str) else str(value["val"])
    return f"{field} {value['op']} {v}"

def compile_leaf(leaf, tc, ic, ret_list, text_pos, img_pos):
    field, pol, value, gid = leaf["field"], leaf["polarity"], leaf["value"], leaf["gold_id"]
    if field in STRUCTURED_FIELDS:
        return _range_sql(field, value) if value is not None else None
    elif field == "text":
        q = text_pos.get(gid)
        if not q: return None
        idx = tc[0]; tc[0] += 1
        ret_list.append({"type": "text", "field": "text", "query": q})
        ph = f"<text_{idx}>"
        return f"product_id IN {ph}" if pol == "pos" else f"product_id NOT IN {ph}"
    elif field == "image_url":
        q = img_pos.get(gid)
        if not q: return None
        idx = ic[0]; ic[0] += 1
        ret_list.append({"type": "image", "field": "image_url", "query": q})
        ph = f"<image_{idx}>"
        return f"product_id IN {ph}" if pol == "pos" else f"product_id NOT IN {ph}"
    return None

def assemble_sql(query, leaf_conds):
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    if k == 1:
        conds = [leaf_conds[i] for g in groups for i in g if leaf_conds.get(i)]
        return f" {inner_op} ".join(conds) if conds else ""
    gi_op = "AND" if outer_op == "OR" else "OR"
    clauses = []
    for group in groups:
        conds = [leaf_conds[i] for i in group if leaf_conds.get(i)]
        if not conds: continue
        clauses.append(conds[0] if len(conds) == 1 else f"({f' {gi_op} '.join(conds)})")
    return f" {outer_op} ".join(clauses) if clauses else ""

def leaf_to_sup_sql(leaf):
    field, value = leaf["field"], leaf["value"]
    if "lo" in value:
        return f"({field} >= {value['lo']} AND {field} <= {value['hi']})"
    v = f"'{value['val']}'" if leaf["ftype"] == "categorical" else str(value["val"])
    return f"{field} {value['op']} {v}"

def compute_superset_sql(query):
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    by_id     = {l["leaf_id"]: l for l in query["leaves"]}
    if k == 1:
        gl = [by_id[i] for i in groups[0]]
        st = [l for l in gl if l["modality"] not in VECTOR_MODALITIES]
        if inner_op == "OR" and any(l["modality"] in VECTOR_MODALITIES for l in gl): return None
        if not st: return None
        return f"WHERE {f' {inner_op} '.join(leaf_to_sup_sql(l) for l in st)}"
    if outer_op == "OR":
        clauses = []
        for g in groups:
            gl = [by_id[i] for i in g]
            st = [l for l in gl if l["modality"] not in VECTOR_MODALITIES]
            if not st: return None
            conds = [leaf_to_sup_sql(l) for l in st]
            clauses.append(f"({' AND '.join(conds)})" if len(conds) > 1 else conds[0])
        return f"WHERE {' OR '.join(clauses)}"
    else:
        clauses = []
        for g in groups:
            gl = [by_id[i] for i in g]
            if any(l["modality"] in VECTOR_MODALITIES for l in gl): continue
            conds = [leaf_to_sup_sql(l) for l in gl]
            clauses.append(f"({' OR '.join(conds)})" if len(conds) > 1 else conds[0])
        return f"WHERE {' AND '.join(clauses)}" if clauses else None


# ── Step 8 函数（shard worker）────────────────────────────────────────────────
def struct_mask(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field not in STRUCTURED_FIELDS: return None
    col = df[field]
    if "lo" in value: return (col >= value["lo"]) & (col <= value["hi"])
    op, val = value["op"], value["val"]
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val

def apply_superset(query, df):
    if query["superset_sql"] is None: return df
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    leaves    = {l["leaf_id"]: l for l in query["leaves"]}
    def gmasks(group):
        return [m for lid in group if leaves[lid]["modality"] not in VECTOR_MODALITIES
                for m in [struct_mask(leaves[lid], df)] if m is not None]
    if k == 1:
        ms = gmasks(groups[0])
        if not ms: return df
        r = ms[0]
        for m in ms[1:]: r = r & m if inner_op == "AND" else r | m
        return df[r]
    elif outer_op == "OR":
        gms = []
        for g in groups:
            ms = gmasks(g)
            if not ms: return df
            gm = ms[0]
            for m in ms[1:]: gm = gm & m
            gms.append(gm)
        r = gms[0]
        for m in gms[1:]: r = r | m
        return df[r]
    else:
        pure = []
        for g in groups:
            if any(leaves[lid]["modality"] in VECTOR_MODALITIES for lid in g): continue
            ms = [struct_mask(leaves[lid], df) for lid in g]
            ms = [m for m in ms if m is not None]
            if not ms: continue
            gm = ms[0]
            for m in ms[1:]: gm = gm | m
            pure.append(gm)
        if not pure: return df
        r = pure[0]
        for m in pure[1:]: r = r & m
        return df[r]

def eval_dsl(query, df, text_sets, image_sets):
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    leaves    = {l["leaf_id"]: l for l in query["leaves"]}
    ti = ii = 0; ph = {}
    for lid in sorted(leaves):
        mod = leaves[lid]["modality"]
        if mod == "text":    ph[lid] = ("text",  ti); ti += 1
        elif mod == "image": ph[lid] = ("image", ii); ii += 1
    def lmask(lid):
        leaf = leaves[lid]
        if leaf["modality"] == "text":
            s = df["product_id"].isin(text_sets[ph[lid][1]])
            return s if leaf["polarity"] == "pos" else ~s
        elif leaf["modality"] == "image":
            s = df["product_id"].isin(image_sets[ph[lid][1]])
            return s if leaf["polarity"] == "pos" else ~s
        return struct_mask(leaf, df)
    def combine(masks, op):
        r = masks[0]
        for m in masks[1:]: r = r & m if op == "AND" else r | m
        return r
    if k == 1:             result = combine([lmask(l) for l in groups[0]], inner_op)
    elif outer_op == "OR": result = combine([combine([lmask(l) for l in g], "AND") for g in groups], "OR")
    else:                  result = combine([combine([lmask(l) for l in g], "OR")  for g in groups], "AND")
    return df[result]["product_id"].tolist()

def run_shard(shard, total):
    import numpy as np, torch, pandas as pd
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"[shard {shard}] Loading data...", flush=True)
    df    = pd.read_parquet(PRODUCTS_FILE).reset_index(drop=True)
    tf    = pd.read_parquet(TEXT_FEAT_FILE)
    imf   = pd.read_parquet(IMG_FEAT_FILE)
    t_ids = tf["product_id"].values
    t_mat = np.stack(tf["text_features"].values).astype(np.float32)
    i_ids = imf["product_id"].values
    i_mat = np.stack(imf["image_features"].values).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shard {shard}] device={device}", flush=True)

    with open(TMP_STEP7_FILE) as f:
        queries = [json.loads(l) for l in f][shard::total]
    print(f"[shard {shard}] {len(queries)} queries", flush=True)

    uq_t = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["type"] == "text"})
    uq_i = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["type"] == "image"})
    print(f"[shard {shard}] encoding {len(uq_t)} text + {len(uq_i)} image...", flush=True)

    # ── 阶段1：用 Qwen3-Embedding 编码文本查询 ──────────────────────────────────
    qwen = SentenceTransformer(QWEN_PATH, device=str(device)); qwen.max_seq_length = 128
    te = qwen.encode(uq_t, batch_size=32, normalize_embeddings=True, show_progress_bar=False) \
         if uq_t else np.empty((0, 4096))
    del qwen; torch.cuda.empty_cache()

    # ── 阶段2：用 Qwen3-VL-Embedding 编码图像查询（文本输入→图像embedding空间） ─
    ie = np.empty((0, 4096))
    if uq_i:
        vl_model = AutoModel.from_pretrained(
            QWEN_VL_PATH,
            dtype=torch.float16,
            device_map=str(device),
        ).eval()
        vl_processor = AutoProcessor.from_pretrained(QWEN_VL_PATH)

        ie_list = []
        for i in range(0, len(uq_i), 32):
            batch_texts = uq_i[i:i+32]
            # 纯文本输入：描述图像内容的query
            messages_batch = [
                [{"role": "user", "content": [{"type": "text", "text": t}]}]
                for t in batch_texts
            ]
            texts = [
                vl_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in messages_batch
            ]
            inputs = vl_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                outputs = vl_model(**inputs, output_hidden_states=True)
                hidden  = outputs.hidden_states[-1]                          # (B, L, D)
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1           # (B,)
                embs = hidden[torch.arange(len(batch_texts), device=device), seq_lens]  # (B, D)
                embs = F.normalize(embs, p=2, dim=-1).float().cpu().numpy()
            ie_list.append(embs)
        ie = np.vstack(ie_list)
        del vl_model, vl_processor; torch.cuda.empty_cache()

    tc = {q: te[i] for i, q in enumerate(uq_t)}
    ic = {q: ie[i] for i, q in enumerate(uq_i)}

    def topk(vec, mat, k):
        v = vec / (np.linalg.norm(vec) + 1e-9); s = mat @ v
        idx = np.argpartition(s, -k)[-k:]; return idx[np.argsort(s[idx])[::-1]]

    stats = {"ok": 0, "empty": 0, "too_large": 0}; results = []
    for q in queries:
        ret = q["dsl"]["retrieval_list"]
        # Step 1: 先用结构化条件缩小语料库
        sup_df = apply_superset(q, df)
        sup_pids = set(sup_df["product_id"])
        # Step 2: 在超集内做 embedding top-k
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
        # Step 3: 填回占位符，在完整语料上执行
        gt  = eval_dsl(q, df, ts, is_)
        if   len(gt) == 0:     q["gt_document_ids"] = None; stats["empty"]     += 1
        elif len(gt) > MAX_GT: q["gt_document_ids"] = None; stats["too_large"] += 1
        else:                  q["gt_document_ids"] = gt;   stats["ok"]        += 1
        results.append(q)

    with open(DATA_DIR / f"_shard{shard}.jsonl", "w") as f:
        for q in results: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[shard {shard}] ok={stats['ok']} empty={stats['empty']} too_large={stats['too_large']}", flush=True)


# ── 主入口 ────────────────────────────────────────────────────────────────────
def main():
    import pandas as pd
    total = len(GPUS)
    random.seed(SEED)

    print("=== Steps 5-7: 生成 queries ===")
    df_prod   = pd.read_parquet(PRODUCTS_FILE)
    all_ids   = df_prod["product_id"].tolist()
    cat_index = defaultdict(list)
    for _, row in df_prod.iterrows(): cat_index[row["category"]].append(row["product_id"])
    id_to_cat = dict(zip(df_prod["product_id"], df_prod["category"]))
    products  = df_prod.set_index("product_id")[
        ["brand","category","price","average_rating","rating_number"]].to_dict("index")

    cat_pools = {"brand": set(), "category": set()}
    for p in products.values():
        for fld in cat_pools:
            v = p.get(fld)
            if v and str(v).strip(): cat_pools[fld].add(str(v))
    cat_pools = {k: sorted(v) for k, v in cat_pools.items()}

    text_lookup, img_lookup = {}, {}
    with open(TEXT_DESC_FILE) as f:
        for line in f: o = json.loads(line); text_lookup[o["product_id"]] = o
    with open(IMG_DESC_FILE) as f:
        for line in f: o = json.loads(line); img_lookup[o["product_id"]]  = o
    text_pos = {pid: o.get("text_pos")  for pid, o in text_lookup.items()}
    img_pos  = {pid: o.get("image_pos") for pid, o in img_lookup.items()}

    print(f"Generating {TOTAL_QUERIES} queries...")
    queries = []
    for i in range(TOTAL_QUERIES):
        q = build_query(f"q_{i+1:05d}", all_ids, cat_index, id_to_cat)
        for leaf in q["leaves"]:
            gid, field, pol = leaf["gold_id"], leaf["field"], leaf["polarity"]
            prod = products.get(gid)
            if prod is None: leaf["value"] = None; continue
            if field in CATEGORICAL_FIELDS:
                gv = prod.get(field)
                leaf["value"] = fill_categorical(str(gv), pol, cat_pools[field]) if gv else None
            elif field in NUMERICAL_FIELDS:
                gv = prod.get(field)
                leaf["value"] = fill_numerical(float(gv), field, pol) if gv else None
            elif field == "text":
                leaf["value"] = {"val": text_lookup.get(gid, {}).get("text_pos" if pol=="pos" else "text_neg")}
            elif field == "image_url":
                leaf["value"] = {"val": img_lookup.get(gid, {}).get("image_pos" if pol=="pos" else "image_neg")}
            else:
                leaf["value"] = None
        tc = [0]; ic = [0]; ret_list = []
        lc = {leaf["leaf_id"]: compile_leaf(leaf, tc, ic, ret_list, text_pos, img_pos)
              for leaf in q["leaves"]}
        body = assemble_sql(q, lc)
        q["dsl"]          = {"sql": f"WHERE {body}" if body else "", "retrieval_list": ret_list}
        q["superset_sql"] = compute_superset_sql(q)
        queries.append(q)

    with open(TMP_STEP7_FILE, "w") as f:
        for q in queries: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Steps 5-7 done → {TMP_STEP7_FILE}")

    print(f"\n=== Step 8: GPU {GPUS} ===")
    procs = []
    for shard, gpu in enumerate(GPUS):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.Popen([PYTHON, __file__, "--_shard", str(shard), "--_total", str(total)], env=env)
        procs.append(p); print(f"  shard {shard} → GPU {gpu} (pid={p.pid})")
    for p in procs: p.wait()
    print("All shards done.")

    print(f"\n=== Merging → {OUTPUT_FILE} ===")
    merged = [None] * len(queries)
    for shard in range(total):
        with open(DATA_DIR / f"_shard{shard}.jsonl") as f:
            for local_idx, line in enumerate(f):
                merged[local_idx * total + shard] = json.loads(line)
    with open(OUTPUT_FILE, "w") as f:
        for q in merged: f.write(json.dumps(q, ensure_ascii=False) + "\n")

    total_q = len(merged); ok = sum(1 for q in merged if q["gt_document_ids"] is not None)
    print(f"Done. {total_q} queries → {OUTPUT_FILE}")
    print(f"  GT found : {ok} ({ok/total_q:.1%})  Invalid : {total_q-ok} ({(total_q-ok)/total_q:.1%})")

    TMP_STEP7_FILE.unlink(missing_ok=True)
    for shard in range(total): (DATA_DIR / f"_shard{shard}.jsonl").unlink(missing_ok=True)
    print("Temp files cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_shard", type=int, default=None)
    parser.add_argument("--_total", type=int, default=len(GPUS))
    args, _ = parser.parse_known_args()
    if args._shard is not None:
        run_shard(args._shard, args._total)
    else:
        main()
