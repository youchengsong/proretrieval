#!/usr/bin/env python3
"""
邮件数据集 Query 构造 Pipeline（Steps 5~8 全合并）

直接运行：
    python pipeline_email.py

内部流程：
    1. Steps 5-7 在内存中完成
    2. 自动拉起 4 个子进程，每个独占一张 GPU，并行完成 Step 8
    3. 等待所有子进程完成后自动合并，输出最终文件

最终输出：
    /data/youchengsong/DSL-R1/data/email/queries_email_step8.jsonl
"""

import argparse, json, os, random, subprocess, sys
from collections import defaultdict
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/data/youchengsong/DSL-R1/data/email")
EMAIL_FILE     = DATA_DIR / "email_train.parquet"
DESC_FILE      = DATA_DIR / "email_descriptions_train.jsonl"
SUBJ_FEAT_FILE = DATA_DIR / "train_subject_features.parquet"
MSG_FEAT_FILE  = DATA_DIR / "train_message_features.parquet"
QWEN_PATH      = "/data/youchengsong/models/Qwen3-Embedding-8B"
PYTHON         = "/data/youchengsong/conda3/envs/DSL-R1/bin/python"

OUTPUT_FILE    = DATA_DIR / "queries_email_train_step8.jsonl"
TMP_STEP7_FILE = DATA_DIR / "_tmp_email_step7_train.jsonl"

GPUS          = [2, 4, 5, 6]
TOTAL_QUERIES = 20_000
SEED          = 42
DATE_CLAMP    = (1998, 2003)

# ── Step 5 参数 ───────────────────────────────────────────────────────────────
N_VALUES = [1, 2, 3, 4, 5]
N_PROBS  = [0.40, 0.20, 0.20, 0.15, 0.05]
CUT_PROB = {2: 0.30, 3: 0.40}
POS_PROB = {1: 1.0, 2: 0.90, 3: 0.80}   # 10/0, 9/1, 8/2

FIELDS = [
    {"name": "From",    "modality": "structured", "ftype": "categorical", "weight": 2},
    {"name": "To",      "modality": "structured", "ftype": "categorical", "weight": 2},
    {"name": "Date",    "modality": "structured", "ftype": "numerical",   "weight": 1},
    {"name": "Subject", "modality": "text",        "ftype": None,          "weight": 2},
    {"name": "Message", "modality": "text",        "ftype": None,          "weight": 3},
]
FIELD_NAMES   = [f["name"]   for f in FIELDS]
FIELD_WEIGHTS = [f["weight"] for f in FIELDS]
FIELD_META    = {f["name"]: f for f in FIELDS}

VECTOR_MODALITIES = {"text"}
STRUCTURED_FIELDS = {"From", "To", "Date"}
CORPUS_SIZE = 10_000
MAX_GT      = CORPUS_SIZE // 2
TOP_K       = 20


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

def sample_golds(groups, outer_op, all_ids, from_index):
    k = len(groups)
    if outer_op == "OR" and k > 1:
        candidates = [ids for ids in from_index.values() if len(ids) >= k]
        if candidates:
            pool = random.choice(candidates); return random.sample(pool, k)
        return random.sample(all_ids, min(k, len(all_ids)))
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

def build_query(qid, all_ids, from_index):
    n = random.choices(N_VALUES, weights=N_PROBS, k=1)[0]
    level = get_level(n)
    groups, inner_op, outer_op = sample_groups(n, level)
    k = len(groups)
    gold_ids = sample_golds(groups, outer_op, all_ids, from_index)
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

def fill_date(gold_year, polarity):
    lo, hi = DATE_CLAMP
    if polarity == "pos":
        val = gold_year
    else:
        val = gold_year + random.choice([-1, 1])
    return {"op": "=", "val": int(max(lo, min(hi, val)))}


# ── Step 7 函数 ───────────────────────────────────────────────────────────────
def compile_leaf_email(leaf, tc, ret_list, pos_lookup):
    field, pol, value, gid = leaf["field"], leaf["polarity"], leaf["value"], leaf["gold_id"]
    if field in ("From", "To"):
        return f'"{field}" {value["op"]} \'{value["val"]}\'' if value else None
    elif field == "Date":
        return f"year {value['op']} {value['val']}" if value else None
    elif field == "Subject":
        q = pos_lookup.get(gid, {}).get("subject_pos")
        if not q: return None
        idx = tc[0]; tc[0] += 1
        ret_list.append({"type": "text", "field": "Subject", "query": q})
        ph = f"<text_{idx}>"
        return f"message_id IN {ph}" if pol == "pos" else f"message_id NOT IN {ph}"
    elif field == "Message":
        q = pos_lookup.get(gid, {}).get("message_pos")
        if not q: return None
        idx = tc[0]; tc[0] += 1
        ret_list.append({"type": "text", "field": "Message", "query": q})
        ph = f"<text_{idx}>"
        return f"message_id IN {ph}" if pol == "pos" else f"message_id NOT IN {ph}"
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

def leaf_to_sup_sql_email(leaf):
    field, value = leaf["field"], leaf["value"]
    if field in ("From", "To"): return f'"{field}" {value["op"]} \'{value["val"]}\''
    return f"year {value['op']} {value['val']}"

def compute_superset_sql_email(query):
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    by_id     = {l["leaf_id"]: l for l in query["leaves"]}
    if k == 1:
        gl = [by_id[i] for i in groups[0]]
        st = [l for l in gl if l["modality"] not in VECTOR_MODALITIES]
        if inner_op == "OR" and any(l["modality"] in VECTOR_MODALITIES for l in gl): return None
        if not st: return None
        return f"WHERE {f' {inner_op} '.join(leaf_to_sup_sql_email(l) for l in st)}"
    if outer_op == "OR":
        clauses = []
        for g in groups:
            gl = [by_id[i] for i in g]
            st = [l for l in gl if l["modality"] not in VECTOR_MODALITIES]
            if not st: return None
            conds = [leaf_to_sup_sql_email(l) for l in st]
            clauses.append(f"({' AND '.join(conds)})" if len(conds) > 1 else conds[0])
        return f"WHERE {' OR '.join(clauses)}"
    else:
        clauses = []
        for g in groups:
            gl = [by_id[i] for i in g]
            if any(l["modality"] in VECTOR_MODALITIES for l in gl): continue
            conds = [leaf_to_sup_sql_email(l) for l in gl]
            clauses.append(f"({' OR '.join(conds)})" if len(conds) > 1 else conds[0])
        return f"WHERE {' AND '.join(clauses)}" if clauses else None


# ── Step 8 函数（shard worker）────────────────────────────────────────────────
def struct_mask_email(leaf, df):
    field, value = leaf["field"], leaf["value"]
    if field not in STRUCTURED_FIELDS: return None
    col = df["year"] if field == "Date" else df[field]
    op, val = value["op"], value["val"]
    if op == "=":  return col == val
    if op == "!=": return col != val
    if op == ">":  return col > val
    if op == ">=": return col >= val
    if op == "<":  return col < val
    if op == "<=": return col <= val

def apply_superset_email(query, df):
    if query["superset_sql"] is None: return df
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    leaves    = {l["leaf_id"]: l for l in query["leaves"]}
    def gmasks(group):
        return [m for lid in group if leaves[lid]["modality"] not in VECTOR_MODALITIES
                for m in [struct_mask_email(leaves[lid], df)] if m is not None]
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
            ms = [struct_mask_email(leaves[lid], df) for lid in g]
            ms = [m for m in ms if m is not None]
            if not ms: continue
            gm = ms[0]
            for m in ms[1:]: gm = gm | m
            pure.append(gm)
        if not pure: return df
        r = pure[0]
        for m in pure[1:]: r = r & m
        return df[r]

def eval_dsl_email(query, df, text_sets):
    k, groups = query["k"], query["groups"]
    inner_op  = query["inner_op"] or "AND"
    outer_op  = query["outer_op"]
    leaves    = {l["leaf_id"]: l for l in query["leaves"]}
    ti = 0; ph = {}
    for lid in sorted(leaves):
        if leaves[lid]["modality"] == "text":
            ph[lid] = ti; ti += 1
    def lmask(lid):
        leaf = leaves[lid]
        if leaf["modality"] == "text":
            s = df["Message_ID"].isin(text_sets[ph[lid]])
            return s if leaf["polarity"] == "pos" else ~s
        return struct_mask_email(leaf, df)
    def combine(masks, op):
        r = masks[0]
        for m in masks[1:]: r = r & m if op == "AND" else r | m
        return r
    if k == 1:             result = combine([lmask(l) for l in groups[0]], inner_op)
    elif outer_op == "OR": result = combine([combine([lmask(l) for l in g], "AND") for g in groups], "OR")
    else:                  result = combine([combine([lmask(l) for l in g], "OR")  for g in groups], "AND")
    return df[result]["Message_ID"].tolist()

def run_shard(shard, total):
    import numpy as np, torch, pandas as pd
    from sentence_transformers import SentenceTransformer

    print(f"[shard {shard}] Loading...", flush=True)
    df = pd.read_parquet(EMAIL_FILE).reset_index(drop=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.year

    sf = pd.read_parquet(SUBJ_FEAT_FILE)
    mf = pd.read_parquet(MSG_FEAT_FILE)
    subj_ids = sf["message_id"].values
    subj_mat = np.stack(sf["text_features"].values).astype(np.float32)
    msg_ids  = mf["message_id"].values
    msg_mat  = np.stack(mf["text_features"].values).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shard {shard}] device={device}", flush=True)
    qwen = SentenceTransformer(QWEN_PATH, device=str(device)); qwen.max_seq_length = 128

    with open(TMP_STEP7_FILE) as f:
        queries = [json.loads(l) for l in f][shard::total]
    print(f"[shard {shard}] {len(queries)} queries", flush=True)

    uq_subj = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["field"] == "Subject"})
    uq_msg  = list({r["query"] for q in queries for r in q["dsl"]["retrieval_list"] if r["field"] == "Message"})
    print(f"[shard {shard}] encoding {len(uq_subj)} subject + {len(uq_msg)} message...", flush=True)

    se = qwen.encode(uq_subj, batch_size=32, normalize_embeddings=True, show_progress_bar=False) \
         if uq_subj else np.empty((0, 4096))
    me = qwen.encode(uq_msg,  batch_size=32, normalize_embeddings=True, show_progress_bar=False) \
         if uq_msg  else np.empty((0, 4096))
    sc = {q: se[i] for i, q in enumerate(uq_subj)}
    mc = {q: me[i] for i, q in enumerate(uq_msg)}
    del qwen; torch.cuda.empty_cache()

    def topk(vec, mat, k):
        v = vec / (np.linalg.norm(vec) + 1e-9); s = mat @ v
        idx = np.argpartition(s, -k)[-k:]; return idx[np.argsort(s[idx])[::-1]]

    stats = {"ok": 0, "empty": 0, "too_large": 0}; results = []
    for q in queries:
        # Step 1: 先用结构化条件缩小语料库
        sup_df = apply_superset_email(q, df)
        sup_mids = set(sup_df["Message_ID"])
        sup_subj_idx = np.array([i for i, mid in enumerate(subj_ids) if mid in sup_mids])
        sup_msg_idx  = np.array([i for i, mid in enumerate(msg_ids)  if mid in sup_mids])
        # Step 2: 在超集内做 embedding top-k
        text_sets = []
        for r in q["dsl"]["retrieval_list"]:
            if r["field"] == "Subject" and len(sup_subj_idx) > 0:
                k_ = min(TOP_K, len(sup_subj_idx))
                text_sets.append(set(subj_ids[sup_subj_idx[topk(sc[r["query"]], subj_mat[sup_subj_idx], k_)]]))
            elif r["field"] == "Subject":
                text_sets.append(set())
            elif len(sup_msg_idx) > 0:
                k_ = min(TOP_K, len(sup_msg_idx))
                text_sets.append(set(msg_ids[sup_msg_idx[topk(mc[r["query"]], msg_mat[sup_msg_idx], k_)]]))
            else:
                text_sets.append(set())
        # Step 3: 填回占位符，在完整语料上执行
        gt = eval_dsl_email(q, df, text_sets)
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
    df_email  = pd.read_parquet(EMAIL_FILE, columns=["Message_ID", "From", "To", "Date"])
    df_email["year"] = pd.to_datetime(df_email["Date"], errors="coerce", utc=True).dt.year
    all_ids   = df_email["Message_ID"].tolist()
    from_index = defaultdict(list)
    for _, row in df_email.iterrows():
        if row["From"]: from_index[row["From"]].append(row["Message_ID"])
    emails = df_email.set_index("Message_ID")[["From","To","year"]].to_dict("index")

    from_pool = sorted({str(v["From"]) for v in emails.values() if v.get("From")})
    to_pool   = sorted({str(v["To"])   for v in emails.values() if v.get("To")})
    cat_pools = {"From": from_pool, "To": to_pool}

    desc_lookup = {}
    with open(DESC_FILE) as f:
        for line in f: o = json.loads(line); desc_lookup[o["message_id"]] = o
    pos_lookup = {mid: {"subject_pos": o.get("subject_pos"), "message_pos": o.get("message_pos")}
                  for mid, o in desc_lookup.items()}

    print(f"Generating {TOTAL_QUERIES} queries...")
    queries = []
    for i in range(TOTAL_QUERIES):
        q = build_query(f"qe_{i+1:05d}", all_ids, from_index)
        for leaf in q["leaves"]:
            gid, field, pol = leaf["gold_id"], leaf["field"], leaf["polarity"]
            email = emails.get(gid)
            if email is None: leaf["value"] = None; continue
            if field in ("From", "To"):
                gv = email.get(field)
                leaf["value"] = fill_categorical(str(gv), pol, cat_pools[field]) if gv else None
            elif field == "Date":
                yr = email.get("year")
                try:
                    leaf["value"] = fill_date(int(yr), pol) if yr and yr == yr else None
                except: leaf["value"] = None
            elif field in ("Subject", "Message"):
                key = ("subject_pos" if field=="Subject" else "message_pos") if pol=="pos" \
                      else ("subject_neg" if field=="Subject" else "message_neg")
                leaf["value"] = {"val": desc_lookup.get(gid, {}).get(key)}
            else:
                leaf["value"] = None
        tc = [0]; ret_list = []
        lc = {leaf["leaf_id"]: compile_leaf_email(leaf, tc, ret_list, pos_lookup)
              for leaf in q["leaves"]}
        body = assemble_sql(q, lc)
        q["dsl"]          = {"sql": f"WHERE {body}" if body else "", "retrieval_list": ret_list}
        q["superset_sql"] = compute_superset_sql_email(q)
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
