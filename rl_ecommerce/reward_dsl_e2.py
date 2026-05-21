"""
E2 Action Space Ablation reward function.
Based on reward_dsl.py but filters DSL by ACTION_SPACE env var before execution.

ACTION_SPACE values:
  - sql_only:  strip retrieval_list
  - sql_text:  strip image retrievals
  - text_only: strip sql conditions, keep only text retrievals
  - sql_image: strip text retrievals
  - full:      no filtering (same as original)
"""

import json, re, sqlite3, os
import numpy as np
import pandas as pd
from collections import Counter

_DB_CONN = None
_T_IDS = None
_T_MAT = None
_I_IDS = None
_I_MAT = None
_TEXT_VEC_MAP = None
_IMAGE_VEC_MAP = None
_INITIALIZED = False

EMBED_SERVER_URL = "http://127.0.0.1:9123"
DATA_DIR = "/home/sunzhen/LocalSearch/data/ecommerce"
TOP_K = 20
ACTION_SPACE = os.environ.get("ACTION_SPACE", "full")

import time as _time
_reward_call_count = 0
_reward_history = []
_reward_start_time = None
_MONITOR_WINDOW = 2000
_ABORT_THRESHOLD = 0.99

def _reward_log(msg):
    ts = _time.strftime("%H:%M:%S")
    print(f"[reward_e2/{ACTION_SPACE} {ts}] {msg}", flush=True)


def _init():
    global _DB_CONN, _T_IDS, _T_MAT, _I_IDS, _I_MAT, _INITIALIZED
    if _INITIALIZED:
        return

    products_file = os.path.join(DATA_DIR, "train/products_train.parquet")
    if not os.path.exists(products_file):
        products_file = os.path.join(DATA_DIR, "test/products_test.json")

    _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)
    cur = _DB_CONN.cursor()
    cur.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            brand TEXT, color TEXT, category TEXT,
            price REAL, average_rating REAL, rating_number INTEGER
        )
    """)

    if products_file.endswith(".json"):
        with open(products_file) as f:
            products = json.load(f)
        for p in products:
            cur.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?)",
                (p["product_id"], p.get("brand"), p.get("color"),
                 p.get("category"), p.get("price"),
                 p.get("average_rating"), p.get("rating_number")))
    else:
        df = pd.read_parquet(products_file)
        for _, p in df.iterrows():
            cur.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?)",
                (p["product_id"], p.get("brand"), p.get("color"),
                 p.get("category"), p.get("price"),
                 p.get("average_rating"), p.get("rating_number")))
    _DB_CONN.commit()

    for split in ["train", "test"]:
        tf_path = os.path.join(DATA_DIR, f"{split}/product_text_features_{split}.parquet")
        if os.path.exists(tf_path):
            tf = pd.read_parquet(tf_path)
            _T_IDS = tf["product_id"].values
            _T_MAT = np.stack(tf["text_features"].values).astype(np.float32)
            _T_MAT = _T_MAT / (np.linalg.norm(_T_MAT, axis=1, keepdims=True) + 1e-9)
            break

    for split in ["train", "test"]:
        if_path = os.path.join(DATA_DIR, f"{split}/product_image_features_{split}.parquet")
        if os.path.exists(if_path):
            imf = pd.read_parquet(if_path)
            _I_IDS = imf["product_id"].values
            _I_MAT = np.stack(imf["image_features"].values).astype(np.float32)
            _I_MAT = _I_MAT / (np.linalg.norm(_I_MAT, axis=1, keepdims=True) + 1e-9)
            break

    _INITIALIZED = True
    _reward_log(f"Initialized: text={_T_MAT.shape if _T_MAT is not None else None}, image={_I_MAT.shape if _I_MAT is not None else None}")


_embed_retry_count = 0

def _call_embed_server(endpoint, queries):
    global _embed_retry_count
    import requests as _req
    for attempt in range(3):
        try:
            resp = _req.post(f"{EMBED_SERVER_URL}/{endpoint}", json={"queries": queries}, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return {}
            return data.get("vectors", {})
        except Exception as e:
            _embed_retry_count += 1
            if attempt < 2:
                import time; time.sleep(1)
    return {}


def _encode_queries(queries, query_type):
    global _TEXT_VEC_MAP, _IMAGE_VEC_MAP
    if query_type == "text":
        if _TEXT_VEC_MAP is None:
            _TEXT_VEC_MAP = {}
        uncached = [q for q in queries if q not in _TEXT_VEC_MAP]
        if uncached:
            vecs = _call_embed_server("encode_text", uncached)
            for q in uncached:
                if q in vecs:
                    _TEXT_VEC_MAP[q] = np.array(vecs[q], dtype=np.float32)
        return {q: _TEXT_VEC_MAP[q] for q in queries if q in _TEXT_VEC_MAP}
    elif query_type == "image":
        if _IMAGE_VEC_MAP is None:
            _IMAGE_VEC_MAP = {}
        uncached = [q for q in queries if q not in _IMAGE_VEC_MAP]
        if uncached:
            vecs = _call_embed_server("encode_image", uncached)
            for q in uncached:
                if q in vecs:
                    _IMAGE_VEC_MAP[q] = np.array(vecs[q], dtype=np.float32)
        return {q: _IMAGE_VEC_MAP[q] for q in queries if q in _IMAGE_VEC_MAP}


def _topk(query_vec, feat_mat, feat_ids, k):
    scores = feat_mat @ query_vec
    k = min(k, len(scores))
    if k == 0:
        return []
    idx = np.argpartition(scores, -k)[-k:]
    return [feat_ids[i] for i in idx]


def _filter_dsl(dsl):
    """Filter DSL based on ACTION_SPACE."""
    if ACTION_SPACE == "sql_only":
        dsl["retrieval_list"] = []
    elif ACTION_SPACE == "sql_text":
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r.get("type") == "text"]
    elif ACTION_SPACE == "text_only":
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r.get("type") == "text"]
    elif ACTION_SPACE == "sql_image":
        dsl["retrieval_list"] = [r for r in dsl.get("retrieval_list", []) if r.get("type") == "image"]
    return dsl


def _execute_dsl(dsl):
    _init()
    dsl = _filter_dsl(dsl)

    sql_body = dsl.get("sql", "").strip()
    retrieval_list = dsl.get("retrieval_list", [])

    if not sql_body:
        return None, "empty_sql"

    text_idx, image_idx = 0, 0
    placeholder_ids = {}

    text_queries = [r["query"] for r in retrieval_list if r.get("type") == "text" and r.get("query")]
    image_queries = [r["query"] for r in retrieval_list if r.get("type") == "image" and r.get("query")]

    text_vecs = _encode_queries(text_queries, "text") if text_queries else {}
    image_vecs = _encode_queries(image_queries, "image") if image_queries else {}

    for r in retrieval_list:
        q = r.get("query", "")
        if r["type"] == "text":
            ph = f"<text_{text_idx}>"; text_idx += 1
            vec = text_vecs.get(q)
            if vec is not None and _T_MAT is not None:
                ids = _topk(vec, _T_MAT, _T_IDS, TOP_K)
                placeholder_ids[ph] = ids
            else:
                placeholder_ids[ph] = []
        elif r["type"] == "image":
            ph = f"<image_{image_idx}>"; image_idx += 1
            vec = image_vecs.get(q)
            if vec is not None and _I_MAT is not None:
                ids = _topk(vec, _I_MAT, _I_IDS, TOP_K)
                placeholder_ids[ph] = ids
            else:
                placeholder_ids[ph] = []

    exec_sql = sql_body
    for ph, ids in placeholder_ids.items():
        if ids:
            id_list = ", ".join(f"'{pid}'" for pid in ids)
            exec_sql = exec_sql.replace(ph, f"({id_list})")
        else:
            exec_sql = exec_sql.replace(ph, "('__EMPTY__')")

    if not exec_sql.upper().startswith("SELECT"):
        full_sql = f"SELECT DISTINCT product_id FROM products {exec_sql}"
    else:
        full_sql = exec_sql

    try:
        cur = _DB_CONN.cursor()
        cur.execute(full_sql)
        result_ids = [row[0] for row in cur.fetchall()]
        return result_ids, None
    except Exception as e:
        return None, str(e)


def compute_score(data_source, solution_str, ground_truth, extra_info):
    global _reward_call_count, _reward_start_time
    _reward_call_count += 1
    if _reward_start_time is None:
        _reward_start_time = _time.time()

    if solution_str is None or not isinstance(solution_str, str):
        _record_and_check(-1.0, "not_str", solution_str)
        return -1.0

    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not m:
        _record_and_check(-1.0, "no_answer_tag", solution_str)
        return -1.0

    try:
        dsl = json.loads(m.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        _record_and_check(-1.0, "json_parse_fail", solution_str)
        return -1.0

    if "sql" not in dsl:
        _record_and_check(-1.0, "no_sql_field", solution_str)
        return -1.0

    if isinstance(ground_truth, (list, np.ndarray)):
        gt_ids = list(ground_truth)
    elif isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            gt_ids = parsed.get("ground_truth", parsed) if isinstance(parsed, dict) else parsed
        except (json.JSONDecodeError, ValueError):
            gt_ids = []
    elif isinstance(ground_truth, dict):
        gt_ids = ground_truth.get("ground_truth", [])
    else:
        gt_ids = []
    if isinstance(gt_ids, str):
        gt_ids = json.loads(gt_ids)

    result_ids, error = _execute_dsl(dsl)

    if result_ids is None:
        _record_and_check(0.0, f"exec_fail:{error}", solution_str)
        return 0.0

    gt_set = set(str(x) for x in gt_ids)
    result_set = set(str(x) for x in result_ids)

    if len(gt_set) == 0:
        coverage = 1.0
    else:
        coverage = len(gt_set & result_set) / len(gt_set)

    if len(result_set) == 0:
        noise = 0.0
    else:
        noise = len(result_set - gt_set) / len(result_set)

    counter = Counter(str(x) for x in result_ids)
    dup_count = sum(max(counter[x] - 1, 0) for x in gt_set if x in counter)
    if len(result_set) == 0:
        redundancy = 0.0
    else:
        redundancy = dup_count / len(result_set)

    score = coverage - 0.5 * noise - 0.5 * redundancy
    score = max(0.0, min(1.0, score))
    _record_and_check(score, "ok", solution_str)
    return score


def _record_and_check(score, reason, solution_str):
    _reward_history.append(score)
    n = _reward_call_count

    if n % 50 == 0:
        recent = _reward_history[-200:] if len(_reward_history) >= 200 else _reward_history
        r_len = len(recent)
        neg1 = sum(1 for r in recent if r == -1.0)
        zero = sum(1 for r in recent if r == 0.0)
        pos = sum(1 for r in recent if r > 0.0)
        avg = sum(recent) / r_len
        elapsed = _time.time() - _reward_start_time
        _reward_log(
            f"#{n} | avg={avg:.3f} | "
            f"format_err(-1)={neg1}/{r_len} exec_fail(0)={zero}/{r_len} valid(>0)={pos}/{r_len} | "
            f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s"
        )

    if len(_reward_history) >= _MONITOR_WINDOW:
        recent = _reward_history[-_MONITOR_WINDOW:]
        r_len = len(recent)
        neg1 = sum(1 for r in recent if r == -1.0)
        zero = sum(1 for r in recent if r == 0.0)
        if neg1 / r_len >= _ABORT_THRESHOLD:
            _reward_log(f"!!! ABORT: {neg1}/{r_len} format errors !!!")
            import sys; sys.exit(1)
        if zero / r_len >= _ABORT_THRESHOLD:
            _reward_log(f"!!! ABORT: {zero}/{r_len} exec failures !!!")
            import sys; sys.exit(1)
        if (neg1 + zero) / r_len >= _ABORT_THRESHOLD:
            _reward_log(f"!!! ABORT: {neg1+zero}/{r_len} failures !!!")
            import sys; sys.exit(1)
