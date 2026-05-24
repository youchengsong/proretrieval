"""
RL 训练的 Reward 函数 (电商)
不依赖外部检索服务，直接在内存中执行 DSL：
  - SQLite 执行结构化条件
  - numpy 做向量 top-K
"""

import json, re, sqlite3, os, traceback
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
DATA_DIR = "/root/autodl-tmp/data/ecommerce"
TOP_K = 20

import time as _time
_reward_call_count = 0
_reward_history = []
_reward_start_time = None
_MONITOR_WINDOW = 200
_ABORT_THRESHOLD = 0.95
_abort_strike = 0

def _reward_log(msg):
    ts = _time.strftime("%H:%M:%S")
    print(f"[reward {ts}] {msg}", flush=True)


def _init():
    global _DB_CONN, _T_IDS, _T_MAT, _I_IDS, _I_MAT, _INITIALIZED
    if _INITIALIZED:
        return

    products_file = os.path.join(DATA_DIR, "train/products_train.parquet")
    if not os.path.exists(products_file):
        products_file = os.path.join(DATA_DIR, "products_test.json")

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
        tf_path = os.path.join(DATA_DIR, f"train/product_text_features_{split}.parquet")
        if os.path.exists(tf_path):
            tf = pd.read_parquet(tf_path)
            _T_IDS = tf["product_id"].values
            _T_MAT = np.stack(tf["text_features"].values).astype(np.float32)
            _T_MAT = _T_MAT / (np.linalg.norm(_T_MAT, axis=1, keepdims=True) + 1e-9)
            break

    for split in ["train", "test"]:
        if_path = os.path.join(DATA_DIR, f"train/product_image_features_{split}.parquet")
        if os.path.exists(if_path):
            imf = pd.read_parquet(if_path)
            _I_IDS = imf["product_id"].values
            _I_MAT = np.stack(imf["image_features"].values).astype(np.float32)
            _I_MAT = _I_MAT / (np.linalg.norm(_I_MAT, axis=1, keepdims=True) + 1e-9)
            break

    _INITIALIZED = True
    print(f"[reward_dsl] Initialized: products loaded, text={_T_MAT.shape if _T_MAT is not None else None}, image={_I_MAT.shape if _I_MAT is not None else None}")


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
        except Exception:
            _embed_retry_count += 1
            if attempt < 2:
                import time; time.sleep(1)
    _reward_log(f"WARN: {endpoint} failed after 3 retries (total retries={_embed_retry_count})")
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

    return {}


def _topk(query_vec, feat_mat, feat_ids, k):
    scores = feat_mat @ query_vec
    k = min(k, len(scores))
    if k == 0:
        return []
    idx = np.argpartition(scores, -k)[-k:]
    return [feat_ids[i] for i in idx]


def _execute_dsl(dsl):
    try:
        _init()

        if not isinstance(dsl, dict):
            return None, "dsl_not_dict"

        sql_body = (dsl.get("sql") or "").strip()
        retrieval_list = dsl.get("retrieval_list") or []

        if not isinstance(retrieval_list, list):
            retrieval_list = []

        if not sql_body:
            return None, "empty_sql"

        text_idx, image_idx = 0, 0
        placeholder_ids = {}

        text_queries = []
        image_queries = []
        for r in retrieval_list:
            if not isinstance(r, dict):
                continue
            rtype = r.get("type")
            query = r.get("query")
            if not isinstance(query, str) or not query:
                continue
            if rtype == "text":
                text_queries.append(query)
            elif rtype == "image":
                image_queries.append(query)

        text_vecs = _encode_queries(text_queries, "text") if text_queries else {}
        image_vecs = _encode_queries(image_queries, "image") if image_queries else {}

        for r in retrieval_list:
            if not isinstance(r, dict):
                continue
            rtype = r.get("type")
            q = r.get("query") or ""
            if not isinstance(q, str):
                q = str(q)

            if rtype == "text":
                ph = f"<text_{text_idx}>"; text_idx += 1
                vec = text_vecs.get(q)
                if vec is not None and _T_MAT is not None:
                    placeholder_ids[ph] = _topk(vec, _T_MAT, _T_IDS, TOP_K)
                else:
                    placeholder_ids[ph] = []
            elif rtype == "image":
                ph = f"<image_{image_idx}>"; image_idx += 1
                vec = image_vecs.get(q)
                if vec is not None and _I_MAT is not None:
                    placeholder_ids[ph] = _topk(vec, _I_MAT, _I_IDS, TOP_K)
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

        cur = _DB_CONN.cursor()
        cur.execute(full_sql)
        result_ids = [row[0] for row in cur.fetchall()]
        return result_ids, None

    except Exception as e:
        return None, str(e)[:200]


def _parse_gt(ground_truth):
    try:
        if isinstance(ground_truth, (list, np.ndarray)):
            return list(ground_truth)
        if isinstance(ground_truth, str):
            parsed = json.loads(ground_truth)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return parsed.get("ground_truth", [])
            return []
        if isinstance(ground_truth, dict):
            return ground_truth.get("ground_truth", [])
    except Exception:
        pass
    return []


def compute_score(data_source, solution_str, ground_truth, extra_info):
    global _reward_call_count, _reward_start_time
    _reward_call_count += 1
    if _reward_start_time is None:
        _reward_start_time = _time.time()

    try:
        score = _compute_score_inner(solution_str, ground_truth)
    except Exception as e:
        _reward_log(f"UNEXPECTED ERROR in compute_score: {e}")
        traceback.print_exc()
        score = 0.0

    _record_and_check(score, "", solution_str)
    return score


def _compute_score_inner(solution_str, ground_truth):
    if not isinstance(solution_str, str):
        return -1.0

    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not m:
        return -1.0

    try:
        dsl = json.loads(m.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return -1.0

    if not isinstance(dsl, dict) or "sql" not in dsl:
        return -1.0

    gt_ids = _parse_gt(ground_truth)
    result_ids, error = _execute_dsl(dsl)

    if result_ids is None:
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
    return max(0.0, min(1.0, score))


def _record_and_check(score, reason, solution_str):
    global _abort_strike
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
        bad = (neg1 / r_len >= _ABORT_THRESHOLD or
               zero / r_len >= _ABORT_THRESHOLD or
               (neg1 + zero) / r_len >= _ABORT_THRESHOLD)

        if bad:
            _abort_strike += 1
            _reward_log(
                f"!!! WARNING (strike {_abort_strike}): "
                f"最近{r_len}个reward中{neg1}个-1+{zero}个0（合计{(neg1+zero)/r_len:.0%}）"
            )
            if _abort_strike <= 3:
                _reward_log(f"最近一个输出: {str(solution_str)[:300]}")
            _reward_history.clear()
        else:
            _abort_strike = 0
