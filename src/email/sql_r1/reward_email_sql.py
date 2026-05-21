#!/usr/bin/env python3
"""SQL-R1 style 4-component reward function for email retrieval.

Faithfully reproduces SQL-R1 paper's reward design (aligned with ecommerce version):
  Sf (Format):    +1 if valid <think>...</think><answer>...</answer> with ```sql block, else -1
  Se (Execution): +2 if SQL executes on SQLite without error, else -2
  Sr (Result):    +3 if any GT email in top-20 results, else -3
  Sl (Length):    0~1.5 bonus for longer think + concise answer (ONLY when Sr > 0)

Total = Sf + Se + Sr + Sl
"""
import os, random, re, sqlite3, time
import pandas as pd

DB_PATH = "/home/sunzhen/LocalSearch/data/email/train/emails_train.db"
TOP_K = 20
LIMIT_LENGTH = 2048

FORMAT_REWARD = 1
EXEC_REWARD = 2
RESULT_REWARD = 3

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DAY_RE = re.compile(r"\b(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{4}\b")

_conn = None


def _parse_date(date_str):
    if not date_str:
        return None, None
    y_m = YEAR_RE.search(date_str)
    d_m = DAY_RE.search(date_str)
    return (int(y_m.group(0)) if y_m else None,
            int(d_m.group(1)) if d_m else None)


def get_conn():
    global _conn
    if _conn is None:
        if os.path.exists(DB_PATH):
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            count = _conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            print(f"[reward_email_sql] SQLite loaded from file: {count} emails")
        else:
            parquet_path = "/home/sunzhen/LocalSearch/data/email/train/email_train.parquet"
            print(f"[reward_email_sql] Building in-memory DB from {parquet_path}")
            df = pd.read_parquet(parquet_path)
            _conn = sqlite3.connect(":memory:", check_same_thread=False)
            cur = _conn.cursor()
            cur.execute("""
                CREATE TABLE emails (
                    message_id TEXT PRIMARY KEY,
                    "From" TEXT, "To" TEXT, Subject TEXT, Message TEXT,
                    Date TEXT, year INTEGER, day INTEGER
                )
            """)
            for _, row in df.iterrows():
                y, d = _parse_date(row.get("Date") or "")
                cur.execute(
                    'INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?)',
                    (row["Message_ID"], row.get("From"), row.get("To"),
                     row.get("Subject"), row.get("Message"), row.get("Date"),
                     y, d),
                )
            _conn.commit()
            count = _conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            print(f"[reward_email_sql] In-memory DB loaded: {count} emails")
    return _conn


def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        processed_str = solution_str

    matches = re.findall(r"<answer>(.*?)</answer>", processed_str, re.DOTALL)
    if matches:
        return matches[-1].strip(), processed_str
    return None, processed_str


def extract_think(processed_str):
    matches = re.findall(r"<think>(.*?)</think>", processed_str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_sql_from_answer(answer_text):
    matches = re.findall(r"```sql\s*(.*?)\s*```", answer_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return answer_text.strip()


def validate_response_structure(processed_str):
    think_start = processed_str.count("<think>")
    think_end = processed_str.count("</think>")
    answer_start = processed_str.count("<answer>")
    answer_end = processed_str.count("</answer>")

    has_sql_block = "```sql" in processed_str and "```" in processed_str[processed_str.find("```sql")+6:]

    if think_start == 1 and think_end == 1 and answer_start == 1 and answer_end == 1:
        p_ts = processed_str.find("<think>")
        p_te = processed_str.find("</think>")
        p_as = processed_str.find("<answer>")
        p_ae = processed_str.find("</answer>")
        return (p_ts < p_te < p_as < p_ae) and has_sql_block

    if think_start == 0 and think_end == 1 and answer_start == 1 and answer_end == 1:
        p_te = processed_str.find("</think>")
        p_as = processed_str.find("<answer>")
        p_ae = processed_str.find("</answer>")
        return (p_te < p_as < p_ae) and has_sql_block

    return False


def execute_where(where_clause):
    conn = get_conn()
    where_clause = where_clause.strip().rstrip(";")
    if not where_clause or where_clause == "1=1":
        return None

    sql = f"SELECT message_id FROM emails WHERE {where_clause} LIMIT {TOP_K}"
    try:
        rows = conn.execute(sql).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return None


def compute_score(solution_str, ground_truth, data_source, extra_info=None):
    """verl reward entry point. Parameter order matches ecommerce SQL-R1."""
    gt_ids = ground_truth.get("gt_email_ids", [])
    if hasattr(gt_ids, "tolist"):
        gt_ids = gt_ids.tolist()
    gt_set = set(gt_ids)

    do_print = random.randint(1, 32) == 1

    answer_text, processed_str = extract_solution(solution_str)
    think_text = extract_think(processed_str)
    format_ok = validate_response_structure(processed_str)

    # Sf: Format reward
    sf = FORMAT_REWARD if format_ok else -FORMAT_REWARD

    # Se: Execution reward
    se = 0.0
    ranked = []
    pred_sql = ""
    if format_ok and answer_text:
        pred_sql = extract_sql_from_answer(answer_text)
        result = execute_where(pred_sql)
        if result is not None:
            se = EXEC_REWARD
            ranked = result
        else:
            se = -EXEC_REWARD
    elif format_ok and not answer_text:
        se = -EXEC_REWARD

    # Sr: Result reward (binary hit — aligned with SQL-R1 paper)
    # Special case: empty GT means no matching emails exist.
    # Returning empty results = correct → full reward.
    # Returning non-empty results = incorrect → penalty.
    sr = 0.0
    if not gt_set:
        sr = RESULT_REWARD if len(ranked) == 0 else -RESULT_REWARD
    elif ranked:
        hit = any(mid in gt_set for mid in ranked)
        sr = RESULT_REWARD if hit else -RESULT_REWARD
    elif se < 0:
        sr = -RESULT_REWARD

    # Sl: Length reward (SQL-R1 exact formula, only when result is correct)
    sl = 0.0
    if format_ok and sr > 0:
        pos_length = len(think_text) + len(answer_text)
        if pos_length <= LIMIT_LENGTH:
            length_sub_score = pos_length / LIMIT_LENGTH * 0.5
        else:
            length_sub_score = 0.5
        if answer_text and pred_sql:
            sql_in_answer_sub_score = len(pred_sql) / max(len(answer_text), 1)
        else:
            sql_in_answer_sub_score = 0.0
        sl = length_sub_score + sql_in_answer_sub_score

    total = sf + se + sr + sl

    if do_print:
        print(f"{'='*60}")
        print(f"Solution: {solution_str[:300]}...")
        print(f"SQL: {pred_sql[:200]}")
        print(f"Ranked: {ranked[:5]}")
        print(f"GT: {list(gt_set)[:5]}")
        print(f"Sf={sf:.1f} Se={se:.1f} Sr={sr:.1f} Sl={sl:.2f} Total={total:.2f}")
        print(f"{'='*60}")

    return total
