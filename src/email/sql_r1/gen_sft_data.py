#!/usr/bin/env python3
"""Generate SFT training data for email SQL-R1.

Converts GT DSL leaves to SQL WHERE clauses with CoT reasoning.
Aligned with ecommerce SQL-R1's 2_gen_sft_data.py methodology.

Usage:
    python gen_sft_data.py
"""
import json, os, re, sqlite3
import pandas as pd

DATA_DIR = "/home/sunzhen/LocalSearch/data/email"
STEP9_FILE = os.path.join(DATA_DIR, "train/queries_email_train_step9.jsonl")
DB_PATH = os.path.join(DATA_DIR, "train/emails_train.db")
OUT_PATH = "/home/sunzhen/SQL_R1_email/sft_train.jsonl"

SYSTEM_PROMPT = """You are a helpful email search SQL expert assistant.
The assistant first thinks about how to write the SQL WHERE clause by analyzing the user query and database schema, then provides the final WHERE clause.
The reasoning process and WHERE clause are enclosed within <think> </think> and <answer> </answer> tags respectively.
The answer must contain the SQL WHERE clause within ```sql ``` tags.

Database Schema:
CREATE TABLE emails (
    message_id TEXT PRIMARY KEY,
    "From" TEXT,
    "To" TEXT,
    Subject TEXT,
    Message TEXT,
    Date TEXT,
    year INTEGER,
    day INTEGER
);

For example:
<think>
The user is looking for emails from john.smith@enron.com about meeting schedules in 2001. Let me analyze:
1. Sender filter: "from john.smith@enron.com" → exact match on "From" field
2. Topic: "meeting schedules" → text search on Subject and Message
3. Year: "in 2001" → year = 2001
I'll combine with AND since all conditions must be satisfied.
</think>
<answer>
The query filters for sender, meeting-related content, and year 2001:
```sql
"From" = 'john.smith@enron.com' AND (Subject LIKE '%meeting%' OR Message LIKE '%meeting%') AND (Subject LIKE '%schedule%' OR Message LIKE '%schedule%') AND year = 2001
```
</answer>"""

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "not", "it", "its", "this", "that", "these", "those",
    "i", "my", "me", "we", "our", "you", "your", "he", "she", "they",
    "his", "her", "their", "what", "which", "who", "when", "where", "how",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "shall", "must", "need", "like",
    "looking", "find", "search", "want", "something", "email", "emails",
    "sent", "received", "about", "than", "more", "some", "any", "all",
}

FIELD_NAMES = {
    "From": "sender",
    "To": "recipient",
    "year": "year",
    "day": "day of month",
    "Subject": "subject",
    "Message": "message body",
}


def escape_sql(val):
    if val is None:
        return "NULL"
    return str(val).replace("'", "''")


def extract_keywords(text, max_kw=4):
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return keywords[:max_kw]


def leaf_to_sql(leaf):
    field = leaf["field"]
    is_pos = leaf["is_pos"]
    value = leaf["value"]

    if field in ("From", "To"):
        val = escape_sql(value["val"])
        if is_pos:
            return f'"{field}" = \'{val}\''
        else:
            return f'"{field}" != \'{val}\''

    elif field in ("year", "day"):
        if "lo" in value and "hi" in value:
            lo, hi = value["lo"], value["hi"]
            if is_pos:
                return f"{field} BETWEEN {lo} AND {hi}"
            else:
                return f"({field} < {lo} OR {field} > {hi})"
        else:
            op = value.get("op", "=")
            val = value["val"]
            if is_pos:
                return f"{field} {op} {val}"
            else:
                neg_ops = {"=": "!=", ">=": "<", "<=": ">", ">": "<=", "<": ">="}
                neg_op = neg_ops.get(op, "!=")
                return f"{field} {neg_op} {val}"

    elif field == "Subject":
        keywords = extract_keywords(str(value["val"]))
        if not keywords:
            keywords = str(value["val"]).split()[:3]
        if is_pos:
            parts = [f"Subject LIKE '%{escape_sql(kw)}%'" for kw in keywords]
            return " AND ".join(parts) if parts else "1=1"
        else:
            parts = [f"Subject NOT LIKE '%{escape_sql(kw)}%'" for kw in keywords]
            return " AND ".join(parts) if parts else "1=1"

    elif field == "Message":
        keywords = extract_keywords(str(value["val"]))
        if not keywords:
            keywords = str(value["val"]).split()[:3]
        if is_pos:
            parts = [f"Message LIKE '%{escape_sql(kw)}%'" for kw in keywords]
            return " AND ".join(parts) if parts else "1=1"
        else:
            parts = [f"Message NOT LIKE '%{escape_sql(kw)}%'" for kw in keywords]
            return " AND ".join(parts) if parts else "1=1"

    return "1=1"


def leaf_to_description(leaf):
    field = leaf["field"]
    is_pos = leaf["is_pos"]
    value = leaf["value"]
    fname = FIELD_NAMES.get(field, field)

    if field in ("From", "To"):
        if is_pos:
            return f'{fname}: "{value["val"]}" (exact match)'
        else:
            return f'{fname}: NOT "{value["val"]}"'
    elif field in ("year", "day"):
        if "lo" in value and "hi" in value:
            return f'{fname}: {"between" if is_pos else "NOT between"} {value["lo"]} and {value["hi"]}'
        else:
            return f'{fname} {value.get("op","=")} {value["val"]}'
    elif field in ("Subject", "Message"):
        return f'{fname} {"contains" if is_pos else "does not contain"} "{str(value["val"])[:60]}"'
    return str(value)


def build_where_clause(query_data):
    leaves = query_data["leaves"]
    inner_op = query_data.get("inner_op", "AND")
    outer_op = query_data.get("outer_op", "OR")

    if not leaves:
        return "1=1"

    grouped = {}
    for leaf in leaves:
        g = leaf["group"]
        if g not in grouped:
            grouped[g] = []
        grouped[g].append(leaf)

    group_clauses = []
    for g_id in sorted(grouped.keys()):
        conditions = [leaf_to_sql(leaf) for leaf in grouped[g_id]]
        conditions = [c for c in conditions if c and c != "1=1"]
        if not conditions:
            continue
        if len(conditions) == 1:
            group_clauses.append(conditions[0])
        else:
            inner = f" {inner_op} ".join(conditions)
            group_clauses.append(f"({inner})")

    if not group_clauses:
        return "1=1"
    if len(group_clauses) == 1:
        return group_clauses[0]
    return f" {outer_op} ".join(group_clauses)


def generate_think_and_answer(query_data, where_clause):
    nl_query = query_data["natural_language_query"]
    leaves = query_data["leaves"]
    inner_op = query_data.get("inner_op", "AND")
    outer_op = query_data.get("outer_op", "OR")
    n_groups = len(set(l["group"] for l in leaves))

    steps = [f'The user is searching for: "{nl_query}". Let me analyze the query components:']
    for i, leaf in enumerate(leaves):
        desc = leaf_to_description(leaf)
        steps.append(f"{i+1}. {desc}")

    if n_groups > 1:
        steps.append(f"These conditions form {n_groups} groups connected by {outer_op}, with {inner_op} within each group.")
    elif len(leaves) > 1:
        steps.append(f"I'll combine these conditions with {inner_op}.")

    think_text = "\n".join(steps)

    if n_groups > 1:
        explanation = f"The query combines {n_groups} condition groups with {outer_op}:"
    else:
        explanation = "The query filters emails based on the identified conditions:"

    answer_text = f"{explanation}\n```sql\n{where_clause}\n```"
    return think_text, answer_text


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run build_email_db.py --split train first.")
        return

    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    print(f"SQLite DB has {count} emails")

    queries = []
    with open(STEP9_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Loaded {len(queries)} training queries")

    good = 0
    total = 0
    hit_gt = 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fout:
        for q in queries:
            total += 1
            if not q.get("leaves"):
                continue

            where_clause = build_where_clause(q)
            if where_clause == "1=1":
                continue

            sql = f"SELECT message_id FROM emails WHERE {where_clause} LIMIT 20"
            try:
                rows = conn.execute(sql).fetchall()
                result_ids = [r[0] for r in rows]
            except Exception:
                continue

            if not result_ids:
                continue

            gt_ids = set(q.get("gt_document_ids") or [])
            has_hit = bool(gt_ids & set(result_ids))
            if has_hit:
                hit_gt += 1

            think_text, answer_text = generate_think_and_answer(q, where_clause)
            assistant_content = f"<think>\n{think_text}\n</think>\n<answer>\n{answer_text}\n</answer>"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q["natural_language_query"]},
                {"role": "assistant", "content": assistant_content},
            ]
            fout.write(json.dumps({"messages": messages, "query_id": q["query_id"]}, ensure_ascii=False) + "\n")
            good += 1

    conn.close()
    print(f"Generated {good}/{total} SFT samples ({good/total*100:.1f}%)")
    print(f"  Samples that hit GT: {hit_gt}/{good} ({hit_gt/good*100:.1f}%)")
    print(f"  Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
