#!/usr/bin/env python3
"""Evaluate email SQL-R1 model on test queries.

Generates SQL WHERE clauses via vLLM, executes against SQLite, computes metrics.

Usage:
    python eval_email_sql.py --model <path_to_merged_model>
    python eval_email_sql.py --model <path> --limit 100  # quick test
"""
import argparse, json, os, re, sqlite3
import pandas as pd
from vllm import LLM, SamplingParams

DATA_DIR = "/home/sunzhen/LocalSearch/data/email"
TEST_FILE = os.path.join(DATA_DIR, "test/queries_email_test_step9.jsonl")
EMAIL_PARQUET = os.path.join(DATA_DIR, "test/email_test.parquet")
OUTPUT_DIR = os.path.join(DATA_DIR, "result")
TOP_K = 20

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DAY_RE = re.compile(r"\b(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{4}\b")

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

Rules:
1. Use exact match (=) for "From" and "To" fields with full email addresses.
2. Use LIKE with % wildcards for Subject and Message text search (case-insensitive).
3. Use exact comparisons (=, >, <, >=, <=, BETWEEN) for year and day.
4. Use AND/OR/NOT to combine conditions as needed.
5. When the query mentions email content or topics, use Message LIKE '%keyword%' or Subject LIKE '%keyword%'.

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
</answer>

Another example:
<think>
The user wants emails to sara.shackleton@enron.com or emails about gas pipeline contracts. Let me analyze:
1. Recipient: "to sara.shackleton@enron.com" → exact match on "To"
2. Topic: "gas pipeline contracts" → text search
These are OR conditions since either one should match.
</think>
<answer>
The query uses OR to combine recipient filter and topic search:
```sql
"To" = 'sara.shackleton@enron.com' OR (Subject LIKE '%gas%' AND Subject LIKE '%pipeline%') OR (Message LIKE '%gas%' AND Message LIKE '%pipeline%' AND Message LIKE '%contract%')
```
</answer>"""


def parse_date(date_str):
    if not date_str:
        return None, None
    y_m = YEAR_RE.search(date_str)
    d_m = DAY_RE.search(date_str)
    return (int(y_m.group(0)) if y_m else None,
            int(d_m.group(1)) if d_m else None)


def build_db():
    print(f"Building SQLite DB from {EMAIL_PARQUET}")
    df = pd.read_parquet(EMAIL_PARQUET)
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE emails (
            message_id TEXT PRIMARY KEY,
            "From" TEXT, "To" TEXT, Subject TEXT, Message TEXT,
            Date TEXT, year INTEGER, day INTEGER
        )
    """)
    for _, row in df.iterrows():
        y, d = parse_date(row.get("Date") or "")
        cur.execute(
            'INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?)',
            (row["Message_ID"], row.get("From"), row.get("To"),
             row.get("Subject"), row.get("Message"), row.get("Date"), y, d),
        )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    print(f"  {n} emails loaded")
    return conn


def extract_sql(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return None
    answer = matches[-1].strip()
    sql_matches = re.findall(r"```sql\s*(.*?)\s*```", answer, re.DOTALL)
    if sql_matches:
        return sql_matches[-1].strip()
    return None


def execute_where(conn, where_clause, top_k=20):
    where_clause = where_clause.strip().rstrip(";")
    if not where_clause:
        return []
    sql = f"SELECT message_id FROM emails WHERE {where_clause} LIMIT {top_k}"
    try:
        rows = conn.execute(sql).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="b3_sql_r1_email")
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    queries = []
    with open(TEST_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    if args.limit:
        queries = queries[:args.limit]
    print(f"Loaded {len(queries)} test queries")

    conn = build_db()

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    tokenizer = llm.get_tokenizer()
    formatted = []
    for q in queries:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["natural_language_query"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    print("Generating SQL WHERE clauses...")
    outputs = llm.generate(formatted, sampling_params)

    predictions = {}
    format_ok = exec_ok = 0
    for q, output in zip(queries, outputs):
        qid = q["query_id"]
        generated = output.outputs[0].text
        sql = extract_sql(generated)
        if sql:
            format_ok += 1
            ranked = execute_where(conn, sql, TOP_K)
            if ranked:
                exec_ok += 1
        else:
            ranked = []
        predictions[qid] = {
            "ranked": ranked,
            "sql": sql or "",
            "generated": generated[:500],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{args.output_name}.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {output_file}")

    hit1 = hit3 = mrr = n = 0
    for q in queries:
        qid = q["query_id"]
        gt = set(q.get("gt_document_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, {}).get("ranked", [])[:TOP_K]
        n += 1
        if ranked and ranked[0] in gt:
            hit1 += 1
        if any(r in gt for r in ranked[:3]):
            hit3 += 1
        for j, r in enumerate(ranked):
            if r in gt:
                mrr += 1.0 / (j + 1)
                break

    print(f"\nEmail SQL-R1 Results (N={n}):")
    print(f"  Hit@1:  {hit1/n:.4f} ({hit1}/{n})")
    print(f"  Hit@3:  {hit3/n:.4f} ({hit3}/{n})")
    print(f"  MRR:    {mrr/n:.4f}")
    print(f"  Format OK: {format_ok}/{len(queries)} ({format_ok/len(queries)*100:.1f}%)")
    print(f"  Exec OK:   {exec_ok}/{len(queries)} ({exec_ok/len(queries)*100:.1f}%)")

    conn.close()


if __name__ == "__main__":
    main()
