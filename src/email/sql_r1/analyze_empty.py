#!/usr/bin/env python3
"""Analyze empty-result SQL queries."""
import json, re, sqlite3
import pandas as pd

PRED_FILE = "/home/sunzhen/LocalSearch/data/email/result/b3_sql_r1_email_step100.json"
TEST_FILE = "/home/sunzhen/LocalSearch/data/email/test/queries_email_test_step9.jsonl"
EMAIL_PARQUET = "/home/sunzhen/LocalSearch/data/email/test/email_test.parquet"

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DAY_RE = re.compile(r"\b(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{4}\b")

def parse_date(date_str):
    if not date_str:
        return None, None
    y_m = YEAR_RE.search(date_str)
    d_m = DAY_RE.search(date_str)
    return (int(y_m.group(0)) if y_m else None,
            int(d_m.group(1)) if d_m else None)

df = pd.read_parquet(EMAIL_PARQUET)
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("""CREATE TABLE emails (
    message_id TEXT PRIMARY KEY,
    "From" TEXT, "To" TEXT, Subject TEXT, Message TEXT,
    Date TEXT, year INTEGER, day INTEGER
)""")
for _, row in df.iterrows():
    y, d = parse_date(row.get("Date") or "")
    cur.execute('INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?)',
        (row["Message_ID"], row.get("From"), row.get("To"),
         row.get("Subject"), row.get("Message"), row.get("Date"), y, d))
conn.commit()

queries = {}
with open(TEST_FILE) as f:
    for line in f:
        q = json.loads(line)
        queries[q["query_id"]] = q

with open(PRED_FILE) as f:
    preds = json.load(f)

# Analyze empty-result queries
empty_with_gt = 0
empty_no_gt = 0
empty_sql_samples = []
like_counts = []

for qid, v in preds.items():
    sql = v["sql"]
    if not sql:
        continue
    if v["ranked"]:
        continue
    gt = queries.get(qid, {}).get("gt_document_ids") or []
    if gt:
        empty_with_gt += 1
        n_like = sql.lower().count("like")
        n_and = sql.lower().count(" and ")
        like_counts.append(n_like)
        if len(empty_sql_samples) < 8:
            empty_sql_samples.append((qid, sql[:300], len(gt), n_like, n_and))
    else:
        empty_no_gt += 1

print(f"Empty result queries with GT: {empty_with_gt}")
print(f"Empty result queries without GT (correct!): {empty_no_gt}")
print(f"Avg LIKE count in empty-with-GT: {sum(like_counts)/max(len(like_counts),1):.1f}")

# Compare with queries that got results
result_like_counts = []
for qid, v in preds.items():
    if v["ranked"] and v["sql"]:
        n_like = v["sql"].lower().count("like")
        result_like_counts.append(n_like)
print(f"Avg LIKE count in has-result: {sum(result_like_counts)/max(len(result_like_counts),1):.1f}")

print(f"\n=== Sample empty-with-GT queries ===")
for qid, sql, n_gt, n_like, n_and in empty_sql_samples:
    nl = queries[qid]["natural_language_query"][:150]
    print(f"QID={qid} (GT={n_gt}, LIKE={n_like}, AND={n_and})")
    print(f"  Query: {nl}")
    print(f"  SQL: {sql}")
    print()

# Check: relax one condition, does it return results?
print("=== Relaxation test (remove last AND clause) ===")
for qid, sql, n_gt, n_like, n_and in empty_sql_samples[:3]:
    parts = re.split(r'\bAND\b', sql, flags=re.IGNORECASE)
    if len(parts) > 1:
        relaxed = " AND ".join(parts[:-1])
        try:
            rows = conn.execute(f"SELECT COUNT(*) FROM emails WHERE {relaxed}").fetchone()[0]
            print(f"  QID={qid}: full=0, relaxed(-1 AND)={rows}")
        except:
            print(f"  QID={qid}: relaxed query failed")

# Level distribution of failures
level_stats = {"1": [0,0], "2": [0,0], "3": [0,0]}
for qid, v in preds.items():
    q = queries.get(qid, {})
    level = str(q.get("level", "?"))
    if level not in level_stats:
        continue
    has_gt = bool(q.get("gt_document_ids"))
    if not has_gt:
        continue
    ranked = v.get("ranked", [])
    gt_set = set(q.get("gt_document_ids", []))
    hit = any(r in gt_set for r in ranked[:1])
    level_stats[level][0] += 1
    if hit:
        level_stats[level][1] += 1
print(f"\n=== Hit@1 by Level ===")
for level, (total, hits) in sorted(level_stats.items()):
    if total > 0:
        print(f"  L{level}: {hits}/{total} = {hits/total:.4f}")
