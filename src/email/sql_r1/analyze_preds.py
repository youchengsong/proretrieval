#!/usr/bin/env python3
"""Analyze email SQL-R1 predictions to find failure modes."""
import json, re, sqlite3
import pandas as pd

PRED_FILE = "/home/sunzhen/LocalSearch/data/email/result/b3_sql_r1_email_step100.json"
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

print("Building DB...")
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
n = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
print(f"  {n} emails loaded")

with open(PRED_FILE) as f:
    preds = json.load(f)

no_sql = 0
exec_error = 0
exec_empty = 0
has_result = 0
error_msgs = []

for qid, v in preds.items():
    sql = v["sql"]
    if not sql:
        no_sql += 1
        continue
    sql = sql.strip().rstrip(";")
    query = f"SELECT message_id FROM emails WHERE {sql} LIMIT 20"
    try:
        rows = conn.execute(query).fetchall()
        if rows:
            has_result += 1
        else:
            exec_empty += 1
    except Exception as e:
        exec_error += 1
        error_msgs.append((qid, sql[:200], str(e)[:150]))

print(f"\n=== Prediction Analysis ===")
print(f"Total: {len(preds)}")
print(f"No SQL extracted (format fail): {no_sql} ({no_sql/len(preds)*100:.1f}%)")
print(f"SQL exec error: {exec_error} ({exec_error/len(preds)*100:.1f}%)")
print(f"SQL exec OK but empty result: {exec_empty} ({exec_empty/len(preds)*100:.1f}%)")
print(f"Has results: {has_result} ({has_result/len(preds)*100:.1f}%)")

print(f"\n=== Sample Exec Errors (first 10) ===")
for qid, sql, err in error_msgs[:10]:
    print(f"  QID={qid}")
    print(f"    SQL: {sql}")
    print(f"    ERR: {err}")
    print()

print(f"\n=== Sample Format Fails (first 5) ===")
count = 0
for qid, v in preds.items():
    if not v["sql"]:
        print(f"  QID={qid}")
        print(f"    Gen: {v['generated'][:300]}")
        print()
        count += 1
        if count >= 5:
            break

# Error type distribution
err_types = {}
for _, _, err in error_msgs:
    key = err.split(":")[0] if ":" in err else err[:50]
    err_types[key] = err_types.get(key, 0) + 1
print(f"\n=== Error Type Distribution ===")
for k, v in sorted(err_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
