#!/usr/bin/env python3
"""Build SQLite database from email parquet for SQL-R1 training/evaluation.

Usage:
    python build_email_db.py --split train  # -> emails_train.db
    python build_email_db.py --split test   # -> emails_test.db
"""
import argparse, json, os, re, sqlite3
import pandas as pd

DATA_DIR = "/home/sunzhen/LocalSearch/data/email"

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DAY_RE = re.compile(r"\b(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{4}\b")


def parse_date(date_str):
    if not date_str:
        return None, None
    y_m = YEAR_RE.search(date_str)
    d_m = DAY_RE.search(date_str)
    return (int(y_m.group(0)) if y_m else None,
            int(d_m.group(1)) if d_m else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parquet_path = os.path.join(args.data_dir, args.split, f"email_{args.split}.parquet")
    if not os.path.exists(parquet_path):
        json_path = os.path.join(args.data_dir, args.split, f"email_{args.split}.json")
        if os.path.exists(json_path):
            parquet_path = json_path
        else:
            raise FileNotFoundError(f"No data file found for split={args.split}")

    output_path = args.output or os.path.join(args.data_dir, args.split, f"emails_{args.split}.db")

    print(f"Reading {parquet_path}")
    if parquet_path.endswith(".parquet"):
        df = pd.read_parquet(parquet_path)
    else:
        with open(parquet_path) as f:
            df = pd.DataFrame(json.load(f))

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS emails")
    cur.execute("""
        CREATE TABLE emails (
            message_id TEXT PRIMARY KEY,
            "From" TEXT,
            "To" TEXT,
            Subject TEXT,
            Message TEXT,
            Date TEXT,
            year INTEGER,
            day INTEGER
        )
    """)

    count = 0
    for _, row in df.iterrows():
        y, d = parse_date(row.get("Date") or "")
        cur.execute(
            'INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?)',
            (row["Message_ID"], row.get("From"), row.get("To"),
             row.get("Subject"), row.get("Message"), row.get("Date"),
             y, d),
        )
        count += 1

    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    print(f"Created {output_path}: {n} emails")

    print("\nSample rows:")
    for row in conn.execute('SELECT message_id, "From", "To", Subject, year FROM emails LIMIT 3'):
        print(f"  {row}")

    print("\nSchema:")
    for row in conn.execute("PRAGMA table_info(emails)"):
        print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    main()
