"""Convert product/email parquet to pyserini JSONL format for BM25 indexing."""
import json
import sys
import os
import pandas as pd

def convert_ecommerce(parquet_path, output_dir):
    df = pd.read_parquet(parquet_path)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "corpus.jsonl")

    with open(jsonl_path, "w") as f:
        for _, row in df.iterrows():
            text = row["text"]
            lines = text.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            doc = {
                "id": row["product_id"],
                "contents": f"{title}\n{body}",
                "title": title,
                "text": body,
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(df)} docs to {jsonl_path}")
    return jsonl_path


def convert_email(parquet_path, output_dir):
    df = pd.read_parquet(parquet_path)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "corpus.jsonl")

    with open(jsonl_path, "w") as f:
        for _, row in df.iterrows():
            subject = str(row.get("Subject", ""))
            message = str(row.get("Message", ""))
            from_addr = str(row.get("From", ""))
            to_addr = str(row.get("To", ""))
            date = str(row.get("Date", ""))
            body = f"From: {from_addr}\nTo: {to_addr}\nDate: {date}\n{message}"
            doc = {
                "id": row["Message_ID"],
                "contents": f"{subject}\n{body}",
                "title": subject,
                "text": body,
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(df)} docs to {jsonl_path}")
    return jsonl_path


if __name__ == "__main__":
    mode = sys.argv[1]  # "ecommerce" or "email"
    parquet = sys.argv[2]
    outdir = sys.argv[3]

    if mode == "ecommerce":
        convert_ecommerce(parquet, outdir)
    elif mode == "email":
        convert_email(parquet, outdir)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
